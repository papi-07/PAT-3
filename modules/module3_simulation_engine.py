"""
modules/module3_simulation_engine.py
Monte Carlo validator + SDE solver for Module 3.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────
# SDE solver (Euler-Maruyama)
# ─────────────────────────────────────────────────────────────

def euler_maruyama_ou(
    theta: float,
    mu: float,
    sigma: float,
    x0: float,
    dt: float = cfg.SDE_DT,
    t_total: float = cfg.SDE_T_TOTAL,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Solve a scalar Ornstein-Uhlenbeck SDE via Euler-Maruyama:
      dX = θ(μ - X)dt + σ dW

    Returns
    -------
    trajectory : 1-D np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(t_total / dt)
    x = np.zeros(n)
    x[0] = x0
    sqrt_dt = np.sqrt(dt)
    for i in range(n - 1):
        x[i + 1] = x[i] + theta * (mu - x[i]) * dt + sigma * rng.normal() * sqrt_dt
    return x


def euler_maruyama_coupled(
    theta_g: float = cfg.SDE_THETA_G,
    mu_g: float    = cfg.SDE_MU_G,
    sigma_g: float = cfg.SDE_SIGMA_G,
    theta_i: float = cfg.SDE_THETA_I,
    mu_i: float    = cfg.SDE_MU_I,
    sigma_i: float = cfg.SDE_SIGMA_I,
    alpha: float   = cfg.SDE_ALPHA,
    dt: float      = cfg.SDE_DT,
    t_total: float = cfg.SDE_T_TOTAL,
    g0: float      = None,
    i0: float      = None,
    rng: np.random.Generator = None,
):
    """
    Coupled glucose-inflammation SDE (Euler-Maruyama).

    Returns
    -------
    G : glucose trajectory
    I : inflammation trajectory
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(t_total / dt)
    G = np.zeros(n)
    I = np.zeros(n)
    G[0] = g0 if g0 is not None else mu_g
    I[0] = i0 if i0 is not None else mu_i
    sqrt_dt = np.sqrt(dt)
    for k in range(n - 1):
        G[k + 1] = (G[k]
                    + theta_g * (mu_g - G[k]) * dt
                    + sigma_g * rng.normal() * sqrt_dt)
        G[k + 1] = np.clip(G[k + 1], 40, 500)

        I[k + 1] = (I[k]
                    + theta_i * (mu_i - I[k]) * dt
                    + sigma_i * rng.normal() * sqrt_dt
                    + alpha * G[k] * dt)
        I[k + 1] = max(I[k + 1], 0.0)
    return G, I


def generate_instability_trajectories(
    n_patients: int = 200,
    seed: int = cfg.RANDOM_SEED,
) -> np.ndarray:
    """
    Generate glycemic instability scores over time for *n_patients* via the SDE.

    Returns
    -------
    trajectories : (n_patients, n_steps) array  (normalised 0–1)
    """
    rng = np.random.default_rng(seed)
    n_steps = int(cfg.SDE_T_TOTAL / cfg.SDE_DT)
    traj = np.zeros((n_patients, n_steps), dtype=np.float32)
    for p in range(n_patients):
        g0 = rng.uniform(80, 300)
        G, _ = euler_maruyama_coupled(g0=g0, rng=rng)
        # Normalise glucose to [0, 1] as instability proxy
        g_min, g_max = G.min(), G.max()
        traj[p] = (G - g_min) / (g_max - g_min + 1e-6)
    return traj


# ─────────────────────────────────────────────────────────────
# Monte Carlo validator
# ─────────────────────────────────────────────────────────────

# Reference distributions from literature (mean, std) for each parameter
REFERENCE_STATS = {
    "mean_glucose":   (150.0, 50.0),
    "cv_pct":         (35.0,  10.0),
    "mage":           (60.0,  25.0),
    "modd":           (30.0,  15.0),
    "conga2":         (40.0,  15.0),
    "crp":            (8.0,   6.0),
    "wbc":            (9000,  3000),
    "spo2":           (93.0,  4.0),
    "temperature":    (33.5,  2.0),
    "moisture_index": (0.5,   0.2),
    "perfusion_index":(2.0,   0.8),
}


def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """KL divergence KL(P || Q) for two Gaussians (closed form)."""
    return (
        np.log(sigma2 / sigma1)
        + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
        - 0.5
    )


def validate_simulation(df_simulated, reference_stats=None) -> dict:
    """
    Compare simulated parameter distributions to reference statistics.

    Parameters
    ----------
    df_simulated   : pandas DataFrame with parameter columns
    reference_stats: dict of {param: (mean, std)}; defaults to REFERENCE_STATS

    Returns
    -------
    report : dict  {param: {kl_div, wasserstein, ks_stat, ks_pval}}
    """
    if reference_stats is None:
        reference_stats = REFERENCE_STATS

    report = {}
    for param, (ref_mu, ref_sigma) in reference_stats.items():
        if param not in df_simulated.columns:
            continue
        sim_vals = df_simulated[param].values.astype(float)

        # Simulated moments
        sim_mu    = float(np.mean(sim_vals))
        sim_sigma = float(np.std(sim_vals)) + 1e-9

        # KL divergence (Gaussian approximation)
        kl  = kl_divergence_gaussian(sim_mu, sim_sigma, ref_mu, ref_sigma)

        # Wasserstein distance (empirical vs. reference Gaussian sample)
        ref_sample = np.random.normal(ref_mu, ref_sigma, len(sim_vals))
        wd  = float(stats.wasserstein_distance(sim_vals, ref_sample))

        # KS test
        ks_stat, ks_pval = stats.kstest(
            sim_vals,
            "norm",
            args=(ref_mu, ref_sigma),
        )

        report[param] = {
            "sim_mean":   sim_mu,
            "sim_std":    sim_sigma,
            "ref_mean":   ref_mu,
            "ref_std":    ref_sigma,
            "kl_div":     float(kl),
            "wasserstein":float(wd),
            "ks_stat":    float(ks_stat),
            "ks_pval":    float(ks_pval),
        }

    return report


def print_validation_report(report: dict):
    """Print a formatted simulation validation table."""
    header = f"{'Parameter':<22} {'KL Div':>8} {'Wasserstein':>13} {'KS-stat':>8} {'KS-pval':>8}"
    print("\n" + "=" * len(header))
    print("  SIMULATION VALIDATION REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for param, m in report.items():
        print(
            f"{param:<22} {m['kl_div']:>8.4f} {m['wasserstein']:>13.4f}"
            f" {m['ks_stat']:>8.4f} {m['ks_pval']:>8.4f}"
        )
    print("=" * len(header))
