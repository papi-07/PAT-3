"""
data/generate_synthetic_data.py
Monte Carlo + SDE synthetic patient data generation for PAT-3.
"""

import os
import random

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────
# SDE solver  (Euler-Maruyama)
# ─────────────────────────────────────────────────────────────

def solve_sde_glucose_inflammation(
    theta_g=cfg.SDE_THETA_G,
    mu_g=cfg.SDE_MU_G,
    sigma_g=cfg.SDE_SIGMA_G,
    theta_i=cfg.SDE_THETA_I,
    mu_i=cfg.SDE_MU_I,
    sigma_i=cfg.SDE_SIGMA_I,
    alpha=cfg.SDE_ALPHA,
    dt=cfg.SDE_DT,
    t_total=cfg.SDE_T_TOTAL,
    g0=None,
    i0=None,
):
    """
    Euler-Maruyama solver for the coupled glucose-inflammation SDE system.

    dG(t) = θ_G(μ_G - G(t))dt + σ_G dW_G
    dI(t) = θ_I(μ_I - I(t))dt + σ_I dW_I + α·G(t)dt

    Returns
    -------
    t_arr : np.ndarray  time points
    G_arr : np.ndarray  glucose trajectory
    I_arr : np.ndarray  inflammation trajectory
    """
    n_steps = int(t_total / dt)
    t_arr = np.linspace(0, t_total, n_steps)
    G_arr = np.zeros(n_steps)
    I_arr = np.zeros(n_steps)

    G_arr[0] = g0 if g0 is not None else mu_g
    I_arr[0] = i0 if i0 is not None else mu_i

    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps - 1):
        dW_G = np.random.randn() * sqrt_dt
        dW_I = np.random.randn() * sqrt_dt
        G_arr[k + 1] = (
            G_arr[k]
            + theta_g * (mu_g - G_arr[k]) * dt
            + sigma_g * dW_G
        )
        I_arr[k + 1] = (
            I_arr[k]
            + theta_i * (mu_i - I_arr[k]) * dt
            + sigma_i * dW_I
            + alpha * G_arr[k] * dt
        )
        # Clamp to physiological bounds
        G_arr[k + 1] = np.clip(G_arr[k + 1], 40, 500)
        I_arr[k + 1] = max(I_arr[k + 1], 0)

    return t_arr, G_arr, I_arr


# ─────────────────────────────────────────────────────────────
# CGM time-series generator
# ─────────────────────────────────────────────────────────────

def generate_cgm_series(
    mean_glucose: float,
    cv_pct: float,
    n_points: int = cfg.CGM_POINTS,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a realistic 24-hour CGM time-series (5-min intervals).

    Adds:
      - postprandial spikes (roughly at 7 AM, 12 PM, 6 PM)
      - nocturnal dip (2–4 AM)
      - stress excursion (random)
    """
    if rng is None:
        rng = np.random.default_rng()

    sd = mean_glucose * cv_pct / 100.0
    base = rng.normal(mean_glucose, sd * 0.3, n_points)
    # clip
    base = np.clip(base, 40, 500)

    t = np.arange(n_points) * 5 / 60  # hours

    # postprandial spikes
    for meal_h, height in [(7, 0.25), (12, 0.30), (18, 0.20)]:
        spike = height * mean_glucose * np.exp(-((t - meal_h) ** 2) / (2 * 0.5 ** 2))
        base += spike

    # nocturnal dip
    dip = 0.15 * mean_glucose * np.exp(-((t - 3) ** 2) / (2 * 0.75 ** 2))
    base -= dip

    # stress excursion (random timing)
    stress_h = rng.uniform(6, 22)
    stress_amp = rng.uniform(0.05, 0.20) * mean_glucose
    base += stress_amp * np.exp(-((t - stress_h) ** 2) / (2 * 0.3 ** 2))

    return np.clip(base, 40, 500).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# GV metric helpers
# ─────────────────────────────────────────────────────────────

def compute_mage(series: np.ndarray) -> float:
    """Mean Amplitude of Glycemic Excursions."""
    sd = np.std(series)
    peaks_idx = []
    nadirs_idx = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            peaks_idx.append(i)
        if series[i] < series[i - 1] and series[i] < series[i + 1]:
            nadirs_idx.append(i)

    excursions = []
    j = 0
    for p in peaks_idx:
        while j < len(nadirs_idx) - 1 and nadirs_idx[j] < p:
            j += 1
        if j < len(nadirs_idx):
            amp = abs(series[p] - series[nadirs_idx[j]])
            if amp > sd:
                excursions.append(amp)
    return float(np.mean(excursions)) if excursions else float(sd)


def compute_cv(series: np.ndarray) -> float:
    """Coefficient of variation (%)."""
    mean = np.mean(series)
    return float(np.std(series) / mean * 100) if mean > 0 else 0.0


def compute_modd(series: np.ndarray, points_per_day: int = 288) -> float:
    """Mean of Daily Differences."""
    if len(series) < 2 * points_per_day:
        return 0.0
    day1 = series[:points_per_day]
    day2 = series[points_per_day: 2 * points_per_day]
    return float(np.mean(np.abs(day1 - day2)))


def compute_conga(series: np.ndarray, n_hours: int = 2, interval_min: int = 5) -> float:
    """Continuous Overall Net Glycemic Action (CONGA-n)."""
    lag = int(n_hours * 60 / interval_min)
    if len(series) <= lag:
        return 0.0
    diffs = series[lag:] - series[:-lag]
    return float(np.std(diffs))


# ─────────────────────────────────────────────────────────────
# HRI label assignment
# ─────────────────────────────────────────────────────────────

def assign_hri_label(
    cv_pct: float,
    crp: float,
    spo2: float,
    temperature: float,
    wbc: float = None,
) -> int:
    """
    Assign HRI class (0=Ready, 1=Borderline, 2=NotReady).

    Criteria (from config):
      Ready      : CV% < 36, CRP < 5,  SpO2 > 95, stable temp (35–37.5 °C)
      Not Ready  : CV% > 50, CRP > 15, SpO2 < 90
      Borderline : everything else
    """
    temp_stable = 35.0 <= temperature <= 37.5

    if cv_pct < 36 and crp < 5 and spo2 > 95 and temp_stable:
        return cfg.CLASS_READY
    if cv_pct > 50 or crp > 15 or spo2 < 90:
        return cfg.CLASS_NOT_READY
    return cfg.CLASS_BORDERLINE


def _sample_class_patient(cls: int, rng: np.random.Generator) -> dict:
    """
    Sample physiological parameters targeted to produce a specific HRI class.
    This ensures a balanced dataset for ML training.
    """
    if cls == cfg.CLASS_READY:
        # CV% < 36, CRP < 5, SpO2 > 95, temp 35–37.5
        cv_pct = rng.uniform(15, 35)
        crp    = rng.uniform(0.1, 4.9)
        spo2   = rng.uniform(95.1, 100)
        temp   = rng.uniform(35.0, 37.5)
        mean_g = rng.uniform(80, 180)
    elif cls == cfg.CLASS_NOT_READY:
        # At least one: CV% > 50 OR CRP > 15 OR SpO2 < 90
        choice = rng.integers(0, 3)
        cv_pct = rng.uniform(15, 60)
        crp    = rng.uniform(0.1, 30)
        spo2   = rng.uniform(70, 100)
        temp   = rng.uniform(28, 38)
        if choice == 0:
            cv_pct = rng.uniform(50.1, 60)
        elif choice == 1:
            crp = rng.uniform(15.1, 30)
        else:
            spo2 = rng.uniform(70, 89.9)
        mean_g = rng.uniform(150, 400)
    else:  # BORDERLINE
        # Not ready AND not not-ready (moderate values)
        cv_pct = rng.uniform(36, 50)
        crp    = rng.uniform(5, 15)
        spo2   = rng.uniform(90, 95)
        temp   = rng.uniform(28, 38)
        mean_g = rng.uniform(100, 250)

    return {
        "mean_g": mean_g,
        "cv_pct": cv_pct,
        "crp":    crp,
        "spo2":   spo2,
        "temp":   temp,
    }


# ─────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_patients: int = cfg.N_PATIENTS,
    seed: int = cfg.RANDOM_SEED,
    save: bool = True,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Generate *n_patients* virtual patient profiles via Monte Carlo sampling.

    Returns a DataFrame with all scalar features + HRI label.
    CGM time-series are saved separately as a NumPy array.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    r = cfg.PARAM_RANGES

    records = []
    cgm_series_list = []

    # Balanced class sampling: equal thirds for Ready / Borderline / Not-Ready
    n_per_class = n_patients // 3
    class_sequence = (
        [cfg.CLASS_READY]      * n_per_class
        + [cfg.CLASS_BORDERLINE] * n_per_class
        + [cfg.CLASS_NOT_READY]  * (n_patients - 2 * n_per_class)
    )
    rng.shuffle(class_sequence)

    for cls in class_sequence:
        params = _sample_class_patient(cls, rng)
        mean_g = params["mean_g"]
        cv_pct = params["cv_pct"]
        crp    = params["crp"]
        spo2   = params["spo2"]
        temp   = params["temp"]

        wbc   = rng.uniform(*r["wbc"])
        moist = rng.uniform(*r["moisture_index"])
        perf  = rng.uniform(*r["perfusion_index"])

        # CGM series
        series = generate_cgm_series(mean_g, cv_pct, rng=rng)
        cgm_series_list.append(series)

        mage  = compute_mage(series)
        modd  = compute_modd(series)
        conga = compute_conga(series)

        records.append({
            "mean_glucose":   float(np.mean(series)),
            "cv_pct":         cv_pct,
            "mage":           mage,
            "modd":           modd,
            "conga2":         conga,
            "crp":            crp,
            "wbc":            wbc,
            "spo2":           spo2,
            "temperature":    temp,
            "moisture_index": moist,
            "perfusion_index": perf,
            "hri_label":      cls,
        })

    df = pd.DataFrame(records)
    cgm_array = np.array(cgm_series_list, dtype=np.float32)

    if save:
        out = output_dir or cfg.DATA_DIR
        os.makedirs(out, exist_ok=True)
        csv_path = os.path.join(out, "synthetic_patients.csv")
        npy_path = os.path.join(out, "cgm_series.npy")
        df.to_csv(csv_path, index=False)
        np.save(npy_path, cgm_array)
        print(f"[DataGen] Saved {n_patients} patients → {csv_path}")
        print(f"[DataGen] CGM series  → {npy_path}")

    return df, cgm_array


if __name__ == "__main__":
    df, cgm = generate_synthetic_dataset()
    print(df["hri_label"].value_counts())
