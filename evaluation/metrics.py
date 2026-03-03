"""
evaluation/metrics.py
All evaluation metrics for PAT-3 (per-module + system-level).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    silhouette_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.special import kl_div


# ─────────────────────────────────────────────────────────────
# Module 1 – Glycemic Variability
# ─────────────────────────────────────────────────────────────

def module1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Regression metrics for GV metric prediction.

    Parameters
    ----------
    y_true, y_pred : arrays of predicted vs. actual GV metric values

    Returns
    -------
    dict with rmse, mae, r2
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────────────────────────
# Module 2 – Tissue State
# ─────────────────────────────────────────────────────────────

def module2_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Classification metrics for tissue state.

    Returns
    -------
    dict with accuracy, kappa, f1_macro
    """
    acc   = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    f1    = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"accuracy": acc, "kappa": kappa, "f1_macro": f1}


# ─────────────────────────────────────────────────────────────
# Module 3 – Simulation Engine
# ─────────────────────────────────────────────────────────────

def module3_metrics(sim_vals: np.ndarray, ref_vals: np.ndarray) -> dict:
    """
    Distribution comparison metrics.

    Parameters
    ----------
    sim_vals : simulated distribution samples
    ref_vals : reference distribution samples

    Returns
    -------
    dict with kl_div, wasserstein, ks_stat, ks_pval
    """
    # KL divergence via histogram (binned)
    n_bins = 50
    combined = np.concatenate([sim_vals, ref_vals])
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)
    sim_hist, _ = np.histogram(sim_vals, bins=bins, density=True)
    ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
    # Smooth to avoid zeros
    eps = 1e-10
    sim_hist += eps
    ref_hist += eps
    sim_hist /= sim_hist.sum()
    ref_hist /= ref_hist.sum()
    kl  = float(np.sum(kl_div(sim_hist, ref_hist)))

    wd  = float(wasserstein_distance(sim_vals, ref_vals))
    ks_stat, ks_pval = ks_2samp(sim_vals, ref_vals)

    return {
        "kl_div":     kl,
        "wasserstein": wd,
        "ks_stat":    float(ks_stat),
        "ks_pval":    float(ks_pval),
    }


# ─────────────────────────────────────────────────────────────
# Module 4 – Fusion
# ─────────────────────────────────────────────────────────────

def module4_metrics(
    recon_mse: float,
    latent_embeddings: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Autoencoder quality metrics.

    Parameters
    ----------
    recon_mse        : reconstruction MSE (scalar)
    latent_embeddings: (N, latent_dim) array
    labels           : (N,) class labels for silhouette score

    Returns
    -------
    dict with recon_mse, silhouette
    """
    n_unique = len(np.unique(labels))
    if n_unique > 1 and len(latent_embeddings) > n_unique:
        sil = float(silhouette_score(latent_embeddings, labels))
    else:
        sil = float("nan")

    return {"recon_mse": float(recon_mse), "silhouette": sil}


# ─────────────────────────────────────────────────────────────
# Module 5 – HRI Engine  (PRIMARY metrics)
# ─────────────────────────────────────────────────────────────

def module5_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Full classification metrics for HRI prediction (3-class).

    Parameters
    ----------
    y_true  : (N,) int array
    y_pred  : (N,) int array
    y_proba : (N, 3) float array

    Returns
    -------
    dict with accuracy, precision, recall, specificity, f1, auc_roc,
             auc_pr, mcc
    """
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    mcc  = float(matthews_corrcoef(y_true, y_pred))

    # Macro-OvR AUC
    try:
        auc_roc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        auc_roc = float("nan")

    try:
        auc_pr = float(
            np.mean([
                average_precision_score(
                    (y_true == c).astype(int),
                    y_proba[:, c],
                )
                for c in range(y_proba.shape[1])
            ])
        )
    except Exception:
        auc_pr = float("nan")

    # Specificity (macro average)
    cm   = confusion_matrix(y_true, y_pred)
    spec_per_class = []
    for i in range(len(cm)):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec_per_class.append(tn / (tn + fp + 1e-9))
    specificity = float(np.mean(spec_per_class))

    return {
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "specificity": specificity,
        "f1":          f1,
        "auc_roc":     auc_roc,
        "auc_pr":      auc_pr,
        "mcc":         mcc,
    }


# ─────────────────────────────────────────────────────────────
# Module 6 – Decision Logic
# ─────────────────────────────────────────────────────────────

def module6_metrics(
    decisions_true: list,
    decisions_pred: list,
    hri_true: np.ndarray = None,
    hri_pred: np.ndarray = None,
) -> dict:
    """
    Decision accuracy and related metrics.

    Parameters
    ----------
    decisions_true : list of str (reference decisions)
    decisions_pred : list of str (model decisions)
    hri_true       : optional ground-truth HRI scores (for state transition analysis)
    hri_pred       : optional predicted HRI scores

    Returns
    -------
    dict with decision_accuracy, false_alarm_rate
    """
    correct = sum(a == b for a, b in zip(decisions_true, decisions_pred))
    dec_acc = correct / len(decisions_true) if decisions_true else 0.0

    # False alarm rate: predicted INTERVENE when true is STABILIZE
    false_alarms = sum(
        1 for a, b in zip(decisions_true, decisions_pred)
        if a == "STABILIZE" and b == "INTERVENE"
    )
    far = false_alarms / max(sum(a == "STABILIZE" for a in decisions_true), 1)

    return {
        "decision_accuracy": float(dec_acc),
        "false_alarm_rate":  float(far),
    }


# ─────────────────────────────────────────────────────────────
# Master evaluation function
# ─────────────────────────────────────────────────────────────

def print_metrics_table(title: str, metrics: dict, targets: dict = None):
    """Print a formatted metrics table to console."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    header = f"  {'Metric':<25} {'Value':>10}"
    if targets:
        header += f"  {'Target':>15}"
    print(header)
    print("-" * width)
    for k, v in metrics.items():
        if isinstance(v, float):
            val_str = f"{v:10.4f}"
        else:
            val_str = f"{str(v):>10}"
        line = f"  {k:<25} {val_str}"
        if targets and k in targets:
            line += f"  {str(targets[k]):>15}"
        print(line)
    print("=" * width)


TARGETS_MODULE5 = {
    "accuracy":  "0.93–0.95",
    "precision": "0.90–0.94",
    "auc_roc":   ">0.95",
}


def run_all_evaluations(results: dict) -> dict:
    """
    Run all module evaluations from a results dictionary.

    Parameters
    ----------
    results : dict populated by main.py with keys per module

    Returns
    -------
    all_metrics : dict of all computed metrics
    """
    all_metrics = {}

    if "m1_y_true" in results and "m1_y_pred" in results:
        m1 = module1_metrics(results["m1_y_true"], results["m1_y_pred"])
        all_metrics["module1"] = m1
        print_metrics_table("MODULE 1 – Glycemic Variability (Regression)", m1)

    if "m2_y_true" in results and "m2_y_pred" in results:
        m2 = module2_metrics(results["m2_y_true"], results["m2_y_pred"])
        all_metrics["module2"] = m2
        print_metrics_table("MODULE 2 – Tissue State Classification", m2)

    if "m3_sim" in results and "m3_ref" in results:
        m3 = module3_metrics(results["m3_sim"], results["m3_ref"])
        all_metrics["module3"] = m3
        print_metrics_table("MODULE 3 – Simulation Validation", m3)

    if "m4_recon_mse" in results and "m4_latent" in results:
        m4 = module4_metrics(
            results["m4_recon_mse"],
            results["m4_latent"],
            results.get("m4_labels", np.zeros(len(results["m4_latent"]))),
        )
        all_metrics["module4"] = m4
        print_metrics_table("MODULE 4 – Feature Fusion (Autoencoder)", m4)

    if "m5_y_true" in results and "m5_y_pred" in results:
        m5 = module5_metrics(
            results["m5_y_true"],
            results["m5_y_pred"],
            results["m5_y_proba"],
        )
        all_metrics["module5"] = m5
        print_metrics_table(
            "MODULE 5 – HRI Engine (PRIMARY METRICS)",
            m5,
            TARGETS_MODULE5,
        )

    if "m6_dec_true" in results and "m6_dec_pred" in results:
        m6 = module6_metrics(results["m6_dec_true"], results["m6_dec_pred"])
        all_metrics["module6"] = m6
        print_metrics_table("MODULE 6 – Decision Logic", m6)

    return all_metrics
