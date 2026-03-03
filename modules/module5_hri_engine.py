"""
modules/module5_hri_engine.py
Adaptive AHP weights + Neural HRI classifier + SHAP explainability (Module 5).
"""

import numpy as np
import torch
import torch.nn as nn
import shap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────
# Adaptive AHP weighting
# ─────────────────────────────────────────────────────────────

# Default pairwise comparison matrix for 5 parameter groups
# (GV stability, Inflammation, Oxygenation, Temperature, Perfusion)
_DEFAULT_AHP_MATRIX = np.array([
    [1,   3,   2,   3,   2],
    [1/3, 1,   1/2, 1,   1/2],
    [1/2, 2,   1,   2,   1],
    [1/3, 1,   1/2, 1,   1/2],
    [1/2, 2,   1,   2,   1],
], dtype=np.float64)


def ahp_priority_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Compute AHP priority weights via the eigenvector method.

    Parameters
    ----------
    matrix : (n, n) positive reciprocal pairwise comparison matrix

    Returns
    -------
    weights : normalised priority weight vector of length n
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_idx = np.argmax(eigenvalues.real)
    w = eigenvectors[:, max_idx].real
    w = np.abs(w)
    return w / w.sum()


def compute_ahp_hri(
    features: np.ndarray,
    feature_names: list = None,
    matrix: np.ndarray = None,
) -> float:
    """
    Compute AHP-weighted HRI score for a single patient.

    Features are first normalised to [0, 1], then weighted by AHP priorities.

    Parameters
    ----------
    features     : 1-D array of patient feature values
    feature_names: optional list (unused – for future grouping)
    matrix       : pairwise comparison matrix; defaults to _DEFAULT_AHP_MATRIX

    Returns
    -------
    hri : float in [0, 1]
    """
    if matrix is None:
        matrix = _DEFAULT_AHP_MATRIX
    weights = ahp_priority_weights(matrix)

    # We use the first len(weights) features
    n = len(weights)
    feat = features[:n].astype(float)
    f_min, f_max = feat.min(), feat.max()
    if f_max > f_min:
        feat_norm = (feat - f_min) / (f_max - f_min)
    else:
        feat_norm = np.zeros_like(feat)

    return float(np.dot(weights, feat_norm))


# ─────────────────────────────────────────────────────────────
# Neural HRI classifier
# ─────────────────────────────────────────────────────────────

class HRIClassifier(nn.Module):
    """
    3-class HRI classifier:  Ready / Borderline / Not-Ready.

    FC(fused_dim, 64) → ReLU → Dropout(0.3) → FC(64, 32) → ReLU → FC(32, 3)
    Also exposes a continuous HRI score from the pre-softmax activations.
    """

    def __init__(self, fused_dim: int, n_classes: int = 3, dropout: float = cfg.HRI_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        """Returns (logits, proba, hri_score)."""
        h       = self.net(x)
        logits  = self.head(h)
        proba   = self.softmax(logits)
        # HRI score: probability of "Ready" class (index CLASS_READY=0).
        # Higher score → more likely the patient is healing-ready.
        hri_score = proba[:, 0]
        return logits, proba, hri_score

    def predict_proba_numpy(self, X: np.ndarray, device=cfg.DEVICE) -> np.ndarray:
        """Convenience wrapper for SHAP: X → softmax probs as np.ndarray."""
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            _, proba, _ = self.forward(X_t)
        return proba.cpu().numpy()


# ─────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────

def train_hri_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fused_dim: int = None,
    epochs: int = cfg.NUM_EPOCHS,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = cfg.LEARNING_RATE,
    device=cfg.DEVICE,
) -> "HRIClassifier":
    """
    Train and return an HRIClassifier.

    Parameters
    ----------
    X_train  : (N, fused_dim) float array
    y_train  : (N,) int array  (0, 1, or 2)

    Returns
    -------
    trained HRIClassifier
    """
    if fused_dim is None:
        fused_dim = X_train.shape[1]

    model = HRIClassifier(fused_dim).to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _, _ = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# SHAP explainability
# ─────────────────────────────────────────────────────────────

def compute_shap_values(
    model: HRIClassifier,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    device=cfg.DEVICE,
    n_background: int = 100,
):
    """
    Compute SHAP values using KernelExplainer.

    Parameters
    ----------
    model        : trained HRIClassifier
    X_background : background dataset for SHAP (use a sample of training data)
    X_explain    : data to explain
    n_background : number of background samples to summarise

    Returns
    -------
    shap_values : list of np.ndarray, one per class
    """
    bg = shap.kmeans(X_background, min(n_background, len(X_background)))

    def predict_fn(X):
        return model.predict_proba_numpy(X, device=device)

    explainer   = shap.KernelExplainer(predict_fn, bg)
    shap_values = explainer.shap_values(X_explain, nsamples=50)
    return shap_values


# ─────────────────────────────────────────────────────────────
# HRI engine wrapper
# ─────────────────────────────────────────────────────────────

class HRIEngine:
    """
    End-to-end HRI computation:
      - AHP-based HRI score
      - Neural HRI classification
      - SHAP feature importance
    """

    def __init__(self, fused_dim: int, device=cfg.DEVICE):
        self.device    = device
        self.fused_dim = fused_dim
        self.model     = None
        self.shap_values = None
        self.X_train   = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = cfg.NUM_EPOCHS,
    ):
        """Train the HRI classifier."""
        self.X_train = X_train
        self.model   = train_hri_classifier(
            X_train, y_train,
            fused_dim=self.fused_dim,
            epochs=epochs,
            device=self.device,
        )
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray):
        """
        Predict HRI class and score for X.

        Returns
        -------
        classes    : (N,) int array
        probas     : (N, 3) float array
        hri_scores : (N,) float array  (prob. of Ready class)
        """
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        _, proba, hri_score = self.model(X_t)
        classes = proba.argmax(dim=1).cpu().numpy()
        return classes, proba.cpu().numpy(), hri_score.cpu().numpy()

    def explain(
        self,
        X_explain: np.ndarray,
        n_background: int = 100,
    ):
        """Run SHAP explanation on X_explain."""
        bg = self.X_train if self.X_train is not None else X_explain
        self.shap_values = compute_shap_values(
            self.model, bg, X_explain,
            device=self.device,
            n_background=n_background,
        )
        return self.shap_values
