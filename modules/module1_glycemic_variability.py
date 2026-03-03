"""
modules/module1_glycemic_variability.py
LSTM + 1D-CNN + Wavelet + Statistical GV metrics for Module 1.
"""

import numpy as np
import torch
import torch.nn as nn
import pywt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────
# Statistical GV metrics
# ─────────────────────────────────────────────────────────────

def compute_gv_metrics(series: np.ndarray) -> dict:
    """
    Compute MAGE, CV%, MODD, CONGA-2 from a glucose time-series.

    Parameters
    ----------
    series : 1-D array of length 288 (5-min samples over 24 h)

    Returns
    -------
    dict with keys: mean_glucose, cv_pct, mage, modd, conga2
    """
    mean_g = float(np.mean(series))
    sd     = float(np.std(series))
    cv_pct = float(sd / mean_g * 100) if mean_g > 0 else 0.0

    # MAGE
    peaks, nadirs = [], []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            peaks.append(i)
        elif series[i] < series[i - 1] and series[i] < series[i + 1]:
            nadirs.append(i)
    excursions = []
    j = 0
    for p in peaks:
        while j < len(nadirs) - 1 and nadirs[j] < p:
            j += 1
        if j < len(nadirs):
            amp = abs(series[p] - series[nadirs[j]])
            if amp > sd:
                excursions.append(amp)
    mage = float(np.mean(excursions)) if excursions else sd

    # MODD  (uses first 288-point window as "yesterday")
    n = len(series)
    half = n // 2
    modd = float(np.mean(np.abs(series[:half] - series[half: half * 2]))) if half > 0 else 0.0

    # CONGA-2
    lag = int(2 * 60 / 5)   # 24 points
    if len(series) > lag:
        conga2 = float(np.std(series[lag:] - series[:-lag]))
    else:
        conga2 = 0.0

    return {
        "mean_glucose": mean_g,
        "cv_pct":  cv_pct,
        "mage":    mage,
        "modd":    modd,
        "conga2":  conga2,
    }


# ─────────────────────────────────────────────────────────────
# LSTM model
# ─────────────────────────────────────────────────────────────

class GlucoseLSTM(nn.Module):
    """
    2-layer LSTM that maps a 288-point CGM sequence → glycemic instability score (0–1).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = cfg.LSTM_HIDDEN_SIZE,
        num_layers: int = cfg.LSTM_NUM_LAYERS,
        dropout: float = cfg.LSTM_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc   = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, 1)
        returns: (batch, 1) instability score
        """
        out, _ = self.lstm(x)
        last    = out[:, -1, :]
        return self.sigmoid(self.fc(last))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return last hidden state as feature vector (batch, hidden_size)."""
        out, _ = self.lstm(x)
        return out[:, -1, :]


# ─────────────────────────────────────────────────────────────
# 1D-CNN model
# ─────────────────────────────────────────────────────────────

class Glucose1DCNN(nn.Module):
    """
    1D-CNN for local spike/dip pattern detection in CGM signals.
    Conv1d(1,32,5) → Conv1d(32,64,3) → MaxPool → FC → feature vector
    """

    def __init__(
        self,
        seq_len: int = cfg.CGM_POINTS,
        out_features: int = 32,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(1, cfg.CNN_CHANNELS_1, kernel_size=cfg.CNN_KERNEL_1, padding=2)
        self.conv2 = nn.Conv1d(cfg.CNN_CHANNELS_1, cfg.CNN_CHANNELS_2, kernel_size=cfg.CNN_KERNEL_2, padding=1)
        self.pool  = nn.AdaptiveMaxPool1d(16)
        self.relu  = nn.ReLU()

        conv_out = cfg.CNN_CHANNELS_2 * 16
        self.fc  = nn.Linear(conv_out, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len) or (batch, 1, seq_len)
        returns: (batch, out_features)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # add channel dim
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.relu(self.fc(x))


# ─────────────────────────────────────────────────────────────
# Wavelet feature extractor
# ─────────────────────────────────────────────────────────────

def extract_wavelet_features(
    series: np.ndarray,
    wavelet: str = cfg.WAVELET_NAME,
    scales: list = None,
) -> np.ndarray:
    """
    Apply Continuous Wavelet Transform and return per-scale energy as a feature vector.

    Parameters
    ----------
    series : 1-D float array
    wavelet : wavelet name (default 'morl')
    scales  : list of scale ints

    Returns
    -------
    energy : np.ndarray of shape (len(scales),)
    """
    if scales is None:
        scales = cfg.WAVELET_SCALES
    coeffs, _ = pywt.cwt(series, scales, wavelet)
    energy = np.sum(coeffs ** 2, axis=1)  # energy per scale
    return energy.astype(np.float32)


def extract_wavelet_features_batch(
    series_batch: np.ndarray,
    wavelet: str = cfg.WAVELET_NAME,
    scales: list = None,
) -> np.ndarray:
    """Batch version – returns (N, n_scales) array."""
    return np.stack(
        [extract_wavelet_features(s, wavelet, scales) for s in series_batch],
        axis=0,
    )


# ─────────────────────────────────────────────────────────────
# Combined glycemic feature extractor
# ─────────────────────────────────────────────────────────────

class GlycemicFeatureExtractor:
    """
    Combines LSTM + 1D-CNN + Wavelet + statistical GV metrics into a single
    glycemic variability feature vector per patient.
    """

    def __init__(self, device=cfg.DEVICE):
        self.device   = device
        self.lstm_model = GlucoseLSTM().to(device)
        self.cnn_model  = Glucose1DCNN().to(device)
        self.lstm_model.eval()
        self.cnn_model.eval()

    def set_trained_models(self, lstm_model: nn.Module, cnn_model: nn.Module):
        """Swap in trained model weights."""
        self.lstm_model = lstm_model.to(self.device)
        self.cnn_model  = cnn_model.to(self.device)
        self.lstm_model.eval()
        self.cnn_model.eval()

    @torch.no_grad()
    def extract(self, cgm_series: np.ndarray) -> np.ndarray:
        """
        Extract features from a single CGM series.

        Parameters
        ----------
        cgm_series : 1-D float array (288,)

        Returns
        -------
        feature_vec : 1-D np.ndarray
            [lstm_features (64) | cnn_features (32) | wavelet_features (32) | stat_metrics (5)]
        """
        series_t = torch.tensor(cgm_series, dtype=torch.float32, device=self.device)

        # LSTM features
        lstm_in = series_t.unsqueeze(0).unsqueeze(-1)   # (1, 288, 1)
        lstm_feat = self.lstm_model.get_features(lstm_in).squeeze(0).cpu().numpy()

        # CNN features
        cnn_in = series_t.unsqueeze(0)                  # (1, 288)
        cnn_feat = self.cnn_model(cnn_in).squeeze(0).cpu().numpy()

        # Wavelet features (32 scales)
        wav_feat = extract_wavelet_features(cgm_series)

        # Statistical GV metrics
        gv = compute_gv_metrics(cgm_series)
        stat_feat = np.array([
            gv["mean_glucose"],
            gv["cv_pct"],
            gv["mage"],
            gv["modd"],
            gv["conga2"],
        ], dtype=np.float32)

        return np.concatenate([lstm_feat, cnn_feat, wav_feat, stat_feat])

    def extract_batch(self, cgm_batch: np.ndarray) -> np.ndarray:
        """
        Extract features for a batch of CGM series.

        Parameters
        ----------
        cgm_batch : (N, 288) float array

        Returns
        -------
        features : (N, feature_dim) float array
        """
        return np.stack([self.extract(s) for s in cgm_batch], axis=0)


# ─────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────

def train_gv_models(
    cgm_train: np.ndarray,
    labels_train: np.ndarray,
    epochs: int = cfg.NUM_EPOCHS,
    batch_size: int = cfg.BATCH_SIZE,
    lr: float = cfg.LEARNING_RATE,
    device=cfg.DEVICE,
):
    """
    Train LSTM and 1D-CNN to predict a binary GV instability label.

    Parameters
    ----------
    cgm_train   : (N, 288) float array
    labels_train: (N,) float array in [0, 1]

    Returns
    -------
    lstm_model, cnn_model (both trained, on *device*)
    """
    lstm_model = GlucoseLSTM().to(device)
    cnn_model  = Glucose1DCNN().to(device)

    X = torch.tensor(cgm_train, dtype=torch.float32)
    y = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt_lstm = torch.optim.Adam(lstm_model.parameters(), lr=lr)
    opt_cnn  = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        lstm_model.train(); cnn_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # LSTM
            opt_lstm.zero_grad()
            lstm_in = xb.unsqueeze(-1)
            pred_lstm = lstm_model(lstm_in)
            loss_lstm = criterion(pred_lstm, yb)
            loss_lstm.backward()
            opt_lstm.step()

            # CNN
            opt_cnn.zero_grad()
            pred_cnn = cnn_model(xb).mean(dim=1, keepdim=True)
            loss_cnn = criterion(pred_cnn, yb)
            loss_cnn.backward()
            opt_cnn.step()

    return lstm_model, cnn_model
