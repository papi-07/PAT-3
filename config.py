"""
config.py
Central configuration for PAT-3: Multi-Parametric Glycemic Variability–Triggered
Tissue Healing Readiness Assessment System.
"""

import torch

# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# Simulation / data
# ─────────────────────────────────────────────────────────────
N_PATIENTS = 5000          # virtual patient profiles
CGM_POINTS  = 288          # 24-hour at 5-min intervals
DATA_DIR    = "data"
RESULTS_DIR = "results"

# ─────────────────────────────────────────────────────────────
# Simulated parameter ranges (literature-grounded)
# ─────────────────────────────────────────────────────────────
PARAM_RANGES = {
    "mean_glucose":  (70,   400),   # mg/dL
    "mage":          (20,   180),   # mg/dL
    "cv_pct":        (15,    60),   # %
    "modd":          (10,   120),   # mg/dL
    "conga2":        (15,   150),   # mg/dL
    "crp":           (0.1,   30),   # mg/L
    "wbc":           (3000, 20000), # cells/μL
    "spo2":          (70,   100),   # %
    "temperature":   (28,    38),   # °C
    "moisture_index":(0,      1),   # normalised
    "perfusion_index":(0.1,   5.0), # normalised
}

# ─────────────────────────────────────────────────────────────
# HRI label criteria
# ─────────────────────────────────────────────────────────────
HRI_LABELS = {
    "READY":      1,
    "BORDERLINE": 0,   # stored as int 1 in 3-class (0-based mapping below)
    "NOT_READY":  2,
}
# 3-class integer mapping used throughout the code
CLASS_READY      = 0
CLASS_BORDERLINE = 1
CLASS_NOT_READY  = 2
CLASS_NAMES      = ["Ready", "Borderline", "Not Ready"]

# HRI thresholds used in decision logic
HRI_THRESHOLD_HIGH = 0.65
HRI_THRESHOLD_LOW  = 0.35

# ─────────────────────────────────────────────────────────────
# SDE parameters  (Ornstein-Uhlenbeck)
# ─────────────────────────────────────────────────────────────
SDE_DT        = 0.1     # time step
SDE_T_TOTAL   = 24.0    # hours
SDE_THETA_G   = 0.3
SDE_MU_G      = 120.0   # mean glucose (mg/dL)
SDE_SIGMA_G   = 15.0
SDE_THETA_I   = 0.2
SDE_MU_I      = 2.0     # baseline inflammation (arbitrary units)
SDE_SIGMA_I   = 0.5
SDE_ALPHA     = 0.01    # glucose→inflammation coupling

# ─────────────────────────────────────────────────────────────
# Module 1 – Glycemic Variability
# ─────────────────────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.3

CNN_CHANNELS_1   = 32
CNN_CHANNELS_2   = 64
CNN_KERNEL_1     = 5
CNN_KERNEL_2     = 3

WAVELET_NAME     = "morl"   # Morlet wavelet
WAVELET_SCALES   = list(range(1, 33))  # 32 scales

# ─────────────────────────────────────────────────────────────
# Module 2 – Tissue State
# ─────────────────────────────────────────────────────────────
XGB_N_ESTIMATORS  = 200
XGB_MAX_DEPTH     = 6
XGB_LR            = 0.1

# ─────────────────────────────────────────────────────────────
# Module 4 – Fusion
# ─────────────────────────────────────────────────────────────
AUTOENCODER_LATENT_DIM = 16
ATTENTION_NUM_HEADS    = 4
FUSION_HIDDEN_DIMS     = [64, 32]

# ─────────────────────────────────────────────────────────────
# Module 5 – HRI Engine
# ─────────────────────────────────────────────────────────────
HRI_HIDDEN_DIMS  = [64, 32]
HRI_DROPOUT      = 0.3

# ─────────────────────────────────────────────────────────────
# Training hyper-parameters
# ─────────────────────────────────────────────────────────────
BATCH_SIZE       = 64
LEARNING_RATE    = 1e-3
NUM_EPOCHS       = 30
TRAIN_SPLIT      = 0.8

# ─────────────────────────────────────────────────────────────
# Bayesian adaptive thresholding priors (Beta distribution)
# ─────────────────────────────────────────────────────────────
BAYES_ALPHA_PRIOR = 2.0
BAYES_BETA_PRIOR  = 2.0
