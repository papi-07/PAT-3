# PAT-3: Multi-Parametric Glycemic Variability–Triggered Tissue Healing Readiness Assessment System

## Overview

PAT-3 is a **simulation-driven (TRL-3), hardware-independent** clinical decision support system
that computes a **Healing Readiness Index (HRI)** for diabetic foot ulcers by fusing glycemic
variability features with tissue state parameters.

The system is based on the patent:
> *"A Multi-Parametric Glycemic Variability–Triggered Tissue Healing Readiness Assessment System
> for Diabetic Foot Ulcers"*

---

## Architecture

```
Synthetic Data (Monte Carlo + SDE)
          │
          ▼
┌─────────────────────┐    ┌─────────────────────┐
│  Module 1           │    │  Module 2            │
│  Glycemic           │    │  Tissue State        │
│  Variability        │    │  Evaluation          │
│  LSTM + 1D-CNN      │    │  Fuzzy Logic +       │
│  + Wavelet + Stats  │    │  XGBoost             │
└─────────┬───────────┘    └──────────┬───────────┘
          │                            │
          └────────────┬───────────────┘
                       ▼
          ┌────────────────────────┐
          │  Module 4              │
          │  Feature Fusion        │
          │  Autoencoder +         │
          │  Attention Network     │
          └────────────┬───────────┘
                       ▼
          ┌────────────────────────┐
          │  Module 5              │
          │  HRI Engine            │
          │  AHP Weighting +       │
          │  Neural Classifier +   │
          │  SHAP Explainability   │
          └────────────┬───────────┘
                       ▼
          ┌────────────────────────┐
          │  Module 6              │
          │  Decision Logic        │
          │  Rule-Based +          │
          │  MDP + Bayesian        │
          │  Thresholding          │
          └────────────────────────┘
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

This will:
1. Generate 5,000 synthetic patient profiles (Monte Carlo + SDE)
2. Extract glycemic variability features (LSTM, 1D-CNN, Wavelet)
3. Evaluate tissue state (Fuzzy Logic + XGBoost)
4. Validate simulation quality (KL Divergence, Wasserstein, KS-test)
5. Fuse features via Autoencoder + Attention
6. Compute HRI and train classifier
7. Run decision logic (Rules + MDP + Bayesian thresholding)
8. Evaluate all metrics
9. Generate all 13 visualisation figures

Results are saved to the `results/` directory.

---

## Directory Structure

```
PAT-3/
├── requirements.txt
├── README.md
├── config.py                          # Central configuration
├── main.py                            # Full pipeline orchestrator
├── data/
│   └── generate_synthetic_data.py    # Monte Carlo + SDE data generation
├── modules/
│   ├── __init__.py
│   ├── module1_glycemic_variability.py
│   ├── module2_tissue_state.py
│   ├── module3_simulation_engine.py
│   ├── module4_fusion.py
│   ├── module5_hri_engine.py
│   └── module6_decision_logic.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── visualization/
│   ├── __init__.py
│   └── plots.py
└── results/                           # Generated plots and tables
```

---

## Module Descriptions

| Module | Description |
|--------|-------------|
| Module 1 | LSTM + 1D-CNN + Wavelet CWT + Statistical GV metrics (MAGE, CV%, MODD, CONGA) |
| Module 2 | Fuzzy Logic tissue scoring + XGBoost tissue state classifier |
| Module 3 | Monte Carlo validation + Euler-Maruyama SDE solver |
| Module 4 | Autoencoder (latent dim=16) + Multi-head attention fusion |
| Module 5 | AHP-weighted HRI + Neural 3-class classifier + SHAP explainability |
| Module 6 | Rule-based expert system + MDP (value iteration) + Bayesian adaptive thresholds |

---

## Expected Outputs

| Metric | Target |
|--------|--------|
| Accuracy | 93–95% |
| Precision | 90–94% |
| AUC-ROC | > 0.95 |

### Generated Figures

| Figure | Description |
|--------|-------------|
| fig01 | Glucose Signal + Wavelet Decomposition |
| fig02 | Clarke Error Grid Analysis |
| fig03 | Fuzzy Membership Functions |
| fig04 | KDE Distribution Overlap (Simulated vs Reference) |
| fig05 | t-SNE Autoencoder Embeddings |
| fig06 | Attention Weight Heatmap |
| fig07 | ROC Curves (3-class OvR) |
| fig08 | Precision-Recall Curves |
| fig09 | HRI Score Distribution (Violin + Histogram) |
| fig10 | MDP State Transition Diagrams |
| fig11 | Radar Chart – PAT-3 vs Prior Art |
| fig12 | SHAP Summary Plot |
| fig13 | SHAP Force Plot |

---

## License

Placeholder — all rights reserved.
