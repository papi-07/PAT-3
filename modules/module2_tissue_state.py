"""
modules/module2_tissue_state.py
Fuzzy Logic + XGBoost tissue state evaluation for Module 2.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from xgboost import XGBClassifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg

# ─────────────────────────────────────────────────────────────
# Fuzzy Logic System
# ─────────────────────────────────────────────────────────────

def build_fuzzy_tissue_system():
    """
    Build and return a scikit-fuzzy ControlSystemSimulation for tissue state.

    Inputs : crp, spo2, temperature, moisture_index, perfusion_index
    Output : tissue_score  (0 = inflamed, 1 = healing-ready)
    """
    # Antecedents
    crp   = ctrl.Antecedent(np.linspace(0, 30, 301),    "crp")
    spo2  = ctrl.Antecedent(np.linspace(70, 100, 301),  "spo2")
    temp  = ctrl.Antecedent(np.linspace(28, 38, 101),   "temperature")
    moist = ctrl.Antecedent(np.linspace(0, 1, 101),     "moisture_index")
    perf  = ctrl.Antecedent(np.linspace(0, 5, 101),     "perfusion_index")

    # Consequent
    score = ctrl.Consequent(np.linspace(0, 1, 101), "tissue_score")

    # ── Membership functions ──────────────────────────────────
    # CRP: low (<3), medium (2–10), high (>8)
    crp["low"]    = fuzz.trapmf(crp.universe,    [0,  0, 3,  6])
    crp["medium"] = fuzz.trapmf(crp.universe,    [3,  6, 10, 14])
    crp["high"]   = fuzz.trapmf(crp.universe,    [10, 14, 30, 30])

    # SpO2: low (<90), medium (88–95), high (>93)
    spo2["low"]    = fuzz.trapmf(spo2.universe, [70, 70, 88, 92])
    spo2["medium"] = fuzz.trapmf(spo2.universe, [88, 92, 95, 97])
    spo2["high"]   = fuzz.trapmf(spo2.universe, [93, 96, 100, 100])

    # Temperature: low (<34), normal (33–37.5), high (>37)
    temp["low"]    = fuzz.trapmf(temp.universe, [28, 28, 33, 35])
    temp["normal"] = fuzz.trapmf(temp.universe, [33, 35, 37, 37.5])
    temp["high"]   = fuzz.trapmf(temp.universe, [37, 37.5, 38, 38])

    # Moisture: dry (<0.3), moist (0.25–0.75), wet (>0.7)
    moist["dry"]   = fuzz.trapmf(moist.universe, [0,    0,    0.25, 0.40])
    moist["moist"] = fuzz.trapmf(moist.universe, [0.25, 0.40, 0.60, 0.75])
    moist["wet"]   = fuzz.trapmf(moist.universe, [0.60, 0.75, 1.0,  1.0])

    # Perfusion: low (<1), medium (0.8–3), high (>2.5)
    perf["low"]    = fuzz.trapmf(perf.universe, [0,   0,   0.8, 1.5])
    perf["medium"] = fuzz.trapmf(perf.universe, [0.8, 1.5, 2.5, 3.5])
    perf["high"]   = fuzz.trapmf(perf.universe, [2.5, 3.5, 5.0, 5.0])

    # Score
    score["inflamed"]       = fuzz.trapmf(score.universe, [0,   0,   0.2, 0.4])
    score["stable"]         = fuzz.trapmf(score.universe, [0.3, 0.45, 0.55, 0.7])
    score["healing_ready"]  = fuzz.trapmf(score.universe, [0.6, 0.75, 1.0, 1.0])

    # ── Rules ────────────────────────────────────────────────
    rules = [
        ctrl.Rule(crp["high"] | spo2["low"] | temp["high"],
                  score["inflamed"]),
        ctrl.Rule(crp["low"] & spo2["high"] & temp["normal"] & perf["high"],
                  score["healing_ready"]),
        ctrl.Rule(crp["low"] & spo2["high"] & temp["normal"] & moist["moist"],
                  score["healing_ready"]),
        ctrl.Rule(crp["medium"] | spo2["medium"] | moist["wet"],
                  score["stable"]),
        ctrl.Rule(crp["medium"] & spo2["medium"],
                  score["stable"]),
        ctrl.Rule(perf["low"] & spo2["low"],
                  score["inflamed"]),
        ctrl.Rule(perf["low"] & crp["medium"],
                  score["stable"]),
        ctrl.Rule(perf["high"] & crp["low"] & spo2["high"],
                  score["healing_ready"]),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def fuzzy_tissue_score(
    crp: float,
    spo2: float,
    temperature: float,
    moisture_index: float,
    perfusion_index: float,
    sim=None,
) -> float:
    """
    Compute a continuous tissue state score (0=inflamed, 1=healing-ready)
    using the fuzzy logic system.

    Parameters
    ----------
    sim : optional pre-built ControlSystemSimulation (for speed)

    Returns
    -------
    float in [0, 1]
    """
    if sim is None:
        sim = build_fuzzy_tissue_system()

    # Clamp inputs to universe ranges
    sim.input["crp"]             = float(np.clip(crp,              0,    30))
    sim.input["spo2"]            = float(np.clip(spo2,            70,   100))
    sim.input["temperature"]     = float(np.clip(temperature,     28,    38))
    sim.input["moisture_index"]  = float(np.clip(moisture_index,   0,     1))
    sim.input["perfusion_index"] = float(np.clip(perfusion_index,  0,     5))

    try:
        sim.compute()
        return float(sim.output["tissue_score"])
    except Exception:
        return 0.5   # fallback on defuzzification failure


def fuzzy_tissue_score_batch(
    df_tissue,
    sim=None,
) -> np.ndarray:
    """
    Batch evaluation of fuzzy tissue scores.

    Parameters
    ----------
    df_tissue : DataFrame or dict-like with columns:
                crp, spo2, temperature, moisture_index, perfusion_index

    Returns
    -------
    scores : np.ndarray (N,)
    """
    if sim is None:
        sim = build_fuzzy_tissue_system()

    scores = []
    cols = ["crp", "spo2", "temperature", "moisture_index", "perfusion_index"]
    for row in (df_tissue[cols].values if hasattr(df_tissue, "values") else df_tissue):
        scores.append(fuzzy_tissue_score(*row, sim=sim))
    return np.array(scores, dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# XGBoost classifier
# ─────────────────────────────────────────────────────────────

TISSUE_FEATURE_COLS = [
    "crp", "wbc", "spo2", "temperature", "moisture_index", "perfusion_index"
]


def build_xgb_tissue_classifier():
    """Return an untrained XGBClassifier with spec hyper-parameters."""
    return XGBClassifier(
        n_estimators=cfg.XGB_N_ESTIMATORS,
        max_depth=cfg.XGB_MAX_DEPTH,
        learning_rate=cfg.XGB_LR,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=cfg.RANDOM_SEED,
        n_jobs=-1,
    )


def train_xgb_tissue_classifier(X_train, y_train):
    """Train and return the XGBoost tissue state classifier."""
    model = build_xgb_tissue_classifier()
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────
# Combined tissue evaluator
# ─────────────────────────────────────────────────────────────

class TissueStateEvaluator:
    """
    Combines fuzzy logic scoring and XGBoost classification for tissue state.
    """

    def __init__(self):
        self.fuzzy_sim = build_fuzzy_tissue_system()
        self.xgb_model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the XGBoost tissue classifier."""
        self.xgb_model = train_xgb_tissue_classifier(X_train, y_train)
        return self

    def evaluate(self, features: dict) -> dict:
        """
        Evaluate tissue state for a single patient.

        Parameters
        ----------
        features : dict with keys matching TISSUE_FEATURE_COLS

        Returns
        -------
        dict with:
          fuzzy_score   : float (0–1, higher = more healing-ready)
          xgb_class     : int   (0=Inflamed, 1=Stable, 2=Healing-Ready)
          xgb_proba     : np.ndarray (3,)
        """
        fz = fuzzy_tissue_score(
            crp=features["crp"],
            spo2=features["spo2"],
            temperature=features["temperature"],
            moisture_index=features["moisture_index"],
            perfusion_index=features["perfusion_index"],
            sim=self.fuzzy_sim,
        )

        xgb_class, xgb_proba = None, None
        if self.xgb_model is not None:
            x_row = np.array([[features[c] for c in TISSUE_FEATURE_COLS]])
            xgb_class = int(self.xgb_model.predict(x_row)[0])
            xgb_proba = self.xgb_model.predict_proba(x_row)[0]

        return {
            "fuzzy_score": fz,
            "xgb_class":   xgb_class,
            "xgb_proba":   xgb_proba,
        }

    def evaluate_batch(self, X: np.ndarray) -> dict:
        """
        Batch evaluation.

        Parameters
        ----------
        X : (N, 6) array in column order of TISSUE_FEATURE_COLS

        Returns
        -------
        dict with fuzzy_scores, xgb_classes, xgb_probas
        """
        fuzzy_scores = []
        for row in X:
            feat = dict(zip(TISSUE_FEATURE_COLS, row))
            fuzzy_scores.append(fuzzy_tissue_score(
                crp=feat["crp"], spo2=feat["spo2"],
                temperature=feat["temperature"],
                moisture_index=feat["moisture_index"],
                perfusion_index=feat["perfusion_index"],
                sim=self.fuzzy_sim,
            ))

        result = {"fuzzy_scores": np.array(fuzzy_scores, dtype=np.float32)}

        if self.xgb_model is not None:
            result["xgb_classes"] = self.xgb_model.predict(X)
            result["xgb_probas"]  = self.xgb_model.predict_proba(X)

        return result
