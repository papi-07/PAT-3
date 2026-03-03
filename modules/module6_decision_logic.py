"""
modules/module6_decision_logic.py
Rule-based expert system + MDP + Bayesian adaptive thresholding (Module 6).
"""

import numpy as np
from typing import List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ─────────────────────────────────────────────────────────────
# Rule-Based Expert System
# ─────────────────────────────────────────────────────────────

TISSUE_STATE_HEALING_READY = 2   # XGBoost label
TISSUE_STATE_INFLAMED      = 0

ACTION_INTERVENE  = "INTERVENE"
ACTION_MONITOR    = "MONITOR"
ACTION_STABILIZE  = "STABILIZE"


def rule_based_decision(
    hri_score: float,
    tissue_class: int,
    threshold_high: float = cfg.HRI_THRESHOLD_HIGH,
    threshold_low:  float = cfg.HRI_THRESHOLD_LOW,
) -> str:
    """
    Apply clinically grounded rules to determine the care action.

    Rules
    -----
    IF HRI > threshold_high AND tissue == Healing-Ready  → INTERVENE
    IF threshold_low < HRI <= threshold_high             → MONITOR
    IF HRI <= threshold_low OR tissue == Inflamed        → STABILIZE

    Returns
    -------
    action : one of {"INTERVENE", "MONITOR", "STABILIZE"}
    """
    if hri_score > threshold_high and tissue_class == TISSUE_STATE_HEALING_READY:
        return ACTION_INTERVENE
    if hri_score <= threshold_low or tissue_class == TISSUE_STATE_INFLAMED:
        return ACTION_STABILIZE
    return ACTION_MONITOR


def apply_rules_batch(
    hri_scores: np.ndarray,
    tissue_classes: np.ndarray,
    threshold_high: float = cfg.HRI_THRESHOLD_HIGH,
    threshold_low:  float = cfg.HRI_THRESHOLD_LOW,
) -> List[str]:
    """Batch version of rule_based_decision."""
    return [
        rule_based_decision(h, t, threshold_high, threshold_low)
        for h, t in zip(hri_scores, tissue_classes)
    ]


# ─────────────────────────────────────────────────────────────
# Markov Decision Process
# ─────────────────────────────────────────────────────────────

# States
MDP_STATES   = ["Unstable", "Stabilizing", "Ready", "Healing"]
MDP_ACTIONS  = ["Stabilize", "Monitor", "Intervene"]

N_STATES  = len(MDP_STATES)
N_ACTIONS = len(MDP_ACTIONS)

# Default transition probability matrix  P[action, state_from, state_to]
# Estimated from simulation heuristics
_DEFAULT_TRANSITIONS = np.array([
    # Stabilize
    [[0.6, 0.3, 0.1, 0.0],
     [0.1, 0.6, 0.2, 0.1],
     [0.0, 0.1, 0.7, 0.2],
     [0.0, 0.0, 0.1, 0.9]],
    # Monitor
    [[0.7, 0.2, 0.1, 0.0],
     [0.1, 0.5, 0.3, 0.1],
     [0.0, 0.1, 0.6, 0.3],
     [0.0, 0.0, 0.0, 1.0]],
    # Intervene
    [[0.5, 0.3, 0.15, 0.05],
     [0.05, 0.3, 0.4, 0.25],
     [0.0, 0.05, 0.45, 0.5],
     [0.0, 0.0, 0.0, 1.0]],
], dtype=np.float64)

# Reward matrix  R[state_from, state_to]
_DEFAULT_REWARDS = np.array([
    [-1.0,  0.5,  1.0,  2.0],   # from Unstable
    [-0.5, -0.5,  1.0,  2.0],   # from Stabilizing
    [-1.0, -0.5,  0.0,  2.0],   # from Ready
    [-2.0, -1.0, -0.5,  1.0],   # from Healing
], dtype=np.float64)


def value_iteration(
    transitions: np.ndarray = None,
    rewards: np.ndarray = None,
    gamma: float = 0.95,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value iteration to find the optimal MDP policy.

    Parameters
    ----------
    transitions : (n_actions, n_states, n_states) transition probability array
    rewards     : (n_states, n_states) reward matrix
    gamma       : discount factor

    Returns
    -------
    V      : (n_states,) optimal value function
    policy : (n_states,) optimal action indices
    """
    if transitions is None:
        transitions = _DEFAULT_TRANSITIONS
    if rewards is None:
        rewards = _DEFAULT_REWARDS

    n_a, n_s, _ = transitions.shape
    V = np.zeros(n_s)

    for _ in range(max_iter):
        V_new = np.zeros(n_s)
        for s in range(n_s):
            q_vals = []
            for a in range(n_a):
                q = sum(
                    transitions[a, s, s_next]
                    * (rewards[s, s_next] + gamma * V[s_next])
                    for s_next in range(n_s)
                )
                q_vals.append(q)
            V_new[s] = max(q_vals)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new

    policy = np.zeros(n_s, dtype=int)
    for s in range(n_s):
        q_vals = []
        for a in range(n_a):
            q = sum(
                transitions[a, s, s_next]
                * (rewards[s, s_next] + gamma * V[s_next])
                for s_next in range(n_s)
            )
            q_vals.append(q)
        policy[s] = int(np.argmax(q_vals))

    return V, policy


def hri_to_mdp_state(hri_score: float) -> int:
    """Map a continuous HRI score to a discrete MDP state index."""
    if hri_score < 0.25:
        return 0   # Unstable
    if hri_score < 0.50:
        return 1   # Stabilizing
    if hri_score < 0.75:
        return 2   # Ready
    return 3       # Healing


# ─────────────────────────────────────────────────────────────
# Bayesian Adaptive Thresholding
# ─────────────────────────────────────────────────────────────

class BayesianAdaptiveThreshold:
    """
    Beta-prior Bayesian adaptive thresholding for HRI decision boundaries.

    The threshold is modelled as a Beta(α, β) distribution.
    Each observation updates the posterior.
    """

    def __init__(
        self,
        alpha_prior: float = cfg.BAYES_ALPHA_PRIOR,
        beta_prior:  float = cfg.BAYES_BETA_PRIOR,
    ):
        self.alpha = alpha_prior
        self.beta  = beta_prior
        self.history: List[float] = []

    @property
    def threshold(self) -> float:
        """Posterior mean as the adaptive threshold."""
        return self.alpha / (self.alpha + self.beta)

    def update(self, hri_score: float, outcome: int):
        """
        Bayesian update given an observed HRI score and clinical outcome.

        Parameters
        ----------
        hri_score : float in [0, 1]
        outcome   : 1 = healing (positive), 0 = non-healing (negative)
        """
        self.history.append(self.threshold)
        if outcome == 1:
            self.alpha += hri_score
        else:
            self.beta += (1 - hri_score)

    def batch_update(self, hri_scores: np.ndarray, outcomes: np.ndarray):
        """Update threshold over a batch of observations."""
        for h, o in zip(hri_scores, outcomes):
            self.update(float(h), int(o))

    def convergence_history(self) -> np.ndarray:
        """Return the threshold at each update step."""
        return np.array(self.history)


# ─────────────────────────────────────────────────────────────
# Decision Logic wrapper
# ─────────────────────────────────────────────────────────────

class DecisionLogic:
    """
    Combines rule-based decisions, MDP optimal policy, and Bayesian thresholds.
    """

    def __init__(self):
        self.bayes_high = BayesianAdaptiveThreshold(
            alpha_prior=cfg.BAYES_ALPHA_PRIOR,
            beta_prior=cfg.BAYES_BETA_PRIOR,
        )
        self.bayes_low  = BayesianAdaptiveThreshold(
            alpha_prior=cfg.BAYES_BETA_PRIOR,
            beta_prior=cfg.BAYES_ALPHA_PRIOR,
        )
        self.V_opt, self.policy_opt = value_iteration()
        self.transitions = _DEFAULT_TRANSITIONS
        self.rewards      = _DEFAULT_REWARDS

    def decide(
        self,
        hri_score: float,
        tissue_class: int,
        use_adaptive: bool = True,
    ) -> dict:
        """
        Make a clinical decision for a single patient.

        Returns
        -------
        dict with:
          rule_action   : str
          mdp_action    : str
          mdp_state     : str
          hri_threshold_high : float (current adaptive threshold)
        """
        t_high = self.bayes_high.threshold if use_adaptive else cfg.HRI_THRESHOLD_HIGH
        t_low  = self.bayes_low.threshold  if use_adaptive else cfg.HRI_THRESHOLD_LOW

        rule_action = rule_based_decision(hri_score, tissue_class, t_high, t_low)
        mdp_state   = hri_to_mdp_state(hri_score)
        mdp_action  = MDP_ACTIONS[self.policy_opt[mdp_state]]

        return {
            "rule_action":          rule_action,
            "mdp_action":           mdp_action,
            "mdp_state":            MDP_STATES[mdp_state],
            "hri_threshold_high":   t_high,
            "hri_threshold_low":    t_low,
        }

    def decide_batch(
        self,
        hri_scores: np.ndarray,
        tissue_classes: np.ndarray,
    ) -> List[dict]:
        """Batch version of decide()."""
        return [
            self.decide(h, t)
            for h, t in zip(hri_scores, tissue_classes)
        ]

    def update_thresholds(
        self,
        hri_scores: np.ndarray,
        outcomes: np.ndarray,
    ):
        """Update Bayesian thresholds with observed outcomes."""
        self.bayes_high.batch_update(hri_scores, outcomes)
        self.bayes_low.batch_update(hri_scores, outcomes)
