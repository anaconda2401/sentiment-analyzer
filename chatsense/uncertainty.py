"""
Uncertainty Quantification & Calibration Module for ContextGuard AI.
Distinguishes negative predictions from insufficient evidence, evaluates margin confidence,
and computes calibration metrics (Expected Calibration Error, Brier Score).

DISCLAIMER:
Uncertainty metrics assess neural model confidence calibration against validation sets.
They DO NOT measure philosophical certainty or objective real-world truth.
"""
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class UncertaintyAssessor:
    """
    Evaluates prediction confidence margins, handles abstention,
    and calculates statistical calibration metrics.
    """

    def __init__(self, default_threshold: float = 0.55, margin_threshold: float = 0.08):
        self.default_threshold = default_threshold
        self.margin_threshold = margin_threshold

    def assess_prediction(
        self,
        scores: Dict[str, float],
        text: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Assesses confidence, borderline proximity, and potential out-of-distribution features.
        Returns confidence metrics and an abstention decision.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        words = str(text).split()
        word_count = len(words)

        if not scores:
            return {
                "decision": "Abstain",
                "reason": "No model output available.",
                "confidence_level": "Insufficient Evidence",
                "margin": 0.0,
                "is_abstain": True,
                "ood_flag": True,
            }

        top_label, top_score = max(scores.items(), key=lambda x: x[1])
        margin = round(abs(top_score - thresh), 4)

        # 1. Out-of-distribution / Unfamiliar-input heuristic
        ood_flag = False
        ood_reasons = []
        if word_count < 2:
            ood_flag = True
            ood_reasons.append("Ultra-short utterance (< 2 words)")
        if word_count > 120:
            ood_flag = True
            ood_reasons.append("Unusually long conversational turn (> 120 words)")

        # Check for repetitive / noisy character patterns
        alpha_count = sum(c.isalpha() for c in text)
        if len(text) > 5 and (alpha_count / len(text)) < 0.4:
            ood_flag = True
            ood_reasons.append("High symbol/emoji/punctuation density")

        # 2. Decision logic: Active Detection vs Negative vs Abstain
        if word_count < 2:
            decision = "Abstain"
            conf_level = "Insufficient Evidence"
            is_abstain = True
        elif margin < self.margin_threshold:
            conf_level = "Borderline / Low Confidence"
            decision = "Flagged (Borderline)" if top_score >= thresh else "Unflagged (Borderline)"
            is_abstain = False
        elif top_score >= thresh:
            conf_level = "High Confidence" if margin >= 0.25 else "Moderate Confidence"
            decision = "Detected Pattern"
            is_abstain = False
        else:
            conf_level = "High Confidence" if margin >= 0.25 else "Moderate Confidence"
            decision = "Negative (No Concerning Pattern)"
            is_abstain = False

        return {
            "top_label": top_label,
            "top_score": top_score,
            "threshold": thresh,
            "margin": margin,
            "decision": decision,
            "confidence_level": conf_level,
            "is_abstain": is_abstain,
            "ood_flag": ood_flag,
            "ood_reasons": ood_reasons,
        }

    @staticmethod
    def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Computes the mean squared error between binary outcomes and predicted probabilities."""
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        return float(np.mean((y_prob - y_true) ** 2))

    @staticmethod
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE).
        Partitions probability predictions into bins and measures weighted difference
        between average accuracy and average confidence.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)

        if n_samples == 0:
            return 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Collect items in current bin
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(round(ece, 4))
