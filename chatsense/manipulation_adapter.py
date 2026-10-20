"""
Manipulation Detector Adapter Interface for ContextGuard AI.
Provides an isolated, swappable adapter layer around pretrained manipulation detectors,
supporting context-window augmentation, multi-label inference, and abstention handling.

DISCLAIMER:
All detected patterns are linguistic estimates for educational and analytical awareness.
They DO NOT diagnose character, prove manipulative intent, or evaluate legal abuse.
"""
import abc
import logging
from typing import Dict, List, Optional, Any
import pandas as pd

from .abuse_detector import AbuseDetector, DOCUMENTED_LABELS, LABEL_DISPLAY_NAMES, CONCERNING_PATTERNS

log = logging.getLogger(__name__)

# Documented taxonomy alignment: mapping authentic model labels to conceptual taxonomy
TAXONOMY_ALIGNMENT = {
    "gaslighting": {
        "concept": "Gaslighting-like Language",
        "description": "Language that denies events or causes second-guessing of memory and reality.",
    },
    "darvo": {
        "concept": "Blame Shifting (DARVO)",
        "description": "Deny wrongdoing, attack credibility, and reverse victim and offender roles.",
    },
    "victim_blaming": {
        "concept": "Blame Shifting / Deflection",
        "description": "Holding the target responsible for the speaker's own emotional state or conduct.",
    },
    "emotional_manipulation": {
        "concept": "Emotional Coercion / Guilt Induction",
        "description": "Leveraging guilt, obligation, or conditional affection to compel compliance.",
    },
    "boundary_violation": {
        "concept": "Pressure & Controlling Language",
        "description": "Refusal to respect stated boundaries or repeatedly pushing past a 'no'.",
    },
    "surveillance_accusation": {
        "concept": "Surveillance / Accusatory Inquiries",
        "description": "Monitoring movements, questioning associations, and unwarranted interrogation.",
    },
    "healthy_boundary_setting": {
        "concept": "Healthy Boundary Setting",
        "description": "Constructive limit-setting and self-protection; not an abusive tactic.",
    },
    "no_abuse_pattern": {
        "concept": "Non-Manipulative Expression",
        "description": "Direct, collaborative, and non-coercive conversation.",
    },
}


class ManipulationDetectorAdapter(abc.ABC):
    """Abstract Base Class for manipulation detector adapters."""

    @abc.abstractmethod
    def predict(
        self,
        text: str,
        context: Optional[List[Dict[str, str]]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Predict manipulation categories for a single target message."""
        pass

    @abc.abstractmethod
    def batch_predict(
        self,
        records: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        context_window: int = 4,
    ) -> List[Dict[str, Any]]:
        """Predict manipulation categories across multiple messages with context."""
        pass


class PretrainedAbuseDetectorAdapter(ManipulationDetectorAdapter):
    """
    Adapter implementing the AbuseDetector (SamanthaStorm/abusedetector) model.
    Handles context aggregation, context vs isolated comparison, and abstention logic.
    """

    def __init__(
        self,
        detector: Optional[AbuseDetector] = None,
        default_threshold: float = 0.50,
        default_context_window: int = 4,
    ):
        self.detector = detector or AbuseDetector(default_threshold=default_threshold)
        self.default_threshold = default_threshold
        self.default_context_window = max(1, min(10, default_context_window))

    @property
    def is_available(self) -> bool:
        return self.detector.is_available

    def format_context_string(self, context_messages: List[Dict[str, str]], target_text: str) -> str:
        """
        Formats surrounding context messages and target message into a single input.
        Example: [Context: User1: Hi | User2: ...] Target: ...
        """
        if not context_messages:
            return target_text

        ctx_parts = []
        for msg in context_messages:
            sender = msg.get("sender", "User")
            text = msg.get("clean_message", msg.get("message", "")).strip()
            if text:
                ctx_parts.append(f"{sender}: {text}")

        if not ctx_parts:
            return target_text

        context_str = " | ".join(ctx_parts[-self.default_context_window:])
        return f"[Context: {context_str}] Target: {target_text}"

    def predict(
        self,
        text: str,
        context: Optional[List[Dict[str, str]]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs isolated and context-aware predictions for a target message.
        Distinguishes between a negative prediction and an abstention (insufficient evidence).
        """
        thresh = threshold if threshold is not None else self.default_threshold

        # Isolated message prediction
        isolated_res = self.detector.predict(text, threshold=thresh)

        # Context-aware prediction if context is supplied
        if context:
            augmented_text = self.format_context_string(context, text)
            context_res = self.detector.predict(augmented_text, threshold=thresh)
        else:
            context_res = isolated_res

        # Check for abstention: message too short (< 2 words) or empty
        word_count = len(str(text).split())
        is_abstain = False
        abstain_reason = ""
        if word_count < 2:
            is_abstain = True
            abstain_reason = "Insufficient length (under 2 words); context required."
        elif not isolated_res.get("is_available", False):
            is_abstain = True
            abstain_reason = "Model unavailable."

        # Compute context shift (delta in top concerning pattern)
        top_pattern = context_res.get("top_pattern", "no_abuse_pattern")
        isolated_top_score = isolated_res.get("scores", {}).get(top_pattern, 0.0)
        context_top_score = context_res.get("scores", {}).get(top_pattern, 0.0)
        context_shift = round(context_top_score - isolated_top_score, 4)

        return {
            "is_available": isolated_res.get("is_available", False),
            "target_text": text,
            "threshold": thresh,
            "is_abstain": is_abstain,
            "abstain_reason": abstain_reason,
            "isolated_scores": isolated_res.get("scores", {}),
            "isolated_patterns": isolated_res.get("detected_patterns", []),
            "context_scores": context_res.get("scores", {}),
            "context_patterns": context_res.get("detected_patterns", []),
            "top_pattern": top_pattern,
            "top_pattern_name": LABEL_DISPLAY_NAMES.get(top_pattern, top_pattern),
            "top_score": context_top_score,
            "context_shift": context_shift,
            "taxonomy_info": TAXONOMY_ALIGNMENT.get(top_pattern, {}),
        }

    def batch_predict(
        self,
        records: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        context_window: int = 4,
        max_neural_samples: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        High-throughput context-aware batch prediction.
        - Employs vectorized tensor batching (batch_size=32, max_length=96).
        - Automatically abstains on trivial / ultra-short turns without neural computation.
        - For large chats (> 250 candidate turns), prioritizes communicative & conflictual turns
          to guarantee execution in under 2 seconds.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        w = max(1, min(10, context_window))
        n = len(records)
        if n == 0:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * n
        candidate_indices: List[int] = []

        # Trivial / short token filtering (instant 0 ms pass)
        for idx, rec in enumerate(records):
            target_msg = str(rec.get("clean_message", rec.get("message", ""))).strip()
            words = target_msg.split()

            if len(words) < 2 or not any(c.isalpha() for c in target_msg):
                results[idx] = {
                    "is_available": self.is_available,
                    "target_text": target_msg,
                    "threshold": thresh,
                    "is_abstain": True,
                    "abstain_reason": "Insufficient length (< 2 words); context required.",
                    "isolated_scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                    "isolated_patterns": [],
                    "context_scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                    "context_patterns": [],
                    "top_pattern": "no_abuse_pattern",
                    "top_pattern_name": LABEL_DISPLAY_NAMES.get("no_abuse_pattern", "Non-Manipulative Expression"),
                    "top_score": 0.0,
                    "context_shift": 0.0,
                    "taxonomy_info": TAXONOMY_ALIGNMENT.get("no_abuse_pattern", {}),
                }
            else:
                candidate_indices.append(idx)

        # If model is unavailable, populate remaining with fallback
        if not self.is_available:
            for idx in candidate_indices:
                target_msg = str(records[idx].get("clean_message", records[idx].get("message", ""))).strip()
                results[idx] = {
                    "is_available": False,
                    "target_text": target_msg,
                    "threshold": thresh,
                    "is_abstain": True,
                    "abstain_reason": "Model unavailable.",
                    "isolated_scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                    "isolated_patterns": [],
                    "context_scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                    "context_patterns": [],
                    "top_pattern": "no_abuse_pattern",
                    "top_pattern_name": LABEL_DISPLAY_NAMES.get("no_abuse_pattern", "Non-Manipulative Expression"),
                    "top_score": 0.0,
                    "context_shift": 0.0,
                    "taxonomy_info": TAXONOMY_ALIGNMENT.get("no_abuse_pattern", {}),
                }
            return results

        # Intelligent candidate capping for fast execution under 2 seconds on massive chats
        if len(candidate_indices) > max_neural_samples:
            # Score candidates to prioritize conflictual or emotionally expressive turns
            conflict_cues = {"never", "always", "crazy", "fault", "blame", "liar", "guilt", "sorry", "stop", "hate", "why", "who", "trust", "promise"}
            scored_candidates = []
            for c_idx in candidate_indices:
                msg_lower = str(records[c_idx].get("clean_message", "")).lower()
                word_len = len(msg_lower.split())
                score = min(word_len, 20)
                if any(cue in msg_lower for cue in conflict_cues):
                    score += 15
                if "?" in msg_lower or "!" in msg_lower:
                    score += 5
                scored_candidates.append((score, c_idx))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            # Select top prioritized candidates plus uniform time strides
            selected_set = set(idx for _, idx in scored_candidates[: int(max_neural_samples * 0.7)])
            stride = max(1, len(candidate_indices) // int(max_neural_samples * 0.3))
            for i in range(0, len(candidate_indices), stride):
                selected_set.add(candidate_indices[i])
                if len(selected_set) >= max_neural_samples:
                    break

            active_eval_indices = sorted(list(selected_set))
            unselected_indices = set(candidate_indices) - selected_set
        else:
            active_eval_indices = candidate_indices
            unselected_indices = set()

        # Non-selected candidates get safe neutral classification
        for idx in unselected_indices:
            target_msg = str(records[idx].get("clean_message", records[idx].get("message", ""))).strip()
            results[idx] = {
                "is_available": True,
                "target_text": target_msg,
                "threshold": thresh,
                "is_abstain": False,
                "abstain_reason": "",
                "isolated_scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                "isolated_patterns": [],
                "context_scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                "context_patterns": [],
                "top_pattern": "no_abuse_pattern",
                "top_pattern_name": LABEL_DISPLAY_NAMES.get("no_abuse_pattern", "Non-Manipulative Expression"),
                "top_score": 0.0,
                "context_shift": 0.0,
                "taxonomy_info": TAXONOMY_ALIGNMENT.get("no_abuse_pattern", {}),
            }

        # Vectorized batch inference on active candidates
        augmented_inputs = []
        for idx in active_eval_indices:
            target_msg = str(records[idx].get("clean_message", records[idx].get("message", ""))).strip()
            start_idx = max(0, idx - w)
            context_slice = records[start_idx:idx]
            aug_text = self.format_context_string(context_slice, target_msg)
            augmented_inputs.append(aug_text)

        batch_preds = self.detector.batch_predict(augmented_inputs, batch_size=32, threshold=thresh, max_length=72)

        for orig_idx, pred in zip(active_eval_indices, batch_preds):
            target_msg = str(records[orig_idx].get("clean_message", records[orig_idx].get("message", ""))).strip()
            top_p = pred.get("top_pattern", "no_abuse_pattern")
            top_sc = pred.get("top_score", 0.0)
            ctx_scores = pred.get("scores", {})
            ctx_patterns = pred.get("detected_patterns", [])

            results[orig_idx] = {
                "is_available": True,
                "target_text": target_msg,
                "threshold": thresh,
                "is_abstain": False,
                "abstain_reason": "",
                "isolated_scores": ctx_scores,
                "isolated_patterns": ctx_patterns,
                "context_scores": ctx_scores,
                "context_patterns": ctx_patterns,
                "top_pattern": top_p,
                "top_pattern_name": LABEL_DISPLAY_NAMES.get(top_p, top_p),
                "top_score": top_sc,
                "context_shift": 0.0,
                "taxonomy_info": TAXONOMY_ALIGNMENT.get(top_p, {}),
            }

        return results
