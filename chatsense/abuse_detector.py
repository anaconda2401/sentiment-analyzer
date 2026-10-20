"""
Abuse & Manipulation Pattern Detector Module for ChatSense AI.
Wraps the pretrained SamanthaStorm/abusedetector model (DistilBERT + Linear Classifier)
for role-aware, multi-label linguistic pattern detection.

DISCLAIMER:
This module provides experimental detection of linguistic patterns for educational
and analytical awareness. Its outputs are uncertain, context-dependent, and MUST NOT
be interpreted as proof of abuse, intent, or clinical diagnosis.
"""
import os
import logging
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel

log = logging.getLogger(__name__)

DOCUMENTED_LABELS: List[str] = [
    "surveillance_accusation",
    "darvo",
    "gaslighting",
    "boundary_violation",
    "emotional_manipulation",
    "victim_blaming",
    "healthy_boundary_setting",
    "no_abuse_pattern",
]

LABEL_DISPLAY_NAMES: Dict[str, str] = {
    "surveillance_accusation": "Surveillance / Accusation",
    "darvo": "DARVO (Deny, Attack, Reverse)",
    "gaslighting": "Gaslighting",
    "boundary_violation": "Boundary Violation",
    "emotional_manipulation": "Emotional Manipulation",
    "victim_blaming": "Victim Blaming",
    "healthy_boundary_setting": "Healthy Boundary Setting",
    "no_abuse_pattern": "No Abuse Pattern",
}

CONCERNING_PATTERNS: List[str] = [
    "surveillance_accusation",
    "darvo",
    "gaslighting",
    "boundary_violation",
    "emotional_manipulation",
    "victim_blaming",
]

DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "models", "abusedetector")
)


class AbusePatternDetector(nn.Module):
    """
    AbusePatternDetector architecture matching modeling_abuse.py and
    config.json from SamanthaStorm/abusedetector.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class AbuseDetector:
    """
    High-level manager for the AbusePatternDetector model.
    Handles device placement, batch inference, thresholding, and graceful fallbacks.
    """
    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        default_threshold: float = 0.50,
        device: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.default_threshold = default_threshold
        self.is_available = False
        self.error_message = ""
        self.tokenizer = None
        self.model = None

        # Select compute device: user-specified, CUDA if available, or CPU fallback
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()

    def _load_model(self) -> None:
        """Loads tokenizer and weights from local directory without downloading."""
        try:
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

            ckpt_path = os.path.join(self.model_dir, "pytorch_model.bin")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

            log.info("Loading AbuseDetector tokenizer from %s", self.model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)

            log.info("Initializing AbusePatternDetector model on %s", self.device)
            model = AbusePatternDetector("distilbert-base-uncased", num_labels=len(DOCUMENTED_LABELS))

            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            except Exception as cuda_err:
                if self.device.type == "cuda":
                    log.warning("Failed to load on CUDA (%s), falling back to CPU", cuda_err)
                    self.device = torch.device("cpu")
                    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                else:
                    raise

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            self.model = model
            self.is_available = True
            log.info("AbuseDetector successfully initialized on device: %s", self.device)

        except Exception as e:
            self.is_available = False
            self.error_message = str(e)
            self.model = None
            self.tokenizer = None
            log.warning("AbuseDetector failed to load (%s). Features will be disabled gracefully.", e)

    def predict(self, text: str, threshold: Optional[float] = None, max_length: int = 96) -> Dict[str, Any]:
        """
        Runs inference on a single text string.
        Returns a dictionary containing all label scores, detected patterns, and top pattern.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        if not self.is_available or self.model is None or self.tokenizer is None:
            return {
                "is_available": False,
                "error": self.error_message or "Model not loaded",
                "scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                "detected_patterns": [],
                "top_pattern": "none",
                "top_score": 0.0,
                "threshold": thresh,
            }

        if not isinstance(text, str) or not text.strip():
            return {
                "is_available": True,
                "scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                "detected_patterns": [],
                "top_pattern": "none",
                "top_score": 0.0,
                "threshold": thresh,
            }

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(logits)[0].cpu().tolist()

            scores = {lbl: round(float(p), 4) for lbl, p in zip(DOCUMENTED_LABELS, probs)}
            detected = [lbl for lbl, sc in scores.items() if sc >= thresh]
            top_pattern = max(scores.items(), key=lambda x: x[1])[0]
            top_score = scores[top_pattern]

            return {
                "is_available": True,
                "scores": scores,
                "detected_patterns": detected,
                "top_pattern": top_pattern,
                "top_score": top_score,
                "threshold": thresh,
            }
        except Exception as err:
            log.error("Inference error in AbuseDetector: %s", err)
            return {
                "is_available": True,
                "error": str(err),
                "scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                "detected_patterns": [],
                "top_pattern": "none",
                "top_score": 0.0,
                "threshold": thresh,
            }

    def batch_predict(
        self, texts: List[str], batch_size: int = 32, threshold: Optional[float] = None, max_length: int = 96
    ) -> List[Dict[str, Any]]:
        """
        Runs batched inference across a list of texts for high throughput.
        Safe against empty lists and non-string elements.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        if not texts:
            return []

        if not self.is_available or self.model is None or self.tokenizer is None:
            return [self.predict("", thresh) for _ in texts]

        results: List[Dict[str, Any]] = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_slice = texts[i : i + batch_size]
            cleaned_slice = [t if isinstance(t, str) and t.strip() else "" for t in batch_slice]

            # Separate non-empty from empty to avoid useless tensor computations
            valid_indices = [idx for idx, t in enumerate(cleaned_slice) if t]
            valid_texts = [cleaned_slice[idx] for idx in valid_indices]

            batch_results = [
                {
                    "is_available": True,
                    "scores": {lbl: 0.0 for lbl in DOCUMENTED_LABELS},
                    "detected_patterns": [],
                    "top_pattern": "none",
                    "top_score": 0.0,
                    "threshold": thresh,
                }
                for _ in batch_slice
            ]

            if valid_texts:
                try:
                    inputs = self.tokenizer(
                        valid_texts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=max_length,
                    )
                    input_ids = inputs["input_ids"].to(self.device)
                    attention_mask = inputs["attention_mask"].to(self.device)

                    with torch.no_grad():
                        logits = self.model(input_ids, attention_mask)
                        probs = torch.sigmoid(logits).cpu().tolist()

                    for local_idx, orig_pos in enumerate(valid_indices):
                        p_list = probs[local_idx]
                        sc_dict = {lbl: round(float(p), 4) for lbl, p in zip(DOCUMENTED_LABELS, p_list)}
                        detected = [lbl for lbl, sc in sc_dict.items() if sc >= thresh]
                        top_p = max(sc_dict.items(), key=lambda x: x[1])[0]
                        batch_results[orig_pos] = {
                            "is_available": True,
                            "scores": sc_dict,
                            "detected_patterns": detected,
                            "top_pattern": top_p,
                            "top_score": sc_dict[top_p],
                            "threshold": thresh,
                        }
                except Exception as b_err:
                    log.error("Batch inference error: %s", b_err)
                    # Fallback to single prediction for this batch
                    for orig_pos, t in enumerate(cleaned_slice):
                        batch_results[orig_pos] = self.predict(t, thresh)

            results.extend(batch_results)

        return results

    def summarize_dataframe(
        self, df: pd.DataFrame, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Computes aggregate pattern statistics for the dashboard.
        Does NOT expose private message text or sensitive identifiers.
        """
        thresh = threshold if threshold is not None else self.default_threshold

        if not self.is_available:
            return {
                "available": False,
                "error": self.error_message or "Manipulation detection model is currently unavailable.",
                "threshold": thresh,
                "total_flagged": 0,
                "pattern_counts": {},
                "participant_patterns": [],
                "disclaimer": (
                    "Experimental Research Model: Linguistic pattern detection is disabled "
                    "because the local model could not be loaded."
                ),
            }

        if df.empty or "abuse_patterns" not in df.columns:
            return {
                "available": True,
                "threshold": thresh,
                "total_flagged": 0,
                "pattern_counts": {LABEL_DISPLAY_NAMES[l]: 0 for l in DOCUMENTED_LABELS},
                "participant_patterns": [],
                "disclaimer": (
                    "Experimental Research Model: Patterns reflect linguistic similarity to "
                    "training samples from a single study and are for educational awareness only."
                ),
            }

        # Count occurrences per pattern across all messages
        pattern_counts = {lbl: 0 for lbl in DOCUMENTED_LABELS}
        total_flagged_messages = 0

        for pattern_list in df["abuse_patterns"]:
            if isinstance(pattern_list, list) and pattern_list:
                # Count non-neutral patterns
                active_patterns = [p for p in pattern_list if p in pattern_counts]
                for p in active_patterns:
                    pattern_counts[p] += 1
                # Only flag message if it exhibits a concerning manipulation pattern (not healthy boundaries)
                if any(p in CONCERNING_PATTERNS for p in active_patterns):
                    total_flagged_messages += 1

        formatted_counts = [
            {
                "key": lbl,
                "name": LABEL_DISPLAY_NAMES.get(lbl, lbl),
                "count": pattern_counts[lbl],
                "pct": round((pattern_counts[lbl] / len(df)) * 100, 1) if len(df) > 0 else 0.0,
                "is_healthy": lbl in ("healthy_boundary_setting", "no_abuse_pattern"),
            }
            for lbl in DOCUMENTED_LABELS
        ]

        # Participant breakdown
        participant_patterns = []
        if "sender" in df.columns:
            for sender, group in df.groupby("sender"):
                sender_counts = {lbl: 0 for lbl in DOCUMENTED_LABELS}
                for plist in group["abuse_patterns"]:
                    if isinstance(plist, list):
                        for p in plist:
                            if p in sender_counts:
                                sender_counts[p] += 1

                # Prioritize concerning patterns detected
                top_detected = [
                    LABEL_DISPLAY_NAMES.get(l, l)
                    for l, c in sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)
                    if c > 0 and l in CONCERNING_PATTERNS
                ][:3]

                participant_patterns.append({
                    "sender": sender,
                    "total_messages": len(group),
                    "patterns_count": sum(
                        c for l, c in sender_counts.items() if l in CONCERNING_PATTERNS
                    ),
                    "top_patterns": top_detected if top_detected else ["None detected"],
                    "healthy_boundaries": sender_counts["healthy_boundary_setting"],
                })

        return {
            "available": True,
            "threshold": thresh,
            "total_flagged": total_flagged_messages,
            "total_messages": len(df),
            "flagged_pct": round((total_flagged_messages / len(df)) * 100, 1) if len(df) > 0 else 0.0,
            "pattern_counts": formatted_counts,
            "participant_patterns": participant_patterns,
            "disclaimer": (
                "Educational Disclaimer: These linguistic indicators represent experimental pattern matching "
                "from SamanthaStorm/abusedetector. They do NOT evaluate character, diagnose personality, "
                "or prove abusive intent."
            ),
        }
