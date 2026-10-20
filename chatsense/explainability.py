"""
Explainability & Evidence Attribution Module for ContextGuard AI.
Computes token-level importance using occlusion (leave-one-out word perturbation)
and formulates alternative interpretations to avoid overconfident attribution.

DISCLAIMER:
Token attribution scores indicate statistical sensitivity of the neural model's logits.
They DO NOT establish psychological causation, guilt, or deliberate human intent.
"""
import re
from typing import Dict, List, Any, Optional

ALTERNATIVE_EXPLANATIONS = {
    "gaslighting": (
        "Alternative interpretation: May indicate an honest recollection discrepancy, "
        "defensive pushback against an accusation, or miscommunication rather than deliberate reality distortion."
    ),
    "darvo": (
        "Alternative interpretation: Speaker may genuinely feel unfairly targeted or attacked, "
        "causing an emotionally reactive counter-defense rather than premeditated role reversal."
    ),
    "victim_blaming": (
        "Alternative interpretation: May reflect exasperation or perceived joint culpability "
        "rather than a systematic attempt to evade personal accountability."
    ),
    "emotional_manipulation": (
        "Alternative interpretation: Could represent an emotional plea, sincere expression of hurt, "
        "or dramatic hyperbole rather than coercive emotional leverage."
    ),
    "boundary_violation": (
        "Alternative interpretation: Could stem from conversational urgency, persistent follow-up, "
        "or social clumsiness rather than contempt for autonomy."
    ),
    "surveillance_accusation": (
        "Alternative interpretation: Might reflect genuine curiosity, planning logistics, "
        "or situational insecurity rather than controlling surveillance."
    ),
    "healthy_boundary_setting": (
        "Constructive signal: Represents appropriate, assertive self-protection and boundary demarcation. "
        "This pattern is healthy, not manipulative."
    ),
    "no_abuse_pattern": (
        "Standard conversational pattern: Language aligns with ordinary, non-adversarial dialogue."
    ),
}


class PredictionExplainer:
    """
    Generates explainability evidence using leave-one-out occlusion attribution
    and provides contextual alternatives and ambiguity notices.
    """

    def __init__(self, detector_adapter=None, max_words_to_evaluate: int = 15):
        self.adapter = detector_adapter
        self.max_words_to_evaluate = max_words_to_evaluate

    def explain_message(
        self,
        text: str,
        target_label: Optional[str] = None,
        context_messages: Optional[List[Dict[str, str]]] = None,
        base_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates occlusion-based token attribution for a target message.
        Identifies words whose removal decreases the target pattern probability.
        """
        words = re.findall(r'\b[\w\'-]+\b', text)
        if not words or not self.adapter or not self.adapter.is_available:
            return {
                "text": text,
                "target_label": target_label or "none",
                "attributions": [],
                "top_contributing_words": [],
                "alternative_interpretation": ALTERNATIVE_EXPLANATIONS.get(target_label, "No pattern detected."),
                "ambiguity_warning": "Model explanation unavailable.",
            }

        # Determine target label and baseline score if not supplied
        if target_label is None or base_score is None:
            pred = self.adapter.predict(text, context=context_messages)
            target_label = pred.get("top_pattern", "no_abuse_pattern")
            base_score = pred.get("top_score", 0.0)

        # Select top significant words to test (cap at 4 for ultra-fast attribution)
        candidate_words = [w for w in words if len(w) > 2][:4]
        if not candidate_words:
            candidate_words = words[:3]

        attributions = []
        if candidate_words and hasattr(self.adapter, "detector") and self.adapter.detector.is_available:
            perturbed_aug_inputs = []
            for word in candidate_words:
                perturbed_text = re.sub(r'\b' + re.escape(word) + r'\b', '', text, count=1).strip() or "..."
                perturbed_aug_inputs.append(self.adapter.format_context_string(context_messages, perturbed_text))

            try:
                batch_pert_preds = self.adapter.detector.batch_predict(
                    perturbed_aug_inputs, batch_size=len(perturbed_aug_inputs), max_length=96
                )
                for word, pert_p in zip(candidate_words, batch_pert_preds):
                    pert_score = pert_p.get("scores", {}).get(target_label, 0.0)
                    drop = round(base_score - pert_score, 4)
                    if drop > 0:
                        attributions.append({"word": word, "importance": drop})
            except Exception as err:
                log.warning("Batch occlusion attribution error: %s", err)

        # Sort by highest positive importance
        attributions.sort(key=lambda x: x["importance"], reverse=True)
        top_words = [item["word"] for item in attributions[:5]]

        # Assess ambiguity
        ambiguity_warning = None
        if base_score < 0.65 and base_score > 0.45:
            ambiguity_warning = (
                "Borderline score: Model confidence is near the classification threshold; "
                "contextual ambiguity is elevated."
            )
        elif len(words) < 4:
            ambiguity_warning = "Short utterance: Limited linguistic evidence may skew attribution."

        return {
            "text": text,
            "target_label": target_label,
            "base_score": base_score,
            "attributions": attributions,
            "top_contributing_words": top_words,
            "alternative_interpretation": ALTERNATIVE_EXPLANATIONS.get(
                target_label, "Language is open to multiple conversational interpretations."
            ),
            "ambiguity_warning": ambiguity_warning,
        }
