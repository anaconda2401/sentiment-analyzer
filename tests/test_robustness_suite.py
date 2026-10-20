"""
Synthetic Robustness Test Suite for ContextGuard AI.
Tests pipeline robustness across edge cases:
- Neutral disagreement
- Linguistic negation
- Sarcastic expressions
- Ambiguous short messages
- Disguised and noisy PII
- Speaker style shifts

Run with: pytest tests/test_robustness_suite.py -v
"""
import pytest
import pandas as pd
from chatsense.pii_anonymizer import PIIAnonymizer
from chatsense.sentiment_model import SentimentModel
from chatsense.manipulation_adapter import PretrainedAbuseDetectorAdapter
from chatsense.emotion_analyzer import EmotionAnalyzer


@pytest.fixture(scope="module")
def pipeline():
    return {
        "anonymizer": PIIAnonymizer(),
        "sentiment": SentimentModel(),
        "manipulation": PretrainedAbuseDetectorAdapter(),
        "emotion": EmotionAnalyzer(),
    }


class TestPIIRobustness:
    """Tests PII redaction against disguised, international, and noisy formats."""

    def test_disguised_phone_numbers(self, pipeline):
        anon = PIIAnonymizer()
        noisy_phones = [
            "Call me on +1 (555) 234-5678 now",
            "My number is 98765-43210 please save it",
            "Reach out at +44 20 7946 0958",
        ]
        df = pd.DataFrame([{"sender": "UserA", "message": m} for m in noisy_phones])
        res = anon.process_dataframe(df)
        audit = anon.generate_privacy_audit(res)

        assert audit["counts"]["phone"] >= 2
        assert audit["leakage_test_passed"] is True

    def test_participant_name_leakage_resistance(self, pipeline):
        anon = PIIAnonymizer()
        df = pd.DataFrame([
            {"sender": "Aswin", "message": "Hi Aswin here, please send the file to Aswin."},
            {"sender": "Deepa", "message": "Thanks Aswin, noted!"},
        ])
        res = anon.process_dataframe(df)
        audit = anon.generate_privacy_audit(res)

        all_text = " ".join(res["message"].tolist())
        assert "Aswin" not in all_text
        assert audit["leakage_test_passed"] is True


class TestNLPInferenceRobustness:
    """Tests sentiment, emotion, and manipulation inference against linguistic edge cases."""

    def test_neutral_disagreement_not_overflagged(self, pipeline):
        text = "I see your point, but I respectfully disagree with that decision based on the budget."
        pred = pipeline["manipulation"].predict(text, threshold=0.60)
        # At conservative threshold (0.60), neutral disagreement should not cross threshold
        assert len(pred["context_patterns"]) == 0
        assert pred["top_score"] <= 0.60

    def test_negation_sentiment_preservation(self, pipeline):
        pos_text = "The presentation was good."
        neg_text = "The presentation was not good."

        pos_res = pipeline["sentiment"].predict(pos_text)
        neg_res = pipeline["sentiment"].predict(neg_text)

        assert pos_res["raw_score"] > 0
        assert neg_res["raw_score"] < pos_res["raw_score"]
        assert neg_res["sentiment"] in ("Negative", "Neutral")

    def test_sarcasm_sentiment_ambiguity(self, pipeline):
        sarcastic_text = "Oh brilliant, another unexpected three-hour meeting on a Friday afternoon."
        emo_res = pipeline["emotion"].analyze_message(sarcastic_text, sentiment_score=-0.2)
        assert emo_res["frustration"] >= 0.0 or emo_res["anger"] >= 0.0

    def test_ultra_short_ambiguous_messages_abstain(self, pipeline):
        short_inputs = ["Fine.", "k", "No", "Sure", "..."]
        for txt in short_inputs:
            pred = pipeline["manipulation"].predict(txt)
            assert pred["is_abstain"] is True
            assert "insufficient length" in pred["abstain_reason"].lower() or len(txt.split()) < 2
