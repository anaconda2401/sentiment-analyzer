"""
Unit and Integration Tests for AbuseDetector and its ChatSense AI integration.
Run with: pytest tests/test_abuse_integration.py -v
"""
import io
import os
import tempfile
import pytest
import pandas as pd
from app import app
from chatsense import AbuseDetector, DOCUMENTED_LABELS


@pytest.fixture(scope="module")
def abuse_detector():
    """Shared AbuseDetector fixture initialized once for testing."""
    detector = AbuseDetector(model_dir="data/models/abusedetector", default_threshold=0.55)
    return detector


class TestAbuseDetectorUnit:
    """Unit tests for the AbuseDetector class and inference engine."""

    def test_model_loads_locally(self, abuse_detector):
        assert abuse_detector.is_available is True
        assert abuse_detector.model is not None
        assert abuse_detector.tokenizer is not None
        assert abuse_detector.error_message == ""

    def test_inference_output_structure_and_labels(self, abuse_detector):
        text = "You are crazy, that never happened. You are imagining things."
        res = abuse_detector.predict(text)

        assert res["is_available"] is True
        assert "scores" in res
        assert "detected_patterns" in res
        assert "top_pattern" in res
        assert "top_score" in res
        assert "threshold" in res

        # Check all 8 labels exist and match exact documented names and ordering
        assert list(res["scores"].keys()) == DOCUMENTED_LABELS
        assert len(res["scores"]) == 8

        # Check score values are finite and within [0, 1]
        for lbl, score in res["scores"].items():
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

        assert res["top_pattern"] in DOCUMENTED_LABELS
        assert res["top_score"] == res["scores"][res["top_pattern"]]

    def test_empty_and_invalid_inputs(self, abuse_detector):
        for bad_input in ["", "   ", "\t\n", None, 12345]:
            res = abuse_detector.predict(bad_input)
            assert res["is_available"] is True
            assert res["detected_patterns"] == []
            assert res["top_pattern"] == "none"
            assert res["top_score"] == 0.0
            for lbl, score in res["scores"].items():
                assert score == 0.0

    def test_batch_prediction_consistency(self, abuse_detector):
        texts = [
            "You are crazy and making this up.",
            "",
            "Please stop yelling at me, I need space.",
            "Meeting at 3 PM.",
            "   ",
        ]
        results = abuse_detector.batch_predict(texts, batch_size=2)
        assert len(results) == len(texts)

        # Non-empty messages should have valid scores
        assert results[0]["is_available"] is True
        assert results[0]["top_score"] > 0.0

        # Empty messages should have 0.0 scores
        assert results[1]["top_score"] == 0.0
        assert results[4]["top_score"] == 0.0

    def test_configurable_threshold(self, abuse_detector):
        text = "Who were you talking to? Show me your phone right now."
        # Low threshold (0.05) should catch multiple patterns
        res_low = abuse_detector.predict(text, threshold=0.05)
        # Very high threshold (0.99) should catch fewer/no patterns
        res_high = abuse_detector.predict(text, threshold=0.99)

        assert len(res_low["detected_patterns"]) >= len(res_high["detected_patterns"])
        assert res_high["threshold"] == 0.99
        assert res_low["threshold"] == 0.05

    def test_cpu_device_fallback(self):
        # Force CPU device execution
        cpu_detector = AbuseDetector(
            model_dir="data/models/abusedetector",
            default_threshold=0.55,
            device="cpu",
        )
        assert cpu_detector.is_available is True
        assert cpu_detector.device.type == "cpu"
        res = cpu_detector.predict("I need some personal space right now.")
        assert res["is_available"] is True
        assert len(res["scores"]) == 8

    def test_missing_model_graceful_fallback(self):
        # Point to a nonexistent directory
        bad_detector = AbuseDetector(model_dir="data/models/nonexistent_dir_12345")
        assert bad_detector.is_available is False
        assert "not found" in bad_detector.error_message.lower()

        res = bad_detector.predict("Sample text")
        assert res["is_available"] is False
        assert "error" in res
        assert res["detected_patterns"] == []

        # Summarizing DataFrame should also return a clean unavailable response
        summary = bad_detector.summarize_dataframe(pd.DataFrame())
        assert summary["available"] is False
        assert "not found" in summary["error"].lower() or "unavailable" in summary["error"].lower()

    def test_summarize_dataframe(self, abuse_detector):
        df = pd.DataFrame([
            {
                "sender": "User1",
                "clean_message": "You made me scream, this is your fault!",
                "abuse_patterns": ["darvo", "victim_blaming"],
            },
            {
                "sender": "User2",
                "clean_message": "I will not speak to you like this. Give me space.",
                "abuse_patterns": ["healthy_boundary_setting"],
            },
            {
                "sender": "User1",
                "clean_message": "Okay fine.",
                "abuse_patterns": ["no_abuse_pattern"],
            },
        ])
        summary = abuse_detector.summarize_dataframe(df, threshold=0.55)
        assert summary["available"] is True
        assert summary["total_messages"] == 3
        # Messages with at least one non-neutral pattern
        assert summary["total_flagged"] >= 1
        assert "disclaimer" in summary
        assert "participant_patterns" in summary
        assert len(summary["participant_patterns"]) == 2


class TestFlaskIntegrationWithAbuseDetector:
    """Integration tests for the Flask web application with AbuseDetector."""

    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_index_page_loads(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "chatsense" in html
        assert "abuse_threshold" in html
        assert "Manipulation Detection Sensitivity" in html

    def test_analyze_route_with_abuse_detection(self, client):
        chat_content = (
            "20/10/2026, 10:00 AM - Alice: You are always making things up and lying to me.\n"
            "20/10/2026, 10:01 AM - Bob: I am not lying. Please stop accusing me.\n"
            "20/10/2026, 10:02 AM - Alice: If you loved me you would show me your texts.\n"
            "20/10/2026, 10:03 AM - Bob: That is a boundary violation. I need space.\n"
        )
        data = {
            "chat_file": (io.BytesIO(chat_content.encode("utf-8")), "chat.txt"),
            "abuse_threshold": "0.55",
        }
        resp = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")

        # Check existing sentiment and dashboard components are intact
        assert "report" in html
        assert "messages" in html
        assert "positive" in html
        assert "negative" in html
        assert "sentiment trend" in html
        assert "participants" in html

        # Check experimental manipulation pattern section is present
        assert "manipulation pattern detection (experimental)" in html
        assert "SamanthaStorm/abusedetector" in html
        assert "Analytical &amp; Educational Notice" in html or "Analytical & Educational Notice" in html
        assert "do not diagnose any person" in html
        assert "messages analyzed" in html
        assert "participant pattern breakdown" in html

    def test_ajax_filter_returns_abuse_stats(self, client):
        # 1. First run /analyze to create a session
        chat_content = (
            "20/10/2026, 10:00 AM - Alice: Hello there\n"
            "20/10/2026, 10:01 AM - Bob: Hi Alice\n"
        )
        data = {
            "chat_file": (io.BytesIO(chat_content.encode("utf-8")), "chat.txt"),
        }
        resp = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")

        # Extract session_id from HTML
        import re
        match = re.search(r"var SID=`([a-f0-9]+)`", html)
        assert match is not None
        session_id = match.group(1)

        # 2. Call /filter via AJAX
        filter_resp = client.post("/filter", json={
            "session_id": session_id,
            "from_date": "2026-10-20",
            "to_date": "2026-10-20",
        })
        assert filter_resp.status_code == 200
        json_data = filter_resp.get_json()
        assert "stats" in json_data
        assert "sender_stats" in json_data
        assert "abuse_stats" in json_data
        assert json_data["abuse_stats"]["available"] is True

    def test_clear_all_route(self, client):
        resp = client.post("/clear_all")
        assert resp.status_code == 200
        json_data = resp.get_json()
        assert json_data["status"] == "cleared"
        assert "message" in json_data
