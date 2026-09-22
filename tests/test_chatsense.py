"""
Unit tests for ChatSense AI pipeline modules.
Run with: pytest tests/ -v
"""
import io
import os
import tempfile
import pandas as pd
# pyrefly: ignore [missing-import]
import pytest

from chatsense.whatsapp_parser import parse_whatsapp_chat, extract_hour
from chatsense.pii_anonymizer import PIIAnonymizer
from chatsense.text_preprocessor import TextPreprocessor
from chatsense.sentiment_model import SentimentModel
from chatsense.context_analyzer import ContextAnalyzer
from chatsense.chat_insights import ChatInsights
from chatsense.evaluation import Evaluator


# ===========================
# Helpers
# ===========================

def _write_chat(lines: list) -> str:
    """Write lines to a temp .txt file and return its path."""
    content = "\n".join(lines)
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(content)
    return path


# ===========================
# WhatsApp Parser
# ===========================

class TestWhatsAppParser:
    def test_basic_android_format(self):
        path = _write_chat([
            "27/08/2026, 8:02 PM - Alice: Hello there",
            "27/08/2026, 8:03 PM - Bob: Hi!",
        ])
        df = parse_whatsapp_chat(path)
        os.unlink(path)
        assert len(df) == 2
        assert df.iloc[0]['sender'] == 'Alice'
        assert df.iloc[0]['message'] == 'Hello there'

    def test_multiline_message(self):
        path = _write_chat([
            "27/08/2026, 8:02 PM - Alice: First line",
            "second line",
            "27/08/2026, 8:03 PM - Bob: Response",
        ])
        df = parse_whatsapp_chat(path)
        os.unlink(path)
        assert 'second line' in df.iloc[0]['message']
        assert len(df) == 2

    def test_system_messages_excluded(self):
        path = _write_chat([
            "27/08/2026, 8:02 PM - Alice: Hello",
            "27/08/2026, 8:03 PM - System: Messages and calls are end-to-end encrypted",
        ])
        df = parse_whatsapp_chat(path)
        os.unlink(path)
        assert len(df) == 1

    def test_media_omitted_excluded(self):
        path = _write_chat([
            "27/08/2026, 8:02 PM - Alice: <Media omitted>",
            "27/08/2026, 8:03 PM - Bob: Hello",
        ])
        df = parse_whatsapp_chat(path)
        os.unlink(path)
        assert len(df) == 1

    def test_empty_file(self):
        path = _write_chat([])
        df = parse_whatsapp_chat(path)
        os.unlink(path)
        assert df.empty

    def test_extract_hour_pm(self):
        assert extract_hour("8:30 PM") == 20

    def test_extract_hour_am(self):
        assert extract_hour("12:00 AM") == 0

    def test_extract_hour_invalid(self):
        assert extract_hour("not a time") is None


# ===========================
# PII Anonymizer
# ===========================

class TestPIIAnonymizer:
    def test_participant_name_replaced(self):
        pii = PIIAnonymizer()
        df = pd.DataFrame([{'sender': 'John', 'message': 'Hello John'}])
        result = pii.process_dataframe(df)
        assert result.iloc[0]['sender'] == 'User1'
        assert 'John' not in result.iloc[0]['message']

    def test_email_replaced(self):
        pii = PIIAnonymizer()
        df = pd.DataFrame([{'sender': 'Alice', 'message': 'Email me at alice@example.com'}])
        result = pii.process_dataframe(df)
        assert '[EMAIL]' in result.iloc[0]['message']
        assert '@' not in result.iloc[0]['message'] or '[EMAIL]' in result.iloc[0]['message']

    def test_phone_replaced(self):
        pii = PIIAnonymizer()
        df = pd.DataFrame([{'sender': 'Bob', 'message': 'Call me at +91 98765 43210'}])
        result = pii.process_dataframe(df)
        assert '[PHONE]' in result.iloc[0]['message']

    def test_url_replaced(self):
        pii = PIIAnonymizer()
        df = pd.DataFrame([{'sender': 'Carol', 'message': 'Check https://example.com/secret'}])
        result = pii.process_dataframe(df)
        assert '[URL]' in result.iloc[0]['message']

    def test_multiple_participants(self):
        pii = PIIAnonymizer()
        df = pd.DataFrame([
            {'sender': 'Alice', 'message': 'Hi'},
            {'sender': 'Bob',   'message': 'Hey'},
        ])
        result = pii.process_dataframe(df)
        assert result.iloc[0]['sender'] in ('User1', 'User2')
        assert result.iloc[1]['sender'] in ('User1', 'User2')
        assert result.iloc[0]['sender'] != result.iloc[1]['sender']

    def test_instances_are_independent(self):
        pii1 = PIIAnonymizer()
        pii2 = PIIAnonymizer()
        df = pd.DataFrame([{'sender': 'Alice', 'message': 'Hi'}])
        pii1.process_dataframe(df)
        result2 = pii2.process_dataframe(df)
        # pii2 should still map Alice -> User1 regardless of pii1's state
        assert result2.iloc[0]['sender'] == 'User1'


# ===========================
# Text Preprocessor
# ===========================

class TestTextPreprocessor:
    def setup_method(self):
        self.tp = TextPreprocessor()

    def test_negation_preserved(self):
        result = self.tp.preprocess("This is not good")
        assert 'not' in result

    def test_emoji_converted(self):
        result = self.tp.preprocess("Great job 😂")
        assert ':face_with_tears_of_joy:' in result

    def test_whitespace_normalised(self):
        result = self.tp.preprocess("Hello    world")
        assert '  ' not in result

    def test_repeated_chars_reduced(self):
        result = self.tp.preprocess("sooooo good!!!!!!")
        # Must not allow more than 2 consecutive identical chars
        import re
        assert not re.search(r'(.)\1{2,}', result)

    def test_empty_string(self):
        result = self.tp.preprocess("")
        assert result == ''

    def test_unicode_normalised(self):
        result = self.tp.preprocess("\u00e9")   # e + combining accent
        assert isinstance(result, str)


# ===========================
# Sentiment Model
# ===========================

class TestSentimentModel:
    def setup_method(self):
        self.sm = SentimentModel()

    def test_positive_prediction(self):
        r = self.sm.predict("This is absolutely amazing and wonderful!")
        assert r['sentiment'] == 'Positive'

    def test_negative_prediction(self):
        r = self.sm.predict("I hate this, it is terrible and awful")
        assert r['sentiment'] == 'Negative'

    def test_negation_not_good(self):
        r = self.sm.predict("This is not good at all")
        # "not good" should NOT be Positive
        assert r['sentiment'] != 'Positive'

    def test_confidence_bounded(self):
        r = self.sm.predict("Maybe okay I guess")
        assert 0.0 <= r['confidence'] <= 1.0

    def test_empty_text_handled(self):
        r = self.sm.predict("")
        assert r['sentiment'] == 'Neutral'
        assert r['is_low_confidence'] is True

    def test_low_confidence_flag(self):
        # Ambiguous short message
        r = self.sm.predict("Fine.")
        assert 'is_low_confidence' in r
        assert isinstance(r['is_low_confidence'], bool)

    def test_raw_score_range(self):
        for text in ["I love this", "I hate this", "Okay"]:
            r = self.sm.predict(text)
            assert -1.0 <= r['raw_score'] <= 1.0


# ===========================
# Context Analyzer
# ===========================

class TestContextAnalyzer:
    def test_context_columns_added(self):
        ca = ContextAnalyzer(window_size=3)
        df = pd.DataFrame({'raw_score': [0.5, 0.6, -0.1]})
        result = ca.apply_context(df)
        assert 'context_score' in result.columns
        assert 'context_sentiment' in result.columns

    def test_positive_context_shifts_ambiguous(self):
        ca = ContextAnalyzer(window_size=3)
        # Previous two messages very positive; last message mildly negative
        df = pd.DataFrame({'raw_score': [0.8, 0.7, -0.04]})
        result = ca.apply_context(df)
        # Context should pull the last score toward Positive
        assert result.iloc[2]['context_sentiment'] == 'Positive'

    def test_empty_dataframe(self):
        ca = ContextAnalyzer()
        df = pd.DataFrame(columns=['raw_score'])
        result = ca.apply_context(df)
        assert result.empty

    def test_fallback_when_score_col_missing(self):
        ca = ContextAnalyzer()
        df = pd.DataFrame({'sentiment': ['Positive', 'Neutral']})
        result = ca.apply_context(df)
        # Should return unchanged (no crash)
        assert 'context_score' not in result.columns


# ===========================
# Chat Insights
# ===========================

class TestChatInsights:
    def _make_df(self):
        return pd.DataFrame({
            'sender':           ['User1', 'User2', 'User1'],
            'message':          ['Great!', 'Fine.', 'Terrible'],
            'context_sentiment': ['Positive', 'Neutral', 'Negative'],
            'context_score':    [0.6, 0.0, -0.5],
            'confidence':       [0.9, 0.5, 0.8],
            'is_low_confidence': [False, True, False],
        })

    def test_counts_correct(self):
        ins = ChatInsights()
        stats = ins.compute_insights(self._make_df())
        assert stats['total_messages'] == 3
        assert stats['positive_count'] == 1
        assert stats['neutral_count'] == 1
        assert stats['negative_count'] == 1

    def test_percentages_sum_to_100(self):
        ins = ChatInsights()
        stats = ins.compute_insights(self._make_df())
        total = stats['positive_pct'] + stats['neutral_pct'] + stats['negative_pct']
        assert abs(total - 100.0) < 0.5

    def test_participant_stats_present(self):
        ins = ChatInsights()
        stats = ins.compute_insights(self._make_df())
        assert len(stats['participant_stats']) == 2

    def test_empty_dataframe(self):
        ins = ChatInsights()
        assert ins.compute_insights(pd.DataFrame()) == {}


# ===========================
# Evaluator
# ===========================

class TestEvaluator:
    def test_evaluation_runs(self):
        ev = Evaluator()
        sm = SentimentModel()
        tp = TextPreprocessor()
        metrics, per_class, cm_path = ev.run_evaluation(sm, tp, output_dir='static/images')
        assert 'accuracy' in metrics
        assert 'Positive' in per_class
        assert 'Negative' in per_class
        assert 'Neutral' in per_class

    def test_metrics_bounded(self):
        ev = Evaluator()
        sm = SentimentModel()
        tp = TextPreprocessor()
        metrics, _, _ = ev.run_evaluation(sm, tp, output_dir='static/images')
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['weighted_f1'] <= 1.0
