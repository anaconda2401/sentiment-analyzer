import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Confidence threshold below which a prediction is flagged as low-confidence
DEFAULT_CONFIDENCE_THRESHOLD = 0.55


class SentimentModel:
    """
    Estimates expressed sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

    VADER was chosen because:
    - It is specifically tuned for short social-media and chat text.
    - It natively handles punctuation emphasis (!!!), capitalisation (GREAT),
      emoji aliases (:thumbs_up:), and negation context (not good).
    - It is fully explainable — no black-box neural weights.
    - It is fast and requires no model training.

    Output:
        sentiment       : 'Positive' | 'Neutral' | 'Negative'
        confidence      : float 0.0 – 1.0
        is_low_confidence: bool (True when confidence < threshold)
        raw_score       : VADER compound score (-1.0 to 1.0)
    """

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.analyzer = SentimentIntensityAnalyzer()
        self.confidence_threshold = confidence_threshold

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'Neutral',
                'confidence': 0.0,
                'is_low_confidence': True,
                'raw_score': 0.0,
            }

        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = 'Positive'
            # Confidence: how strongly positive relative to any negative content
            confidence = min(abs(compound) + scores['pos'] * 0.5, 1.0)
        elif compound <= -0.05:
            sentiment = 'Negative'
            confidence = min(abs(compound) + scores['neg'] * 0.5, 1.0)
        else:
            sentiment = 'Neutral'
            # Neutral confidence: distinguish genuine neutral text from absence of evidence
            # Strip emoji aliases (e.g. :thumbs_up:), digits, and punctuation to inspect genuine words
            stripped_alpha = re.sub(r':[a-zA-Z0-9_]+:', ' ', text)
            words = [w for w in stripped_alpha.split() if any(c.isalpha() for c in w)]

            if not words:
                # Insufficient evidence: emoji-only, digits, symbols, or unmapped tokens
                # Represent uncertainty appropriately with low confidence
                confidence = 0.25
            elif len(words) < 2:
                # Ambiguous single-word response (e.g., 'fine', 'k')
                confidence = 0.40
            else:
                # Genuine multi-word factual/neutral statement
                confidence = min(scores['neu'] * 0.85, 0.85)

        confidence = round(max(0.0, min(1.0, confidence)), 2)
        is_low_confidence = confidence < self.confidence_threshold

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'is_low_confidence': is_low_confidence,
            'raw_score': round(compound, 4),
        }
