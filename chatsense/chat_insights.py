import pandas as pd


def _pct(count: int, total: int) -> float:
    return round((count / total) * 100, 1) if total > 0 else 0.0


class ChatInsights:
    """
    Computes aggregate statistics from an anonymised, sentiment-labelled DataFrame.

    All statistics are described as 'expressed sentiment' — not as a measure of
    the person's true psychological state.
    """

    def compute_insights(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}

        total = len(df)

        # Use context-adjusted sentiment when available
        sentiment_col = 'context_sentiment' if 'context_sentiment' in df.columns else 'sentiment'
        score_col = 'context_score' if 'context_score' in df.columns else 'raw_score'

        counts = df[sentiment_col].value_counts().to_dict()
        pos = counts.get('Positive', 0)
        neg = counts.get('Negative', 0)
        neu = counts.get('Neutral', 0)

        avg_score = round(df[score_col].mean(), 3) if score_col in df.columns else 0.0
        avg_confidence = round(df['confidence'].mean(), 2) if 'confidence' in df.columns else 0.0
        low_conf_count = int(df['is_low_confidence'].sum()) if 'is_low_confidence' in df.columns else 0

        participants = df['sender'].nunique() if 'sender' in df.columns else 0

        participant_stats = []
        if 'sender' in df.columns:
            grp = df.groupby('sender').agg(
                message_count=('message', 'count'),
                avg_sentiment=(score_col, 'mean'),
                avg_confidence=('confidence', 'mean'),
            ).reset_index()
            grp['avg_sentiment'] = grp['avg_sentiment'].round(3)
            grp['avg_confidence'] = grp['avg_confidence'].round(2)
            participant_stats = grp.to_dict('records')

        return {
            'total_messages': total,
            'positive_count': pos,
            'negative_count': neg,
            'neutral_count': neu,
            'positive_pct': _pct(pos, total),
            'negative_pct': _pct(neg, total),
            'neutral_pct': _pct(neu, total),
            'average_score': avg_score,
            'average_confidence': avg_confidence,
            'low_confidence_count': low_conf_count,
            'participant_count': participants,
            'participant_stats': participant_stats,
        }
