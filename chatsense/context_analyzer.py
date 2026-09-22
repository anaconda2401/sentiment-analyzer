import pandas as pd

DEFAULT_WINDOW = 4
BLEND_CURRENT = 0.7   # 70% current message, 30% context
BLEND_CONTEXT = 0.3


def _categorize(score: float) -> str:
    if score >= 0.05:
        return 'Positive'
    if score <= -0.05:
        return 'Negative'
    return 'Neutral'


class ContextAnalyzer:
    """
    Applies a sliding-window context adjustment to sentiment scores.

    Rationale:
        Short ambiguous messages like "Fine." or "Okay." are hard to classify
        in isolation. By blending the current score with a rolling average of
        the previous N messages, conversational context can shift the estimate.

    This does NOT claim to detect the person's true emotional state — it only
    adjusts the expressed sentiment estimate based on conversational context.

    If context is unavailable (first messages), min_periods=1 ensures
    the single-message score is used as-is.
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW):
        self.window_size = window_size

    def apply_context(self, df: pd.DataFrame, score_col: str = 'raw_score') -> pd.DataFrame:
        """
        Adds 'context_score' and 'context_sentiment' columns.
        Falls back gracefully if score_col is missing.
        """
        if df.empty or score_col not in df.columns:
            return df.copy()

        df_ctx = df.copy()

        rolling_avg = (
            df_ctx[score_col]
            .rolling(window=self.window_size, min_periods=1)
            .mean()
        )

        df_ctx['context_score'] = (
            BLEND_CURRENT * df_ctx[score_col] + BLEND_CONTEXT * rolling_avg
        ).round(4)

        df_ctx['context_sentiment'] = df_ctx['context_score'].apply(_categorize)

        return df_ctx
