import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

COLORS = {
    'Positive': '#10b981',
    'Negative': '#ef4444',
    'Neutral':  '#6b7280',
}

DARK_RC = {
    'axes.facecolor':   '#1a1d24',
    'figure.facecolor': '#0f1115',
    'grid.color':       '#2b303b',
    'text.color':       '#f0f2f5',
    'axes.labelcolor':  '#9ea3b0',
    'xtick.color':      '#9ea3b0',
    'ytick.color':      '#9ea3b0',
    'axes.edgecolor':   '#2b303b',
}


def _apply_theme():
    plt.style.use('dark_background')
    plt.rcParams.update(DARK_RC)


class Visualizer:
    def __init__(self, output_dir: str = 'static/images'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, name: str) -> str:
        """Save current figure and return the static-relative path."""
        path = os.path.join(self.output_dir, name)
        plt.savefig(path, transparent=True, bbox_inches='tight', dpi=120)
        plt.close()
        return f'images/{name}'

    def create_sentiment_distribution(self, df: pd.DataFrame) -> str:
        _apply_theme()
        col = 'context_sentiment' if 'context_sentiment' in df.columns else 'sentiment'
        if df.empty or col not in df.columns:
            return ''

        counts = df[col].value_counts()
        labels = counts.index.tolist()
        colors = [COLORS.get(c, '#ffffff') for c in labels]

        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'color': '#f0f2f5', 'fontsize': 11},
            pctdistance=0.78,
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_color('#f0f2f5')
        ax.set_title('Expressed Sentiment Distribution', color='#f0f2f5', fontsize=13, pad=12)
        return self._save('dist.png')

    def create_trend_over_time(self, df: pd.DataFrame) -> str:
        _apply_theme()
        if df.empty or 'date' not in df.columns:
            return ''

        score_col = 'context_score' if 'context_score' in df.columns else 'raw_score'
        df_copy = df.copy()

        try:
            # dayfirst=True handles common WhatsApp date formats like 27/08/2026
            df_copy['_dt'] = pd.to_datetime(df_copy['date'], dayfirst=True, errors='coerce')
            df_copy = df_copy.dropna(subset=['_dt'])
            if df_copy.empty:
                return ''

            trend = df_copy.groupby(df_copy['_dt'].dt.date)[score_col].mean()

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.fill_between(trend.index, trend.values, alpha=0.15, color='#3b82f6')
            ax.plot(trend.index, trend.values, color='#3b82f6', marker='o', markersize=4, linewidth=1.8)
            ax.axhline(0, color='#6b7280', linestyle='--', linewidth=0.8)
            ax.set_title('Sentiment Trend Over Time', color='#f0f2f5', fontsize=13)
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Avg Sentiment Score', fontsize=10)
            plt.xticks(rotation=45, ha='right')
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            plt.tight_layout()
            return self._save('trend.png')
        except Exception:
            return ''

    def create_participant_chart(self, df: pd.DataFrame) -> str:
        _apply_theme()
        if df.empty or 'sender' not in df.columns:
            return ''

        score_col = 'context_score' if 'context_score' in df.columns else 'raw_score'
        grp = df.groupby('sender')[score_col].mean().sort_values(ascending=True)

        if grp.empty:
            return ''

        bar_colors = [COLORS['Positive'] if v >= 0.05 else COLORS['Negative'] if v <= -0.05 else COLORS['Neutral']
                      for v in grp.values]

        fig, ax = plt.subplots(figsize=(8, max(3, len(grp) * 0.6)))
        ax.barh(grp.index, grp.values, color=bar_colors, height=0.5)
        ax.axvline(0, color='#6b7280', linewidth=0.8)
        ax.set_title('Avg Expressed Sentiment by Participant', color='#f0f2f5', fontsize=13)
        ax.set_xlabel('Average Sentiment Score', fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        plt.tight_layout()
        return self._save('participants.png')

    def create_confidence_histogram(self, df: pd.DataFrame) -> str:
        _apply_theme()
        if df.empty or 'confidence' not in df.columns:
            return ''

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['confidence'], bins=20, color='#3b82f6', edgecolor='#0f1115', alpha=0.85)
        ax.axvline(0.55, color='#f59e0b', linestyle='--', linewidth=1.2, label='Low-confidence threshold')
        ax.set_title('Prediction Confidence Distribution', color='#f0f2f5', fontsize=13)
        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_ylabel('Number of Messages', fontsize=10)
        ax.legend(facecolor='#1a1d24', edgecolor='#2b303b', labelcolor='#f0f2f5')
        plt.tight_layout()
        return self._save('confidence.png')
