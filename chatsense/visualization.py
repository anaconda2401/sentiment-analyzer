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
        """Save current figure and return the path relative to Flask's static folder.

        output_dir is always 'static/images' or 'static/images/<session_id>'.
        We strip the leading 'static/' so the returned value can be passed
        directly to url_for('static', filename=...) or prepended with '/static/'.
        """
        path = os.path.join(self.output_dir, name)
        plt.savefig(path, transparent=True, bbox_inches='tight', dpi=120)
        plt.close()
        # Normalize to forward slashes (Windows uses backslashes in os.path.join)
        norm = self.output_dir.replace('\\', '/')
        # Strip the 'static/' prefix to get the Flask-static-relative path
        if norm.startswith('static/'):
            norm = norm[len('static/'):]
        return f'{norm}/{name}'

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
            # Reuse pre-parsed datetime if available, otherwise parse
            if '_dt' not in df_copy.columns or df_copy['_dt'].dropna().empty:
                df_copy['_dt'] = pd.to_datetime(df_copy['date'], dayfirst=True, errors='coerce')
            df_copy = df_copy.dropna(subset=['_dt'])
            if df_copy.empty:
                return ''

            trend = df_copy.groupby(df_copy['_dt'].dt.date)[score_col].mean()

            fig, ax = plt.subplots(figsize=(10, 4))
            if len(trend) <= 1:
                # Single day: plot turn-by-turn trajectory across conversation turns
                turns = list(range(1, len(df_copy) + 1))
                scores = df_copy[score_col].tolist()
                ax.plot(turns, scores, color='#3b82f6', marker='o', markersize=4, linewidth=1.8, label='Turn Polarity')
                if len(scores) >= 5:
                    w = max(2, min(5, len(scores) // 3))
                    rolling = pd.Series(scores).rolling(window=w, min_periods=1).mean()
                    ax.plot(turns, rolling, color='#10b981', linestyle='--', linewidth=1.5, label='Rolling Trend')
                    ax.legend(loc='upper right', facecolor='#111113', edgecolor='#2b303b', fontsize=9)
                ax.fill_between(turns, scores, alpha=0.15, color='#3b82f6')
                single_date_str = str(trend.index[0]) if len(trend) > 0 else 'Single Day'
                ax.set_title(f'Turn-by-Turn Sentiment Progression ({single_date_str})', color='#f0f2f5', fontsize=13)
                ax.set_xlabel('Conversational Message Sequence (Turn #)', fontsize=10)
                ax.set_ylabel('Sentiment Polarity (-1 to +1)', fontsize=10)
                ax.set_ylim(-1.05, 1.05)
            else:
                ax.fill_between(trend.index, trend.values, alpha=0.15, color='#3b82f6')
                ax.plot(trend.index, trend.values, color='#3b82f6', marker='o', markersize=4, linewidth=1.8)
                ax.set_title('Sentiment Trend Over Time', color='#f0f2f5', fontsize=13)
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Sentiment Polarity (-1 to +1)', fontsize=10)
                ax.set_ylim(-1.05, 1.05)
                plt.xticks(rotation=45, ha='right')
            ax.axhline(0, color='#6b7280', linestyle='--', linewidth=0.8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.2f'))
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

        fig, ax = plt.subplots(figsize=(8, max(3, len(grp) * 0.8)))
        bars = ax.barh(grp.index, grp.values, color=bar_colors, height=0.5)
        ax.axvline(0, color='#6b7280', linewidth=0.8)
        ax.set_title('Avg Expressed Sentiment by Participant', color='#f0f2f5', fontsize=13)
        ax.set_xlabel('Sentiment Polarity (-1.0 Negative to +1.0 Positive)', fontsize=10)
        ax.set_xlim(-1.05, 1.05)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%+.2f'))

        if len(grp) == 1:
            # Single participant — annotate the bar value directly since axis ticks may be sparse
            for bar, val in zip(bars, grp.values):
                ax.text(
                    val + (0.01 if val >= 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}',
                    va='center',
                    ha='left' if val >= 0 else 'right',
                    color='#f0f2f5', fontsize=10,
                )

        plt.tight_layout()
        return self._save('participants.png')

    def create_confidence_histogram(self, df: pd.DataFrame) -> str:
        _apply_theme()
        if df.empty or 'confidence' not in df.columns:
            return ''

        valid_conf = df['confidence'].dropna()
        if valid_conf.empty:
            return ''

        n_msgs = len(valid_conf)
        # Auto-scale bins: at least 1, at most 20; for very small sets just use 1 bin per message
        bins = max(1, min(20, n_msgs))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid_conf, bins=bins, color='#3b82f6', edgecolor='#0f1115', alpha=0.85)
        ax.axvline(0.55, color='#f59e0b', linestyle='--', linewidth=1.2, label='Low-confidence threshold')
        ax.set_title('Prediction Confidence Distribution', color='#f0f2f5', fontsize=13)
        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_ylabel('Number of Messages', fontsize=10)
        ax.legend(facecolor='#1a1d24', edgecolor='#2b303b', labelcolor='#f0f2f5')

        if n_msgs == 1:
            # Single message — show exact value as annotation instead of an empty histogram bar
            val = float(valid_conf.iloc[0])
            ax.text(
                val, 0.5, f'{val:.2f}',
                ha='center', va='bottom', color='#f0f2f5', fontsize=11,
                transform=ax.get_xaxis_transform(),
            )

        plt.tight_layout()
        return self._save('confidence.png')

    def create_emotion_trajectory(self, df: pd.DataFrame) -> str:
        _apply_theme()
        if df.empty or 'emo_roll_joy' not in df.columns:
            return ''

        fig, ax = plt.subplots(figsize=(10, 4.2))
        x = range(len(df))

        emo_colors = {
            'emo_roll_joy': ('Joy', '#10b981'),
            'emo_roll_anger': ('Anger', '#ef4444'),
            'emo_roll_sadness': ('Sadness', '#60a5fa'),
            'emo_roll_anxiety': ('Anxiety', '#c084fc'),
            'emo_roll_frustration': ('Frustration', '#fb923c'),
        }

        any_plotted = False
        for col, (label, color) in emo_colors.items():
            if col in df.columns and df[col].max() > 0.05:
                ax.plot(x, df[col], label=label, color=color, linewidth=1.8, alpha=0.85)
                any_plotted = True

        if not any_plotted:
            # Fallback if all values are low
            for col, (label, color) in emo_colors.items():
                if col in df.columns:
                    ax.plot(x, df[col], label=label, color=color, linewidth=1.5, alpha=0.6)

        ax.set_title('Temporal Emotion Trajectories (5-turn rolling average)', color='#f0f2f5', fontsize=13)
        ax.set_xlabel('Conversational Turn Index', fontsize=10)
        ax.set_ylabel('Estimated Intensity', fontsize=10)
        ax.set_ylim(-0.02, 1.0)
        ax.legend(facecolor='#1a1d24', edgecolor='#2b303b', labelcolor='#f0f2f5', loc='upper right')
        plt.tight_layout()
        return self._save('emotions_trajectory.png')

