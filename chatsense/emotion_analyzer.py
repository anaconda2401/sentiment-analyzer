"""
Emotion Trajectory & Change-Point Analysis Module for ContextGuard AI.
Estimates expressed emotional signals (joy, sadness, anger, fear, frustration, anxiety)
across conversational turns using lexicon-grounded affect cues, computing rolling-window
trajectories and identifying sudden change-points.

DISCLAIMER:
Expressed linguistic cues indicate conversational tone and stylistic shifts only.
They DO NOT establish a participant's true internal emotional or psychiatric state.
"""
import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Emotion lexicon terms grounded in NRC / Plutchik affect associations
EMOTION_LEXICON: Dict[str, set] = {
    "joy": {
        "happy", "glad", "great", "awesome", "love", "wonderful", "excited", "congrats",
        "yay", "delighted", "fantastic", "pleasure", "blessed", "enjoy", "smile", "laugh",
        "fun", "beautiful", "sweet", "good", "nice", "cheerful", "grateful", "celebrate"
    },
    "sadness": {
        "sad", "depressed", "unhappy", "sorrow", "grief", "heartbroken", "crying", "tears",
        "miserable", "hurt", "lonely", "down", "disappointed", "hopeless", "miss", "regret",
        "mourn", "gloomy", "melancholy", "painful", "despair"
    },
    "anger": {
        "angry", "mad", "furious", "hate", "rage", "irritated", "annoyed", "pissed",
        "yelling", "screaming", "disgusted", "offensive", "shut", "stupid", "idiot",
        "ridiculous", "horrible", "awful", "hostile", "bitter", "provoked"
    },
    "fear": {
        "scared", "afraid", "terrified", "panic", "horror", "threat", "danger", "dread",
        "alarmed", "shaking", "frightened", "creepy", "intimidated", "paralyzed"
    },
    "frustration": {
        "frustrated", "exhausted", "tired", "annoying", "fed", "stuck", "pointless",
        "useless", "impossible", "bother", "bothered", "waste", "ugh", "give", "done",
        "confused", "overwhelmed", "struggling", "interrupted"
    },
    "anxiety": {
        "anxious", "nervous", "worried", "worry", "stress", "stressed", "tense", "uneasy",
        "dreading", "apprehensive", "insecure", "jittery", "troubled", "hesitant", "scared"
    },
}

SUPPORTED_EMOTIONS = ["joy", "sadness", "anger", "fear", "frustration", "anxiety"]


class EmotionAnalyzer:
    """
    Computes message-level emotion distributions, rolling temporal trajectories,
    and change-points in conversational time-series.
    """

    def __init__(self, rolling_window: int = 5, change_point_threshold: float = 0.35):
        self.rolling_window = rolling_window
        self.change_point_threshold = change_point_threshold

    def analyze_message(self, text: str, sentiment_score: float = 0.0) -> Dict[str, float]:
        """
        Analyzes a single message and returns scores for each supported emotion in [0.0, 1.0].
        """
        if not isinstance(text, str) or not text.strip():
            return {emo: 0.0 for emo in SUPPORTED_EMOTIONS}

        words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
        if not words:
            return {emo: 0.0 for emo in SUPPORTED_EMOTIONS}

        raw_scores = {}
        for emo, lexicon in EMOTION_LEXICON.items():
            overlap = words.intersection(lexicon)
            base_score = len(overlap) / max(len(words), 1)

            # Modulate with sentiment alignment
            if emo == "joy":
                if sentiment_score > 0.1:
                    base_score += sentiment_score * 0.3
            elif emo in ("sadness", "anger", "fear", "frustration", "anxiety"):
                if sentiment_score < -0.1:
                    base_score += abs(sentiment_score) * 0.25

            raw_scores[emo] = round(float(min(1.0, max(0.0, base_score))), 3)

        return raw_scores

    def process_dataframe(self, df: pd.DataFrame, score_col: str = "raw_score") -> pd.DataFrame:
        """
        Enriches DataFrame with message-level emotion scores and rolling trajectory averages.
        """
        if df.empty or "clean_message" not in df.columns:
            for emo in SUPPORTED_EMOTIONS:
                df[f"emo_{emo}"] = 0.0
                df[f"emo_roll_{emo}"] = 0.0
            df["dominant_emotion"] = "neutral"
            return df

        scores_list = []
        dominant_list = []

        for _, row in df.iterrows():
            msg = row.get("clean_message", row.get("message", ""))
            s_score = row.get(score_col, 0.0)
            emo_dict = self.analyze_message(msg, s_score)
            scores_list.append(emo_dict)

            # Determine dominant emotion if any score exceeds 0.15
            max_emo = max(emo_dict.items(), key=lambda x: x[1])
            dominant_list.append(max_emo[0] if max_emo[1] >= 0.15 else "neutral")

        # Assign individual emotion columns
        for emo in SUPPORTED_EMOTIONS:
            df[f"emo_{emo}"] = [s[emo] for s in scores_list]
            df[f"emo_roll_{emo}"] = (
                df[f"emo_{emo}"]
                .rolling(window=self.rolling_window, min_periods=1)
                .mean()
                .round(3)
            )

        df["dominant_emotion"] = dominant_list
        return df

    def detect_change_points(self, df: pd.DataFrame, score_col: str = "raw_score") -> List[Dict[str, Any]]:
        """
        Detects sudden shifts in sentiment or intense negative emotions between consecutive turns.
        Returns a list of change-point event dictionaries.
        """
        if df.empty or len(df) < 2:
            return []

        change_points = []
        for i in range(1, len(df)):
            prev_row = df.iloc[i - 1]
            curr_row = df.iloc[i]

            prev_sent = prev_row.get(score_col, 0.0)
            curr_sent = curr_row.get(score_col, 0.0)
            sent_delta = curr_sent - prev_sent

            # 1. Sentiment Plunge
            if sent_delta <= -0.45:
                change_points.append({
                    "index": i,
                    "date": curr_row.get("date", ""),
                    "time": curr_row.get("time", ""),
                    "sender": curr_row.get("sender", "Unknown"),
                    "type": "Sentiment Plunge",
                    "description": f"Shift from {prev_sent:+.2f} to {curr_sent:+.2f} (Δ {sent_delta:.2f})",
                    "severity": "high" if sent_delta <= -0.7 else "medium",
                    "snippet": str(curr_row.get("clean_message", ""))[:80],
                })

            # 2. Sudden Anger Spike
            prev_anger = prev_row.get("emo_anger", 0.0)
            curr_anger = curr_row.get("emo_anger", 0.0)
            if (curr_anger - prev_anger) >= self.change_point_threshold:
                change_points.append({
                    "index": i,
                    "date": curr_row.get("date", ""),
                    "time": curr_row.get("time", ""),
                    "sender": curr_row.get("sender", "Unknown"),
                    "type": "Anger Escalation",
                    "description": f"Sudden surge in hostile or angry expressions (+{curr_anger - prev_anger:.2f})",
                    "severity": "medium",
                    "snippet": str(curr_row.get("clean_message", ""))[:80],
                })

            # 3. Sudden Anxiety / Distress Spike
            prev_anx = prev_row.get("emo_anxiety", 0.0)
            curr_anx = curr_row.get("emo_anxiety", 0.0)
            if (curr_anx - prev_anx) >= self.change_point_threshold:
                change_points.append({
                    "index": i,
                    "date": curr_row.get("date", ""),
                    "time": curr_row.get("time", ""),
                    "sender": curr_row.get("sender", "Unknown"),
                    "type": "Anxiety / Distress Surge",
                    "description": f"Acute rise in expressed anxiety or distress markers (+{curr_anx - prev_anx:.2f})",
                    "severity": "medium",
                    "snippet": str(curr_row.get("clean_message", ""))[:80],
                })

        return change_points[:15]  # Limit to top 15 most prominent events
