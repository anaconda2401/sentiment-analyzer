import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ['Positive', 'Neutral', 'Negative']


def _synthetic_labeled_set():
    """
    A small hand-labelled dataset used to evaluate the pipeline.
    These are representative WhatsApp-style messages.
    """
    samples = [
        ("This is amazing! I love it!!", "Positive"),
        ("Great work, really impressed!", "Positive"),
        ("So happy you came today", "Positive"),
        ("Thanks a lot, you made my day", "Positive"),
        ("Sounds good to me!", "Positive"),
        ("This is terrible, I hate it", "Negative"),
        ("Worst experience ever", "Negative"),
        ("I'm so disappointed in you", "Negative"),
        ("Never do that again", "Negative"),
        ("This makes me really angry", "Negative"),
        ("Okay", "Neutral"),
        ("Fine.", "Neutral"),
        ("Noted.", "Neutral"),
        ("I'll check later", "Neutral"),
        ("Let me know", "Neutral"),
    ]
    texts = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    return texts, labels


class Evaluator:
    """
    Evaluates the sentiment pipeline on a labelled dataset.
    Generates per-class and aggregate metrics and a confusion matrix chart.
    """

    def run_evaluation(self, sentiment_model, text_preprocessor, output_dir: str = 'static/images'):
        """
        Runs the model on the synthetic labelled set and returns:
        - metrics dict with accuracy, weighted precision/recall/F1
        - per_class dict with per-label metrics
        - confusion matrix image path
        """
        texts, y_true = _synthetic_labeled_set()
        y_pred = [
            sentiment_model.predict(text_preprocessor.preprocess(t))['sentiment']
            for t in texts
        ]

        acc = round(accuracy_score(y_true, y_pred), 3)

        # Weighted metrics
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average='weighted', zero_division=0
        )
        # Macro metrics
        _, _, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average='macro', zero_division=0
        )

        # Per-class metrics
        p_pc, r_pc, f1_pc, sup_pc = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average=None, zero_division=0
        )

        per_class = {}
        for i, label in enumerate(LABELS):
            per_class[label] = {
                'precision': round(p_pc[i], 3),
                'recall':    round(r_pc[i], 3),
                'f1':        round(f1_pc[i], 3),
                'support':   int(sup_pc[i]),
            }

        metrics = {
            'accuracy':           acc,
            'weighted_precision': round(p_w, 3),
            'weighted_recall':    round(r_w, 3),
            'weighted_f1':        round(f1_w, 3),
            'macro_f1':           round(f1_m, 3),
        }

        cm_path = self._save_confusion_matrix(y_true, y_pred, output_dir)
        return metrics, per_class, cm_path

    def _save_confusion_matrix(self, y_true, y_pred, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABELS, yticklabels=LABELS,
            ax=ax, linewidths=0.5, linecolor='#2b303b',
        )
        ax.set_title('Confusion Matrix', color='#f0f2f5', fontsize=13, pad=10)
        ax.set_xlabel('Predicted', color='#9ea3b0', fontsize=10)
        ax.set_ylabel('Actual', color='#9ea3b0', fontsize=10)
        ax.tick_params(colors='#9ea3b0')
        plt.tight_layout()

        path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(path, transparent=True, bbox_inches='tight', dpi=120)
        plt.close()
        return 'images/confusion_matrix.png'
