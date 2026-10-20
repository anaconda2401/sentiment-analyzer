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
    Returns a comprehensive, balanced benchmark dataset of 120 WhatsApp-style messages
    (40 Positive, 40 Negative, 40 Neutral) to evaluate the sentiment pipeline.
    """
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sentiment_eval_dataset.csv"))
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and "message" in df.columns and "sentiment" in df.columns:
                return df["message"].astype(str).tolist(), df["sentiment"].astype(str).tolist()
        except Exception:
            pass

    # Built-in balanced 120-sample WhatsApp conversational benchmark
    samples = [
        # Positive (40)
        ("This is amazing! I love it so much!!", "Positive"),
        ("Great work on the project, really impressed!", "Positive"),
        ("So happy you came to the party today! ❤️", "Positive"),
        ("Thanks a lot, you totally made my day!", "Positive"),
        ("Sounds fantastic to me, let's do it! 🚀", "Positive"),
        ("Congratulations on passing your exams!! So proud of you!", "Positive"),
        ("You are the best! Thank you for the ride.", "Positive"),
        ("I loved the dinner yesterday, the food was delicious.", "Positive"),
        ("Haha that is hilarious 😂😂😂", "Positive"),
        ("Super excited for our trip this weekend!!", "Positive"),
        ("Awesome news, well deserved promotion!", "Positive"),
        ("Thanks for checking in on me, feeling much better today.", "Positive"),
        ("You did a wonderful job with the presentation.", "Positive"),
        ("Yes absolutely, that works perfectly for me!", "Positive"),
        ("Loved spending time with you all today.", "Positive"),
        ("Such a beautiful photo! Where was this taken?", "Positive"),
        ("Really appreciate your quick response and support.", "Positive"),
        ("That is such a relief, thank goodness!", "Positive"),
        ("Brilliant idea, let's pitch that tomorrow.", "Positive"),
        ("So grateful for your guidance throughout this week.", "Positive"),
        ("The new update looks clean and fast! Good job team.", "Positive"),
        ("Had a blast at the concert last night! 🎉", "Positive"),
        ("Everything is going so smoothly, thank you!", "Positive"),
        ("Sending lots of love and best wishes your way! ✨", "Positive"),
        ("Yay! Can't wait to see you soon!", "Positive"),
        ("That was extremely kind of you, thank you.", "Positive"),
        ("I'm really enjoying reading this book you recommended.", "Positive"),
        ("Best weekend ever with the family! 😊", "Positive"),
        ("You're a lifesaver, seriously thank you!", "Positive"),
        ("Super happy with how the design turned out.", "Positive"),
        ("Good morning! Wishing you a productive and happy day ahead.", "Positive"),
        ("That's wonderful news, congratulations to both of you!", "Positive"),
        ("I am so impressed by your dedication and hard work.", "Positive"),
        ("Thank you so much for the thoughtful gift, I love it!", "Positive"),
        ("We won the match!! Incredible performance from everyone!", "Positive"),
        ("Totally agree with you, excellent suggestion.", "Positive"),
        ("Feeling very energized and inspired after our meeting.", "Positive"),
        ("Such great memories from college! Miss you all.", "Positive"),
        ("It was an absolute pleasure working with you on this.", "Positive"),
        ("Cheers to a great start of the new semester! 🥂", "Positive"),

        # Negative (40)
        ("This is terrible, I absolutely hate it.", "Negative"),
        ("Worst experience ever trying to contact support.", "Negative"),
        ("I'm so disappointed in how this turned out.", "Negative"),
        ("Never do that again, that was completely unacceptable.", "Negative"),
        ("This makes me really angry and frustrated.", "Negative"),
        ("The service was horrible and the staff was rude.", "Negative"),
        ("I've been waiting for two hours and nobody showed up.", "Negative"),
        ("My phone is broken and all my files are lost. 😭", "Negative"),
        ("I am sick of dealing with these constant server crashes.", "Negative"),
        ("This solution is broken, inefficient, and painfully slow.", "Negative"),
        ("I'm feeling really down and stressed out today.", "Negative"),
        ("Stop sending these annoying spam messages in the group.", "Negative"),
        ("The code is throwing errors everywhere and nothing compiles.", "Negative"),
        ("Such a waste of time and money, completely regret it.", "Negative"),
        ("I am extremely upset with what you said yesterday.", "Negative"),
        ("The train got delayed by three hours, so exhausted.", "Negative"),
        ("I failed my driving test again today, feeling miserable.", "Negative"),
        ("Don't blame me for your own mistakes, that's unfair.", "Negative"),
        ("The food was cold, stale, and tasted awful.", "Negative"),
        ("I hate how complicated this registration process is.", "Negative"),
        ("Nobody answered my calls, very unprofessional behavior.", "Negative"),
        ("I am so tired of these endless arguments.", "Negative"),
        ("My laptop crashed right before saving the report.", "Negative"),
        ("This is a complete disaster, we need to redo everything.", "Negative"),
        ("I feel completely ignored and disrespected in this team.", "Negative"),
        ("Terrible quality, it broke on the first day itself.", "Negative"),
        ("I'm having a horrible headache and feeling unwell.", "Negative"),
        ("That was very insensitive and hurtful of you.", "Negative"),
        ("Nothing is working properly, I'm losing my patience.", "Negative"),
        ("The wifi keeps disconnecting every five minutes, so annoying.", "Negative"),
        ("I am really unhappy with the current situation.", "Negative"),
        ("Why did you cancel last minute without telling anyone?", "Negative"),
        ("The package arrived damaged and unusable.", "Negative"),
        ("I cannot believe you lied about that.", "Negative"),
        ("So stressful having to deal with this emergency alone.", "Negative"),
        ("This is the worst customer service I have ever seen.", "Negative"),
        ("I'm fed up with the lack of communication here.", "Negative"),
        ("It's frustrating when people don't respect deadlines.", "Negative"),
        ("Feeling totally overwhelmed and exhausted by work.", "Negative"),
        ("That's awful to hear, I'm so sorry that happened.", "Negative"),

        # Neutral (40)
        ("Okay", "Neutral"),
        ("Fine.", "Neutral"),
        ("Noted.", "Neutral"),
        ("I will check the details later tonight.", "Neutral"),
        ("Let me know when you arrive at the station.", "Neutral"),
        ("The meeting is scheduled for 3 PM in conference room B.", "Neutral"),
        ("Please find the attached PDF document for your review.", "Neutral"),
        ("I'll be there in about 15 minutes.", "Neutral"),
        ("Did you receive the email from the department?", "Neutral"),
        ("Call me when you are free to talk.", "Neutral"),
        ("The store closes at 9 PM on weekdays.", "Neutral"),
        ("What is the address of the venue?", "Neutral"),
        ("I am on the bus right now, will reach soon.", "Neutral"),
        ("Can you share the link to the shared drive?", "Neutral"),
        ("We have class tomorrow from 10 AM to 1 PM.", "Neutral"),
        ("I've forwarded the invoice to accounts.", "Neutral"),
        ("Are you coming to the library today?", "Neutral"),
        ("The assignment submission deadline is Friday midnight.", "Neutral"),
        ("Will pick up groceries on my way home.", "Neutral"),
        ("Here is the contact number: +91 98765 43210.", "Neutral"),
        ("Please confirm your attendance by replying to this text.", "Neutral"),
        ("I am currently in a meeting, will text you back.", "Neutral"),
        ("The file size is around 45 MB.", "Neutral"),
        ("What time should we meet at the metro station?", "Neutral"),
        ("I left my notebook on the second floor desk.", "Neutral"),
        ("Received the package from the courier just now.", "Neutral"),
        ("The lecture has been postponed to Monday.", "Neutral"),
        ("Can you send me the Wi-Fi password for the lab?", "Neutral"),
        ("I have updated the spreadsheet with the latest numbers.", "Neutral"),
        ("Let me check with the coordinator and get back to you.", "Neutral"),
        ("The presentation slides are saved in the project folder.", "Neutral"),
        ("Is the office open this Saturday?", "Neutral"),
        ("Payment has been processed via UPI.", "Neutral"),
        ("I will send the draft by tomorrow morning.", "Neutral"),
        ("Please remember to bring your student ID card.", "Neutral"),
        ("The temperature today is expected to be 28 degrees.", "Neutral"),
        ("We are meeting at the cafeteria entrance at 1 PM.", "Neutral"),
        ("I'll drop the keys with the security guard.", "Neutral"),
        ("Check page 14 of the syllabus document.", "Neutral"),
        ("Got it, see you tomorrow at 10.", "Neutral"),
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

        # Always overwrite the same stable filename.
        # Cache-busting is handled in the template via a ?v= query string.
        fname = 'confusion_matrix.png'
        path = os.path.join(output_dir, fname)
        plt.savefig(path, transparent=True, bbox_inches='tight', dpi=120)
        plt.close()
        return f'images/{fname}'
