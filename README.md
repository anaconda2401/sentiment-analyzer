# ChatSense AI
### Privacy-Aware WhatsApp Conversation Analytics

> An NLP system that analyses exported WhatsApp conversations to **estimate expressed sentiment**, identify communication patterns, and generate visual insights — while anonymising all personally identifiable information before any analysis is performed.

---

## Objectives

- Parse exported WhatsApp `.txt` files and extract structured message data.
- Detect and anonymise PII (names, phones, emails, URLs, addresses) **before** sentiment analysis.
- Estimate expressed sentiment at both message and context-aware levels.
- Produce interpretable metrics, charts, and a model evaluation report.
- Never expose raw participant identities in the dashboard or charts.

---

## Features

| Feature | Description |
|---|---|
| WhatsApp parser | Handles Android and iOS export formats, multiline messages, and system messages |
| PII anonymisation | Names → User1/User2, Emails → [EMAIL], Phones → [PHONE], URLs → [URL], Addresses → [ADDRESS] |
| Text preprocessing | Unicode normalisation, emoji-to-alias conversion, negation preservation |
| VADER sentiment | Message-level: Positive / Neutral / Negative with compound score |
| Context-aware sentiment | Sliding-window (4 messages) blending for ambiguous messages like "Fine." |
| Confidence scoring | Per-prediction confidence with low-confidence flag |
| Chat insights | Counts, percentages, averages, busiest hour, per-participant breakdown |
| Visualisations | Sentiment distribution, trend over time, participant chart, confidence histogram |
| Model evaluation | Accuracy, Precision, Recall, F1 (weighted + macro + per-class), Confusion Matrix |
| Limitations section | Sarcasm, short-message ambiguity, language scope |

---

## NLP Techniques Used

- **VADER** (Valence Aware Dictionary and sEntiment Reasoner) — lexicon-based sentiment analysis tuned for social text
- **Emoji demojization** — converts emoji to text aliases so VADER can process them
- **Sliding-window context blending** — adjusts ambiguous message scores using conversational history
- **spaCy NER** — named entity recognition for detecting and redacting person names and locations
- **Regular expressions** — email, phone, and URL detection

---

## Architecture

```
WhatsApp .txt
    |
    v
whatsapp_parser.py    -- Extracts date, time, sender, message; filters system/media lines
    |
    v
pii_anonymizer.py     -- Regex + spaCy NER: replaces PII with labelled placeholders
    |
    v
text_preprocessor.py  -- Unicode norm, whitespace, emoji->alias, repeated-char collapse
    |
    v
sentiment_model.py    -- VADER: Positive / Neutral / Negative + confidence score
    |
    v
context_analyzer.py   -- Sliding-window context-aware score adjustment
    |
    v
chat_insights.py      -- Aggregated statistics (counts, %, averages, per-participant)
    |
    v
visualization.py      -- matplotlib/seaborn charts (saved to static/images/)
    |
    v
evaluation.py         -- Accuracy, F1, Confusion Matrix on synthetic labelled dataset
    |
    v
Flask dashboard        -- results.html: all sections rendered with anonymised data only
```

---

## Privacy Approach

1. A fresh `PIIAnonymizer` instance is created for every upload — state never persists between users.
2. Participant names are replaced with `User1`, `User2`, etc. before any text processing.
3. Regular expressions strip emails, phone numbers, and URLs before spaCy runs.
4. spaCy NER redacts remaining person names and location entities.
5. No raw PII is stored beyond the duration of the HTTP request.
6. Uploaded files are deleted immediately after parsing.
7. The dashboard, charts, and all exports exclusively use anonymised identifiers.

---

## Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate      # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy language model
python -m spacy download en_core_web_sm
```

---

## How to Run

```bash
python app.py
# Open http://localhost:5000
```

---

## How to Upload a WhatsApp Export

1. Open a WhatsApp chat on your phone.
2. Tap the **three-dot menu** > **More** > **Export chat**.
3. Choose **Without Media**.
4. Save the `.txt` file to your computer.
5. Upload it on the ChatSense AI home page.

---

## Example Workflow

```
Upload chat.txt
    -> 1124 messages parsed
    -> PII anonymised (2 participants: User1, User2)
    -> 1124 messages preprocessed
    -> Sentiment: 112 Positive, 972 Neutral, 40 Negative
    -> Context adjustment applied (window=4)
    -> Charts generated: dist.png, trend.png, participants.png, confidence.png
    -> Evaluation: Accuracy 86.7%, Weighted F1 0.847
    -> Report rendered at /analyze
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Limitations

- **Sarcasm and irony** are not detected. "Great job!" with sarcastic intent will likely be classified as Positive.
- **Short ambiguous messages** ("Fine.", "Okay", "Lol") are inherently difficult to classify, even with context blending.
- **Language** — optimised for English. Mixed-language or transliterated messages may produce inaccurate results.
- **No psychological claims** — results reflect patterns in expressed text only, not true emotional states.
- **PII edge cases** — unusual names, informal references, or abbreviations may not be fully detected.

---

## Future Enhancements

- Sarcasm classifier (e.g., rule-based patterns or fine-tuned lightweight model)
- Keyword / topic extraction per sentiment class
- Weekly and monthly trend breakdowns
- Multi-language support
- PDF export of the analysis report
- Real WhatsApp labeled dataset for more reliable evaluation

---

## Disclaimer

> ChatSense AI estimates **expressed sentiment** from conversational text. It does not detect true emotions, read minds, determine psychological states, or diagnose mental health conditions. All statistics should be interpreted as patterns in written communication, not as definitive truths about any individual.
