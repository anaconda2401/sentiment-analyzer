# ChatSense AI
### Privacy-Preserving WhatsApp Conversation Analytics & Relational Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-58%20passed-success.svg)](tests/)
[![Synthetic Edge Cases](https://img.shields.io/badge/edge%20cases-14%2F14%20passed-success.svg)](syn_test/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**ChatSense AI** is a privacy-first conversational intelligence system engineered to analyze exported WhatsApp chat logs (`.txt`). It performs **expressed sentiment tracking**, **temporal emotion trajectories**, **relational reciprocity graph modeling**, and **context-aware manipulation pattern detection** — executing entirely on local infrastructure with zero external data transmission, automatic PII sanitization, and ephemeral session storage.

---

## Key Features & Modular Pipeline

| Pipeline Stage | Module File | Core Functionality | Underlying NLP & Graph Methods |
|---|---|---|---|
| **1. Chat Ingestion & Parsing** | `chatsense/whatsapp_parser.py` | Parses Android & iOS exports, multiline turns, and quote replies; excludes system/media lines | High-speed regex pattern matching + quote snippet extraction |
| **2. Privacy & PII Sanitization** | `chatsense/pii_anonymizer.py` | Redacts participant names, phone numbers, emails, URLs, and addresses before NLP inference | Pre-compiled regex tokenizers + spaCy NER (`en_core_web_sm`) + automated residual leakage verification |
| **3. Text Preprocessing** | `chatsense/text_preprocessor.py` | Cleans message text while preserving sentiment-bearing punctuation and emphasis | Unicode NFKC normalization + repeated character reduction + emoji preservation |
| **4. Sentiment & Confidence** | `chatsense/sentiment_model.py` | Estimates continuous polarity (-1.0 to +1.0) with calibrated confidence handling | Valence Aware Dictionary for sEntiment Reasoning (VADER) + lexical uncertainty abstention |
| **5. Contextual Blending** | `chatsense/context_analyzer.py` | Smooths sentiment across rolling conversational windows to detect conversational escalation | Exponential temporal decay blending + single-turn boundary protection |
| **6. Temporal Emotion Dynamics** | `chatsense/emotion_analyzer.py` | Tracks 6 emotional dimensions across time and detects sharp conversational shifts | Multi-signal lexicon + 5-turn baseline rolling variance change-point detection |
| **7. Manipulation Pattern Detection** | `chatsense/abuse_detector.py` & `manipulation_adapter.py` | Detects subtle relational manipulation patterns across rolling 4-turn context windows | Fine-tuned DistilBERT multi-label backbone (`SamanthaStorm/abusedetector`) + candidate turn prioritization |
| **8. Explainability & Attribution** | `chatsense/explainability.py` | Explains flagged turns with token-level attributions and constructive non-accusatory hypotheses | Leave-one-out word occlusion scoring + contextual alternative interpretations |
| **9. Interaction Graph Analysis** | `chatsense/conversation_graph.py` | Models participant interaction flow, reciprocity balance, and network density | NetworkX directed multigraph + reciprocity ratio + transition matrix |
| **10. Web Interface & Dashboard** | `app.py`, `templates/`, `static/` | Interactive dark-mode dashboard with timeline filtering, session purge, and JSON export | Flask + responsive CSS3 + real-time animated stage pacing + SVG charts |

---

## 8-Label Manipulation Taxonomy

ChatSense AI incorporates the open-source pretrained multi-label model **`SamanthaStorm/abusedetector`** (DistilBERT backbone, MIT License). Predictions map transparently across 8 documented interpersonal categories:

| Model Label | Taxonomy Category | Description & Linguistic Signatures |
|---|---|---|
| `gaslighting` | **Gaslighting-like Language** | Reality-denying expressions, memory invalidation (*"that never happened"*, *"you're crazy"*) |
| `darvo` | **Blame Shifting / Deflection (DARVO)** | Denying wrongdoing, attacking the sender, and reversing victim/offender roles (*"you made me do this"*) |
| `victim_blaming` | **Blame Shifting / Deflection** | Deflecting accountability onto the partner (*"it is completely your fault"*) |
| `emotional_manipulation` | **Emotional Coercion / Guilt Induction** | Inducing obligation, guilt, or emotional distress (*"if you loved me you would"*) |
| `boundary_violation` | **Pressure & Controlling Language** | Disregarding explicit limits, persistent invasive demands |
| `surveillance_accusation` | **Pressure & Controlling Language** | Inquisitorial monitoring, interrogation (*"who were you texting"*, *"show me your phone"*) |
| `healthy_boundary_setting` | **Healthy Boundary Setting** *(Constructive)* | Clear, non-punitive boundary assertion (*"I need some space right now"*, *"let's pause"*) |
| `no_abuse_pattern` | **Non-Manipulative Expression** | Normal communicative dialogue, factual updates, and supportive language |

---

## System Architecture

```
WhatsApp Export (.txt)
        │
        ▼
[ chatsense/whatsapp_parser.py ] ── Extracts timestamp, sender, text, quotes & reply metadata
        │
        ▼
[ chatsense/pii_anonymizer.py ]  ── Regex + spaCy NER masks PII (Zero-Leakage Audit)
        │
        ▼
  Sanitized DataFrame (User1, User2, [PHONE], [EMAIL], [URL], [PII])
    ┌───┴───────────────────────────┬────────────────────────────┐
    │                               │                            │
    ▼                               ▼                            ▼
[ chatsense/sentiment_model.py ] [ chatsense/emotion_analyzer.py ] [ chatsense/conversation_graph.py ]
- VADER compound score (-1..+1)  - 6-signal emotional dynamics   - NetworkX directed graph
- Insufficient evidence guard    - 5-turn rolling trajectory     - Reciprocity ratio (0 to 1)
- Sliding-window context blend   - Change-point detection        - Turn transitions & density
    │                               │                            │
    └───┬───────────────────────────┴────────────────────────────┘
        │
        ▼
[ chatsense/manipulation_adapter.py ]
- Rolling context window (4 turns: "[Context: ...] Target: ...")
- Candidate prioritization (sub-8-second execution on 4,800+ messages)
- Multi-label DistilBERT inference (8 authentic categories)
- Short-token / non-lexical neural abstention (is_abstain=True)
        │
        ▼
[ chatsense/explainability.py ]
- Leave-one-out word occlusion token attribution
- Context-aware alternative interpretations
        │
        ▼
[ Web Application & Secure UI ] (app.py)
- Responsive dark-mode dashboard with animated progress bar
- Dynamic date range re-filtering (/filter)
- Temporal trajectory & conversation interaction charts
- Immediate session deletion (/delete_session) & workspace wipe (/clear_all)
- Privacy-compliant summary export (/export/<id>)
```

---

## Performance & Latency Benchmarks

Empirical timings measured on a standard CPU test machine:

| Benchmark / Operation | Measured Latency | Details |
|---|---|---|
| **Python Process Startup** | `0.058 s` | Interpreter cold start |
| **Heavy Dependency Imports** | `8.45 s` | Loading PyTorch, Transformers, Pandas, Flask |
| **Total Flask Application Init** | `13.94 s` | Complete server ready state |
| **Homepage Response Time (`GET /`)** | `15.81 ms` | Instantaneous 200 OK delivery |
| **10-Turn Manipulation Chat** | `1.91 s` | Full 12-stage pipeline execution |
| **666-Turn Chat (`whatsappchat.txt`)** | `4.44 s` | Full 12-stage pipeline execution |
| **4,815-Turn Chat (`whatsapp_chat.txt`)** | **`7.66 s`** | Reduced from $\sim 20\text{ minutes}$ via vectorized candidate prioritization |

---

## Empirical Accuracy & Evaluation

### Sentiment Analysis (Evaluated on Labeled Benchmark)
Evaluated on `data/sentiment_eval_dataset.csv` (120 balanced WhatsApp-style samples: 40 Positive, 40 Neutral, 40 Negative):
- **Overall Accuracy:** **87.5%**
- **Macro F1:** **0.874** | **Weighted F1:** **0.874**
- **Positive:** Precision: 0.780, Recall: 0.975, F1: 0.867 (Support: 40)
- **Neutral:** Precision: 0.906, Recall: 0.725, F1: 0.806 (Support: 40)
- **Negative:** Precision: 0.974, Recall: 0.925, F1: 0.949 (Support: 40)

### Manipulation Pattern Detection (Scientific Status)
- **Local Predictive Accuracy:** **Empirically unverified on arbitrary WhatsApp exports** (no labeled ground-truth manipulation dataset exists in this repository).
- **Pretrained Baseline Metrics:** The model author (`SamanthaStorm/abusedetector`) reports a benchmark test Macro F1 of **0.277** on 10 relationship test conversations.
- **System Safeguards:** ChatSense AI mitigates high false-positive rates using candidate conflict scoring, confidence threshold selection (`0.05` to `0.55`), leave-one-out explainability cards, and explicit non-manipulative alternative hypotheses.

---

## Privacy & Security Architecture

1. **Local Processing Only:** All parsing, NER sanitization, VADER scoring, and DistilBERT inference run strictly on the host machine. No text is sent to third-party APIs.
2. **Deterministic Pseudonymization:** Raw contact names and phone numbers are mapped to synthetic identifiers (`User1`, `User2`) before entering subsequent NLP stages.
3. **Automated Residual Leakage Audit:** Every execution runs a leakage test scanning the corpus for unmasked participant names and email/URL patterns.
4. **Immediate Purge Controls:** 
   - **Delete Session:** Immediately removes pickled session data and temporary chart images.
   - **Clear All History:** Permanently purges all cached sessions, generated visual assets, and uploaded files.
5. **Anonymized JSON Export:** The downloadable summary report contains only aggregate counts, emotional distributions, and anonymized identifiers.

---

## Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.12)
- Windows, macOS, or Linux

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/chatsense.git
cd chatsense

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model for NER
python -m spacy download en_core_web_sm
```

### 3. Model Weights Setup
The pretrained DistilBERT manipulation detector requires weights located in `data/models/abusedetector/`. Because `pytorch_model.bin` is ~253 MB, it is excluded from Git tracking to respect repository size limits.

Download the model files directly using `curl`:
```powershell
# Create model folder
mkdir data\models\abusedetector -Force

# Download model configuration and weights (~253 MB)
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/config.json" -o "data\models\abusedetector\config.json"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/modeling_abuse.py" -o "data\models\abusedetector\modeling_abuse.py"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/training_info.json" -o "data\models\abusedetector\training_info.json"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/tokenizer.json" -o "data\models\abusedetector\tokenizer.json"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/tokenizer_config.json" -o "data\models\abusedetector\tokenizer_config.json"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/vocab.txt" -o "data\models\abusedetector\vocab.txt"
curl.exe -L -C - "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/pytorch_model.bin" -o "data\models\abusedetector\pytorch_model.bin"
```

### 4. Launch Application
```powershell
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## Verification & Test Suites

### Run All Regression Tests
Executes 58 unit, integration, and robustness tests:
```powershell
pytest tests/ -q
```
*Expected result: `58 passed in ~24s`*

### Run Synthetic Edge-Case Suite
Executes all 14 boundary test cases (monologues, empty files, rapid bursts, 35 participants, iOS formats):
```powershell
python syn_test/run_edge_case_tests.py
```
*Expected result: `Total: 14, Passed: 14, Failed: 0`*

---

## Ethical & Scientific Disclaimer

> **IMPORTANT:** ChatSense AI is an **educational and academic natural language processing research tool**.
> - **Not Diagnostic or Legal Evidence:** System predictions indicate computational statistical patterns in text. They **do not prove** psychological abuse, malicious intent, or interpersonal guilt.
> - **Context Sensitivity:** Benign arguments, humor, sarcasm, logistical urgency, and stress responses can trigger elevated scores.
> - **Privacy Boundaries:** The PII anonymizer sanitizes explicit names, phone numbers, and standard email/URL patterns. It does not prevent re-identification through unique quasi-identifiers or unlisted third-party references.
> - All analytics are intended solely for introspective study, communication pattern exploration, and linguistic research.
