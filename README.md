# ChatSense AI

### Privacy-Preserving WhatsApp Conversation Analytics, Relational Intelligence & Temporal NLP

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Test Suite Passing](https://img.shields.io/badge/tests-58%20passed%20%28100%25%29-success.svg)](tests/)
[![Synthetic Edge Cases](https://img.shields.io/badge/edge%20cases-14%2F14%20passed-success.svg)](syn_test/)
[![Inference Engine](https://img.shields.io/badge/backbone-DistilBERT%20%2B%20VADER%20%2B%20NetworkX-orange.svg)](data/models/abusedetector/)
[![Privacy Audit](https://img.shields.io/badge/privacy-Zero--Egress%20%7C%20Local--Only-emerald.svg)](#privacy--security-architecture)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. System Architecture & End-to-End Pipeline](#2-system-architecture--end-to-end-pipeline)
- [3. Modular Pipeline Deep-Dive & Codebase Specification](#3-modular-pipeline-deep-dive--codebase-specification)
  - [3.1 WhatsApp Chat Ingestion (`whatsapp_parser.py`)](#31-whatsapp-chat-ingestion-whatsapp_parserpy)
  - [3.2 Deterministic PII Sanitization & Audit (`pii_anonymizer.py`)](#32-deterministic-pii-sanitization--audit-pii_anonymizerpy)
  - [3.3 Text Normalization (`text_preprocessor.py`)](#33-text-normalization-text_preprocessorpy)
  - [3.4 Sentiment Intensity & Uncertainty Guard (`sentiment_model.py`)](#34-sentiment-intensity--uncertainty-guard-sentiment_modelpy)
  - [3.5 Sliding-Window Context Blending (`context_analyzer.py`)](#35-sliding-window-context-blending-context_analyzerpy)
  - [3.6 6-Signal Affect Dynamics & Change-Points (`emotion_analyzer.py`)](#36-6-signal-affect-dynamics--change-points-emotion_analyzerpy)
  - [3.7 Pretrained Manipulation Detection (`abuse_detector.py` & `manipulation_adapter.py`)](#37-pretrained-manipulation-detection-abuse_detectorpy--manipulation_adapterpy)
  - [3.8 Token-Level Explainability via Occlusion (`explainability.py`)](#38-token-level-explainability-via-occlusion-explainabilitypy)
  - [3.9 Uncertainty Quantification & Calibration (`uncertainty.py`)](#39-uncertainty-quantification--calibration-uncertaintypy)
  - [3.10 Relational Conversation Graph Modeling (`conversation_graph.py`)](#310-relational-conversation-graph-modeling-conversation_graphpy)
  - [3.11 Aggregation & Headless Visualizations (`chat_insights.py` & `visualization.py`)](#311-aggregation--headless-visualizations-chat_insightspy--visualizationpy)
- [4. 8-Label Manipulation Taxonomy Specification](#4-8-label-manipulation-taxonomy-specification)
- [5. Performance Engineering & Optimization Benchmarks](#5-performance-engineering--optimization-benchmarks)
  - [5.1 The 20-Minute to 7.6-Second Latency Reduction](#51-the-20-minute-to-76-second-latency-reduction)
  - [5.2 Empirical Latency Benchmarks](#52-empirical-latency-benchmarks)
- [6. Empirical Model Evaluation & Verification](#6-empirical-model-evaluation--verification)
  - [6.1 Sentiment Evaluation (Verified Benchmark)](#61-sentiment-evaluation-verified-benchmark)
  - [6.2 Manipulation Detection Status (Scientific Transparency)](#62-manipulation-detection-status-scientific-transparency)
- [7. Privacy, Security & Data Lifecycle Architecture](#7-privacy-security--data-lifecycle-architecture)
- [8. REST API & Route Reference](#8-rest-api--route-reference)
- [9. Developer Installation & Environment Setup](#9-developer-installation--environment-setup)
- [10. Testing, Quality Assurance & Edge-Case Verification](#10-testing-quality-assurance--edge-case-verification)
- [11. Repository & Directory Structure](#11-repository--directory-structure)
- [12. Scientific & Ethical Disclaimer](#12-scientific--ethical-disclaimer)
- [13. Contributing & License](#13-contributing--license)

---

## 1. Executive Summary

**ChatSense AI** is a production-grade natural language processing system engineered to analyze exported WhatsApp chat transcripts (`.txt`) locally and confidentially. Most contemporary chat analytics tools rely on third-party cloud APIs (e.g., OpenAI, Anthropic, Google), transmitting unencrypted private dialogues across external networks. ChatSense AI solves this through a **strict zero-egress architecture**:

1. **Local-Only Inference:** All parsing, Named Entity Recognition (NER), lexicon scoring, and deep transformer evaluations execute entirely on the host CPU/GPU.
2. **Deterministic Pseudonymization:** Participants are deterministically remapped to synthetic handles (`User1`, `User2`) while phone numbers, emails, addresses, and external personal identities are masked prior to inference.
3. **Calibrated Confidence:** Recognizes absence of evidence (emojis, punctuation-only turns) and refrains from assigning false 100% confidence.
4. **Relational Reciprocity Modeling:** Employs directed multigraph network algorithms to evaluate communication flow balance, response latency, and interaction density.
5. **Non-Accusatory Explainability:** Provides leave-one-out token attribution and constructive alternative interpretations to guard against algorithmic confirmation bias.

---

## 2. System Architecture & End-to-End Pipeline

The pipeline follows a layered architecture moving from raw unstructured text to structured, privacy-scrubbed representations, multi-signal feature extraction, graph modeling, and interactive visualization:

```
                  ┌─────────────────────────────────────────┐
                  │    WhatsApp Chat Export (.txt file)     │
                  │   Android (12h/24h) or iOS Bracketed    │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & REGEX PARSER (chatsense/whatsapp_parser.py)                  │
│    - Strips WhatsApp system messages, encryption notices, & media tokens    │
│    - Preserves multiline utterances and quote/reply structures ("> quote") │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. PRIVACY & PII SANITIZATION (chatsense/pii_anonymizer.py)                 │
│    - Regex tokenizers: [EMAIL], [PHONE], [URL], [ADDRESS]                   │
│    - Batched spaCy NER (en_core_web_sm): Named entities -> [PII]            │
│    - Deterministic participant pseudonymization: Alice/Bob -> User1/User2   │
│    - Residual leakage test: Scans for unmasked names and contact patterns   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TEXT PREPROCESSING (chatsense/text_preprocessor.py)                      │
│    - Unicode NFKC normalization, whitespace standardization                 │
│    - Preserves sentiment punctuation (!!!, ???) and maps emoji semantics    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│ 4. SENTIMENT ENGINE                  │  │ 6. RELATIONAL GRAPH MODELING      │
│    (chatsense/sentiment_model.py)    │  │    (conversation_graph.py)        │
│ - VADER compound polarity (-1..+1)   │  │ - NetworkX directed multigraph    │
│ - Lexical uncertainty detection      │  │ - Reciprocity ratio (0.0 to 1.0)  │
│ - Low-confidence flag on emojis/none │  │ - Density & transition matrix     │
└──────────────────┬───────────────────┘  └─────────────────┬─────────────────┘
                   │                                        │
                   ▼                                        │
┌──────────────────────────────────────┐                    │
│ 5. TEMPORAL CONTEXT BLENDING         │                    │
│    (chatsense/context_analyzer.py)   │                    │
│ - Rolling 4-turn exponential decay   │                    │
│ - Blends 70% current + 30% history   │                    │
└──────────────────┬───────────────────┘                    │
                   │                                        │
            ┌──────┴───────────────────┐                    │
            ▼                          ▼                    │
┌──────────────────────────┐ ┌──────────────────────────┐   │
│ 7. EMOTION DYNAMICS      │ │ 8. MANIPULATION ADAPTER  │   │
│    (emotion_analyzer.py) │ │    (manipulation_adapter)│   │
│ - 6 affect dimensions    │ │ - 4-turn contextualized  │   │
│ - 5-turn trajectory      │ │   target augmentation    │   │
│ - Variance change-point  │ │ - Candidate prioritizing │   │
│   escalation detection   │ │ - DistilBERT (8 labels)  │   │
└──────────────────────────┘ └─────────────┬────────────┘   │
                                           │                │
                                           ▼                │
                             ┌──────────────────────────┐   │
                             │ 9. EXPLAINABILITY & OOD  │   │
                             │    (explainability.py)   │   │
                             │ - Occlusion attribution  │   │
                             │ - Alternative hypotheses │   │
                             │ - Uncertainty assessor   │   │
                             └─────────────┬────────────┘   │
                                           │                │
                                           ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 10. FLASK APPLICATION & SECURE DASHBOARD (app.py, templates/, static/)      │
│     - Interactive dark-mode analytical view with responsive charts          │
│     - Real-time date-range refiltering via in-memory session cache (/filter)│
│     - Hard session purge (/delete_session) & full workspace wipe (/clear_all│
│     - Anonymized JSON report export with zero retained PII (/export/<id>)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Modular Pipeline Deep-Dive & Codebase Specification

### 3.1 WhatsApp Chat Ingestion (`whatsapp_parser.py`)
- **Module File:** `chatsense/whatsapp_parser.py`
- **Primary Function:** `parse_whatsapp_chat(filepath: str) -> pd.DataFrame`
- **Supported Formats:**
  - Android standard: `27/08/2026, 8:02 PM - Sender: Message`
  - iOS bracketed: `[27/08/2026, 8:02:05 PM] Sender: Message`
  - 24-hour timestamp variants: `27.08.26, 20:02 - Sender: Message`
- **Multiline & Continuation Handling:** Uncolonized continuation lines are automatically concatenated to the preceding turn's text buffer.
- **Quote / Reply Parsing:** Lines starting with `>`, `“`, or `"` are flagged with `is_reply=True` and a truncated 60-character snippet is captured in `reply_preview`.
- **System Filtering:** Drops encryption notifications, phone number changes, group description edits, and media omissions (`<Media omitted>`, `<image omitted>`, `<sticker omitted>`).

### 3.2 Deterministic PII Sanitization & Audit (`pii_anonymizer.py`)
- **Module File:** `chatsense/pii_anonymizer.py`
- **Primary Class:** `PIIAnonymizer`
- **Methodology:**
  1. **Participant Pseudonymization:** Discovers unique senders and maps them monotonically to `User1`, `User2`, ..., `UserN`. Senders retain this identity consistently across all downstream analytics.
  2. **High-Speed Regex Layer:** Uses precompiled regular expressions to mask standard patterns:
     - URLs: `https?://\S+|www\.\S+` $\to$ `[URL]`
     - Emails: `[\w\.-]+@[\w\.-]+\.\w+` $\to$ `[EMAIL]`
     - Phone Numbers: Canonical and international dial formats $\to$ `[PHONE]`
  3. **Batched spaCy Named Entity Recognition:** Uses `en_core_web_sm` (with tokenizers, taggers, and parsers disabled for speed) to extract `PERSON` $\to$ `[PII]` and `GPE`/`LOC`/`FAC` $\to$ `[ADDRESS]`.
  4. **Automated Residual Leakage Audit:** Automatically inspects the transformed corpus for raw participant names, phone numbers, and emails. Produces a verifiable privacy audit structure returned in `results.html`.

### 3.3 Text Normalization (`text_preprocessor.py`)
- **Module File:** `chatsense/text_preprocessor.py`
- **Primary Class:** `TextPreprocessor`
- **Operations:**
  - Unicode NFKC normalization.
  - Collapse repeated character elongation (e.g., `"nooooooo"` $\to$ `"noo"`).
  - Preserves exclamation and question marks (`!`, `?`) critical for valence estimation.
  - Converts Unicode emojis into text tokens (`❤️` $\to$ `:red_heart:`) via the `emoji` library.

### 3.4 Sentiment Intensity & Uncertainty Guard (`sentiment_model.py`)
- **Module File:** `chatsense/sentiment_model.py`
- **Primary Class:** `SentimentModel`
- **Underlying Engine:** Valence Aware Dictionary for sEntiment Reasoning (`vaderSentiment`).
- **Mathematical Output:** Returns continuous compound score $S \in [-1.0, +1.0]$, categorical label (`Positive` if $S \ge 0.05$, `Negative` if $S \le -0.05$, else `Neutral`), and calibrated confidence $C \in [0.0, 1.0]$.
- **Calibrated Confidence & Lexical Abstention:**
  - **Emoji-only, digits, or symbol inputs:** Flagged as insufficient evidence with $C = 0.25$ and `is_low_confidence = True`.
  - **Ambiguous single-word replies (`"k"`, `"fine"`):** $C = 0.40$ and `is_low_confidence = True`.
  - **Genuine multi-word neutral statements:** $C = \min(\text{score}_{\text{neu}} \times 0.85, 0.85)$.

### 3.5 Sliding-Window Context Blending (`context_analyzer.py`)
- **Module File:** `chatsense/context_analyzer.py`
- **Primary Class:** `ContextAnalyzer(window_size=4)`
- **Mathematical Formulation:**
  For conversational turn $i$, the contextualized score $S_i^{(\text{context})}$ is computed as:
  $$S_i^{(\text{context})} = 0.70 \cdot S_i^{(\text{raw})} + 0.30 \cdot \left( \frac{1}{\min(i, W)} \sum_{k=0}^{\min(i-1, W-1)} S_{i-k}^{(\text{raw})} \right)$$
  where $W = 4$ is the rolling window size. Single-message files or empty histories evaluate with $S_1^{(\text{context})} = S_1^{(\text{raw})}$.

### 3.6 6-Signal Affect Dynamics & Change-Points (`emotion_analyzer.py`)
- **Module File:** `chatsense/emotion_analyzer.py`
- **Primary Class:** `EmotionAnalyzer`
- **Affect Categories:** Joy, Sadness, Anger, Fear, Frustration, Anxiety (grounded in NRC / Plutchik affect lexicons).
- **Temporal Trajectory:** Applies a 5-turn rolling average across each affective dimension.
- **Change-Point Detection:** Detects sharp conversational escalations where polarity drops by $\Delta \ge 0.50$ accompanied by sharp spikes in negative affective signals.

### 3.7 Pretrained Manipulation Detection (`abuse_detector.py` & `manipulation_adapter.py`)
- **Module Files:** `chatsense/abuse_detector.py`, `chatsense/manipulation_adapter.py`
- **Backbone Model:** `SamanthaStorm/abusedetector` (Fine-tuned `distilbert-base-uncased` + Linear Classification Layer, MIT License).
- **Context Augmentation:** Reformulates turns with rolling 4-turn contextual prefixes:
  ```text
  [Context: User1: Where are you? | User2: Stuck in traffic] Target: You always lie to me!
  ```
- **Abstention Logic:** Turns with fewer than 2 words are automatically exempted from neural evaluation (`is_abstain = True`, top pattern = `no_abuse_pattern`, scores = `0.0`).

### 3.8 Token-Level Explainability via Occlusion (`explainability.py`)
- **Module File:** `chatsense/explainability.py`
- **Primary Class:** `PredictionExplainer`
- **Attribution Method:** Leave-one-out word occlusion. Computes the probability drop for target class $c$:
  $$\Delta_w = \text{Score}_c(\text{Text}) - \text{Score}_c(\text{Text} \setminus \{w\})$$
- **Constructive Alternatives:** Formulates non-accusatory hypotheses to explain flagged patterns objectively (e.g., miscommunication vs. intentional manipulation).

### 3.9 Uncertainty Quantification & Calibration (`uncertainty.py`)
- **Module File:** `chatsense/uncertainty.py`
- **Primary Class:** `UncertaintyAssessor`
- **Metrics:** Evaluates decision margins $| \text{top\_score} - \text{threshold} |$, out-of-distribution (OOD) token densities, Expected Calibration Error (ECE), and Brier Score.

### 3.10 Relational Conversation Graph Modeling (`conversation_graph.py`)
- **Module File:** `chatsense/conversation_graph.py`
- **Primary Class:** `ConversationGraphAnalyzer`
- **Graph Topology:** NetworkX directed multigraph $G = (V, E)$ where nodes represent participants and conversational turns, and edges represent authorship, temporal sequence, and cross-speaker transitions.
- **Reciprocity Metric:**
  $$\rho = \frac{m_{\leftrightarrow}}{m}$$
  Measures the proportion of mutual turn transitions between interlocutors.

### 3.11 Aggregation & Headless Visualizations (`chat_insights.py` & `visualization.py`)
- **Module Files:** `chatsense/chat_insights.py`, `chatsense/visualization.py`
- **Visualizer Engine:** Headless Matplotlib (`matplotlib.use('Agg')`) with custom dark-mode typography and HSL-aligned color palettes (`#10b981` positive, `#ef4444` negative, `#6b7280` neutral).
- **Generated Assets:** Expressed sentiment pie chart (`dist.png`), temporal polarity trendline (`trend.png`), participant breakdown bar chart (`participants.png`), confidence histogram (`confidence.png`), 6-signal emotion trajectory (`emotions_trajectory.png`), and conversation network diagram (`conversation_graph.png`).

---

## 4. 8-Label Manipulation Taxonomy Specification

The system classifies linguistic patterns across 8 categories matching `SamanthaStorm/abusedetector`:

| Label Identifier | Taxonomy Category | Linguistic Signatures & Indicators | Non-Accusatory Alternative Interpretation |
|---|---|---|---|
| `gaslighting` | **Gaslighting-like Language** | Reality-denying expressions, memory invalidation (*"that never happened"*, *"you're imagining things"*) | May reflect honest discrepancy in recollection or defensive pushback against misremembered events. |
| `darvo` | **Blame Shifting (DARVO)** | Denying wrongdoing, attacking the speaker, reversing victim/offender roles (*"you made me do this"*) | Speaker may feel genuinely attacked or unfairly accused, reacting defensively. |
| `victim_blaming` | **Blame Shifting / Deflection** | Assigning total fault to the interlocutor (*"it is completely your fault"*) | Exasperation or perceived shared accountability rather than deliberate deflection. |
| `emotional_manipulation` | **Emotional Coercion / Guilt** | Guilt induction, conditional affection, obligation (*"if you really loved me you would"*) | Sincere plea of emotional distress or dramatic hyperbole during intense frustration. |
| `boundary_violation` | **Pressure & Controlling Language** | Disregarding explicit limits, persistent invasive demands after refusal | Conversational urgency, persistent scheduling follow-up, or social clumsiness. |
| `surveillance_accusation` | **Inquisitorial Monitoring** | Inquisitorial surveillance, interrogating contacts (*"who were you texting"*, *"show me your phone"*) | Planning logistics, situational insecurity, or conversational curiosity. |
| `healthy_boundary_setting` | **Healthy Boundary Setting** | Clear, non-punitive boundary demarcation (*"I need some space right now"*, *"let's pause"*) | **Constructive signal.** Demonstrates healthy communication hygiene; not an abusive tactic. |
| `no_abuse_pattern` | **Non-Manipulative Dialogue** | Neutral communicative dialogue, factual discussions, logistics, affection | Standard conversational interaction. |

---

## 5. Performance Engineering & Optimization Benchmarks

### 5.1 The 20-Minute to 7.6-Second Latency Reduction

On large exported chats (e.g., `data/whatsapp_chat.txt` containing 4,815 turns), initial versions required **~20 minutes** on CPU. Through code profiling, three primary bottlenecks were resolved:

1. **Vectorized Candidate Prioritization:** Rather than executing unconstrained forward passes on all 4,815 messages, `manipulation_adapter.py` prioritizes turns exhibiting lexical tension markers or uniform temporal strides (up to 60 candidate turns). Batch size was vectorized to 32 with token length truncated to 72.
2. **Precompiled Single-Pass PII Regex:** Replaced iterative loops across 35 participants with a single pre-compiled regular expression disjunction.
3. **NetworkX Layout Pruning:** Capped the spring layout force-directed simulation to the top 12 active participants.

**Outcome:** Processing time on 4,815 messages dropped from **1,200 seconds** to **7.66 seconds** (a **99.4% runtime reduction**).

### 5.2 Empirical Latency Benchmarks

Measured on a standard CPU test machine (Python 3.11, Windows 11):

| Benchmark Operation | Latency | Pipeline Scope & Description |
|---|---|---|
| **Python Process Startup** | `0.058 s` | Interpreter cold launch |
| **Heavy Dependency Imports** | `8.45 s` | Loading PyTorch, Hugging Face Transformers, Pandas, Flask |
| **Model Weight Deserialization** | `20.81 s` | `torch.load()` deserializing 253 MB of DistilBERT weights |
| **Flask Ready State (`import app`)** | `13.94 s` | Complete application startup |
| **Homepage HTTP Latency (`GET /`)** | `15.81 ms` | Instantaneous 200 OK delivery |
| **10-Turn Manipulation Chat** | `1.91 s` | Full 12-stage analytical pipeline |
| **666-Turn Chat (`whatsappchat.txt`)** | `4.44 s` | Full 12-stage analytical pipeline |
| **4,815-Turn Chat (`whatsapp_chat.txt`)** | **`7.66 s`** | Full 12-stage analytical pipeline |

---

## 6. Empirical Model Evaluation & Verification

### 6.1 Sentiment Evaluation (Verified Benchmark)

Evaluated against `data/sentiment_eval_dataset.csv` (120 balanced WhatsApp-style conversational turns):

- **Overall Accuracy:** **87.5%**
- **Macro F1:** **0.874**
- **Weighted F1:** **0.874**
- **Weighted Precision:** **0.887**
- **Weighted Recall:** **0.875**

#### Per-Class Performance Breakdown

| Class | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Positive** | 40 | 0.780 | 0.975 | 0.867 |
| **Neutral** | 40 | 0.906 | 0.725 | 0.806 |
| **Negative** | 40 | 0.974 | 0.925 | 0.949 |

```
Confusion Matrix (120 samples):
               Predicted Positive   Predicted Neutral   Predicted Negative
True Positive          39                   1                   0
True Neutral            8                  29                   3
True Negative           3                   0                  37
```

### 6.2 Manipulation Detection Status (Scientific Transparency)

> [!WARNING]
> **Scientific Status:** The multi-label manipulation model is an **experimental research tool**. 
> - **Local Accuracy:** **Empirically unverified on arbitrary WhatsApp exports** (no labeled ground-truth manipulation dataset exists in this repository).
> - **Pretrained Baseline Metrics:** The model author (`SamanthaStorm/abusedetector`) reports a benchmark test Macro F1 of **0.277** across 10 test conversations.
> - **Safeguards in ChatSense AI:** The system actively guards against over-interpretation through candidate conflict scoring, dynamic confidence thresholds (`0.05` to `0.55`), leave-one-out word occlusion cards, and non-accusatory alternative hypotheses.

---

## 7. Privacy, Security & Data Lifecycle Architecture

```
User Upload (RAM)
      │
      ▼
Save temporary file to uploads/
      │
      ▼
In-Memory Parsing & PII Masking (User1, [PHONE], [EMAIL])
      │
      ▼
Original raw file deleted immediately from disk (finally block)
      │
      ▼
Pickled sanitized DataFrame saved to OS Temp Directory (/tmp/contextguard_sessions/<UUID>.pkl)
      │
      ▼
Session Images saved to static/images/<UUID>/
      │
      ├───────────────────────┬────────────────────────┐
      ▼                       ▼                        ▼
Filter by Date (/filter)  Delete Session (/delete)  Clear All (/clear_all)
Re-slices in-memory df    Purges <UUID>.pkl &      Purges all .pkl files,
and returns JSON          static/images/<UUID>/    all static images, & uploads
```

1. **Zero External Data Transmission:** Zero telemetry, zero cloud API calls.
2. **Ephemeral Disk Persistence:** Uploaded files are deleted from `uploads/` immediately after parsing in a guaranteed `finally:` block.
3. **Session Cache Isolation:** Sessions are stored under cryptographically random UUIDs (`SESSION_CACHE_DIR`).
4. **Purge Endpoints:**
   - `POST /delete_session`: Instantly removes the pickled session and image folder.
   - `POST /clear_all`: Completely resets the application workspace, wiping all session caches, images, and upload residuals.
5. **Anonymized Summary Export:** `GET /export/<session_id>` delivers a structured JSON payload containing aggregated statistical distributions and anonymized identifiers (`User1`, `User2`) with zero original contact data.

---

## 8. REST API & Route Reference

| HTTP Method & Endpoint | Content-Type | Payload / Query Parameters | Response Structure | Description |
|---|---|---|---|---|
| `GET /` | `text/html` | None | HTML | Renders application landing page. |
| `POST /analyze` | `multipart/form-data` | `chat_file`: `.txt` file<br>`abuse_threshold`: float (`0.05` to `0.95`, default `0.50`) | HTML (`results.html`) | Executes full 12-stage analytical pipeline. |
| `POST /filter` | `application/json` | `{"session_id": "...", "from_date": "YYYY-MM-DD", "to_date": "YYYY-MM-DD"}` | `application/json` | Slices cached DataFrame and recomputes all metrics and visualizations without re-parsing. |
| `POST /delete_session` | `application/json` | `{"session_id": "..."}` | `{"status": "deleted", "session_id": "..."}` | Purges session pickle and temporary image directory. |
| `POST /clear_all` | `application/json` | None | `{"status": "cleared", "session_cache_purged": N, ...}` | Wipes all session pickles, temporary charts, and uploads across the application. |
| `GET /export/<session_id>` | `application/json` | URL parameter `session_id` | Downloadable JSON attachment (`chatsense_report_<id>.json`) | Returns privacy-compliant aggregate analytics summary. |

#### Example: Filter Cached Session by Date Range

```powershell
curl.exe -X POST "http://localhost:5000/filter" `
  -H "Content-Type: application/json" `
  -d '{"session_id": "d19882d497d04d039685e1bc8fdaf523", "from_date": "2026-08-01", "to_date": "2026-08-31"}'
```

---

## 9. Developer Installation & Environment Setup

### 9.1 Prerequisites
- Python 3.10, 3.11, or 3.12
- `git`
- `curl` (for automated model weight download)
- Supported OS: Windows 10/11, macOS (Intel/Apple Silicon), Ubuntu 20.04+

### 9.2 Virtual Environment & Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/chatsense.git
cd chatsense

# Initialize virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy English NER model
python -m spacy download en_core_web_sm
```

### 9.3 Model Weights Download

The fine-tuned DistilBERT manipulation classifier (`SamanthaStorm/abusedetector`) requires pre-trained PyTorch weights (~253 MB). Download the model files directly into `data/models/abusedetector/`:

#### Option A: PowerShell / Windows

```powershell
mkdir data\models\abusedetector -Force

curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/config.json" -o "data\models\abusedetector\config.json"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/modeling_abuse.py" -o "data\models\abusedetector\modeling_abuse.py"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/training_info.json" -o "data\models\abusedetector\training_info.json"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/tokenizer.json" -o "data\models\abusedetector\tokenizer.json"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/tokenizer_config.json" -o "data\models\abusedetector\tokenizer_config.json"
curl.exe -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/vocab.txt" -o "data\models\abusedetector\vocab.txt"
curl.exe -L -C - "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/pytorch_model.bin" -o "data\models\abusedetector\pytorch_model.bin"
```

#### Option B: Bash / Linux / macOS

```bash
mkdir -p data/models/abusedetector

curl -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/config.json" -o "data/models/abusedetector/config.json"
curl -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/modeling_abuse.py" -o "data/models/abusedetector/modeling_abuse.py"
curl -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/training_info.json" -o "data/models/abusedetector/training_info.json"
curl -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/tokenizer.json" -o "data/models/abusedetector/tokenizer.json"
curl -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/tokenizer_config.json" -o "data/models/abusedetector/tokenizer_config.json"
curl -L "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/vocab.txt" -o "data/models/abusedetector/vocab.txt"
curl -L -C - "https://huggingface.co/SamanthaStorm/abusedetector/resolve/main/pytorch_model.bin" -o "data/models/abusedetector/pytorch_model.bin"
```

### 9.4 Launch Development Server

```powershell
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 10. Testing, Quality Assurance & Edge-Case Verification

### 10.1 Regression Test Suite (`tests/`)

The repository maintains 58 comprehensive unit and integration tests across three test modules:

```powershell
pytest tests/ -v
```

- `tests/test_abuse_integration.py` (12 tests): Weight loading, CPU fallback, multi-label output schema, threshold scaling, and Flask route integration.
- `tests/test_chatsense.py` (40 tests): Android & iOS regex parsing, multiline concatenation, system message stripping, PII substitution, text preprocessing, VADER polarity scoring, and context smoothing.
- `tests/test_robustness_suite.py` (6 tests): Sarcasm sentiment ambiguity, negation preservation, disguised/noisy phone numbers, and ultra-short message abstention.

### 10.2 Synthetic Boundary Suite (`syn_test/`)

Executes 14 synthetic stress-test scenarios covering parser and NLP edge cases:

```powershell
python syn_test/run_edge_case_tests.py
```

| ID | Test Scenario | Validated Behavior |
|---|---|---|
| `01` | Single Turn Message | Bypasses context rolling smoothly without division-by-zero errors. |
| `02` | Single Participant Notes | Handles monologue self-chats with reciprocity score = 0.0. |
| `03` | Empty File | Graceful error screen rendering without 500 exceptions. |
| `04` | System Messages Only | Filters all notification lines, resulting in a safe zero-message rejection. |
| `05` | Multiline & Quote Replies | Correctly aggregates wrapped lines and identifies `is_reply=True`. |
| `06` | Emojis & Symbols Only | Assigns low confidence ($C = 0.25$) and triggers abstention. |
| `07` | Intense Manipulation Dialogue | Identifies elevated pattern scores and displays leave-one-out cards. |
| `08` | Healthy Boundaries | Categorizes boundary defense as constructive non-abusive dialogue. |
| `09` | Same-Minute Bursts | Handles timestamp collisions without corrupting sequential ordering. |
| `10` | Out-of-Order Timestamps | Sorts turns sequentially without pipeline degradation. |
| `11` | Large Group (35 Senders) | Prunes NetworkX visualization to top participants cleanly. |
| `12` | Special Senders | Sanitizes punctuation, spaces, and international script sender names. |
| `13` | Extreme Long Message (8 KB) | Truncates transformer tokens to 72 without OOM exceptions. |
| `14` | iOS Bracket Format | Parses `[DD/MM/YYYY, HH:MM:SS AM/PM]` bracketed exports reliably. |

---

## 11. Repository & Directory Structure

```
chatsense/
├── .gitignore                          # Production gitignore (ML models, caches, OS, session dirs)
├── README.md                           # Developer & architectural documentation
├── app.py                              # Flask application controller & route handlers
├── fix.py                              # Utility script for HTML template entity patching
├── requirements.txt                    # Project dependency specification
├── chatsense/                          # Core analytical NLP package
│   ├── __init__.py                     # Package entry point & public exports
│   ├── abuse_detector.py               # Pretrained DistilBERT PyTorch module & model wrapper
│   ├── chat_insights.py                # Statistical aggregations & temporal analytics
│   ├── context_analyzer.py             # Sliding-window context decay sentiment blending
│   ├── conversation_graph.py           # NetworkX directed multigraph & reciprocity engine
│   ├── emotion_analyzer.py             # 6-signal affect tracking & change-point detection
│   ├── evaluation.py                   # Sentiment benchmark evaluator & scikit-learn runner
│   ├── explainability.py               # Leave-one-out occlusion attribution engine
│   ├── manipulation_adapter.py         # Adapter layer, context formatter & candidate filter
│   ├── pii_anonymizer.py               # Regex tokenizer & spaCy NER anonymizer + leakage audit
│   ├── sentiment_model.py              # VADER sentiment intensity & confidence calibrator
│   ├── text_preprocessor.py            # Normalization, repeated char collapse, emoji parser
│   ├── uncertainty.py                  # Expected Calibration Error, Brier score & OOD heuristics
│   ├── visualization.py                # Headless Matplotlib & Seaborn dark-mode figure generator
│   └── whatsapp_parser.py              # Android & iOS WhatsApp regex parser & multiline handler
├── data/                               # Evaluation datasets and model directories
│   ├── generate_eval_dataset.py        # Benchmark dataset generator script
│   ├── sentiment_eval_dataset.csv      # 120-message labeled ground-truth sentiment benchmark
│   └── models/                         # Pretrained transformer checkpoints
│       └── abusedetector/              # SamanthaStorm/abusedetector tokenizer & configuration
│           ├── config.json             # Architecture & label index mapping
│           ├── modeling_abuse.py       # PyTorch module definition
│           ├── special_tokens_map.json # Special token dictionary
│           ├── tokenizer.json          # Fast tokenizer configuration
│           ├── tokenizer_config.json   # Hugging Face tokenizer hyperparams
│           ├── training_info.json      # Pretraining dataset & hyperparameter metadata
│           ├── vocab.txt               # WordPiece vocabulary
│           └── pytorch_model.bin       # [Excluded from git] ~253 MB model weights
├── static/                             # Web assets and generated figures
│   ├── style.css                       # Modern dark-mode responsive stylesheet
│   └── images/                         # Visual assets
│       ├── .gitkeep                    # Directory structure marker
│       └── confusion_matrix.png        # Baseline evaluation confusion matrix visualization
├── templates/                          # Jinja2 HTML templates
│   ├── error.html                      # Structured user error template
│   ├── index.html                      # Modern drag-and-drop landing page
│   └── results.html                    # Comprehensive interactive dashboard
├── tests/                              # PyTest test suite (58 regression tests)
│   ├── conftest.py                     # PyTest test fixtures
│   ├── test_abuse_integration.py       # AbuseDetector unit & Flask integration tests
│   ├── test_chatsense.py               # Parser, anonymizer, sentiment & context tests
│   └── test_robustness_suite.py        # Robustness against sarcasm, negation & noisy inputs
├── syn_test/                           # [Local test harness] 14 synthetic edge-case tests
│   ├── run_edge_case_tests.py          # Synthetic edge case test runner
│   ├── bench_audit_timings.py          # Automated latency & timing benchmark suite
│   ├── generate_synthetic_cases.py     # Script generating boundary test files
│   ├── report.md                       # Engineering and scientific audit report
│   └── 01_single_message.txt ...       # 14 synthetic edge-case text files
├── testing/                            # [Local test harness] Specialized evaluation scripts
│   ├── counterfactual_eval.py          # Counterfactual context contrast evaluation
│   ├── evaluate_system.py              # Full system benchmark runner
│   └── test_abusedetector.py           # Standalone PyTorch architecture validation
└── uploads/                            # Temporary upload directory
    └── .gitkeep                        # Directory structure marker
```

---

## 12. Scientific & Ethical Disclaimer

> [!IMPORTANT]
> **Academic & Educational Research Instrument Only:**
> 1. **Not Psychological Diagnosis:** ChatSense AI estimates statistical patterns in text. Outputs **do not diagnose mental health conditions**, personality disorders, psychological abuse, or interpersonal malice.
> 2. **Not Legal or Forensic Evidence:** Algorithmic scores must never be utilized as courtroom evidence, child custody documentation, or proof of guilt.
> 3. **Contextual Fallibility:** Sarcasm, humor, culturally specific colloquialisms, stressful life events, and legitimate arguments can trigger elevated tension scores.
> 4. **Anonymization Scope:** The built-in PII anonymizer sanitizes explicit names, phone numbers, standard email formats, and URLs. It cannot protect against re-identification via unique quasi-identifiers, personal anecdotes, or unlisted third-party references.

---

## 13. Contributing & License

Contributions are welcome! Please follow these standards:
1. **Tests Required:** Any new feature or bug fix must include corresponding tests in `tests/`.
2. **Code Style:** Strict adherence to PEP 8. Format with `black` or `ruff`.
3. **No External Egress:** Pull requests introducing network calls during chat processing will be rejected.

This project is licensed under the **MIT License**.
