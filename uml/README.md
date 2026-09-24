# ChatSense — UML Diagrams

**Project:** Privacy-Aware WhatsApp Psychological Manipulation Detection  
**Repo:** https://github.com/anaconda2401/sentiment-analyzer  
**Tool:** PlantUML (`.puml` files — render with VS Code, IntelliJ, or `plantuml.jar`)

---

## Architecture Summary

```
WhatsApp .txt
  → WhatsAppParser       (parse, filter system/media messages)
  → PIIAnonymizer        (regex + spaCy NER — one instance per request)
  → TextPreprocessor     (NFKC, emoji->alias, whitespace, repeats)
  → VADER SentimentModel (Positive / Neutral / Negative + confidence)
  → DistilBERT ManipulationDetector  ← future
     (Sarcasm / Guilt Tripping / Emotional Pressure /
      Gaslighting / Coercive Patterns)
  → ContextAnalyzer      (sliding-window blend 70/30, window=4)
  → ChatInsights         (aggregates, participant breakdown, manip counts)
  → Visualizer           (5 Matplotlib/Seaborn charts)
  → Flask Dashboard      (GET / | POST /analyze | POST /filter AJAX)
```

Both **VADER** (sentiment) and **DistilBERT** (manipulation) run in parallel
on each preprocessed message. They are separate, independent components.

---

## Diagram Index

| # | File | Diagram Type | Description |
|---|---|---|---|
| 0 | `00_main_architecture.puml` | **Overview** | Full system data-flow: parser → PII → preprocessing → VADER + DistilBERT → context → insights → Flask dashboard |
| 1 | `01_class_diagram.puml` | **Class** | All classes, attributes, methods, relationships (incl. `ManipulationDetector`) |
| 2 | `02_use_case_diagram.puml` | **Use-Case** | User/Server actor flows; DistilBERT and VADER as system actors |
| 3 | `03_sequence_analyze.puml` | **Sequence** | Full `POST /analyze` message flow — dual model inference shown in parallel |
| 4 | `04_sequence_filter.puml` | **Sequence** | AJAX `POST /filter` — date-filtered reload reuses already-computed columns |
| 5 | `05_activity_pipeline.puml` | **Activity** | Branching pipeline with fork for VADER + DistilBERT parallel inference |
| 6 | `06_component_diagram.puml` | **Component** | All components, third-party libs (HF, PyTorch), storage, wiring |
| 7 | `07_state_message.puml` | **State Machine** | State transitions of a single message from RawText → Persisted |
| 8 | `08_deployment_diagram.puml` | **Deployment** | Dev machine, browser, ML training env, file system, GPU |

---

## DistilBERT Integration Notes

| Item | Detail |
|---|---|
| Model | `distilbert-base-uncased` fine-tuned on D2 |
| Training data | `testing/final_train.csv` — 12,433 sentences |
| Classes | Safe (0) / Manipulative (1) |
| Techniques detected | Sarcasm, Guilt Tripping, Emotional Pressure, Gaslighting, Coercive Patterns |
| Hardware | RTX 3050 Laptop (4 GB VRAM), batch size 16–32 |
| Status | **Deferred** — datasets prepared, training not yet started |
| VADER | Remains the sentiment (tone) analyser — completely separate |

---

## How to Render

### VS Code (recommended)
1. Install **PlantUML** extension (`jebbs.plantuml`)
2. Open any `.puml` file
3. Press `Alt+D` to preview

### Command line (requires Java)
```bash
java -jar plantuml.jar uml/*.puml -o uml/rendered/
```

### Online
Paste file contents at https://www.plantuml.com/plantuml/uml/