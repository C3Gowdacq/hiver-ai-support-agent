# SpotifyCares AI Support Agent (Hiver SDE Intern Take-Home)

An end-to-end, production-grade Customer Support AI Agent built for **`@SpotifyCares`** on Twitter / X, benchmarked against the Kaggle Customer Support dataset (`twcs.csv`).

Decouples **semantic reasoning** from **safety routing** using LangChain LCEL, dense FAISS vector retrieval, and deterministic explainable escalation.

---

## ⚡ < 15-Minute Fast Reproduction Run

All responses, embeddings, and intermediate evaluation runs are cached to disk via SHA-256 content addressing. **Reviewers can reproduce the full benchmark in minutes with zero external API costs.**

### 1. Environment Setup
```bash
# Clone and enter directory
cd c:/Chetan/Hiver

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate     # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create or edit `.env` in the project root:
```env
# Groq API Key (get free at https://console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# Supported Models: openai/gpt-oss-120b, qwen/qwen3.8-27b, llama-3.3-70b-versatile
LLM_MODEL=openai/gpt-oss-120b
LLM_PROVIDER=groq
```

### 3. Step-by-Step Pipeline Execution
```bash
# Step 1: Run Classical Baselines (Trivial & TF-IDF + Logistic Regression)
python src/baselines.py

# Step 2: Test FAISS Vector Index Retrieval (<1 ms local retrieval)
python src/retriever.py

# Step 3: Run Full LLM Pipeline on Golden Test Set (N=200, cached for instant replay)
python src/pipeline.py

# Step 4: Run Automated Evaluation Harness (Accuracy, Macro-F1, Escalation Matrix)
python src/evaluate.py

# Step 5: Run LLM-as-a-Judge & Inter-Rater Reliability Audit (Cohen's Kappa)
python src/judge.py
```

---

## 📊 Benchmark Results Summary

Evaluated on the balanced 200-example Golden Evaluation Set (`data/golden_set.csv`) strictly purged from the retrieval corpus (**Zero Data Leakage**):

| Benchmark Metric | Trivial Baseline | Simple Baseline (TF-IDF + LR) | Full LLM Pipeline (LangChain + FAISS) | Performance Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 15.0% | 80.0% | **62.0%** | **-18.0%** (Un-finetuned LLM) |
| **Intent Macro-F1** | 0.037 | 0.807 | **0.617** | **-0.190** |
| **Security F1 (`account_access`)** | 0.000 | 0.842 | **0.774** | **-0.068** |
| **Billing F1 (`billing_subscription`)** | 0.000 | 0.815 | **0.764** | **-0.051** |
| **Escalation Accuracy** | 28.5% | 86.5% | **87.0%** | **+0.5%** |
| **Escalation Precision** | 28.5% | 91.1% | **73.1%** | **-18.0%** |
| **Escalation Recall** | 100.0% | 89.5% | **86.0%** | **-3.5%** |
| **Safety Risk (Missed Escalations)** | 0.0% (Over-escalates) | **10.5%** | **14.0%** (8 missed of 57) | **+3.5%** |
| **LLM Judge Score (1–5)** | 1.80 | 2.85 | **3.73 / 5.00** | **+0.88** (Substantial Quality Lead) |
| **Judge Inter-Rater Reliability** | N/A | N/A | **94.4% Adj / Kappa: 0.914** | **Almost Perfect Agreement** |

---

## 🏗️ Architecture & Core Components

```
                           [ Inbound Customer Tweet ]
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │  Stage 1: Few-Shot Intent Classifier  │
                   │  (qwen/qwen3.8-27b / gpt-oss-120b)    │
                   │  Outputs: Intent, Confidence, Reason  │
                   └───────────────────┬───────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
        ┌─────────────────────────┐         ┌─────────────────────────┐
        │ Stage 2: Dense FAISS    │         │ Stage 3: Explainable    │
        │ Retriever (IndexFlatIP) │         │ Escalation Engine       │
        │ Top-3 Grounding Threads │         │ (Invariant Policy Layer)│
        └────────────┬────────────┘         └────────────┬────────────┘
                     │                                   │
                     ▼                                   │
        ┌─────────────────────────┐                      │
        │ Grounded Reply Drafter  │                      │
        │ Authentic Spotify Tone  │                      │
        │ Official DM Link Guard  │                      │
        └────────────┬────────────┘                      │
                     │                                   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ Human-in-the-Loop Routing Engine  │
                     └─────────────────┬─────────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 │                                           │
                 ▼                                           ▼
      [ Escalation Queue ]                        [ Auto-Handled Stream ]
   (Account, Billing, Legal,                                 │
     Low Confidence <0.70)                       ┌───────────┴───────────┐
                                                 ▼                       ▼
                                       [ Spot-Check Queue ]       [ Published Reply ]
                                         (7.5% QA Audit)
```

1. **Stage 1 — Intent Classification**: Structured output (`IntentResult`) over 7 operational classes:
   - `playback_streaming`, `account_access`, `billing_subscription`, `content_availability`, `feature_request_howto`, `venting_feedback`, `other`.
2. **Stage 2 — Grounded Reply Drafting**: Top-3 retrieval via FAISS `IndexFlatIP` over 6,300 historical Spotify pairs. Negative constraints prevent fake URLs and preserve signature `/Initial` agent sign-offs.
3. **Stage 3 — Explainable Escalation**: Invariant rule layer guaranteeing zero unhandled security breaches (`account_access`, `billing_subscription`, legal/churn triggers) with explicit audit reasons.
4. **Human-in-the-Loop (HITL)**: Two-tier logging to `eval/hitl_review_log.jsonl` (Escalation Queue + 7.5% Bernoulli Spot-Check Queue).

---

## 📁 Repository Structure

```
├── .env                              # API configuration (Groq / Gemini)
├── cache/                            # Content-addressed SHA-256 cache files
│   ├── embeddings_model.pkl          # Persisted 256-dim TruncatedSVD model
│   └── faiss_index/                  # Persisted FAISS vector index (6,300 vectors)
├── data/
│   ├── twcs.csv                      # Raw Kaggle dataset (2.81M rows)
│   ├── spotify_pairs_subsample.csv   # Filtered 6,500 SpotifyCares interaction pairs
│   ├── golden_set.csv                # 200 balanced, stratified ground-truth test set
│   └── intent_taxonomy.md            # Empirical 7-class intent definitions
├── eval/
│   ├── baseline_results.json         # Trivial and Simple baseline performance
│   ├── pipeline_predictions.csv      # Full LLM pipeline outputs on test set
│   ├── evaluation_comparison.json    # Consolidated benchmark comparison metrics
│   └── hitl_review_log.jsonl         # Escalation and spot-check human audit queue
├── report/
│   ├── REPORT.md                     # Comprehensive technical & failure-mode analysis
│   └── DECISION_LOG.md               # 14 real-time engineering trade-off logs
└── src/
    ├── baselines.py                  # Scikit-learn baseline implementations
    ├── cache.py                      # SHA-256 disk caching utility
    ├── evaluate.py                   # Automated metric computation harness
    ├── judge.py                      # LLM-as-a-judge rubric & Cohen's Kappa
    ├── pipeline.py                   # 3-stage LangChain LCEL pipeline orchestrator
    └── retriever.py                  # Dense FAISS vector retriever
```

---

## 🛠️ Verification & Testing
Run the quick smoke test to verify all pipeline components in under 5 seconds:
```bash
python src/test_pipeline.py
```
Outputs:
```text
============================================================
Tweet ID: T1 | Text: I cant play my songs offline even though I have premium
Intent: playback_streaming | Confidence: 0.95 | Escalate: False
Drafted Reply: Hey there! Sorry you're having trouble with offline playback...
============================================================
Tweet ID: T2 | Text: Someone hacked my Spotify account please help
Intent: account_access | Confidence: 0.95 | Escalate: True
Reason: Intent 'account_access' requires private account/billing verification via DM.
============================================================
Tweet ID: T3 | Text: Why is Taylor Swift reputation not on Spotify yet??
Intent: content_availability | Confidence: 0.95 | Escalate: False
Drafted Reply: Hey there! Taylor Swift's 'Reputation' isn't available to stream just yet...
```

---

## 📑 Hiver Take-Home Deliverables Mapping

| # | Hiver Deliverable Requirement | Repository Location & Status |
| :--- | :--- | :--- |
| **1** | **Runnable Pipeline & Fast Reproduction (<15 min)** | [`README.md`](file:///c:/Chetan/Hiver/README.md) (Step-by-step commands, disk-cached runs in <5 min) |
| **2** | **Golden Evaluation Set (150–250 hand-labelled examples)** | [`data/golden_set.csv`](file:///c:/Chetan/Hiver/data/golden_set.csv) (N=200 balanced & stratified) + [`data/GOLDEN_SET_METHODOLOGY.md`](file:///c:/Chetan/Hiver/data/GOLDEN_SET_METHODOLOGY.md) |
| **3** | **Evaluation Harness & LLM-as-a-Judge** | [`src/evaluate.py`](file:///c:/Chetan/Hiver/src/evaluate.py) (automated metrics) + [`src/judge.py`](file:///c:/Chetan/Hiver/src/judge.py) (4-dim rubric, Cohen's Kappa = 0.914) |
| **4** | **Comprehensive Engineering Report (Max 6 pages)** | [`report/REPORT.md`](file:///c:/Chetan/Hiver/report/REPORT.md) (Problem framing, 2 baselines, top 5 failure modes, mandatory headline critique, 1-week roadmap) |
| **5** | **Decision Log (10–15 non-obvious engineering decisions)** | [`report/DECISION_LOG.md`](file:///c:/Chetan/Hiver/report/DECISION_LOG.md) (14 recorded decisions with context, decision, and trade-offs) |

