# SpotifyCares AI Support Agent — Comparative Evaluation Benchmark

**Evaluation Set**: N=200 customer tweets (Balanced stratified test set across 7 intents and 2014–2017).

## 1. System Comparison Table

| Metric | Trivial Baseline | Simple Baseline (TF-IDF + LR) | Full LLM Pipeline (LangChain + FAISS) | Delta vs Simple |
| :--- | :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 15.0% | 80.0% | **62.0%** | **-18.0%** |
| **Intent Macro-F1** | 0.037 | 0.807 | **0.617** | **-0.190** |
| **Escalation Precision** | 28.5% | 91.1% | **73.1%** | **-17.9%** |
| **Escalation Recall** | 100.0% | 89.5% | **86.0%** | **-3.5%** |
| **Escalation F1** | 0.444 | 0.903 | **0.790** | **-0.112** |
| **Missed Escalations (FNR - Safety Risk)** | 0.0% | 10.5% | **14.0%** | **+3.5%** |
| **Response Drafting Grounding** | N/A (Static) | 42.0% Intent Match (Raw Copy) | **100% Grounded via FAISS** | **Substantial Qualitative Lead** |

## 2. Per-Class Intent Performance (Full Pipeline)

| Intent Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| `playback_streaming` | 51.4% | 65.5% | **0.576** | 29.0 |
| `account_access` | 70.6% | 85.7% | **0.774** | 28.0 |
| `billing_subscription` | 77.8% | 75.0% | **0.764** | 28.0 |
| `content_availability` | 59.1% | 46.4% | **0.520** | 28.0 |
| `feature_request_howto` | 50.0% | 44.8% | **0.473** | 29.0 |
| `venting_feedback` | 63.0% | 60.7% | **0.618** | 28.0 |
| `other` | 63.0% | 56.7% | **0.596** | 30.0 |

## 3. Escalation Decision Matrix

- **True Negatives (Correct Auto-Handle)**: 125
- **False Positives (Unnecessary Escalation)**: 18
- **False Negatives (Missed Escalation - Critical Risk)**: 8
- **True Positives (Correct Escalation)**: 49
