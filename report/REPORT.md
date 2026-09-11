# SpotifyCares AI Support Agent — Engineering & Evaluation Report

**Candidate**: **Chetan C Chikkegowda**  
**Email**: [chetanccgowda@gmail.com](mailto:chetanccgowda@gmail.com) | **Phone**: +91 9611225645  
**GitHub**: [https://github.com/C3Gowdacq](https://github.com/C3Gowdacq)  
**Portfolio**: [https://c3-gowdacq-github-io.vercel.app/](https://c3-gowdacq-github-io.vercel.app/)  
**LinkedIn**: [https://www.linkedin.com/in/chetan-c-chikkegowda-724b7b37b](https://www.linkedin.com/in/chetan-c-chikkegowda-724b7b37b)  

**Project**: SpotifyCares AI Support Pipeline (Hiver SDE Intern Take-Home)  
**Target Brand**: `@SpotifyCares` (Customer Support on Twitter / X)  
**Tech Stack**: Python 3.12, LangChain LCEL, Groq LPUs (`openai/gpt-oss-120b` / `qwen/qwen3.8-27b`), FAISS (`IndexFlatIP`), scikit-learn, Pydantic v2  
**Dataset**: Kaggle Customer Support on Twitter (`twcs.csv`, ~2.81M rows filtered to 6,500 SpotifyCares pairs)  
**Evaluation Set**: N=200 stratified balanced customer tweets across 7 intents and 3 temporal terciles (2014–2017)  

---

## 1. Executive Summary & Problem Framing

Automating customer support for high-volume consumer platforms like Spotify presents a critical tension: **maximizing resolution velocity while eliminating hallucination and security risk**. A customer complaining about a playlist can safely receive an automated troubleshooting link, but a customer experiencing an unauthorized charge or hacked account must be transitioned to private, authenticated support immediately.

### Problem Framing: What "Good" Means for @SpotifyCares
1. **Zero Security & Financial Compromise**: Spotify users regularly report compromised credentials, unauthorized renewals, or region locking. "Good" support strictly forbids discussing credentials or account PII on a public Twitter timeline; it mandates immediate, deterministic routing to Spotify's authenticated Direct Message (DM) portal with canonical links (`https://t.co/ldFdZRiNAt`).
2. **Authentic Brand Voice & Actionable Grounding**: Spotify support has a signature conversational cadence: a friendly greeting (*"Hey there!"*), empathetic validation of frustration, concrete self-serve steps, and personal agent initials (*"/AR"*, *"/GS"*, *"/CB"*). "Good" drafting reflects this brand personality without generic robotic boilerplate.
3. **Elimination of Fabricated URLs**: Hallucinating dead links or outdated settings (e.g., non-existent offline storage toggles) destroys user trust. A good agent only outputs verified Spotify domains grounded in historical brand resolutions.
4. **High Resolution Velocity with Explainable Safety**: Safely auto-resolving standard technical and playlist queries while guaranteeing that 100% of high-risk cases are escalated with an explicit, auditable reason.

### What We Chose NOT to Build (and Why)
1. **No Autonomous Account Mutation Tools**: We explicitly chose *not* to provide the LLM with write access to Spotify account databases or billing APIs. Granting an unconstrained LLM write permissions over public Twitter messages introduces severe prompt-injection and account takeover vulnerabilities.
2. **No Black-Box Prompted Escalation**: We rejected asking the LLM `"Should you escalate this? (yes/no)"` via open-ended prompts. LLMs exhibit calibration drift, sycophancy, and unpredictable thresholding on high-liability queries. Instead, we built a deterministic Python invariant policy layer (Rules 1–4) so safety guarantees are provable in code.
3. **No Overcomplicated Agent Graph Frameworks (LangGraph / Multi-Agent Swarms)**: The support triage pipeline is an invariant linear sequence: Classification → Retrieval → Grounded Drafting → Safety Routing. Introducing cyclic multi-agent loops introduces non-deterministic latency, token exhaustion, and debugging complexity with zero customer benefit.
4. **No Remote Proprietary Embedding API Calls**: Rather than incurring per-query latency and dollar costs via remote embedding endpoints (e.g., OpenAI text-embedding-3), we engineered a local dense TruncatedSVD + FAISS vector store that runs in <1 ms offline with zero API cost.

---

## 2. Benchmark Comparison Table

The full pipeline was evaluated against two independent baseline architectures on the balanced 200-example Golden Evaluation Set (`data/golden_set.csv`). All test cases were strictly excluded from the training and retrieval corpus (Zero-Leakage Protocol).

| Evaluation Dimension | Metric | Trivial Baseline | Simple Baseline (TF-IDF + LR) | Full LLM Pipeline (LangChain + FAISS) | Performance Delta vs Simple |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Classification** | **Overall Accuracy** | 15.0% | 80.0% | **62.0%** | **-18.0%** |
| | **Macro-Averaged F1** | 0.037 | 0.807 | **0.617** | **-0.190** |
| | **Security Intent F1 (`account_access`)** | 0.000 | 0.842 | **0.774** | **-0.068** |
| | **Billing Intent F1 (`billing_subscription`)**| 0.000 | 0.815 | **0.764** | **-0.051** |
| **Escalation Routing** | **Escalation Accuracy** | 28.5% | 86.5% | **87.0%** | **+0.5%** |
| | **Escalation Precision** | 28.5% | 91.1% | **73.1%** | **-18.0%** |
| | **Escalation Recall** | 100.0% | 89.5% | **86.0%** | **-3.5%** |
| | **Escalation F1** | 0.444 | 0.903 | **0.790** | **-0.113** |
| | **Missed Escalation Rate (FNR - Safety Risk)**| **0.0%** (Over-escalates) | **10.5%** | **14.0%** | **+3.5%** |
| | **Unnecessary Escalation Rate (FPR - Agent Burden)**| 100.0% | 8.9% | **12.6%** | **+3.7%** |
| **Response Quality** | **Retrieval Intent Relevance** | N/A | 42.0% (Lexical overlap) | **100.0%** (Dense FAISS) | **+58.0%** |
| | **Hallucinated URLs / Policies** | High | High (Stale copy-paste) | **0.0%** (Grounded prompting)| **Eliminated** |
| | **Tone & Empathy Compliance** | 0.0% | 35.0% | **92.5%** | **+57.5%** |
| | **LLM-as-a-Judge Overall Score (1–5)**| 1.80 | 2.85 | **3.73 / 5.00** | **+0.88** |
| | **Judge Inter-Rater Reliability** | N/A | N/A | **94.4% Adj / Kappa: 0.914**| **Almost Perfect Agreement** |

### Benchmark Analysis & Engineering Trade-Offs

1. **Why Linear TF-IDF Shines on Classification vs Prompted LLMs**:
   - The classical TF-IDF + Logistic Regression model was trained directly on 6,300 in-domain Spotify interaction pairs with 5,000 sublinear n-gram features. Because Twitter queries are brief (often <15 words) and feature distinctive trigger tokens (e.g. *"doku"*, *"receipt"*, *"offline"*, *"playlist"*), the linear model achieves strong lexical separation (80.0% accuracy).
   - In contrast, the zero/few-shot LLM (`openai/gpt-oss-120b` via Groq) operates without task-specific weight fine-tuning and occasionally misattributes polysemous queries between adjacent intents (e.g. classifying a feature inquiry about offline storage as `playback_streaming` rather than `feature_request_howto`). However, on high-stakes intents (`account_access` F1: 0.774, `billing_subscription` F1: 0.764), the LLM remains robust.

2. **The Qualitative Leap in Generation Quality (The True Bottleneck)**:
   - While the simple baseline scores well on classification, **its generation capability fails completely in production**. Nearest-neighbor retrieval on raw TF-IDF matches the true customer intent only 42.0% of the time, pasting verbatim third-party replies that include stale customer usernames (`@12345 2: ...`) and broken historical links.
   - The full LLM pipeline with dense FAISS retrieval achieves a **3.73 / 5.00** overall judge rating (vs 2.85 for simple copy-paste), with **4.05/5 Correctness & Safety** and **3.98/5 Brand Tone Compliance**, strictly injecting official Spotify DM routing (`https://t.co/ldFdZRiNAt`) while keeping replies empathetic and contextually aware.

3. **Escalation Layer Invariant Protection**:
   - By decoupling semantic classification from the deterministic Python safety layer (Decision 10), the pipeline achieves an **87.0% Escalation Accuracy** and catches 49 of 57 true escalations (86.0% recall), preventing private account or financial disclosures on public social media.

## 3. Architecture Deep-Dive

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

### Stage 1: Intent Classification
Spotify support issues map naturally to 7 operational categories derived through empirical n-gram and semantic clustering of raw customer messages:
1. `playback_streaming`: Offline playback bugs, audio stutter, device connect/Chromecast disconnects, app crashes.
2. `account_access`: Password recovery, hacked/compromised credentials, Facebook OAuth disconnects, country/region locks.
3. `billing_subscription`: Duplicate charges, student discount verification, family plan management, refunds.
4. `content_availability`: Region-locked albums, greyed-out songs, release timing expectations (e.g. Taylor Swift).
5. `feature_request_howto`: UI navigation, collaborative playlist configuration, local file syncing.
6. `venting_feedback`: Emotional rants regarding recommendations (Discover Weekly), UI redesigns, general brand feedback.
7. `other`: Ambiguous fragments, greetings, multilingual noise.

### Stage 2: Grounded Retrieval-Augmented Drafting
Rather than allowing the LLM to freely generate advice, Stage 2 enforces strict context grounding:
- The customer query is projected into a 256-dimensional semantic space via TruncatedSVD over sublinear TF-IDF n-grams.
- FAISS executes an exact inner-product search (`IndexFlatIP`) against 6,300 verified historical `@SpotifyCares` resolution threads in <1 ms.
- Top-3 past threads are formatted into few-shot demonstration context.
- System prompt negative constraints explicitly forbid generating unverified URLs, mandating the canonical `@SpotifyCares` DM portal (`https://t.co/ldFdZRiNAt`) when private information is required.

### Stage 3: Explainable Escalation Decision Engine
While large language models excel at generation, delegating safety-critical routing decisions entirely to an unconstrained prompt introduces non-deterministic hallucinations. The pipeline applies a deterministic Python rule matrix:
- **Rule 1 (Privacy & Security Invariant)**: Any query classified as `account_access` or `billing_subscription` triggers mandatory escalation. Spotify agents require internal CRM lookups and customer verification that cannot occur over a public Twitter timeline.
- **Rule 2 (Legal & Churn Severity Invariant)**: High-risk keywords (`sue`, `lawsuit`, `fraud`, `scam`, `stolen`, `cancel subscription`) bypass automated handling immediately.
- **Rule 3 (Confidence Floor)**: Classification confidence below 0.70 routes to human review.
- **Rule 4 (Retrieval Novelty Guard)**: Maximum FAISS similarity below 0.40 indicates an unseen platform bug or novel incident, triggering human review.

---

## 4. Top 5 Failure Modes (Failure Analysis with Real Examples & Hypotheses)

A rigorous evaluation must examine boundary failures to understand where the system breaks. Below are the top 5 operational failure modes observed across the 200-sample golden evaluation set.

### Failure Mode 1: Polysemous Sarcasm & Pop-Culture Venting
* **Customer Tweet** (`186494`): *"@SpotifyCares you feelin strong? Auto play that Andy Grammer song that's on 0 of my playlists one more time and see what happens"*
* **Ground Truth**: Intent: `venting_feedback` | Should Escalate: `0` (Humorous complaint about recommendation algorithms).
* **Pipeline Output**: Predicted Intent: `playback_streaming` (Confidence: 0.75) | Escalate: `False`.
* **Drafted Action**: Sent a generic playback caching guide (*"Try clearing your cache and reinstalling..."*).
* **Hypothesis & Root Cause**: Lexical co-occurrence of tokens *"Auto play"*, *"playlists"*, and *"song"* biased the classification prompt toward technical audio bugs rather than algorithmic dissatisfaction. The model lacks a dedicated sentiment/sarcasm feature representation.
* **Remedy**: Introduce a zero-shot sentiment/irony detector or split `venting_feedback` into a dedicated `recommendations_algorithm` sub-intent.

### Failure Mode 2: Multi-Intent Composite Requests
* **Customer Tweet** (`2043225`): *"@115888 I cannot access my account despite paying monthly to subscribe!?! Help please!!"*
* **Ground Truth**: Intent: `billing_subscription` | Should Escalate: `1` (Paid account inaccessible).
* **Pipeline Output**: Predicted Intent: `account_access` (Confidence: 0.88) | Escalate: `True` (Reason: Account access requires private verification).
* **Hypothesis & Root Cause**: The query is genuinely multi-intent (`account_access` + `billing_subscription`). Strict single-label evaluation marks this as an intent misclassification (reducing accuracy to 62.0%), but the invariant escalation engine correctly intercepted the safety risk and routed the customer to a human agent via DM.
* **Remedy**: Transition from mutually exclusive single-label classification to multi-label intent tagging with priority-ranked routing.

### Failure Mode 3: Truncated Third-Party Regional Payment Entities
* **Customer Tweet** (`1270548`): *"@SpotifyCares hi, i paid with doku and the payment is successful but my account is still free"*
* **Ground Truth**: Intent: `billing_subscription` | Should Escalate: `1`.
* **Simple Baseline Prediction**: Classified as `other` (TF-IDF had never encountered the localized Indonesian payment gateway token *"doku"* in training).
* **LLM Pipeline Prediction**: Classified as `billing_subscription` (Semantic comprehension recognized *"paid"*, *"payment"*, *"still free"*), Correctly Escalated.
* **Hypothesis & Root Cause**: Lexical baselines suffer severe out-of-vocabulary (OOV) drops on regional payment gateways, whereas semantic LLMs generalize cleanly from contextual syntax.

### Failure Mode 4: Colloquial Churn Phrasing Bypassing Exact Regex (Safety False Negative)
* **Customer Tweet** (`2653080`): *"@115888 plz tell me why Velvet by Stoney LaRue suddenly got deleted. Guess I’m ending my subscription. Bye."*
* **Ground Truth**: Intent: `billing_subscription` / Churn | Should Escalate: `1` (Imminent customer churn).
* **Pipeline Output**: Predicted Intent: `content_availability` (Confidence: 0.94) | Escalate: `False`.
* **Drafted Action**: Automatically replied explaining artist song licensing agreements without human notification.
* **Hypothesis & Root Cause**: Rule 2 scans for exact churn phrases like `"cancel subscription"`. The colloquial expression *"ending my subscription"* failed exact string matching. Simultaneously, the prominent song title *"Velvet by Stoney LaRue"* dominated the classification prompt, classifying it as `content_availability`.
* **Remedy**: Replace rigid string matching in Rule 2 with lemmatized regex patterns (`(end|cancel|stop|drop|quit).{0,10}(sub|subscription|premium)`) and an explicit LLM-detected churn flag.

### Failure Mode 5: Generic Security Inquiries Triggering False Escalation (False Positive)
* **Customer Tweet** (`1098081`): *"Security is important, @spotifycares. We'd like it if you supported two factor auth. https://t.co/dFe7f0NAmS #SupportTwoFactorAuth"*
* **Ground Truth**: Intent: `feature_request_howto` | Should Escalate: `0` (Public product feedback).
* **Pipeline Output**: Predicted Intent: `feature_request_howto` (Confidence: 0.95) | Escalate: `True` (Reason: *"Security/financial keywords detected: security."*).
* **Drafted Action**: Unnecessarily routed a public feature suggestion to human agent queue.
* **Hypothesis & Root Cause**: Over-eager invariant trigger. The isolated presence of the word *"security"* triggered the safety rule regardless of whether the customer was reporting an active breach or making a generic product suggestion.
* **Remedy**: Condition security keyword triggers on first-person distress indicators (*"my account"*, *"hacked"*, *"unauthorized"*) or restrict keyword scanning to `account_access` intents.

---

## 5. "What is misleading about my headline number?" (Mandatory Section)

A candid engineering evaluation requires dismantling headline metrics to expose their hidden caveats:

1. **The 87.0% Escalation Accuracy Hides Class Imbalance & Real Safety Risk**:
   - In our 200-sample balanced golden set, 143 cases are non-escalations (71.5%) and 57 are escalations (28.5%).
   - A trivial "Never Escalate" dummy agent would achieve **71.5% accuracy** with zero intelligence. Thus, an 87.0% headline accuracy only provides a marginal +15.5% delta over doing nothing.
   - More critically, accuracy conceals the **14.0% False Negative Rate (FNR)**: the pipeline failed to escalate **8 out of 57 real escalations**. In an enterprise production environment handling 50,000 queries daily, a 14% missed escalation rate would leave **hundreds of paying users with compromised accounts or unauthorized charges receiving automated bot replies**. In safety-critical routing, Recall and FNR are the only metrics that truly matter.

2. **The 62.0% Intent Accuracy Penalizes Multi-Intent Realities**:
   - Forcing single-label evaluation on natural customer dialogue produces an artificially deflated accuracy score. As demonstrated in Failure Case 2 (*"cannot access account despite paying"*), classifying as `account_access` instead of `billing_subscription` was marked as an error (-1) despite leading to the identical, correct downstream resolution (DM escalation).

3. **Stratified Benchmark vs Real-World Distribution Shift**:
   - Our 200-sample Golden Set enforced an artificial equal quota of ~14.3% per intent (28–30 examples each) to stress-test low-frequency security cases. In real-world Twitter data, `playback_streaming` and `other` constitute >60% of all inbound volume, while `account_access` represents <8%. The true production distribution will yield higher raw accuracy (boosted by easy playback queries) while masking security edge cases.

4. **Single-Turn Evaluation Ignores Multi-Turn Context**:
   - Many tweets in the Kaggle dataset are turn-2 or turn-3 follow-ups (*"I tried that, still didn't work"*, *"Thanks, it worked!"*). Evaluating tweets in isolation without prior conversational state degrades classification confidence.

5. **LLM-as-a-Judge Shared Prior Bias**:
   - While our judge achieved an exceptional **Cohen's Kappa of 0.914** against human auditing, both the generator and judge share underlying transformer architectures. Judges systematically reward polite, articulate, well-formatted replies (*"Hey there! Sorry to hear that..."*) even when a blunt, one-sentence direct link would resolve the customer's problem faster.

---

## 6. What I'd Do Next With One More Week

If given one additional week of engineering time, here is the exact execution roadmap:

1. **Multi-Turn Dialogue State Tracking (DST)**:
   - Group Kaggle tweets by `author_id` and temporal clustering (<2 hours) into coherent conversation sessions. Pass the conversation history window into Stage 1, resolving follow-up ambiguity.
2. **Domain Fine-Tuning of Compact On-Premise Models (LoRA / QLoRA)**:
   - Fine-tune a lightweight 7B/8B model (e.g. `Qwen-2.5-7B` or `Llama-3.1-8B`) directly on the 6,300 filtered Spotify interaction pairs. This would elevate intent accuracy from 62.0% to >88%, eliminate third-party API rate limits, and cut per-query latency to <150 ms.
3. **Context-Aware Dependency Parsing for Safety Rules**:
   - Replace flat keyword lists with dependency-parsed subject-verb-object rules (e.g., detecting `[Subject: User] + [Action: End/Cancel] + [Object: Subscription]`) to eliminate false-positive feature request escalations (Failure Mode 5) and catch colloquial churn threats (Failure Mode 4).
4. **Mock CRM / Billing Tool Integration**:
   - Equip the agent with read-only tool functions (`get_subscription_status(user_id)`, `check_outage_map(region)`) so the agent can autonomously diagnose account issues before deciding whether human intervention is necessary.
5. **Direct Integration with Hiver Shared Inbox Workflow (Copilot Mode)**:
   - Build a lightweight webhook integration that feeds drafted replies directly into Hiver's shared inbox UI as suggested responses, allowing human agents to approve or edit with 1 click.

---

## 7. Citations & Attributions

- **Dataset**: *Customer Support on Twitter* (`twcs.csv`), curated by thoughtvector on Kaggle (~2.81 million customer support interactions across brands).
- **LangChain**: LangChain LCEL (`langchain-core`, `langchain-groq`) for structured chain composition and Pydantic schema validation.
- **FAISS**: Facebook AI Similarity Search (`faiss-cpu`, Meta Research) for exact inner product vector similarity retrieval.
- **Scikit-Learn**: Pedregosa et al., for TF-IDF vectorization, TruncatedSVD dimensionality reduction, Logistic Regression baselines, and Cohen's Quadratic Weighted Kappa computation.
- **Inference Infrastructure**: Groq LPUs (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) providing high-throughput inference.

