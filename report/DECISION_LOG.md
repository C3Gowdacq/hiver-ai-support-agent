# Decision Log: SpotifyCares AI Support Agent

This log records non-obvious architectural, algorithmic, and engineering trade-offs made during development, captured in real time.

---

### Decision 1: Streaming Two-Pass Filter for Subsample Extraction
* **Context**: The raw Kaggle `twcs.csv` dataset contains ~2.81 million rows (~516 MB). In-memory loading and pandas merging of the entire dataset exhausts memory and is needlessly slow.
* **Decision**: Implemented a streaming chunked two-pass extraction in `src/filter_subsample.py`:
  - *Pass 1*: Scanned in chunks of 100,000 rows to extract brand replies (`author_id == 'SpotifyCares'`) and indexed their parent `in_response_to_tweet_id`. Found 41,734 total brand replies.
  - *Pass 2*: Scanned for matching inbound customer tweets matching those IDs. Found 41,697 matching tweets, yielding 41,681 valid reply pairs.
* **Trade-off**: Favoring a deterministic, streaming Python filter avoids large memory footprints and guarantees that `twcs.csv` is only parsed once end-to-end.

---

### Decision 2: Subsample Size Selection (6,500 Pairs)
* **Context**: The prompt specifies a working subsample of ~5,000–8,000 customer↔brand reply pairs.
* **Decision**: Fixed the subsample size to **6,500 pairs** with a fixed random seed (`seed=42`) across the full chronological span (2014-05-15 to 2017-12-03).
* **Rationale**: 6,500 pairs provides high semantic coverage for FAISS vector search across all Spotify product areas (playlists, offline downloads, billing, connect/chromecast, account recovery) while keeping FAISS embedding time under ~2 minutes and index size under 20 MB.

---

### Decision 3: Framework & Architecture Choice (LangChain LCEL over LangGraph)
* **Context**: Need a traceable, inspectable 3-stage pipeline (Intent Classification → FAISS Retrieval → Grounded Drafting → Explainable Escalation).
* **Decision**: Selected LangChain LCEL (`|` piping) with Pydantic structured output, explicitly bypassing LangGraph.
* **Rationale**: The pipeline is an invariant linear sequence without dynamic state cycles or multi-agent branching. LangChain LCEL keeps the execution graph inspectable, transparent, and easy to explain live during technical review.

---

### Decision 4: Empirical 7-Class Intent Taxonomy Design
* **Context**: The prompt requires defining 5–8 intents grounded directly in real Spotify customer messages, avoiding fabricated categories.
* **Decision**: Formulated 7 mutually exclusive, empirically supported intent classes in `data/intent_taxonomy.md`:
  1. `playback_streaming`
  2. `account_access`
  3. `billing_subscription`
  4. `content_availability`
  5. `feature_request_howto`
  6. `venting_feedback`
  7. `other`
* **Rationale**: Derived from frequency and n-gram analysis of 200 real customer tweets. Each intent has a distinct operational resolution path: `account_access` and `billing_subscription` require private direct message routing and security verification (mandatory escalation), while `content_availability` and `feature_request_howto` lend themselves directly to grounded auto-replies.

---

### Decision 5: Two-Factor Balanced Stratification for Golden Evaluation Set
* **Context**: Naive sampling (e.g. first 200 rows or pure random sampling) suffers from class imbalance (over-representing dominant intents) and temporal bias (clustering around specific historical weeks).
* **Decision**: Implemented a two-factor balanced stratified sampling algorithm in `src/sample_golden_set.py`:
  - *Factor 1 (Temporal)*: 3 equal terciles across time (Early 33.5%, Mid 33.0%, Late 33.5%).
  - *Factor 2 (Semantic)*: Equal quota allocation across all 7 intent classes (28–30 examples each, ~14–15%).
* **Rationale**: Balanced test sets prevent majority-class bias from inflating headline accuracy/F1 scores, providing an honest, rigorous benchmark across high-stakes security/billing issues as well as standard playback bugs.

---

### Decision 6: Non-LLM Baselines & Leak-Free Split Architecture
* **Context**: The pipeline needs honest, inspectable reference benchmarks to establish true performance delta.
* **Decision**: Built two classical baselines in `src/baselines.py` using scikit-learn:
  - *Trivial*: Majority-class intent prediction (15.0% accuracy on balanced test set) and unconditional escalation ("Always Escalate", 28.5% precision).
  - *Simple*: TF-IDF (1-2 ngrams, 5,000 features) + Logistic Regression (80.0% accuracy, 80.7% macro-F1) with raw nearest-neighbor copy-paste retrieval.
  - *Zero Leakage Protocol*: Golden set tweet IDs (N=200) are strictly purged from the training and retrieval corpus (6,300 pairs remaining).
* **Key Finding**: Pure TF-IDF nearest-neighbor reply retrieval only retrieves the same intent 42.0% of the time and produces disjointed, fragmented tweets (e.g. "@user 2: ..."), proving the necessity of semantic FAISS vector retrieval and grounded LLM drafting.

---

### Decision 7: Dense TruncatedSVD + L2-Normalized FAISS Vector Store
* **Context**: Grounded reply generation requires retrieving semantically similar past customer↔agent pairs. Remote embedding APIs (e.g. OpenAI `text-embedding-3-small`) incur external network overhead, ongoing dollar cost, and API rate limits.
* **Decision**: Built a local dense representation using 256-dimensional TruncatedSVD over sublinear TF-IDF word n-grams (1-2 ngrams, English stop words removed), paired with a FAISS `IndexFlatIP` (inner product on L2-normalized unit vectors).
* **Rationale**: Cosine similarity is computed locally via native C++ FAISS in under 1 millisecond per query. The entire model (`cache/embeddings_model.pkl`) and index (`cache/faiss_index`) take ~8.5 MB of disk space, load in 0.15 seconds, and run 100% offline with zero per-query API cost.

---

### Decision 8: Content-Addressed SHA-256 Disk Cache
* **Context**: LLM inference over hundreds of test tweets involves repeated API round-trips that can fail on transient network errors or run up costs during development and grading.
* **Decision**: Implemented an inspectable file-based disk cache in `src/cache.py` storing individual JSON files keyed by `SHA256(model + extra_salt + prompt)`.
* **Rationale**: 
  - Makes all evaluation runs 100% deterministic and instantaneously reproducible for reviewers without requiring new API spend.
  - Granular per-prompt JSON files allow granular inspection of prompt inputs and model outputs without monolithic database dependencies.

---

### Decision 9: Model Selection on Groq Infrastructure (`qwen/qwen3.8-27b`)
* **Context**: Groq's active model endpoints evolve. While initial documentation suggested legacy Llama variants, live inspection revealed newer high-throughput models (`qwen/qwen3.8-27b`, `openai/gpt-oss-120b`).
* **Decision**: Evaluated available models for native tool calling and LangChain Pydantic structured output (`with_structured_output`). Selected `qwen/qwen3.8-27b` as the primary production engine with automated schema parsing.
* **Trade-off**: `qwen/qwen3.8-27b` delivers sub-second inference latency on Groq's LPU hardware while reliably conforming to strict Pydantic models for intent classification (`IntentResult`) and reply drafting (`ReplyDraft`).

---

### Decision 10: Hybrid Escalation Architecture (Deterministic Rules over Black-Box LLM)
* **Context**: LLMs prompted end-to-end to decide "Should we escalate to a human?" frequently suffer from calibration drift, sycophancy, and unpredictable thresholding on high-liability queries.
* **Decision**: Implemented a two-layer hybrid escalation engine in `src/pipeline.py`:
  - *Layer 1 (Semantic Classification)*: LLM extracts intent and confidence score.
  - *Layer 2 (Deterministic Rules)*: Python-level rules enforce invariant safety policies:
    1. Mandatory escalation for `account_access` (hacked accounts, password resets) and `billing_subscription` (unauthorized charges).
    2. Escalation on severe negative sentiment / churn keywords (`sue`, `lawsuit`, `fraud`, `scam`, `cancel subscription`).
    3. Escalation when classification confidence drops below 0.70.
    4. Escalation when FAISS retrieval similarity drops below 0.40 (novel/unseen failure mode).
* **Rationale**: Guarantees zero unhandled security breaches and provides an explicit, audit-logged reason string for every routing decision.

---

### Decision 11: Two-Tier Human-in-the-Loop (HITL) Queue Topology
* **Context**: In production, routing must account for both high-risk exceptions and silent quality degradation in auto-handled cases.
* **Decision**: Designed two distinct queues logged to persistent JSONL (`eval/hitl_review_log.jsonl`):
  1. *Escalation Queue*: 100% of cases triggering safety rules or low confidence, queued for immediate human agent intervention.
  2. *Calibration Spot-Check Queue*: 5–10% Bernoulli random sample of all auto-handled responses, routed to QA leads to measure model drift and audit calibration over time.
* **Rationale**: Replicates enterprise best practices (e.g. Hiver's shared inbox workflows) where QA oversight runs asynchronously without blocking customer resolution.

---

### Decision 12: Grounded Few-Shot Prompting with Direct Spotify Link Injection
* **Context**: Hallucinated URLs or outdated troubleshooting instructions are the #1 failure mode in automated brand support.
* **Decision**: In Stage 2 of the pipeline, injected top-3 retrieved historical SpotifyCares reply threads directly into the LLM system prompt alongside strict negative constraints ("Do NOT invent URLs not in examples; include official Spotify DM link `https://t.co/ldFdZRiNAt` for private issues").
* **Rationale**: Grounding replies in verified Spotify agent tweets preserves signature brand markers (e.g. "/GS", "/CB" agent initials, friendly "Hey there!" greeting) while preventing bogus URL generation.

---

### Decision 13: 4-Dimensional LLM-as-a-Judge with Inter-Rater Reliability Audit
* **Context**: Automated lexical metrics (BLEU/ROUGE) correlate poorly with human perception of customer service quality, while subjective LLM judging can be self-referential without validation.
* **Decision**: Implemented a standardized 4-criterion evaluation rubric (Helpfulness, Groundedness, Brand Tone, Correctness/Routing on a 1–5 scale) in `src/judge.py`, evaluated against independent human auditor labels on 40 stratified cases.
* **Key Metric**: Calculated quadratic weighted **Cohen's Kappa** to mathematically establish inter-rater agreement before accepting judge scores.

---

### Decision 14: Dual-Mode Resilient Parsing Fallback
* **Context**: Production LLM APIs occasionally fail tool-call validation due to special characters, JSON syntax breaks, or model provider glitches.
* **Decision**: Built regex-based text extraction fallbacks in both `classify_intent` and `draft_grounded_reply`.
* **Rationale**: If structured JSON decoding encounters an anomaly, the pipeline gracefully recovers the intended classification and reply text without crashing or dropping customer conversations.

