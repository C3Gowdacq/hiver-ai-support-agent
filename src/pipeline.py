"""
SpotifyCares AI Support Pipeline — 3-stage LangChain LCEL implementation.

Stages:
  1. Intent Classification: Few-shot LLM prompt with Pydantic structured output.
  2. Grounded Reply Drafting: FAISS top-3 retrieval → transparent few-shot LCEL chain.
  3. Escalation Decision: Deterministic Python rules + inspectable stated reason.

Human-in-the-Loop:
  - Escalation Queue: All escalated messages logged for human review.
  - Spot-Check Queue: ~5-10% of auto-handled messages randomly sampled for calibration audit.
  - Correction Logging: Persistent JSONL log of human verdicts and corrections.

Tech Stack: LangChain LCEL (RunnableSequence), langchain-groq, FAISS, Pydantic.
No LangGraph — pipeline is a fixed linear sequence, not a branching agent.
"""

import os
import sys
import json
import random
import hashlib
import time
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add src to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache import get_cached_response, save_cached_response
from retriever import build_faiss_index, retrieve_similar_cases

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*")
warnings.filterwarnings("ignore", message=".*deprecat.*")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic Models for Structured Output
# ---------------------------------------------------------------------------

class IntentResult(BaseModel):
    """Structured output for intent classification."""
    intent: str = Field(description="One of: playback_streaming, account_access, billing_subscription, content_availability, feature_request_howto, venting_feedback, other")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="One-sentence reasoning for this classification")


class ReplyDraft(BaseModel):
    """Structured output for grounded reply drafting."""
    reply: str = Field(description="The drafted customer support reply")
    grounding_note: str = Field(description="Brief note on which past case(s) informed this reply")


# ---------------------------------------------------------------------------
# LLM Initialization
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.1, max_tokens: int = 300):
    """Initialize the LLM based on .env configuration. Supports Groq and Gemini."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY not set in .env file. Please add your Groq API key.")
        return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env file.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_output_tokens=max_tokens, google_api_key=api_key)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'groq' or 'gemini'.")


def invoke_with_retry(runnable, inputs, max_retries: int = 4, base_delay: float = 5.0):
    """Invoke a LangChain runnable with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return runnable.invoke(inputs)
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or "tokens per" in err_str:
                wait_time = base_delay * (attempt + 1)
                print(f"    [Rate limit] Waiting {wait_time:.1f}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    return runnable.invoke(inputs)



# ---------------------------------------------------------------------------
# Stage 1: Intent Classification
# ---------------------------------------------------------------------------

INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SpotifyCares customer support intent classifier.

Classify the customer tweet into exactly ONE of these 7 intents:

1. playback_streaming — Technical issues: can't play, offline failure, audio glitches, device connect, shuffle/skip bugs, crashes
2. account_access — Login issues, password resets, hacked accounts, Facebook auth, email/username problems, country settings
3. billing_subscription — Charges, refunds, cancellations, Premium activation, payment methods, student/family plans, pricing
4. content_availability — Missing songs/albums, greyed-out tracks, artist catalog, regional licensing, new release availability
5. feature_request_howto — How-to questions, feature suggestions, playlist management, settings, UI navigation, storage
6. venting_feedback — Complaints about algorithms (Discover Weekly, Daily Mix), UI frustration, emotional venting without a specific technical ask
7. other — Ambiguous, incomplete, mid-conversation fragments, non-English without clear intent, general greetings

Here are representative examples:

CUSTOMER: "@SpotifyCares I can't listen to songs offline despite being a premium user"
INTENT: playback_streaming | CONFIDENCE: 0.95

CUSTOMER: "@SpotifyCares been hacked need some help"
INTENT: account_access | CONFIDENCE: 0.90

CUSTOMER: "@SpotifyCares I cancelled my subscription but you charged me again"
INTENT: billing_subscription | CONFIDENCE: 0.95

CUSTOMER: "Why isn't Taylor Swift's new album on Spotify?"
INTENT: content_availability | CONFIDENCE: 0.90

CUSTOMER: "How do I add songs to a collaborative playlist?"
INTENT: feature_request_howto | CONFIDENCE: 0.85

CUSTOMER: "Discover Weekly is trash this week, worst recommendations ever"
INTENT: venting_feedback | CONFIDENCE: 0.85

CUSTOMER: "@SpotifyCares thanks!"
INTENT: other | CONFIDENCE: 0.70

Respond with the intent, confidence (0.0-1.0), and a one-sentence reasoning."""),
    ("human", "Classify this customer tweet:\n\n\"{tweet_text}\"")
])


def classify_intent(tweet_text: str, llm=None) -> dict:
    """
    Stage 1: Classify a customer tweet's intent using few-shot LLM prompting.
    Uses disk cache to avoid redundant API calls.
    """
    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    # Check cache first
    cached = get_cached_response(tweet_text, model=model_name, extra="intent_v1")
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            pass  # Cache corrupted, regenerate

    if llm is None:
        llm = get_llm(temperature=0.0)

    # Use structured output for clean parsing (try native tool call, then json_mode)
    try:
        try:
            structured_llm = llm.with_structured_output(IntentResult)
            chain = INTENT_CLASSIFICATION_PROMPT | structured_llm
            result = invoke_with_retry(chain, {"tweet_text": tweet_text})
        except Exception:
            structured_llm = llm.with_structured_output(IntentResult, method="json_mode")
            chain = INTENT_CLASSIFICATION_PROMPT | structured_llm
            result = invoke_with_retry(chain, {"tweet_text": tweet_text})
        output = {
            "intent": result.intent,
            "confidence": result.confidence,
            "reasoning": result.reasoning
        }
    except Exception as e:
        # Fallback: parse from raw text if structured output fails
        chain = INTENT_CLASSIFICATION_PROMPT | llm | StrOutputParser()
        raw = invoke_with_retry(chain, {"tweet_text": tweet_text})
        output = _parse_intent_fallback(raw)

    # Cache the result
    save_cached_response(tweet_text, json.dumps(output), model=model_name, extra="intent_v1")
    return output


def _parse_intent_fallback(raw_text: str) -> dict:
    """Parse intent from raw LLM text output as fallback."""
    valid_intents = [
        'playback_streaming', 'account_access', 'billing_subscription',
        'content_availability', 'feature_request_howto', 'venting_feedback', 'other'
    ]
    text_lower = raw_text.lower()
    detected_intent = 'other'
    for intent in valid_intents:
        if intent in text_lower:
            detected_intent = intent
            break
    return {
        "intent": detected_intent,
        "confidence": 0.5,
        "reasoning": f"Fallback parse from raw output: {raw_text[:100]}"
    }


# ---------------------------------------------------------------------------
# Stage 2: Grounded Reply Drafting
# ---------------------------------------------------------------------------

REPLY_DRAFTING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SpotifyCares customer support agent. Draft a helpful, friendly reply to the customer tweet below.

IMPORTANT RULES:
1. Your reply MUST be grounded in the real past SpotifyCares replies shown below — follow their tone, structure, and resolution patterns.
2. Do NOT invent solutions, URLs, or promises not seen in the examples.
3. Keep the reply concise (1-3 sentences), warm, and on-brand (use /XX agent initials at the end).
4. If the issue requires private info (account, billing), ask the customer to DM you.
5. Include the SpotifyCares DM link (https://t.co/ldFdZRiNAt) when asking for DMs.

Here are the most similar past resolved cases for grounding context:

{grounding_context}

Now draft a reply for the new customer tweet."""),
    ("human", """Customer tweet: "{tweet_text}"

Classified intent: {intent}

Draft a grounded reply following the patterns from the past cases above.""")
])


def draft_grounded_reply(
    tweet_text: str,
    intent: str,
    retrieved_cases: list[dict],
    llm=None
) -> dict:
    """
    Stage 2: Draft a reply grounded in real historical SpotifyCares responses.
    Uses FAISS-retrieved past cases as few-shot context.
    """
    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    # Format grounding context from retrieved cases
    grounding_lines = []
    for i, case in enumerate(retrieved_cases):
        grounding_lines.append(
            f"--- Past Case {i+1} (similarity: {case['similarity_score']:.2f}) ---\n"
            f"Customer: {case['customer_text']}\n"
            f"SpotifyCares Reply: {case['reply_text']}"
        )
    grounding_context = "\n\n".join(grounding_lines)

    # Cache key includes the grounding context hash for determinism
    cache_extra = f"reply_v1|{hashlib.sha256(grounding_context.encode()).hexdigest()[:16]}"
    cached = get_cached_response(tweet_text, model=model_name, extra=cache_extra)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            pass

    if llm is None:
        llm = get_llm(temperature=0.3)  # Slightly higher temp for natural replies

    try:
        try:
            structured_llm = llm.with_structured_output(ReplyDraft)
            chain = REPLY_DRAFTING_PROMPT | structured_llm
            result = invoke_with_retry(chain, {
                "tweet_text": tweet_text,
                "intent": intent,
                "grounding_context": grounding_context
            })
        except Exception:
            structured_llm = llm.with_structured_output(ReplyDraft, method="json_mode")
            chain = REPLY_DRAFTING_PROMPT | structured_llm
            result = invoke_with_retry(chain, {
                "tweet_text": tweet_text,
                "intent": intent,
                "grounding_context": grounding_context
            })
        output = {
            "reply": result.reply,
            "grounding_note": result.grounding_note,
            "grounding_cases_used": len(retrieved_cases),
            "top_similarity": retrieved_cases[0]['similarity_score'] if retrieved_cases else 0.0
        }
    except Exception as e:
        # Fallback: plain text generation
        chain = REPLY_DRAFTING_PROMPT | llm | StrOutputParser()
        raw_reply = invoke_with_retry(chain, {
            "tweet_text": tweet_text,
            "intent": intent,
            "grounding_context": grounding_context
        })
        output = {
            "reply": raw_reply.strip(),
            "grounding_note": "Generated via fallback (unstructured output)",
            "grounding_cases_used": len(retrieved_cases),
            "top_similarity": retrieved_cases[0]['similarity_score'] if retrieved_cases else 0.0
        }

    save_cached_response(tweet_text, json.dumps(output), model=model_name, extra=cache_extra)
    return output


# ---------------------------------------------------------------------------
# Stage 3: Escalation Decision (Deterministic Rules)
# ---------------------------------------------------------------------------

# Keywords that trigger mandatory escalation (security/financial sensitivity)
ESCALATION_KEYWORDS = {
    'hacked', 'stolen', 'unauthorized', 'fraud', 'identity',
    'refund', 'charged', 'billing', 'payment', 'bank',
    'cancel subscription', 'legal', 'lawsuit', 'police',
    'password', 'security', 'breach', 'compromised'
}

# Strong negative sentiment indicators
ANGER_KEYWORDS = {
    'fucking', 'fuck', 'shit', 'bullshit', 'scam',
    'sue', 'lawyer', 'report you', 'worst company',
    'never again', 'disgusting', 'criminal'
}

# Similarity threshold below which we don't trust auto-handling
SIMILARITY_THRESHOLD = 0.35


def decide_escalation(
    tweet_text: str,
    intent: str,
    confidence: float,
    retrieved_cases: list[dict]
) -> dict:
    """
    Stage 3: Deterministic, inspectable escalation decision.
    
    Rules (in priority order):
    1. account_access or billing_subscription intents → ALWAYS escalate (PII/financial)
    2. Security/financial keywords detected → escalate
    3. Strong anger/abuse keywords → escalate
    4. Low intent confidence (< 0.6) → escalate (ambiguous)
    5. Low retrieval similarity (< threshold) → escalate (no precedent)
    6. Otherwise → auto-handle
    
    Returns dict with: escalate (bool), reason (str), rule_triggered (str)
    """
    text_lower = tweet_text.lower()
    top_sim = retrieved_cases[0]['similarity_score'] if retrieved_cases else 0.0
    reasons = []

    # Rule 1: Sensitive intent categories always escalate
    if intent in ('account_access', 'billing_subscription'):
        return {
            "escalate": True,
            "reason": f"Intent '{intent}' requires private account/billing verification via DM.",
            "rule_triggered": "sensitive_intent"
        }

    # Rule 2: Security/financial keywords
    triggered_keywords = [kw for kw in ESCALATION_KEYWORDS if kw in text_lower]
    if triggered_keywords:
        return {
            "escalate": True,
            "reason": f"Security/financial keywords detected: {', '.join(triggered_keywords[:3])}.",
            "rule_triggered": "security_keywords"
        }

    # Rule 3: Strong negative sentiment / abuse
    anger_hits = [kw for kw in ANGER_KEYWORDS if kw in text_lower]
    if anger_hits:
        return {
            "escalate": True,
            "reason": f"Strong negative sentiment detected: {', '.join(anger_hits[:2])}. Requires human tone moderation.",
            "rule_triggered": "anger_detected"
        }

    # Rule 4: Low intent confidence (ambiguous message)
    if confidence < 0.6:
        return {
            "escalate": True,
            "reason": f"Low classification confidence ({confidence:.2f}). Message may be ambiguous or multi-intent.",
            "rule_triggered": "low_confidence"
        }

    # Rule 5: Low retrieval similarity (no historical precedent)
    if top_sim < SIMILARITY_THRESHOLD:
        return {
            "escalate": True,
            "reason": f"No similar past case found (top similarity: {top_sim:.2f}). No grounded precedent for auto-reply.",
            "rule_triggered": "low_similarity"
        }

    # Rule 6: Default auto-handle
    return {
        "escalate": False,
        "reason": f"Intent '{intent}' (confidence: {confidence:.2f}) with strong retrieval match (sim: {top_sim:.2f}). Safe to auto-handle.",
        "rule_triggered": "auto_handle"
    }


# ---------------------------------------------------------------------------
# Human-in-the-Loop Logging
# ---------------------------------------------------------------------------

HITL_LOG_PATH = "data/hitl_corrections.jsonl"
SPOT_CHECK_RATE = 0.08  # 8% of auto-handled messages sampled for spot-check


def log_hitl_entry(entry: dict):
    """Append a human-in-the-loop review entry to the corrections log."""
    entry['logged_at'] = datetime.now().isoformat()
    with open(HITL_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def should_spot_check() -> bool:
    """Randomly determine if an auto-handled message should be spot-checked."""
    return random.random() < SPOT_CHECK_RATE


# ---------------------------------------------------------------------------
# Full Pipeline Orchestrator
# ---------------------------------------------------------------------------

def process_tweet(
    tweet_text: str,
    tweet_id: str = "",
    vectorstore=None,
    llm=None,
    log_hitl: bool = True
) -> dict:
    """
    Process a single customer tweet through the full 3-stage pipeline.
    
    Returns a dict with all intermediate outputs for full traceability:
      - intent, confidence, reasoning
      - retrieved_cases (top-3 past resolved threads)
      - drafted_reply, grounding_note
      - escalate, escalation_reason, rule_triggered
      - hitl_queue (escalation_queue | spot_check_queue | none)
    """
    # Stage 1: Intent Classification
    intent_result = classify_intent(tweet_text, llm=llm)
    
    # Stage 2: Retrieve similar past cases + draft grounded reply
    if vectorstore is None:
        vectorstore, _ = build_faiss_index()
    
    retrieved_cases = retrieve_similar_cases(vectorstore, tweet_text, k=3)
    reply_result = draft_grounded_reply(
        tweet_text, intent_result['intent'], retrieved_cases, llm=llm
    )
    
    # Stage 3: Escalation Decision
    escalation = decide_escalation(
        tweet_text,
        intent_result['intent'],
        intent_result['confidence'],
        retrieved_cases
    )
    
    # Determine HITL queue assignment
    hitl_queue = "none"
    if escalation['escalate']:
        hitl_queue = "escalation_queue"
    elif should_spot_check():
        hitl_queue = "spot_check_queue"

    # Assemble full pipeline output
    output = {
        "tweet_id": tweet_id,
        "tweet_text": tweet_text,
        # Stage 1
        "intent": intent_result['intent'],
        "confidence": intent_result['confidence'],
        "intent_reasoning": intent_result['reasoning'],
        # Stage 2
        "retrieved_cases": retrieved_cases,
        "drafted_reply": reply_result['reply'],
        "grounding_note": reply_result.get('grounding_note', ''),
        "top_similarity": reply_result.get('top_similarity', 0.0),
        # Stage 3
        "escalate": escalation['escalate'],
        "escalation_reason": escalation['reason'],
        "rule_triggered": escalation['rule_triggered'],
        # HITL
        "hitl_queue": hitl_queue,
        # Placeholders for human review
        "human_verdict": "",  # accept / edit / reject
        "human_correction": ""
    }

    # Log to HITL if applicable
    if log_hitl and hitl_queue != "none":
        log_hitl_entry({
            "tweet_id": tweet_id,
            "text": tweet_text,
            "intent": intent_result['intent'],
            "ai_draft_reply": reply_result['reply'],
            "escalation_reason": escalation['reason'],
            "hitl_queue": hitl_queue,
            "human_verdict": "",
            "human_correction": ""
        })

    return output


def run_pipeline_on_golden_set(
    golden_csv: str = "data/golden_set.csv",
    output_csv: str = "eval/pipeline_predictions.csv",
    output_json: str = "eval/pipeline_results_detail.json"
):
    """
    Run the full pipeline on the golden evaluation set and save all predictions.
    """
    print("="*60)
    print("  Running Full LLM Pipeline on Golden Evaluation Set")
    print("="*60)

    df = pd.read_csv(golden_csv)
    print(f"Loaded {len(df)} golden set examples.")

    # Initialize shared resources
    llm = get_llm(temperature=0.0)
    vectorstore, _ = build_faiss_index()

    results = []
    for idx, row in df.iterrows():
        tweet_text = str(row['text'])
        tweet_id = str(row['tweet_id'])

        if idx % 20 == 0:
            print(f"  Processing {idx+1}/{len(df)}...")

        output = process_tweet(
            tweet_text=tweet_text,
            tweet_id=tweet_id,
            vectorstore=vectorstore,
            llm=llm,
            log_hitl=True
        )
        results.append(output)

    # Save predictions CSV (for metric computation)
    pred_df = pd.DataFrame([{
        'tweet_id': r['tweet_id'],
        'text': r['tweet_text'],
        'pred_intent': r['intent'],
        'pred_confidence': r['confidence'],
        'intent_reasoning': r['intent_reasoning'],
        'drafted_reply': r['drafted_reply'],
        'grounding_note': r['grounding_note'],
        'top_similarity': r['top_similarity'],
        'pred_escalate': 1 if r['escalate'] else 0,
        'escalation_reason': r['escalation_reason'],
        'rule_triggered': r['rule_triggered'],
        'hitl_queue': r['hitl_queue']
    } for r in results])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pred_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nPipeline predictions saved to '{output_csv}'.")

    # Save detailed JSON (includes retrieved cases for inspection)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to '{output_json}'.")

    # Print summary
    n_escalated = sum(1 for r in results if r['escalate'])
    n_spot_check = sum(1 for r in results if r['hitl_queue'] == 'spot_check_queue')
    print(f"\nSummary: {n_escalated} escalated, {n_spot_check} spot-checked, "
          f"{len(results) - n_escalated} auto-handled")

    return results


if __name__ == "__main__":
    run_pipeline_on_golden_set()
