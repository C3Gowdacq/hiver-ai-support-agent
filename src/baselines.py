"""
Non-LLM Reference Baselines for SpotifyCares Customer Support Pipeline.

Implements:
1. Trivial Baseline:
   - Intent: Majority-class predictor
   - Escalation: "Always Escalate" policy
   - Reply: Static generic brand macro

2. Simple Baseline:
   - Intent: TF-IDF + Logistic Regression
   - Escalation: Heuristic rule-based (keyword / sensitive intent detection)
   - Reply: Pure Nearest-Neighbor copy-paste of past brand reply (no LLM generation)

Evaluates on data/golden_set.csv and saves metrics to eval/baseline_results.json.
Guarantees NO data leakage: golden set examples are strictly excluded from the training/retrieval corpus.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_baselines(
    subsample_csv="data/spotify_pairs_subsample.csv",
    golden_set_csv="data/golden_set.csv",
    output_json="eval/baseline_results.json"
):
    print("=== Loading Data ===")
    df_subsample = pd.read_csv(subsample_csv)
    df_golden = pd.read_csv(golden_set_csv)

    print(f"Total subsample pairs: {len(df_subsample):,}")
    print(f"Total golden test pairs: {len(df_golden):,}")

    # Prevent Data Leakage: Golden set IDs must not be in training corpus
    golden_ids = set(df_golden['tweet_id'].astype(str))
    df_train = df_subsample[~df_subsample['customer_tweet_id'].astype(str).isin(golden_ids)].copy()
    print(f"Training corpus size (after excluding test set): {len(df_train):,} pairs")

    # Clean text
    df_train['customer_text'] = df_train['customer_text'].fillna('')
    df_golden['text'] = df_golden['text'].fillna('')

    y_test_intent = df_golden['my_intent_label'].astype(str)
    y_test_escalate = df_golden['should_escalate'].astype(int)

    # -------------------------------------------------------------
    # 1. TRIVIAL BASELINE
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("           EVALUATING TRIVIAL BASELINE             ")
    print("="*50)
    
    # Majority class from golden or heuristic subsample
    majority_intent = y_test_intent.value_counts().index[0]
    y_pred_trivial_intent = [majority_intent] * len(df_golden)
    y_pred_trivial_escalate = [1] * len(df_golden)  # Always escalate

    trivial_metrics = {
        "intent_accuracy": float(accuracy_score(y_test_intent, y_pred_trivial_intent)),
        "intent_macro_f1": float(f1_score(y_test_intent, y_pred_trivial_intent, average='macro', zero_division=0)),
        "intent_weighted_f1": float(f1_score(y_test_intent, y_pred_trivial_intent, average='weighted', zero_division=0)),
        "escalate_precision": float(precision_score(y_test_escalate, y_pred_trivial_escalate, zero_division=0)),
        "escalate_recall": float(recall_score(y_test_escalate, y_pred_trivial_escalate, zero_division=0)),
        "escalate_f1": float(f1_score(y_test_escalate, y_pred_trivial_escalate, zero_division=0))
    }

    print(f"Majority Class Intent: {majority_intent}")
    print(f"Intent Accuracy:    {trivial_metrics['intent_accuracy']:.4f}")
    print(f"Intent Macro-F1:    {trivial_metrics['intent_macro_f1']:.4f}")
    print(f"Escalate Precision: {trivial_metrics['escalate_precision']:.4f}")
    print(f"Escalate Recall:    {trivial_metrics['escalate_recall']:.4f}")
    print(f"Escalate F1:        {trivial_metrics['escalate_f1']:.4f}")

    # -------------------------------------------------------------
    # 2. SIMPLE BASELINE (TF-IDF + Logistic Regression & NN Copy-Paste)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("            EVALUATING SIMPLE BASELINE             ")
    print("="*50)

    # To train Logistic Regression on full train set, we need silver/heuristic labels for df_train
    from sample_golden_set import heuristic_classify_intent
    df_train['train_intent'] = df_train['customer_text'].apply(heuristic_classify_intent)

    # Fit TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(df_train['customer_text'])
    X_test_vec = vectorizer.transform(df_golden['text'])

    # Train Logistic Regression
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_train_vec, df_train['train_intent'])

    y_pred_simple_intent = clf.predict(X_test_vec)

    # Simple Escalation Rule: Escalate if classified intent is account_access or billing_subscription
    # or if high-risk keywords appear
    def simple_escalate_rule(text, pred_intent):
        t = str(text).lower()
        if pred_intent in ['account_access', 'billing_subscription']:
            return 1
        if any(k in t for k in ['hacked', 'stolen', 'unauthorized', 'cancel subscription', 'refund', 'charge back']):
            return 1
        return 0

    y_pred_simple_escalate = [
        simple_escalate_rule(text, intent)
        for text, intent in zip(df_golden['text'], y_pred_simple_intent)
    ]

    # Nearest Neighbor Copy-Paste Retrieval
    # Compute cosine similarity between each test query and all training tweets
    sim_matrix = cosine_similarity(X_test_vec, X_train_vec)
    top1_indices = np.argmax(sim_matrix, axis=1)

    retrieved_replies = df_train.iloc[top1_indices]['reply_text'].tolist()
    retrieved_intents = df_train.iloc[top1_indices]['train_intent'].tolist()
    retrieval_sim_scores = [sim_matrix[i, top1_indices[i]] for i in range(len(df_golden))]

    # Measure retrieval quality: did top-1 retrieved past case have the same intent as query?
    retrieval_intent_match = [
        1 if ret_int == true_int else 0
        for ret_int, true_int in zip(retrieved_intents, y_test_intent)
    ]
    retrieval_intent_accuracy = float(np.mean(retrieval_intent_match))

    simple_metrics = {
        "intent_accuracy": float(accuracy_score(y_test_intent, y_pred_simple_intent)),
        "intent_macro_f1": float(f1_score(y_test_intent, y_pred_simple_intent, average='macro', zero_division=0)),
        "intent_weighted_f1": float(f1_score(y_test_intent, y_pred_simple_intent, average='weighted', zero_division=0)),
        "escalate_precision": float(precision_score(y_test_escalate, y_pred_simple_escalate, zero_division=0)),
        "escalate_recall": float(recall_score(y_test_escalate, y_pred_simple_escalate, zero_division=0)),
        "escalate_f1": float(f1_score(y_test_escalate, y_pred_simple_escalate, zero_division=0)),
        "retrieval_intent_accuracy": retrieval_intent_accuracy,
        "mean_top1_cosine_sim": float(np.mean(retrieval_sim_scores))
    }

    print(f"Intent Accuracy:           {simple_metrics['intent_accuracy']:.4f}")
    print(f"Intent Macro-F1:           {simple_metrics['intent_macro_f1']:.4f}")
    print(f"Escalate Precision:        {simple_metrics['escalate_precision']:.4f}")
    print(f"Escalate Recall:           {simple_metrics['escalate_recall']:.4f}")
    print(f"Escalate F1:               {simple_metrics['escalate_f1']:.4f}")
    print(f"Retrieval Top-1 Match Acc: {simple_metrics['retrieval_intent_accuracy']:.4f}")
    print(f"Mean Top-1 Cosine Sim:     {simple_metrics['mean_top1_cosine_sim']:.4f}")

    print("\nClassification Report (Simple Baseline):")
    print(classification_report(y_test_intent, y_pred_simple_intent, zero_division=0))

    # Save baseline comparisons and predictions
    results = {
        "trivial_baseline": trivial_metrics,
        "simple_baseline": simple_metrics,
        "sample_predictions": []
    }

    for i in range(min(5, len(df_golden))):
        results["sample_predictions"].append({
            "tweet_id": str(df_golden.iloc[i]['tweet_id']),
            "customer_text": str(df_golden.iloc[i]['text']),
            "gold_intent": str(y_test_intent.iloc[i]),
            "gold_escalate": int(y_test_escalate.iloc[i]),
            "simple_pred_intent": str(y_pred_simple_intent[i]),
            "simple_pred_escalate": int(y_pred_simple_escalate[i]),
            "nn_copy_paste_reply": str(retrieved_replies[i]),
            "retrieval_cosine_sim": float(retrieval_sim_scores[i])
        })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Baseline benchmark results successfully saved to '{output_json}'.")

    # Save simple baseline predictions for judge comparison
    df_simple_eval = df_golden.copy()
    df_simple_eval['pred_intent'] = y_pred_simple_intent
    df_simple_eval['pred_escalate'] = y_pred_simple_escalate
    df_simple_eval['drafted_reply'] = retrieved_replies
    df_simple_eval['retrieval_sim'] = retrieval_sim_scores
    df_simple_eval.to_csv("eval/simple_baseline_predictions.csv", index=False, encoding="utf-8")
    print("Saved 'eval/simple_baseline_predictions.csv'.")

    return results

if __name__ == "__main__":
    run_baselines()
