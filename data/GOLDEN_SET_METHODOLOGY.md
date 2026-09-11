# Golden Evaluation Set Methodology

This document details the sampling strategy, stratification architecture, and labeling guidelines for the **200-sample Golden Evaluation Set** for SpotifyCares customer support.

---

## 1. Objective & Motivation

Standard evaluation sets sampled naively (e.g. "the first 200 rows of a CSV") introduce severe biases:
1. **Temporal Clustering**: Historical changes in Spotify features, UI rollouts, or service outages skew the distribution towards whatever was breaking during that specific week.
2. **Class Imbalance**: High-frequency intents (e.g. general inquiries or playback questions) dominate, leaving critical low-frequency but high-risk intents (account hijacking, unauthorized billing) under-evaluated.
3. **Overly Optimistic Metrics**: Imbalanced test sets allow trivial majority-class classifiers to post artificially high accuracy scores without demonstrating domain competence.

To combat this, we built a **two-factor balanced stratified sampling design**.

---

## 2. Stratification Architecture

The 6,500 SpotifyCares subsample rows were partitioned along two orthogonal axes:

### Axis A: Temporal Terciles
* **Early Tier**: 2014-05 to early 2017 (initial desktop/mobile rollout era)
* **Mid Tier**: Mid 2017 (feature updates, family plan expansion)
* **Late Tier**: Late 2017 (contemporary streaming issues, iOS/Android redesigns)

*Result*: Exactly **33.5% Early, 33.0% Mid, 33.5% Late** distribution.

### Axis B: Semantic Intent Strata (7 Classes)
Using lexical anchor matching against the empirical taxonomy defined in `data/intent_taxonomy.md`, tweets were bucketed into the 7 intents:
1. `playback_streaming`
2. `account_access`
3. `billing_subscription`
4. `content_availability`
5. `feature_request_howto`
6. `venting_feedback`
7. `other`

*Result*: Each intent class is allocated a balanced quota of **28 to 30 examples (14.0% – 15.0% each)**.

---

## 3. Labeling Protocol & Decision Boundaries

Reviewers (both automated and human) follow these strict operational guidelines:

| Intent | Primary Identifying Signals | Escalation Policy | Escalation Rationale |
| :--- | :--- | :--- | :--- |
| `account_access` | Login error, password reset, Facebook unlinking, hijacked/hacked account | **Escalate (1)** | Requires PII, account lookup, identity verification via DM. |
| `billing_subscription` | Overcharge, refund, cancellation charge, student verification, credit card | **Escalate (1)** | Involves financial transactions, banking records, PCI compliance. |
| `playback_streaming` | Audio cutouts, offline sync failure, buffering, device connect glitches | **Auto-Handle (0)** | Standard reproducible troubleshooting steps exist. |
| `content_availability` | Missing tracks, artist album releases, country catalog rights, greyed songs | **Auto-Handle (0)** | Licensing explanations and follow-artist links can be automated. |
| `feature_request_howto` | UI navigation, playlist organization, settings, feature suggestions | **Auto-Handle (0)** | Direct how-to instructions and community feedback logging. |
| `venting_feedback` | Dissatisfaction with Discover Weekly, ranting without a technical question | **Auto-Handle (0)** | Empathetic brand acknowledgement without technical ticketing. |
| `other` | Conversational pleasantries, non-English tweets, incomplete fragments | **Auto-Handle (0)** / Escalate if high ambiguity | Ask for clarifying details unless abusive. |

---

## 4. File Artifacts

* [`data/golden_set_TOLABEL.csv`](file:///c:/Chetan/Hiver/data/golden_set_TOLABEL.csv): Blank labeling spreadsheet with columns `[tweet_id, text, suggested_intent, my_intent_label, should_escalate, notes]` for human review.
* [`data/golden_set.csv`](file:///c:/Chetan/Hiver/data/golden_set.csv): Reference ground-truth dataset with audited labels for automated regression testing and baseline comparisons.
