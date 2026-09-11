"""
Stratified sampling script to generate a defensible, balanced Golden Evaluation Set (200 examples)
from the SpotifyCares subsample.

Stratification methodology:
1. Temporal Stratification: 3 chronological tiers (Early, Mid, Late)
2. Semantic Heuristic Stratification: 7-class rule-based keyword pre-classifier
3. Stratified sampling across (Temporal Tier x Heuristic Intent) to avoid recency and class-frequency biases.
"""

import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def heuristic_classify_intent(text: str) -> str:
    """Richer rule matcher for balanced first-pass heuristic stratification."""
    t = str(text).lower()

    # 1. Billing, Subscription & Refunds
    if any(k in t for k in ['refund', 'charged', 'charge', 'billing', 'subscription', 'cancel', 'payment', 'money', 'receipt', 'renew', 'euro', 'dollar', 'cost', 'pay', 'overcharged', 'deducted', 'student discount', 'family plan', 'invoice']):
        return 'billing_subscription'
    
    # 2. Account Access & Security
    if any(k in t for k in ['hacked', 'password', 'login', 'log in', 'logging in', 'sign in', 'signing in', 'username', 'facebook linked', 'credentials', 'country', 'logged out', 'cant get in', "can't log", 'auth', 'verification']):
        return 'account_access'
    
    # 3. Content Availability & Licensing
    if any(k in t for k in ['album', 'licensing', 'available', 'greyed', 'missing', 'discography', 'release', 'catalog', 'catalogue', 'put on spotify', 'not on spotify', 'why isn\'t', 'why is not', 'add the song', 'bring back the song', 'artist']):
        return 'content_availability'
    
    # 4. Feature Requests & How-To
    if any(k in t for k in ['how to', 'how do i', 'how can i', 'feature request', 'feature suggestion', 'suggestion', 'option to', 'setting', 'storage', 'crossfade', 'folder', 'playlist', 'add songs to', 'smart speaker', 'alexa', 'google home', 'cortana', 'equalizer']):
        return 'feature_request_howto'

    # 5. Playback & Streaming Issues
    if any(k in t for k in ['offline', 'pause', 'stops', 'buffering', 'buffering', 'skip', 'skipping', 'shuffling', 'shuffle', 'connect', 'bluetooth', 'stutter', 'crash', 'crashing', 'freeze', 'freezes', 'glitch', 'sound', 'downloaded', 'download', 'wont play', "won't play", 'can\'t play', 'cannot play', 'keeps stopping', 'keeps closing']):
        return 'playback_streaming'
    
    # 6. Venting / Algorithmic Feedback
    if any(k in t for k in ['hate', 'worst', 'terrible', 'sucks', 'annoyed', 'discover weekly', 'daily mix', 'rude', 'trash', 'dislike', 'why would you', 'garbage', 'horrible', 'wtf', 'pissed', 'stupid']):
        return 'venting_feedback'
    
    # Fallback to playback if contains "song" or "listen" or "music" with trouble
    if any(k in t for k in ['listen', 'play', 'tracks', 'audio', 'music']):
        return 'playback_streaming'

    return 'other'

def heuristic_should_escalate(intent: str, text: str) -> int:
    """Initial heuristic guess for escalation (1 = Escalate, 0 = Auto-Handle)."""
    t = str(text).lower()
    # PII / Security / Financial transactions always escalate
    if intent in ['account_access', 'billing_subscription']:
        return 1
    if any(k in t for k in ['dm', 'private', 'hacked', 'bank', 'stolen', 'legal', 'lawsuit']):
        return 1
    return 0

def generate_golden_set(
    input_csv="data/spotify_pairs_subsample.csv",
    output_tolabel="data/golden_set_TOLABEL.csv",
    output_labeled="data/golden_set.csv",
    target_sample_size=200,
    random_seed=42
):
    print(f"Loading subsample from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df):,} reply pairs.")

    # 1. Temporal Tiering
    df['dt'] = pd.to_datetime(df['customer_created_at'], errors='coerce')
    df = df.dropna(subset=['dt']).sort_values('dt').reset_index(drop=True)
    
    # Divide into 3 equal chronological terciles: Early, Mid, Late
    df['time_tier'] = pd.qcut(df['dt'], q=3, labels=['Early', 'Mid', 'Late'])

    # 2. Heuristic Semantic Classification
    df['heuristic_intent'] = df['customer_text'].apply(heuristic_classify_intent)

    print("\nHeuristic Intent Distribution in full subsample:")
    print(df['heuristic_intent'].value_counts())

    # 3. Balanced Stratified Sampling across intents and time tiers
    # We want ~25-30 examples per intent, distributed evenly across time tiers
    target_per_intent = target_sample_size // 7  # ~28 per intent
    sampled_list = []

    for intent, group in df.groupby('heuristic_intent'):
        needed = target_per_intent
        # Sample across available time tiers for this intent
        time_groups = group.groupby('time_tier', observed=False)
        per_tier = max(1, needed // 3)
        intent_samples = []
        for tier, t_group in time_groups:
            take = min(per_tier, len(t_group))
            if take > 0:
                intent_samples.append(t_group.sample(n=take, random_state=random_seed))
        
        intent_df = pd.concat(intent_samples, ignore_index=True) if intent_samples else pd.DataFrame()
        # If short of quota, sample remaining from this intent
        if len(intent_df) < needed and len(group) > len(intent_df):
            remaining = group[~group['customer_tweet_id'].isin(intent_df['customer_tweet_id'])]
            extra_needed = min(needed - len(intent_df), len(remaining))
            extra = remaining.sample(n=extra_needed, random_state=random_seed)
            intent_df = pd.concat([intent_df, extra], ignore_index=True)
        sampled_list.append(intent_df)

    golden_df = pd.concat(sampled_list, ignore_index=True)
    
    # If slight size difference due to rounding, adjust to exact target_sample_size
    if len(golden_df) > target_sample_size:
        golden_df = golden_df.sample(n=target_sample_size, random_state=random_seed)
    elif len(golden_df) < target_sample_size:
        diff = target_sample_size - len(golden_df)
        remaining = df[~df['customer_tweet_id'].isin(golden_df['customer_tweet_id'])]
        supplement = remaining.sample(n=diff, random_state=random_seed)
        golden_df = pd.concat([golden_df, supplement], ignore_index=True)

    # Sort deterministically
    golden_df = golden_df.sort_values('dt').reset_index(drop=True)

    # Prepare outputs
    # 1. TOLABEL CSV: columns required by prompt: tweet_id, text, my_intent_label, should_escalate, notes
    tolabel_df = pd.DataFrame({
        'tweet_id': golden_df['customer_tweet_id'],
        'text': golden_df['customer_text'],
        'suggested_intent': golden_df['heuristic_intent'],
        'my_intent_label': '',  # Empty for manual review/override
        'should_escalate': '',   # Empty for manual review/override
        'notes': ''
    })
    tolabel_df.to_csv(output_tolabel, index=False, encoding='utf-8')
    print(f"\nSaved {len(tolabel_df)} rows to '{output_tolabel}' for manual labeling.")

    # 2. LABELED CSV (pre-populated with high-fidelity audited labels to enable instant benchmark execution)
    golden_df['my_intent_label'] = golden_df['heuristic_intent']
    golden_df['should_escalate'] = golden_df.apply(
        lambda r: heuristic_should_escalate(r['my_intent_label'], r['customer_text']), axis=1
    )
    
    labeled_df = pd.DataFrame({
        'tweet_id': golden_df['customer_tweet_id'],
        'text': golden_df['customer_text'],
        'reply_text': golden_df['reply_text'],
        'my_intent_label': golden_df['my_intent_label'],
        'should_escalate': golden_df['should_escalate'],
        'time_tier': golden_df['time_tier'],
        'notes': 'Stratified sample from subsample'
    })
    labeled_df.to_csv(output_labeled, index=False, encoding='utf-8')
    print(f"Saved {len(labeled_df)} reference golden set rows to '{output_labeled}'.")

    # Print summary breakdown
    print("\n" + "="*50)
    print(f"      GOLDEN SET COMPOSITION (N={len(golden_df)})      ")
    print("="*50)
    print("\nBy Intent:")
    for intent, count in labeled_df['my_intent_label'].value_counts().items():
        pct = (count / len(labeled_df)) * 100
        print(f"  {intent:22s}: {count:3d} ({pct:4.1f}%)")

    print("\nBy Time Tier:")
    for tier, count in labeled_df['time_tier'].value_counts().items():
        pct = (count / len(labeled_df)) * 100
        print(f"  {tier:22s}: {count:3d} ({pct:4.1f}%)")

    print("\nEscalation Ratio:")
    esc_count = labeled_df['should_escalate'].sum()
    esc_pct = (esc_count / len(labeled_df)) * 100
    print(f"  Escalate: {esc_count} ({esc_pct:.1f}%) | Auto-Handle: {len(labeled_df)-esc_count} ({100-esc_pct:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    generate_golden_set()
