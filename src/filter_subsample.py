"""
Filter and subsample SpotifyCares customer-brand reply pairs from Kaggle twcs.csv.

Dataset schema:
    tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

Steps:
    1. Pass 1: Stream twcs.csv to collect all brand replies where author_id == 'SpotifyCares'
       and in_response_to_tweet_id is present.
    2. Pass 2: Stream twcs.csv to find matching inbound customer tweets.
    3. Pair customer tweet with SpotifyCares reply.
    4. Clean and subsample to target range (~5,000–8,000 pairs, default 6,500).
    5. Save to data/spotify_pairs_subsample.csv and print summary statistics.
"""

import os
import sys
import pandas as pd
import numpy as np

def filter_and_subsample(
    input_csv="data/twcs.csv",
    output_csv="data/spotify_pairs_subsample.csv",
    target_sample_size=6500,
    random_seed=42,
    chunksize=100000
):
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
        sys.exit(1)

    print(f"=== Step 1: Scanning '{input_csv}' for SpotifyCares replies ===")
    spotify_replies = []
    # Pass 1: find all SpotifyCares outbound tweets that respond to another tweet
    chunk_idx = 0
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False, dtype=str):
        chunk_idx += 1
        # Brand replies from SpotifyCares
        mask = (chunk['author_id'].str.strip() == 'SpotifyCares') & chunk['in_response_to_tweet_id'].notna()
        matching = chunk[mask][['tweet_id', 'author_id', 'created_at', 'text', 'in_response_to_tweet_id']].copy()
        if len(matching) > 0:
            spotify_replies.append(matching)

    if not spotify_replies:
        print("Error: No SpotifyCares replies found in dataset.")
        sys.exit(1)

    df_replies = pd.concat(spotify_replies, ignore_index=True)
    # in_response_to_tweet_id could be float string like '12345.0' or integer string
    df_replies['in_response_to_tweet_id'] = df_replies['in_response_to_tweet_id'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and x != '' else ''
    )
    # Drop duplicates if multiple replies to the same customer tweet (keep first)
    df_replies = df_replies.drop_duplicates(subset=['in_response_to_tweet_id'], keep='first')
    target_customer_ids = set(df_replies['in_response_to_tweet_id'])
    print(f"Found {len(df_replies):,} unique SpotifyCares replies linking to customer tweets.")

    print(f"=== Step 2: Scanning '{input_csv}' for matching customer inbound tweets ===")
    customer_tweets = []
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False, dtype=str):
        # Clean tweet_id
        chunk['tweet_id_clean'] = chunk['tweet_id'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and x != '' else ''
        )
        mask = chunk['tweet_id_clean'].isin(target_customer_ids)
        matching = chunk[mask][['tweet_id_clean', 'author_id', 'inbound', 'created_at', 'text']].copy()
        if len(matching) > 0:
            customer_tweets.append(matching)

    df_customers = pd.concat(customer_tweets, ignore_index=True)
    df_customers = df_customers.drop_duplicates(subset=['tweet_id_clean'], keep='first')
    print(f"Found {len(df_customers):,} matching customer tweets.")

    print("=== Step 3: Merging customer tweets and SpotifyCares replies ===")
    merged = pd.merge(
        df_customers,
        df_replies,
        left_on='tweet_id_clean',
        right_on='in_response_to_tweet_id',
        suffixes=('_customer', '_reply')
    )

    # Rename & clean columns
    pairs = pd.DataFrame({
        'customer_tweet_id': merged['tweet_id_clean'],
        'customer_author_id': merged['author_id_customer'],
        'customer_created_at': merged['created_at_customer'],
        'customer_text': merged['text_customer'].str.strip(),
        'reply_tweet_id': merged['tweet_id'],
        'reply_author_id': merged['author_id_reply'],
        'reply_created_at': merged['created_at_reply'],
        'reply_text': merged['text_reply'].str.strip()
    })

    # Drop empty or very short junk messages (e.g. just @handle)
    pairs = pairs[pairs['customer_text'].str.len() > 10]
    pairs = pairs[pairs['reply_text'].str.len() > 10]
    total_valid_pairs = len(pairs)
    print(f"Total valid customer <-> brand pairs extracted: {total_valid_pairs:,}")

    # Subsample if greater than target
    if total_valid_pairs > target_sample_size:
        print(f"Subsampling to target size: {target_sample_size:,} pairs (random_seed={random_seed})")
        # Sort chronologically before sampling or sample evenly across time
        pairs['customer_datetime'] = pd.to_datetime(pairs['customer_created_at'], errors='coerce')
        # Sort by datetime to ensure consistent ordering, then sample with fixed seed
        pairs = pairs.sort_values('customer_datetime').reset_index(drop=True)
        subsample = pairs.sample(n=target_sample_size, random_state=random_seed).copy()
        subsample = subsample.sort_values('customer_datetime').reset_index(drop=True)
        subsample = subsample.drop(columns=['customer_datetime'])
    else:
        print(f"Total pairs ({total_valid_pairs}) <= target ({target_sample_size}), keeping all.")
        subsample = pairs

    # Save to disk
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    subsample.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved {len(subsample):,} pairs to '{output_csv}'.")

    # Summary statistics
    print("\n" + "="*50)
    print("        DATASET SUBSAMPLE SUMMARY STATISTICS        ")
    print("="*50)
    print(f"Total Rows (Reply Pairs): {len(subsample):,}")
    print(f"Unique Customers: {subsample['customer_author_id'].nunique():,}")
    
    # Date range
    dates = pd.to_datetime(subsample['customer_created_at'], errors='coerce').dropna()
    if len(dates) > 0:
        print(f"Date Range: {dates.min().strftime('%Y-%m-%d %H:%M:%S')} to {dates.max().strftime('%Y-%m-%d %H:%M:%S')}")

    # Text length distributions
    subsample['cust_char_len'] = subsample['customer_text'].str.len()
    subsample['cust_word_len'] = subsample['customer_text'].str.split().apply(len)
    subsample['reply_char_len'] = subsample['reply_text'].str.len()
    subsample['reply_word_len'] = subsample['reply_text'].str.split().apply(len)

    print("\n--- Customer Tweet Text Length ---")
    print(f"  Character length: Mean = {subsample['cust_char_len'].mean():.1f}, "
          f"Median = {subsample['cust_char_len'].median():.0f}, "
          f"Min = {subsample['cust_char_len'].min()}, Max = {subsample['cust_char_len'].max()}")
    print(f"  Word count:       Mean = {subsample['cust_word_len'].mean():.1f}, "
          f"Median = {subsample['cust_word_len'].median():.0f}, "
          f"Min = {subsample['cust_word_len'].min()}, Max = {subsample['cust_word_len'].max()}")

    print("\n--- SpotifyCares Reply Text Length ---")
    print(f"  Character length: Mean = {subsample['reply_char_len'].mean():.1f}, "
          f"Median = {subsample['reply_char_len'].median():.0f}, "
          f"Min = {subsample['reply_char_len'].min()}, Max = {subsample['reply_char_len'].max()}")
    print(f"  Word count:       Mean = {subsample['reply_word_len'].mean():.1f}, "
          f"Median = {subsample['reply_word_len'].median():.0f}, "
          f"Min = {subsample['reply_word_len'].min()}, Max = {subsample['reply_word_len'].max()}")
    print("="*50 + "\n")

    # Sample preview of 3 pairs
    print("--- Sample Pairs Preview ---")
    for idx, row in subsample.head(3).iterrows():
        print(f"\n[Pair #{idx+1}] Tweet ID: {row['customer_tweet_id']}")
        print(f"  Customer: {row['customer_text']}")
        print(f"  SpotifyCares: {row['reply_text']}")
    print("\nFilter & subsampling completed successfully.")

if __name__ == "__main__":
    filter_and_subsample()
