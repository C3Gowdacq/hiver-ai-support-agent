"""Smoke test for the full pipeline on 3 sample tweets."""
import sys, os
sys.path.insert(0, 'src')
os.chdir('c:/Chetan/Hiver')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pipeline import process_tweet, get_llm
from retriever import build_faiss_index

llm = get_llm(temperature=0.0)
vs, _ = build_faiss_index()

test_tweets = [
    ('T1', "I cant play my songs offline even though I have premium"),
    ('T2', "Someone hacked my Spotify account please help"),
    ('T3', "Why is Taylor Swift reputation not on Spotify yet??"),
]

for tid, text in test_tweets:
    r = process_tweet(text, tid, vectorstore=vs, llm=llm, log_hitl=False)
    print("=" * 60)
    print("Tweet ID:", tid)
    print("Text:", text)
    print("Intent:", r["intent"], "| Confidence:", r["confidence"])
    print("Escalate:", r["escalate"], "| Reason:", r["escalation_reason"])
    print("Drafted Reply:", r["drafted_reply"][:150])
    print()
