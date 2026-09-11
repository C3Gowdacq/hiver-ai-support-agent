"""
FAISS Vector Retriever for SpotifyCares grounded reply drafting.

Builds a dense FAISS index from the SpotifyCares subsample using TF-IDF + TruncatedSVD
embeddings (fully local, zero API cost). Each document in the index is a customer tweet
paired with SpotifyCares' actual historical reply, enabling retrieval-grounded generation.

Persists the FAISS index and fitted embeddings to disk so re-runs don't re-embed.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Suppress langchain-community deprecation warning
import warnings
warnings.filterwarnings("ignore", message=".*langchain-community.*")

FAISS_INDEX_DIR = "cache/faiss_index"
EMBEDDINGS_PATH = "cache/embeddings_model.pkl"


class DenseTfidfEmbeddings(Embeddings):
    """
    Local dense embeddings using TF-IDF + TruncatedSVD (LSA).
    
    No API calls needed — runs entirely offline using scikit-learn.
    Produces normalized 128-dimensional dense vectors suitable for FAISS.
    """
    
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.svd = None
        self._fitted = False

    def fit(self, texts: list[str]):
        """Fit TF-IDF vectorizer and SVD on the corpus."""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        n_comp = min(self.n_components, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf_matrix)
        self._fitted = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents into dense vectors."""
        tfidf_matrix = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf_matrix).astype(np.float32)
        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]

    def save(self, path: str):
        """Persist the fitted embeddings model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'svd': self.svd}, f)
        self._fitted = True

    def load(self, path: str):
        """Load a previously fitted embeddings model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.svd = data['svd']
        self._fitted = True


def build_faiss_index(
    subsample_csv: str = "data/spotify_pairs_subsample.csv",
    golden_csv: str = "data/golden_set.csv",
    force_rebuild: bool = False
) -> tuple[FAISS, DenseTfidfEmbeddings]:
    """
    Build or load a persisted FAISS index from the SpotifyCares subsample.
    
    Each document contains:
      - page_content: the customer tweet text (used for similarity matching)
      - metadata: customer_tweet_id, reply_text (the actual SpotifyCares reply)
    
    Golden set tweet IDs are excluded from the index to prevent data leakage.
    
    Returns:
        (faiss_vectorstore, embeddings_model)
    """
    embeddings = DenseTfidfEmbeddings(n_components=128)
    
    # Try loading persisted index
    if not force_rebuild and os.path.exists(FAISS_INDEX_DIR) and os.path.exists(EMBEDDINGS_PATH):
        print("Loading persisted FAISS index from disk...")
        embeddings.load(EMBEDDINGS_PATH)
        vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
        )
        print(f"FAISS index loaded. Contains {vectorstore.index.ntotal} vectors.")
        return vectorstore, embeddings

    # Build from scratch
    print("Building FAISS index from subsample...")
    df = pd.read_csv(subsample_csv)
    
    # Exclude golden set to prevent data leakage
    if os.path.exists(golden_csv):
        df_golden = pd.read_csv(golden_csv)
        golden_ids = set(df_golden['tweet_id'].astype(str))
        df = df[~df['customer_tweet_id'].astype(str).isin(golden_ids)].copy()
        print(f"Excluded {len(golden_ids)} golden set IDs. Indexing {len(df)} pairs.")

    # Clean text
    df['customer_text'] = df['customer_text'].fillna('').str.strip()
    df['reply_text'] = df['reply_text'].fillna('').str.strip()
    df = df[df['customer_text'].str.len() > 5].reset_index(drop=True)

    # Fit embeddings on the corpus
    corpus_texts = df['customer_text'].tolist()
    print(f"Fitting TF-IDF + SVD embeddings on {len(corpus_texts)} documents...")
    embeddings.fit(corpus_texts)

    # Create LangChain Documents with metadata
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row['customer_text'],
            metadata={
                'customer_tweet_id': str(row['customer_tweet_id']),
                'reply_text': str(row['reply_text']),
                'customer_author_id': str(row.get('customer_author_id', '')),
            }
        )
        documents.append(doc)

    print(f"Building FAISS index with {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Persist to disk
    os.makedirs(os.path.dirname(FAISS_INDEX_DIR), exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)
    embeddings.save(EMBEDDINGS_PATH)
    print(f"FAISS index saved to '{FAISS_INDEX_DIR}'. Embeddings saved to '{EMBEDDINGS_PATH}'.")

    return vectorstore, embeddings


def retrieve_similar_cases(
    vectorstore: FAISS,
    query: str,
    k: int = 3
) -> list[dict]:
    """
    Retrieve top-k most similar past SpotifyCares cases for a given customer query.
    
    Returns list of dicts with keys:
      - customer_text: the past customer tweet
      - reply_text: SpotifyCares' actual reply
      - similarity_score: cosine similarity (higher = more similar)
    """
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    
    cases = []
    for doc, score in results_with_scores:
        # FAISS returns L2 distance by default; convert to similarity
        # For normalized vectors, similarity = 1 - (distance^2 / 2)
        similarity = max(0.0, 1.0 - (score / 2.0))
        cases.append({
            'customer_text': doc.page_content,
            'reply_text': doc.metadata.get('reply_text', ''),
            'customer_tweet_id': doc.metadata.get('customer_tweet_id', ''),
            'similarity_score': float(similarity)
        })
    
    return cases


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    # Build or load the index
    vs, emb = build_faiss_index(force_rebuild=True)
    
    # Test retrieval
    test_queries = [
        "I can't play songs offline on my phone",
        "You charged me after I cancelled my subscription",
        "Where is Taylor Swift's new album?",
    ]
    
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        cases = retrieve_similar_cases(vs, q, k=2)
        for i, c in enumerate(cases):
            print(f"  [{i+1}] Sim={c['similarity_score']:.3f}")
            print(f"      Customer: {c['customer_text'][:100]}")
            print(f"      Reply:    {c['reply_text'][:100]}")
