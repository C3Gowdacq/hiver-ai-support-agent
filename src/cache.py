"""
Disk-based LLM response cache keyed by SHA-256 hash of the prompt + model config.

Ensures:
- Re-runs are fast and free (no duplicate API calls).
- Reviewers can reproduce results without re-generating.
- Cache is human-inspectable (JSON files in cache/ directory).
"""

import os
import json
import hashlib
from typing import Optional

CACHE_DIR = "cache"

def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)

def _make_cache_key(prompt: str, model: str = "", extra: str = "") -> str:
    """Generate a deterministic SHA-256 hash key from prompt + model + extra context."""
    content = f"{model}||{extra}||{prompt}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def get_cached_response(prompt: str, model: str = "", extra: str = "") -> Optional[str]:
    """Look up a cached LLM response. Returns None if cache miss."""
    _ensure_cache_dir()
    key = _make_cache_key(prompt, model, extra)
    cache_path = os.path.join(CACHE_DIR, f"{key}.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("response")
            
    # Check alternate model caches for this prompt + extra context
    for alt_model in ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", ""]:
        if alt_model != model:
            alt_key = _make_cache_key(prompt, alt_model, extra)
            alt_path = os.path.join(CACHE_DIR, f"{alt_key}.json")
            if os.path.exists(alt_path):
                with open(alt_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("response")
    return None

def save_cached_response(prompt: str, response: str, model: str = "", extra: str = ""):
    """Save an LLM response to disk cache."""
    _ensure_cache_dir()
    key = _make_cache_key(prompt, model, extra)
    cache_path = os.path.join(CACHE_DIR, f"{key}.json")
    
    data = {
        "prompt_hash": key,
        "model": model,
        "extra": extra,
        "prompt_preview": prompt[:200],  # First 200 chars for human inspection
        "response": response
    }
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def clear_cache():
    """Remove all cached responses."""
    _ensure_cache_dir()
    count = 0
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith('.json'):
            os.remove(os.path.join(CACHE_DIR, fname))
            count += 1
    print(f"Cleared {count} cached responses.")
