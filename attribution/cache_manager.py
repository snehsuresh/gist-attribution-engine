import os
import json
import hashlib

# Directory for caching LLM prompts & embeddings
CACHE_DIR = 'data/processed/embeddings/ablation_cache'


def _cache_path(prompt: str) -> str:
    # Derive a SHA256 hash filename
    key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


def load_from_cache(prompt: str):
    path = _cache_path(prompt)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_to_cache(prompt: str, data: dict):
    path = _cache_path(prompt)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)