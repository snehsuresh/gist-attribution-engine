#!/usr/bin/env python3
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from explanation_utils import build_explanation_prompt, call_llm_for_explanation
from utils.helpers import sanitize_filename

def clean_snippet(text):
    """Normalize whitespace in text snippet."""
    return ' '.join(text.strip().split())


def explain_chunk(chunk, query, full_response):
    """Build and call LLM prompt for a single chunk, returning its explanation."""
    prompt = build_explanation_prompt(query, full_response, [chunk])
    raw = call_llm_for_explanation(prompt)
    try:
        explanation = json.loads(raw)
    except json.JSONDecodeError:
        try:
            explanation = json.loads(f"[{raw.strip()}]")
        except Exception:
            explanation = []
    return chunk['chunk_id'], explanation


def main():
    parser = argparse.ArgumentParser(
        description="Generate parallelized explanations for attributed chunks."
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Path to attribution JSON file"
    )
    parser.add_argument(
        "--metadata_path", required=True,
        help="Path to metadata.pkl file"
    )
    parser.add_argument(
        "--output_dir", default="data/processed/output/explanation_results",
        help="Directory to save explanation output"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Minimum influence threshold to include a chunk"
    )
    args = parser.parse_args()

    start_time = time()
    # Load attribution results
    with open(args.input_path, 'r') as f:
        attribution = json.load(f)

    # Load metadata and build chunk->text map
    metadata = pd.read_pickle(args.metadata_path)
    chunk_text_map = metadata.set_index("chunk_id")["chunk_text"].to_dict()

    # Filter chunks by threshold and clean snippet
    explained_chunks = [
        {
            'chunk_id': cid,
            'influence': float(score),
            'text_snippet': clean_snippet(chunk_text_map.get(cid, ""))
        }
        for cid, score in attribution['influence_by_chunk'].items()
        if score >= args.threshold
    ]

    print(f"📌 Explaining {len(explained_chunks)} chunks in parallel...")

    query = attribution['query']
    full_response = attribution['full_response']

    # Parallel explanation calls
    explanations = {}
    max_workers = min(len(explained_chunks), os.cpu_count() or len(explained_chunks))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(explain_chunk, chunk, query, full_response): chunk
            for chunk in explained_chunks
        }
        for future in as_completed(future_to_chunk):
            cid, explanation = future.result()
            explanations[cid] = explanation

    # Convert explanations dict to list to match frontend expectations
    explanations_list = [
        {"chunk_id": cid, "explanation": explanations[cid]}
        for cid in explanations
    ]

    # Save structured output
    os.makedirs(args.output_dir, exist_ok=True)
    out_fname = sanitize_filename(query) + "_explanation.json"
    out_path = os.path.join(args.output_dir, out_fname)
    output_data = {
        'query': query,
        'explained_chunks': explained_chunks,
        'explanations': explanations_list
    }
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    elapsed = time() - start_time
    print(f"✅ Saved explanations to {out_path} (took {elapsed:.2f}s)")

if __name__ == "__main__":
    main()
