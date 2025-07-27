#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
from explanation_utils import build_explanation_prompt, call_llm_for_explanation


def main():
    parser = argparse.ArgumentParser(
        description="Generate filtered, structured explanations for attributed chunks."
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Path to Phase 3 attribution JSON file"
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

    # Load attribution results
    with open(args.input_path, 'r') as f:
        attribution = json.load(f)

    # Load metadata to map chunk_id -> chunk_text
    metadata = pd.read_pickle(args.metadata_path)
    chunk_text_map = {row.chunk_id: row.chunk_text for _, row in metadata.iterrows()}

    # Filter chunks by influence threshold and build list
    explained_chunks = []
    for cid, score in attribution['influence_by_chunk'].items():
        if score >= args.threshold:
            explained_chunks.append({
                'chunk_id': cid,
                'influence': float(score),
                'text_snippet': chunk_text_map.get(cid, '').strip().replace('\n', ' ')
            })

    # Build explanation prompt and call LLM
    prompt = build_explanation_prompt(
        attribution['query'],
        attribution['full_response'],
        explained_chunks
    )
    raw = call_llm_for_explanation(prompt)
    # parse it into a Python list
    try:
        explanation_list = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: wrap in array if needed
        explanation_list = json.loads(f"[{raw.strip()}]")

    # Save structured output
    os.makedirs(args.output_dir, exist_ok=True)
    out_fname = os.path.basename(args.input_path).replace('.json', '_explanation.json')
    out_path = os.path.join(args.output_dir, out_fname)
    with open(out_path, 'w') as f:
        json.dump({
            'query': attribution['query'],
            'explained_chunks': explained_chunks,
            'explanations': explanation_list
        }, f, indent=2)
    print(f"✅ Explanation saved to {out_path}")


if __name__ == '__main__':
    main()
