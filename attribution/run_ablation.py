#!/usr/bin/env python3
import argparse
import os
import json
import faiss
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Tuple

# project imports
from attribution.ablation_utils import (
    embed_text,
    build_full_prompt,
    build_ablated_prompt,
    rollup_to_documents,
    compute_combined_score,
    get_top_k_chunks
)
from attribution.cache_manager import load_from_cache, save_to_cache
from openai import OpenAI
from utils.helpers import sanitize_filename
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm_with_cache(prompt: str) -> Tuple[str, np.ndarray]:
    """LLM call + caching of both response text and its embedding."""
    cached = load_from_cache(prompt)
    if cached:
        return (
            cached["response"],
            np.array(cached.get("embedding", []), dtype="float32")
        )
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0
    )
    text = res.choices[0].message.content.strip()
    emb = embed_text(text)
    save_to_cache(prompt, {"response": text, "embedding": emb.tolist()})
    return text, emb


def stream_first_response(
    query: str,
    embeddings_path: str,
    metadata_path: str,
    top_k: int = 5
):
    """Stream only the first full LLM response token-by-token."""
    # Fetch top-K chunks for the full prompt
    chunks = get_top_k_chunks(
        query,
        embeddings_path,
        metadata_path,
        top_k=top_k,
        use_gpu=False,
        filter_articles=None
    )
    prompt = build_full_prompt(query, chunks)

    # Enable OpenAI streaming
    response_stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0,
        stream=True
    )

    # Yield tokens as they arrive
    for chunk in response_stream:
        token = chunk.choices[0].delta.content
        if token := chunk.choices[0].delta.content:
            yield token


def stream_ablation(
    query: str,
    embeddings_path: str,
    metadata_path: str,
    top_k: int,
    scoring_mode: str,
    threshold: float
) -> Generator[str, None, None]:
    # 1) fetch top-K
    chunks = get_top_k_chunks(
        query,
        embeddings_path,
        metadata_path,
        top_k=top_k,
        use_gpu=False,
        filter_articles=None
    )

    # 2) full response + embedding
    full_prompt = build_full_prompt(query, chunks)
    r_full, e_full = call_llm_with_cache(full_prompt)

    # stream full response immediately
    yield json.dumps({
        "type": "full_response",
        "data": {"full_response": r_full}
    }) + "\n"

    # let frontend know title/source for each chunk
    chunks_info = [
        {
            "chunk_index": i,
            "chunk_id": c["chunk_id"],
            "title": c.get("title", c["chunk_id"]),
            "source": c.get("source", "unknown")
        }
        for i, c in enumerate(chunks)
    ]
    yield json.dumps({
        "type": "chunk_info",
        "data": {"chunks_info": chunks_info}
    }) + "\n"

    # 3) parallel ablation + explanation per chunk
    influence = {}
    explanations = {}
    chunk_to_article = {c["chunk_id"]: c["article_id"] for c in chunks}

    def process_chunk(idx: int, chunk: dict):
        ablated_prompt = build_ablated_prompt(query, chunks, idx)
        _, e_abl = call_llm_with_cache(ablated_prompt)

        drift_score = 1.0 - np.dot(e_full, e_abl) if e_abl.size else 0.0
        final_score = compute_combined_score(
            drift=drift_score,
            faiss_score=chunk["faiss_score"],
            chunk_text=chunk["chunk_text"],
            full_response=r_full,
            mode=scoring_mode
        )

        expl_prompt = (
            f"Explain how the following chunk influenced the answer:\n\n"
            f"'{chunk['chunk_text']}'\n\n"
            f"Full answer:\n'{r_full}'\n\n"
            "Keep it concise."
        )
        explanation, _ = call_llm_with_cache(expl_prompt)
        snippet = chunk["chunk_text"][:200]
        return idx, chunk["chunk_id"], float(final_score), explanation, snippet

    with ThreadPoolExecutor(max_workers=top_k) as executor:
        futures = {
            executor.submit(process_chunk, i, c): i
            for i, c in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx, cid, score, explanation, snippet = future.result()
            if score < threshold:
                score = 0.0
                explanation = ""
            influence[cid] = score
            explanations[cid] = explanation

            yield json.dumps({
                "type": "chunk",
                "data": {
                    "index": idx,
                    "chunk_id": cid,
                    "influence": score,
                    "text_snippet": snippet,
                    "explanation": explanation
                }
            }) + "\n"

    # 4) roll up and normalize
    total_score = sum(influence.values()) or 1.0
    influence_by_chunk = {cid: sc / total_score for cid, sc in influence.items()}
    influence_by_doc = rollup_to_documents(influence_by_chunk, chunk_to_article)

    # 5) Stream final metrics
    yield json.dumps({
        "type": "final_metrics",
        "data": {
            "influence_by_chunk": influence_by_chunk,
            "influence_by_doc": influence_by_doc
        }
    }) + "\n"

    # 6) Save output to disk
    out_dir = "data/processed/output/attribution_results"
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{sanitize_filename(query)}.json")
    with open(fname, "w") as f:
        json.dump({
            "query": query,
            "full_response": r_full,
            "influence_by_chunk": influence_by_chunk,
            "influence_by_doc": influence_by_doc,
            "explanations": explanations,
            "top_chunks": [c["chunk_id"] for c in chunks]
        }, f, indent=2)

    print(f"✅ Attribution JSON saved to: {fname}", flush=True)

    # 7) Signal to frontend that we’re done
    yield json.dumps({"type": "done"}) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run ablation attribution with optional streaming.")
    parser.add_argument("--query", required=True, help="User query text")
    parser.add_argument("--embeddings_path", required=True, help=".npy of chunk embeddings")
    parser.add_argument("--metadata_path", required=True, help=".pkl of chunk metadata")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks")
    parser.add_argument("--scoring_mode", choices=["drift", "drift_faiss", "drift_overlap"], default="drift")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true", help="Stream JSONL output")
    args = parser.parse_args()

    if args.stream:
        for line in stream_ablation(
            args.query,
            args.embeddings_path,
            args.metadata_path,
            args.top_k,
            args.scoring_mode,
            args.threshold
        ):
            print(line, end="", flush=True)
    else:
        print("⚠️ Non-stream mode is deprecated in this script.")
        print("Please use the --stream flag.")
        return

if __name__ == "__main__":
    main()