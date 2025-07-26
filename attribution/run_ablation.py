#!/usr/bin/env python3
import argparse
import json
import os
import faiss
import numpy as np
import pandas as pd
from ablation_utils import (
    embed_text,
    build_full_prompt,
    build_ablated_prompt,
    cosine_drift,
    rollup_to_documents,
)
from cache_manager import load_from_cache, save_to_cache
from openai import OpenAI
client = OpenAI()

"""
python attribution/run_ablation.py   --query "Why is inflation rising?"   --embeddings_path data/processed/embeddings/embeddings.npy   --metadata_path data/processed/embeddings/metadata.pkl   --top_k 5
"""



def convert(o):
    if isinstance(o, np.float32) or isinstance(o, np.float64):
        return float(o)
    return o

def get_top_k_chunks(query, embeddings_path, metadata_path, top_k=5, use_gpu=False):
    """
    Retrieve the top-K most similar chunks for the given query.
    """
    emb_mat = np.load(embeddings_path)
    if emb_mat.dtype != np.float32:
        emb_mat = emb_mat.astype('float32')
    metadata = pd.read_pickle(metadata_path)

    # Build FAISS index
    d = emb_mat.shape[1]
    index = faiss.IndexFlatIP(d)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(emb_mat)

    # Embed query
    q_emb = embed_text(query).astype('float32')
    D, I = index.search(q_emb.reshape(1, -1), top_k)

    # Gather chunk info
    chunks = []
    for score, idx in zip(D[0], I[0]):
        row = metadata.iloc[idx]
        chunks.append({
            'chunk_id': row.chunk_id,
            'article_id': row.article_id,
            'chunk_text': row.chunk_text,
            'title': row.title,
            'score': float(score)
        })
    return chunks


def call_llm(prompt: str):
    """
    Call the LLM (OpenAI) with caching of prompts and embeddings.
    """
    cached = load_from_cache(prompt)
    if cached:
        return cached['response'], np.array(cached['embedding'])

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

    # Save to cache
    save_to_cache(prompt, {'response': text, 'embedding': emb.tolist()})
    return text, emb


def main():
    parser = argparse.ArgumentParser(description="Run chunk-level and document-level ablation attribution.")
    parser.add_argument("--query",           required=True, help="User query text")
    parser.add_argument("--embeddings_path", required=True, help=".npy file of chunk embeddings")
    parser.add_argument("--metadata_path",   required=True, help=".pkl file of chunk metadata")
    parser.add_argument("--top_k",           type=int, default=5, help="Number of chunks to retrieve and ablate")
    parser.add_argument("--use_gpu",         action='store_true', help="Use FAISS GPU index")
    args = parser.parse_args()

    # Step 1: Retrieve top-K chunks
    chunks = get_top_k_chunks(
        args.query,
        args.embeddings_path,
        args.metadata_path,
        top_k=args.top_k,
        use_gpu=args.use_gpu
    )

    # Step 2: Full LLM response and embedding
    full_prompt = build_full_prompt(args.query, chunks)
    r_full, e_full = call_llm(full_prompt)

    # Step 3: Ablation per chunk
    influence = {}
    for idx, chunk in enumerate(chunks):
        ablated_prompt = build_ablated_prompt(args.query, chunks, idx)
        _, e_abl = call_llm(ablated_prompt)
        drift = cosine_drift(e_full, e_abl)
        influence[chunk['chunk_id']] = drift

    # Step 4: Normalize chunk-level scores
    total = sum(influence.values()) or 1.0
    influence_by_chunk = {cid: score/total for cid, score in influence.items()}

    # Step 5: Roll up to document-level
    chunk_to_article = {c['chunk_id']: c['article_id'] for c in chunks}
    influence_by_doc = rollup_to_documents(influence_by_chunk, chunk_to_article)

    # Step 6: Save results
    out = {
        'query': args.query,
        'full_response': r_full,
        'influence_by_chunk': influence_by_chunk,
        'influence_by_doc': influence_by_doc
    }
    out_dir = 'data/processed/output/attribution_results'
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{args.query.replace(' ', '_')}.json")

    with open(fname, 'w') as f:
        json.dump(out, f, indent=2, default=convert)

    print(f"✅ Saved attribution results to {fname}")

if __name__ == '__main__':
    main()