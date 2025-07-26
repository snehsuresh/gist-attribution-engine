#!/usr/bin/env python3
"""
fast_vector_search.py
conda install -c pytorch faiss-gpu


A high‑performance vector search using:
  • Sentence-Transformer for query embedding (GPU or CPU)
  • FAISS for sub‑millisecond inner‑product search
  • Preloaded float32 embeddings + in‑memory metadata

Requirements:
  pip install faiss-cpu sentence-transformers pandas numpy
  # or 'faiss-gpu' if you want FAISS on GPU support

Before running:
  1) Dump your chunk embeddings to embeddings.npy (shape: N×D, dtype=float32)
  2) Dump your metadata to metadata.pkl (a pandas DataFrame with columns
     ['chunk_id','article_id','chunk_text','title'] indexed 0…N-1)


python fast_vector_search.py   --query "Your query"   --embeddings_path data/processed/embeddings/embeddings.npy   --metadata_path data/processed/embeddings/metadata.pkl   --top_k 5

"""

import argparse
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

def parse_args():
    p = argparse.ArgumentParser(
        description="Ultra‑fast FAISS vector search over pre‑loaded embeddings"
    )
    p.add_argument(
        "--query", required=True, help="Search query text"
    )
    p.add_argument(
        "--embeddings_path",
        required=True,
        help="Path to .npy file containing float32 embeddings (N×D)",
    )
    p.add_argument(
        "--metadata_path",
        required=True,
        help="Path to .pkl file containing pandas.DataFrame of metadata",
    )
    p.add_argument(
        "--model_name",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model checkpoint",
    )
    p.add_argument(
        "--top_k", type=int, default=5, help="Number of top results to return"
    )
    p.add_argument(
        "--use_gpu",
        action="store_true",
        help="If set, move FAISS index to GPU (requires faiss-gpu).",
    )
    return p.parse_args()

def build_faiss_index(emb_mat, use_gpu: bool):
    d = emb_mat.shape[1]
    index = faiss.IndexFlatIP(d)          # exact inner‑product index
    if use_gpu:
        # initialize one GPU resource and move index there
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(emb_mat)
    return index

def main():
    args = parse_args()

    # 1) Load embeddings + metadata once
    print("Loading embeddings from", args.embeddings_path)
    emb_mat = np.load(args.embeddings_path)
    if emb_mat.dtype != np.float32:
        emb_mat = emb_mat.astype("float32")

    print("Loading metadata from", args.metadata_path)
    metadata = pd.read_pickle(args.metadata_path)

    # 2) Build FAISS index
    print(f"Building FAISS index (use_gpu={args.use_gpu})…")
    index = build_faiss_index(emb_mat, args.use_gpu)

    # 3) Load SentenceTransformer
    device = "cuda" if args.use_gpu else "cpu"
    print(f"Loading SentenceTransformer('{args.model_name}') on {device}…")
    model = SentenceTransformer(args.model_name, device=device)

    # 4) Embed query
    print("Encoding query…")
    qvec = model.encode(
        args.query, normalize_embeddings=True
    ).astype("float32")

    # 5) Search
    print(f"Searching top {args.top_k}…")
    D, I = index.search(qvec.reshape(1, -1), args.top_k)
    scores, idxs = D[0], I[0]

    # 6) Print results
    print(f"\nTop {args.top_k} results for: '{args.query}'\n" + "-"*60)
    for rank, (score, idx) in enumerate(zip(scores, idxs), 1):
        row = metadata.iloc[idx]
        print(f"{rank:2d}. Score: {score:.4f}")
        print(f"    Article: {row.article_id} — {row.title}")
        print(f"    Chunk:   {row.chunk_id}")
        print(f"    Text:    {row.chunk_text.strip()[:200]}{'…' if len(row.chunk_text)>200 else ''}\n")

if __name__ == "__main__":
    main()
