#!/usr/bin/env python3
import argparse
import duckdb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def parse_args():
    p = argparse.ArgumentParser(description="Vector search + show chunk text & title")
    p.add_argument("--query",   required=True, help="Search query text")
    p.add_argument("--top_k",   type=int, default=5, help="Number of top results")
    p.add_argument("--db_path", default="data/processed/duckdb/articles.duckdb",
                   help="Path to DuckDB database file")
    p.add_argument("--model",   default="all-MiniLM-L6-v2",
                   help="SentenceTransformer model for query embedding")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Embed the query
    model = SentenceTransformer(args.model)
    qvec = model.encode(args.query, normalize_embeddings=True)

    # 2) Load embeddings + metadata (chunk text & title) from DuckDB
    con = duckdb.connect(args.db_path)
    df = con.execute("""
        SELECT
          ce.chunk_id,
          ac.article_id,
          ac.chunk_text    AS chunk_text,
          ar.title         AS title,
          ce.embedding
        FROM chunk_embeddings AS ce
        JOIN article_chunks  AS ac USING(chunk_id)
        JOIN articles        AS ar ON ac.article_id = ar.id
    """).df()
    con.close()

    # 3) Build embeddings matrix
    emb_mat = np.vstack(df["embedding"].tolist())  # shape (N, D)

    # 4) Compute cosine scores via dot-product (they’re normalized)
    scores = emb_mat.dot(qvec)                     # shape (N,)

    # 5) Grab top_k indices
    top_idx = np.argsort(-scores)[: args.top_k]

    # 6) Print results with context
    print(f"\nTop {args.top_k} results for query: '{args.query}'\n")
    for i in top_idx:
        print(f"Score:   {scores[i]:.3f}")
        print(f"Article: {df.at[i, 'article_id']} — {df.at[i, 'title']}")
        print(f"Chunk:   {df.at[i, 'chunk_id']}")
        print(f"Text:    {df.at[i, 'chunk_text']}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
