import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the saved FAISS index from disk
index = faiss.read_index("faiss_index_cosine.idx")

# Load metadata dataframe from Parquet
meta = pd.read_parquet("vector_metadata.parquet")

# Load your embedding model (same model you used before)
model = SentenceTransformer("all-MiniLM-L6-v2")

def query_faiss_index(query_text, k=5):
    # Encode and normalize query vector
    query_vec = model.encode([query_text])
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    # Search FAISS index for top k similar vectors
    D, I = index.search(np.array(query_vec), k)

    print(f"Top {k} results for query: '{query_text}'")
    for rank, idx in enumerate(I[0], start=1):
        print(f"Rank {rank} | Similarity Score: {D[0][rank-1]:.4f}")
        print(meta.iloc[idx]["chunk_text"][:300])  # Show first 300 chars of matched text
        print("-" * 80)

# Example queries:
queries = [
    " Ukraine's president proposed reviving talks brokered by the Trump administration, which seemed stalled a month ago.",
]

for q in queries:
    query_faiss_index(q)
