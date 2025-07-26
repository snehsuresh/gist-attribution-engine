# dump_embeddings.py
import duckdb
import numpy as np
import pandas as pd

con = duckdb.connect("data/processed/duckdb/articles.duckdb")

df = con.execute("""
    SELECT
        ce.chunk_id,
        ac.article_id,
        ac.chunk_text,
        ar.title,
        ce.embedding
    FROM chunk_embeddings AS ce
    JOIN article_chunks ac USING(chunk_id)
    JOIN articles ar ON ac.article_id = ar.id
""").df()

# Save embedding matrix
emb_mat = np.vstack(df["embedding"].tolist()).astype("float32")
np.save("data/processed/embeddings.npy", emb_mat)

# Save metadata
metadata = df.drop(columns=["embedding"])
metadata.to_pickle("data/processed/metadata.pkl")

print("✅ Saved embeddings.npy and metadata.pkl")
