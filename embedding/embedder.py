# === embedder.py ===
#!/usr/bin/env python3
"""
embed_chunks.py: Efficiently embed paragraph chunks from DuckDB with batching,
progress tracking, and no primary key constraints.
"""
import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DB_PATH = "data/processed/duckdb/articles.duckdb"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 5000            # Number of chunks to process per batch
ENCODE_BATCH_SIZE = 64       # Batch size passed to the embedding model

# Initialize embedding model with progress bar
model = SentenceTransformer(MODEL_NAME)

# Connect to DuckDB
db = duckdb.connect(DB_PATH)

# Drop existing embeddings table to ensure a clean slate
db.execute("DROP TABLE IF EXISTS chunk_embeddings;")
# Create table without primary key to avoid unique constraint issues
db.execute(
    "CREATE TABLE chunk_embeddings (chunk_id TEXT, embedding FLOAT[]);")

# Fetch distinct chunk_id and chunk_text to avoid duplicates
rows = db.execute(
    "SELECT DISTINCT chunk_id, chunk_text FROM article_chunks;"
).fetchall()

total = len(rows)
print(f"🔍 Found {total} unique chunks to embed.")

# Batch embedding with progress tracking
for start in tqdm(range(0, total, BATCH_SIZE), desc="Embedding batches"):
    end = min(start + BATCH_SIZE, total)
    batch = rows[start:end]
    ids, texts = zip(*batch)

    # Encode the batch of texts
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Build a DataFrame for bulk insertion
    df = pd.DataFrame({
        "chunk_id": ids,
        "embedding": list(embeddings)
    })

    # Register and insert into DuckDB in one operation
    db.register("to_insert", df)
    db.execute("INSERT INTO chunk_embeddings SELECT * FROM to_insert;")
    db.unregister("to_insert")

# Close the database connection
db.close()
print("✅ Completed embedding all chunks.")