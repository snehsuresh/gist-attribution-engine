# === embed_chunks.py ===
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


# === vector_search.py ===
#!/usr/bin/env python3
"""
vector_search.py: Perform vector-based cosine similarity search over chunk embeddings in DuckDB.
"""
import argparse
import duckdb
from sentence_transformers import SentenceTransformer

# Parse CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Vector search in DuckDB.")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results")
    parser.add_argument(
        "--db_path", default="data/processed/duckdb/articles.duckdb",
        help="Path to DuckDB database file"
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for query embedding"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and encode the query
    model = SentenceTransformer(args.model)
    qvec = model.encode(args.query, normalize_embeddings=True).tolist()

    # Connect to DuckDB
    db = duckdb.connect(args.db_path)

    # Perform cosine similarity search
    sql = f"""
        SELECT
          ce.chunk_id,
          ac.article_id,
          1 - cosine_distance(ce.embedding, {qvec}) AS score
        FROM chunk_embeddings AS ce
        JOIN article_chunks AS ac USING(chunk_id)
        ORDER BY score DESC
        LIMIT {args.top_k};
    """
    results = db.execute(sql).fetchall()
    db.close()

    # Display results
    print(f"\nTop {args.top_k} results for query: '{args.query}'\n")
    print(f"{'Score':>6}  {'Article ID':<30}  {'Chunk ID':<40}")
    print('-' * 80)
    for chunk_id, article_id, score in results:
        print(f"{score:6.3f}  {article_id:<30}  {chunk_id:<40}")

if __name__ == '__main__':
    main()
