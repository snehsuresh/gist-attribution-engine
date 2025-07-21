#!/usr/bin/env python3
import duckdb
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EntityCleaner")

# Paths
CHUNKS_PATH = Path("data/processed/article_chunks.parquet")
ENTITIES_PATH = Path("data/processed/doc_entities.parquet")
CLEANED_PATH = Path("data/processed/doc_entities_cleaned.parquet")
DB_PATH = Path("data/processed/duckdb/articles.duckdb")

def main():
    logger.info("🔗 Connecting to DuckDB...")
    conn = duckdb.connect(str(DB_PATH))

    # Load tables if not already created
    logger.info("📥 Loading article_chunks and doc_entities tables...")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS article_chunks AS
        SELECT * FROM parquet_scan('{CHUNKS_PATH}/*.parquet')
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS doc_entities AS
        SELECT * FROM parquet_scan('{ENTITIES_PATH}/*.parquet')
    """)

    # Clean entities and save to new parquet
    logger.info("🧹 Cleaning entities and writing to cleaned Parquet file...")
    conn.execute(f"""
        COPY (
            SELECT *
            FROM doc_entities
            WHERE 
                LENGTH(entity) > 2
                AND NOT entity GLOB '##*'
                AND NOT entity ~ '^[0-9\\s]+$'
                AND NOT entity ~ '^[A-Z]$'
                AND NOT entity ~ '^[A-Z]{{2}}$'
        ) TO '{CLEANED_PATH}' (FORMAT 'parquet');
    """)
    logger.info(f"✅ Cleaned entities written to: {CLEANED_PATH}")

    # Check if chunk_index starts at 0
    invalid_start = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT article_id, MIN(chunk_index) AS min_idx
            FROM article_chunks
            GROUP BY article_id
            HAVING min_idx > 0
        )
    """).fetchone()[0]
    logger.info(f"🔍 Articles with non-zero starting chunk_index: {invalid_start}")
    
    # Check if chunk_index is contiguous
    non_contiguous = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT article_id
            FROM article_chunks
            GROUP BY article_id
            HAVING COUNT(*) != MAX(chunk_index) + 1
        )
    """).fetchone()[0]
    logger.info(f"🔍 Articles with non-contiguous chunk_index values: {non_contiguous}")


    # Validation: Count checks
    logger.info("🧪 Running validation checks...")

    chunk_count = conn.execute("SELECT COUNT(*) FROM article_chunks").fetchone()[0]
    entity_count = conn.execute("SELECT COUNT(*) FROM doc_entities").fetchone()[0]
    cleaned_count = conn.execute(f"SELECT COUNT(*) FROM parquet_scan('{CLEANED_PATH}')").fetchone()[0]

    logger.info(f"📊 article_chunks: {chunk_count}")
    logger.info(f"📊 doc_entities: {entity_count}")
    logger.info(f"📊 doc_entities_cleaned: {cleaned_count}")

    # Check joinability
    missing_chunks = conn.execute("""
        SELECT COUNT(*) 
        FROM doc_entities 
        WHERE chunk_id NOT IN (SELECT chunk_id FROM article_chunks)
    """).fetchone()[0]

    logger.info(f"🔍 Entities with unmatched chunk_id: {missing_chunks}")

    conn.close()
    logger.info("🏁 Validation complete.")

if __name__ == "__main__":
    main()
