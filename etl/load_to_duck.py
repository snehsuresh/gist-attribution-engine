#!/usr/bin/env python3
import duckdb
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DuckDBLoader")

# Paths
CSV_PATH = Path("data/processed/articles_export.csv")
DUCKDB_PATH = Path("data/processed/duckdb/articles.duckdb")
TABLE_NAME = "articles"

def main():
    # Ensure output directory exists
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to DuckDB
    logger.info(f"🔗 Connecting to {DUCKDB_PATH} ...")
    conn = duckdb.connect(str(DUCKDB_PATH))

    # Drop if already exists to refresh
    logger.info(f"📦 Creating table '{TABLE_NAME}' from CSV...")
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME};")
    conn.execute(f"""
        CREATE TABLE {TABLE_NAME} AS
        SELECT * FROM read_csv_auto('{CSV_PATH}')
    """)

    # Show count and sample
    count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    logger.info(f"✅ Loaded {count} rows into '{TABLE_NAME}'.")

    logger.info("🔍 Preview of loaded data:")
    preview = conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 5;").fetchall()
    for row in preview:
        logger.info(row)

    conn.close()
    logger.info("🏁 Done.")

if __name__ == "__main__":
    main()
