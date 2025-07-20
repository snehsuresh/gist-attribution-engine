import os
import csv
import logging
from datetime import datetime
import duckdb
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Input and output paths
RAW_CSV_PATH = "data/raw/news/articles_enhanced.csv"
OUTPUT_DB_PATH = "data/processed/duckdb/articles.duckdb"
TABLE_NAME = "articles"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_DB_PATH), exist_ok=True)


def normalize_date(year: str, full_date: str):
    """Format to ISO date (YYYY-MM-DD) if possible, else fallback to year."""
    try:
        if full_date:
            return datetime.strptime(full_date, "%Y-%m-%d").date().isoformat()
        elif year and len(year) == 4:
            return f"{year}-01-01"
    except Exception:
        return year or "unknown"


def transform_row(row, index):
    """Map raw CSV row to structured schema."""
    title = row.get("Title", "").strip()
    source = row.get("Publisher", "unknown").strip()
    author = row.get("Authors") or "Unknown"
    year = row.get("Year", "").strip()
    full_date = row.get("Full_Date", "").strip()
    pub_date = normalize_date(year, full_date)

    return {
        "id": f"{source}{pub_date}{index}",
        "title": title,
        "source": source,
        "author": author,
        "published_at": pub_date,
        "content": row.get("Text_Sample") or "<no content>",
        "section": row.get("Publication_Type", "General"),
        "tags": ["inflation", "interest rates", "economy"],
    }


def read_and_transform_csv(path):
    """Read CSV and return transformed records."""
    transformed_data = []
    with open(path, newline="", encoding="ISO-8859-1") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            transformed_data.append(transform_row(row, i))
    return transformed_data


def save_to_duckdb(data, db_path, table_name):
    """Save list of dicts to DuckDB."""
    df = pd.DataFrame(data)
    con = duckdb.connect(db_path)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    con.close()


def main():
    logging.info("Starting ingestion pipeline...")
    data = read_and_transform_csv(RAW_CSV_PATH)
    logging.info(f"Read and transformed {len(data)} records.")
    save_to_duckdb(data, OUTPUT_DB_PATH, TABLE_NAME)
    logging.info(f"Data written to DuckDB at {OUTPUT_DB_PATH}, table: {TABLE_NAME}")


if __name__ == "__main__":
    main()
