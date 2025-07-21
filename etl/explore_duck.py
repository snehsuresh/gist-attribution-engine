#!/usr/bin/env python3
import duckdb
from pathlib import Path

# Set your DuckDB file path here
DB_PATH = Path("C:\\Users\\Govind\\Downloads\\articles.duckdb")

def explore_duckdb(db_path):
    conn = duckdb.connect(str(db_path))

    # 1. List all tables
    print("📋 Tables in database:")
    tables = conn.execute("SHOW TABLES").fetchall()
    for table in tables:
        print(f" - {table[0]}")

    # 2. Print schema of each table
    for table in tables:
        table_name = table[0]
        print(f"\n🧬 Schema of table: {table_name}")
        schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
        for col in schema:
            print(f"   - {col[0]} ({col[1]})")

    # 3. Preview top 5 rows from each table
    for table in tables:
        print(f"\n🔍 Top 5 rows from {table[0]}:")
        preview = conn.execute(f"SELECT * FROM {table[0]} LIMIT 5").fetchdf()
        print(preview)

    conn.close()

if __name__ == "__main__":
    explore_duckdb(DB_PATH)
