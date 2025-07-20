import pandas as pd
import duckdb
import ast
import numpy as np
from pathlib import Path

# Paths
csv_files = [
    "data/raw/news/mediastack_topics_news_final.csv",
    "data/raw/news/mediastack_topics_news.csv",
    "data/raw/news/news_api_final.csv",
    "data/raw/news/news1.csv",
    "data/raw/news/wikipedia_articles.csv",
]
duckdb_path = "data/processed/duckdb/articles.duckdb"
duckdb_table = "articles"

# Ensure output dir exists
Path(duckdb_path).parent.mkdir(parents=True, exist_ok=True)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize schema, normalize tags and dates, drop empty rows.
    """
    required_cols = [
        "id",
        "title",
        "source",
        "author",
        "published_at",
        "content",
        "section",
        "tags",
    ]
    # Keep only known columns
    df = df[[c for c in required_cols if c in df.columns]]

    # Add any missing columns
    for c in required_cols:
        if c not in df.columns:
            df[c] = None

    # Parser for the tags field
    def parse_tags(t):
        # Case 1: already list/array-like
        if isinstance(t, (list, tuple, set, np.ndarray, pd.Series)):
            return [str(x).strip().lower() for x in t]

        # Case 2: null or NaN
        if t is None or (isinstance(t, float) and pd.isna(t)):
            return []

        # Case 3: string (possibly a list or single tag)
        if isinstance(t, str):
            try:
                parsed = ast.literal_eval(t)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip().lower() for x in parsed]
            except Exception:
                # fallback: comma-separated
                return [seg.strip().lower() for seg in t.split(",") if seg.strip()]
            return [str(parsed).strip().lower()]

        # Case 4: fallback — wrap as list
        return [str(t).strip().lower()]

    # Apply tag parsing
    df["tags"] = df["tags"].apply(parse_tags)

    # Normalize dates
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    # Drop rows lacking title or content
    df = df.dropna(subset=["title", "content"])

    # Flatten tags back to comma-string for storage
    df["tags"] = df["tags"].apply(lambda lst: ", ".join(lst))

    return df


# 1) Load and clean each CSV
csv_dfs = []
for path in csv_files:
    try:
        raw = pd.read_csv(path)
    except Exception:
        raw = pd.read_csv(path, encoding="latin1")
    csv_dfs.append(clean_df(raw))

# 2) Connect to DuckDB and try to load existing table
con = duckdb.connect(duckdb_path)
try:
    db_raw = con.execute(f"SELECT * FROM {duckdb_table}").fetchdf()
    db_df = clean_df(db_raw)
except duckdb.CatalogException:
    print(f"⚠️ No existing table '{duckdb_table}' found — starting fresh")
    db_df = pd.DataFrame(
        columns=[
            "id",
            "title",
            "source",
            "author",
            "published_at",
            "content",
            "section",
            "tags",
        ]
    )

# 3) Combine all sources
full_df = pd.concat(csv_dfs + [db_df], ignore_index=True)

# 4) Deduplicate
full_df = full_df.drop_duplicates(subset=["id"])
full_df = full_df.drop_duplicates(subset=["title", "content"])

# 5) Overwrite the DuckDB table
#    Register the pandas DF so DuckDB can see it
con.register("full_df_view", full_df)
con.execute(f"CREATE OR REPLACE TABLE {duckdb_table} AS SELECT * FROM full_df_view")

con.close()
print(
    f"✅ Unified + cleaned articles written to: {duckdb_path} → table `{duckdb_table}`"
)
