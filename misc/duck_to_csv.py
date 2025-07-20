import duckdb

con = duckdb.connect("data/processed/duckdb/articles.duckdb")
con.execute(
    """
    COPY articles TO 'data/processed/articles_export.csv' (HEADER, DELIMITER ',');
"""
)
con.close()

print("✅ Exported to data/processed/articles_export.csv")
