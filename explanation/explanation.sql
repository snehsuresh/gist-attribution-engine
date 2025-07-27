"""
SQL schema to store generated explanations (document-level or chunk-level).
"""
CREATE TABLE IF NOT EXISTS explanations (
  query TEXT,
  chunk_id TEXT,
  explanation TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);