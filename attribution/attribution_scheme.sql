CREATE TABLE IF NOT EXISTS attribution_scores (
  query TEXT,
  chunk_id TEXT,
  article_id TEXT,
  influence_score DOUBLE,
  normalized_score DOUBLE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);