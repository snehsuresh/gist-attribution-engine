#!/usr/bin/env python3
"""
ChunkNERPipeline: Named Entity Recognition on Article Paragraphs using PySpark + Hugging Face + DuckDB

Overview:
---------
This script performs scalable Named Entity Recognition (NER) over a dataset of news or long-form articles.
Each article is split into paragraph chunks, passed through a Hugging Face NER model, and the resulting
entities are saved for downstream use. It uses PySpark for parallelism, Hugging Face transformers for NER,
and DuckDB for local SQL-based querying and integration.

Key Steps:
----------
1. **Setup & Logging**: Suppresses noisy logs and initializes logging and Spark session.

2. **Data Load**:
   - Reads a CSV file (`articles_export.csv`) containing article `id` and `content` (full text).

3. **Paragraph Chunking**:
   - Splits each article into paragraph-level chunks using double newlines (`\n\n`).
   - Each paragraph becomes a chunk with a unique `chunk_id`.

4. **NER Inference (Batched)**:
   - Uses `dslim/bert-base-NER` model via Hugging Face's `pipeline` API.
   - Batched inference is done inside each Spark partition using `mapPartitions` for GPU-efficient throughput.
   - Extracts recognized entities along with type and model confidence score.

5. **Data Output**:
   - Writes `article_chunks` (chunked paragraphs) and `doc_entities` (NER output) to Parquet files.
   - Loads both datasets into a local DuckDB database (`articles.duckdb`) for analytics.

6. **Schema**:
   - `article_chunks`: chunk_id, article_id, chunk_index, chunk_text
   - `doc_entities`: article_id, chunk_id, entity, entity_type, relevance_score

Use Cases:
----------
- Downstream entity-level summarization, visualization, attribution
- Fine-tuned entity filtering and trend analysis across geopolitical or technical domains
- Lightweight integration with vector search, LLM summarizers, or dashboards

Dependencies:
-------------
- PySpark (for distributed processing)
- Transformers (Hugging Face NER pipeline)
- DuckDB (in-process SQL for fast local analytics)
- GPU-enabled `torch` if available (optional, but recommended for speed)

Note:
-----
- Designed to work locally or on a cluster.
- Batch NER improves GPU utilization and avoids Hugging Face's sequential pipeline warning.
"""

import os
import logging
import traceback

import torch
from transformers import pipeline

from pathlib import Path
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf, explode, concat_ws, col
from pyspark.sql.types import ArrayType, StringType, StructType, StructField, DoubleType
import pyspark.sql.functions as F

import duckdb

# Suppress noisy PySpark logs
os.environ["PYSPARK_LOG_LEVEL"] = "ERROR"
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ChunkNERPipeline")

def split_paragraphs(text):
    if text is None:
        return []
    return [p.strip() + "." for p in text.split(". ") if p.strip()]

split_udf = udf(split_paragraphs, ArrayType(StringType()))

entity_schema = StructType([
    StructField("article_id", StringType(), False),
    StructField("chunk_id", StringType(), False),
    StructField("entity", StringType(), False),
    StructField("entity_type", StringType(), False),
    StructField("relevance_score", DoubleType(), False),
])

def ner_partition(rows):
    try:
        device = 0 if torch.cuda.is_available() else -1
        ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            device=device,
            aggregation_strategy="simple",
            batch_size=16
        )
        logger.info(f"NER model loaded on {'GPU' if device == 0 else 'CPU'}")

        batch = list(rows)
        if not batch:
            return

        texts       = [r.chunk_text for r in batch]
        article_ids = [r.article_id for r in batch]
        chunk_ids   = [r.chunk_id for r in batch]

        try:
            batch_results = ner(texts)
            for i, ents in enumerate(batch_results):
                for ent in ents:
                    yield Row(
                        article_id=article_ids[i],
                        chunk_id=chunk_ids[i],
                        entity=ent["word"],
                        entity_type=ent["entity_group"],
                        relevance_score=float(ent["score"]),
                    )
        except Exception as e:
            logger.error(f"Batch NER error: {e}")
            traceback.print_exc()

    except Exception as e:
        logger.error(f"NER initialization error: {e}")
        traceback.print_exc()

def main():
    logger.info("Starting Spark session")
    spark = (
        SparkSession.builder.appName("ChunkNERPipeline")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )

    # Load articles
    articles_df = (
        spark.read.option("header", True)
        .csv("data/processed/articles_export.csv")
        .select("id", "content")
    )
    total_articles = articles_df.count()
    logger.info(f"Loaded {total_articles} articles")

    # Chunk articles into paragraphs in original order, with per-article indexing
    chunks_df = (
        articles_df
        .withColumn("chunk_text_array", split_udf(col("content")))
        .select(
            col("id"),
            F.posexplode("chunk_text_array").alias("chunk_index", "chunk_text")
        )
        .withColumn("chunk_id", concat_ws("_", col("id"), col("chunk_index")))
        .select(
            col("chunk_id"),
            col("id").alias("article_id"),
            col("chunk_index"),
            col("chunk_text")
        )
    )
    total_chunks = chunks_df.count()
    logger.info(f"Created {total_chunks} chunks")

    # Write article_chunks to Parquet
    chunks_df.write.mode("overwrite").parquet("data/processed/article_chunks.parquet")
    logger.info("Wrote article_chunks.parquet")

    # Repartition for parallel NER
    chunks_df = chunks_df.repartition(8)

    # Extract entities using Spark mapPartitions with batch NER
    entities_rdd = chunks_df.rdd.mapPartitions(ner_partition)
    entities_df = spark.createDataFrame(entities_rdd, schema=entity_schema)
    entity_count = entities_df.count()
    logger.info(f"Extracted {entity_count} entities")

    # Write doc_entities to Parquet
    entities_df.write.mode("overwrite").parquet("data/processed/doc_entities.parquet")
    logger.info("Wrote doc_entities.parquet")

    # Load into DuckDB for downstream use—drop existing tables first
    db_path = "data/processed/duckdb/articles.duckdb"
    conn = duckdb.connect(db_path)

    conn.execute("DROP TABLE IF EXISTS article_chunks;")
    conn.execute("DROP TABLE IF EXISTS doc_entities;")

    conn.execute(f"""
        CREATE TABLE article_chunks AS
        SELECT * FROM parquet_scan('data/processed/article_chunks.parquet/*.parquet');
    """)
    conn.execute(f"""
        CREATE TABLE doc_entities AS
        SELECT * FROM parquet_scan('data/processed/doc_entities.parquet/*.parquet');
    """)
    conn.close()
    logger.info("Loaded tables into DuckDB")

    spark.stop()
    logger.info("Pipeline complete")

if __name__ == "__main__":
    main()