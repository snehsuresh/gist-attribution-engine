#!/usr/bin/env python3
import os
import logging
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)
import torch
from transformers import pipeline
import traceback

# ---- Suppress noisy PySpark logs ----
os.environ["PYSPARK_LOG_LEVEL"] = "ERROR"
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

# ---- Setup Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("NER-Pipeline")


def main():
    try:
        logger.info("🔄 Starting Spark session...")
        spark = (
            SparkSession.builder.appName("HF-NER-PySpark-GPU")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        logger.info("✅ Spark session created.")

        logger.info("📥 Reading articles CSV...")
        articles_df = (
            spark.read.option("header", True)
            .csv("data/processed/articles_export.csv")
            .selectExpr("id", "content")
        )
        total_articles = articles_df.count()
        logger.info(f"✅ Loaded {total_articles} articles.")

        logger.info("🧱 Creating article_chunks DataFrame...")
        chunks_df = articles_df.selectExpr(
            "concat(id, '_chunk_0') as chunk_id",
            "id as article_id",
            "cast(0 as int) as chunk_index",
            "content as chunk_text",
        )
        logger.info("✅ article_chunks DataFrame created.")

        # Partition function for entity extraction
        def ner_partition(rows):
            from transformers import pipeline

            try:
                device = 0 if torch.cuda.is_available() else -1
                ner = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    device=device,
                    aggregation_strategy="simple",
                )
                print(f"[NER PARTITION] Model loaded on {'GPU' if device == 0 else 'CPU'}")

                buffer, ids = [], []
                for r in rows:
                    if r.content:
                        ids.append(r.id)
                        buffer.append(r.content)

                print(f"[NER PARTITION] Processing {len(buffer)} documents")

                if not buffer:
                    return

                results = ner(buffer, batch_size=16)
                print(f"[NER PARTITION] Got NER results for {len(results)} docs")

                for doc_id, ents in zip(ids, results):
                    chunk_id = f"{doc_id}_chunk_0"
                    for ent in ents:
                        print(f"[NER RESULT] {doc_id}: {ent['word']} ({ent['entity_group']})")
                        yield Row(
                            article_id=doc_id,
                            chunk_id=chunk_id,
                            entity=ent["word"],
                            entity_type=ent["entity_group"],
                            relevance_score=float(ent["score"]),
                        )

            except Exception:
                print("[NER ERROR] An exception occurred in ner_partition:")
                traceback.print_exc()
                return

        logger.info("🧠 Running entity extraction...")
        entities_rdd = articles_df.rdd.mapPartitions(ner_partition)

        logger.info("🧾 Converting to entities DataFrame...")
        entities_df = spark.createDataFrame(
            entities_rdd,
            schema=StructType(
                [
                    StructField("article_id", StringType(), False),
                    StructField("chunk_id", StringType(), False),
                    StructField("entity", StringType(), False),
                    StructField("entity_type", StringType(), False),
                    StructField("relevance_score", DoubleType(), False),
                ]
            ),
        )

        entity_count = entities_df.count()
        logger.info(f"✅ Extracted {entity_count} entities.")

        if entity_count == 0:
            logger.warning("⚠️ No entities extracted. Check content or model issues.")
        else:
            logger.info("💾 Writing article_chunks to Parquet...")
            chunks_df.write.mode("overwrite").parquet("data/processed/article_chunks.parquet")
            logger.info("💾 Writing doc_entities to Parquet...")
            entities_df.write.mode("overwrite").parquet("data/processed/doc_entities.parquet")
            logger.info("✅ Parquet write complete.")

        spark.stop()
        logger.info("🛑 Spark session stopped. Pipeline finished.")

    except Exception as e:
        logger.exception(f"🔥 Pipeline failed with error: {e}")


if __name__ == "__main__":
    main()
