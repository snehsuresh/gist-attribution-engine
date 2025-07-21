#!/usr/bin/env python3
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    ArrayType,
)
import torch


def main():
    # 1. Spark session (tune for your cluster / GPU resources)
    spark = (
        SparkSession.builder.appName("HF-NER-PySpark-GPU")
        # if you're on Databricks or EMR with GPU support, set executors/resources here
        .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate()
    )

    # 2. Read articles (exported CSV with id,content)
    articles_df = (
        spark.read.option("header", True)
        .csv("data/processed/articles_export.csv")
        .selectExpr("id", "content")
    )

    # 3. Create article_chunks DF: one chunk per article
    chunks_df = articles_df.selectExpr(
        "concat(id, '_chunk_0') as chunk_id",
        "id as article_id",
        "cast(0 as int) as chunk_index",
        "content as chunk_text",
    )

    # 4. Define partition processor that loads the HF pipeline once
    def ner_partition(rows):
        from transformers import pipeline

        # send model to GPU if available
        device = 0 if torch.cuda.is_available() else -1
        ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            device=device,
            aggregation_strategy="simple",
        )
        buffer = []
        ids = []
        for r in rows:
            ids.append(r.id)
            buffer.append(r.content)
        if not buffer:
            return

        # run NER in one batch (adjust batch_size if you run out of memory)
        results = ner(buffer, batch_size=16)
        # results is a list of lists of entity dicts
        for doc_id, ents in zip(ids, results):
            chunk_id = f"{doc_id}_chunk_0"
            for ent in ents:
                yield Row(
                    article_id=doc_id,
                    chunk_id=chunk_id,
                    entity=ent["word"],
                    entity_type=ent["entity_group"],
                    relevance_score=float(ent["score"]),
                )

    # 5. Apply mapPartitions and convert to DataFrame
    entities_rdd = articles_df.rdd.mapPartitions(ner_partition)
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

    # 6. Write outputs in parallel Parquet (fast) — import back to DuckDB later if desired
    chunks_df.write.mode("overwrite").parquet("data/processed/article_chunks.parquet")
    entities_df.write.mode("overwrite").parquet("data/processed/doc_entities.parquet")

    spark.stop()
    print("Done: article_chunks + doc_entities written as Parquet.")


if __name__ == "__main__":
    main()
