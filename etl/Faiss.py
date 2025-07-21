import duckdb
import pandas as pd
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
import faiss                   # Ensure faiss-cpu is installed: pip install faiss-cpu
import numpy as np
import logging
import os

# Check JAVA_HOME (required for PySpark)
print("JAVA_HOME:", os.environ.get("JAVA_HOME"))

# === Logging Configuration ===
# Using Python's logging module to provide info/debug messages during execution
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# === Step 1: Load data from DuckDB ===
# DuckDB is an in-process SQL OLAP database — fast for analytical queries on local data
logger.info("Connecting to DuckDB and loading data...")
logger.info("Connecting to DuckDB and loading data...")
con = duckdb.connect("C:/Users/Govind/Downloads/articles.duckdb")
df = con.execute("SELECT chunk_text, published_at FROM article_chunks_with_dates").fetch_df()
logger.info(f"Loaded {df.shape[0]} rows from DuckDB.")

# === Step 2: Parallelize vectorization with PySpark ===
logger.info("Initializing Spark session...")
spark = SparkSession.builder.appName("VectorizeChunks").getOrCreate()

logger.info("Converting Pandas DataFrame to Spark DataFrame...")
spark_df = spark.createDataFrame(df)

logger.info("Creating RDD for text chunks...")
rdd = spark_df.select("chunk_text").rdd.map(lambda row: row["chunk_text"])
rdd = rdd.repartition(42)
logger.info("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
sc = spark.sparkContext
model_broadcast = sc.broadcast(model)  # sc is the SparkContext

# Define the partition function using the broadcasted model
def embed_partition(partition):
    try:
        # Load model inside the partition (each executor will do this independently)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = list(partition)
        if not texts:
            return iter([])
        vectors = model.encode(texts, batch_size=32)  # Tune batch_size if needed
        return iter(vectors)
    except Exception as e:
        import traceback
        logger.error("Error during partition embedding:")
        traceback.print_exc()
        return iter([])

# Step 4: Apply vectorization
logger.info("Vectorizing text chunks in parallel using Spark...")
vectors = rdd.mapPartitions(embed_partition)

# Step 5: Collect and flatten the vectors
logger.info("Flattening vectors...")
flat_vectors = np.vstack(vectors.collect()) # .collect() needed to pull all partitions together

# === Normalize vectors for cosine similarity ===
logger.info("Normalizing vectors for cosine similarity...")
flat_vectors = flat_vectors / np.linalg.norm(flat_vectors, axis=1, keepdims=True)

# === Step 3: Save FAISS index ===
logger.info("Creating FAISS index with IndexFlatIP for cosine similarity...")
index = faiss.IndexFlatIP(flat_vectors.shape[1])
index.add(flat_vectors)

logger.info("Saving FAISS index to disk...")
faiss.write_index(index, "faiss_index_cosine.idx")
logger.info("FAISS index written to faiss_index_cosine.idx")

# Save metadata for retrieval
logger.info("Saving metadata as Parquet...")
df.reset_index(drop=True).to_parquet("vector_metadata.parquet")
logger.info("Metadata written to vector_metadata.parquet")

# === Step 4: Query with a sample string ===
query_text = "climate policy in Europe"
logger.info(f"Encoding query text: '{query_text}'")
query_vec = model.encode([query_text])
query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

logger.info("Searching FAISS index using cosine similarity...")
D, I = index.search(np.array(query_vec), k=5)

logger.info("Loading metadata and printing top results...")
meta = pd.read_parquet("vector_metadata.parquet")
for i, idx in enumerate(I[0]):
    print(f"\nRank {i+1} | Cosine Similarity: {D[0][i]:.4f}")
    print(meta.iloc[idx]["chunk_text"][:300])
