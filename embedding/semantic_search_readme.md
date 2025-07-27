# Semantic Search with DuckDB, FAISS & SentenceTransformers

This project implements a fast, modular semantic search system over paragraph-level news articles using **DuckDB**, **SentenceTransformers**, and **FAISS**. It is built for scalable, high-speed search over dense text embeddings and optimized for both experimentation and production search pipelines.

---

## Files Overview

### 1. `embed_chunks.py`

Efficiently embeds text chunks from the DuckDB database using a SentenceTransformer model and stores them in a new table for downstream search.

**Main Responsibilities:**

- Connects to `articles.duckdb`
- Fetches distinct `chunk_id`, `chunk_text` pairs from the `article_chunks` table
- Uses `SentenceTransformer("all-MiniLM-L6-v2")` to encode texts in mini-batches
- Normalizes all embeddings for cosine similarity use
- Writes output to a new table: `chunk_embeddings (chunk_id TEXT, embedding FLOAT[])`

This ensures that even millions of records can be processed in memory-efficient batches with full progress tracking using `tqdm`.

### 2. `dump_embedding.py`

Converts data in DuckDB into NumPy and Pandas formats for fast FAISS search.

**Outputs:**

- `embeddings.npy`: a 2D float32 array of shape `(num_chunks, embedding_dim)`
- `metadata.pkl`: a pickled DataFrame containing `chunk_id`, `article_id`, `chunk_text`, `title`

This separation of embedding matrix and metadata enables blazing-fast in-memory retrieval.

### 3. `fast_vector_search.py`

Performs **ultra-fast** top-K vector search using the FAISS library.

**Workflow:**

- Loads `embeddings.npy` and `metadata.pkl`
- Initializes a `faiss.IndexFlatIP` (cosine similarity on normalized vectors)
- Encodes the query using the same embedding model
- Searches top K most similar vectors
- Returns human-readable metadata for those results

Supports GPU search if FAISS-GPU is installed and `--use_gpu` is passed.

### 4. `vector_search.py`

A pure-SQL fallback that uses DuckDB to search via cosine distance between a query vector and chunk embeddings.

**Key Features:**

- No need to export `.npy` or `.pkl`
- Less performant, but easier to integrate in SQL-based pipelines
- Ideal for quick tests or smaller datasets

---

## Setup Instructions

Install all dependencies with:

```bash
pip install sentence-transformers pandas numpy faiss-cpu
# or use: pip install faiss-gpu if running with CUDA support
```

Ensure that `data/processed/duckdb/articles.duckdb` exists and contains the following:

- `article_chunks (chunk_id, article_id, chunk_text)`
- `articles (id, title)`

---

## Usage Workflow

### Step 1: Embed all article chunks into DuckDB

```bash
python embed_chunks.py
```

This creates the `chunk_embeddings` table.

### Step 2: Dump vectors and metadata for FAISS

```bash
python dump_embedding.py
```

Creates `embeddings.npy` and `metadata.pkl`.

### Step 3: Run a FAISS-based search

```bash
python fast_vector_search.py \
  --query "trump" \
  --embeddings_path data/processed/embeddings.npy \
  --metadata_path data/processed/metadata.pkl \
  --top_k 5
```

Or run a DuckDB-only SQL search:

```bash
python vector_search.py --query "trump"
```

---

##  Data Model (DuckDB Schema)

\*\*Table: \*\*\`\`

- `chunk_id` (TEXT): Unique ID for the paragraph chunk
- `article_id` (TEXT): Foreign key to article
- `chunk_text` (TEXT): Raw paragraph text

\*\*Table: \*\*\`\`

- `id` (TEXT): Unique article ID
- `title` (TEXT): Article title

\*\*Table: \*\*\`\`

- `chunk_id` (TEXT): Foreign key to `article_chunks`
- `embedding` (FLOAT[]): 384-dimensional vector (MiniLM output)

---

## Speed & Architecture Considerations

### Architecture Overview:

```
[DuckDB] → [SentenceTransformer Encoder] → [chunk_embeddings Table]
     ↓
[metadata.pkl, embeddings.npy] → [FAISS Index] → [Top-K Similar Results]
```

### Why This Is Fast:

- All embeddings are computed once and stored efficiently (no re-encoding at query time)
- FAISS uses an in-memory vector index (`IndexFlatIP`) for instant dot-product retrieval
- DuckDB enables structured joins and filtering pre-embedding
- Query encoding takes \~5–10ms, and FAISS search \~1–3ms even for 1M+ vectors
- GPU support via `faiss-gpu` enables sub-millisecond search

### Comparison of Search Options:

| Method                          | Speed     | Scalability | Storage                  |
| ------------------------------- | --------- | ----------- | ------------------------ |
| `vector_search.py`              | 🐢 Slower | Medium      | Pure SQL, no export      |
| `fast_vector_search.py` + FAISS | ⚡ Fastest | High        | Uses .npy/.pkl in memory |

---

## Interpreting Cosine Similarity Scores

Cosine similarity is used to rank results. All embeddings are L2-normalized, so FAISS uses dot-product for cosine similarity.

**Score Ranges:**

- `0.55 – 0.70`: Highly relevant (e.g. same event or person)
- `0.40 – 0.55`: Topical similarity
- `0.25 – 0.40`: Mild semantic overlap
- `< 0.25`: Often noise

The **relative ranking** of results is more important than absolute scores.

---

## Future Enhancements

-  Replace MiniLM with `all-mpnet-base-v2` or `bge-base` for better semantic precision
-  Add re-ranking step using `CrossEncoder` to refine top-K
-  Switch from `IndexFlatIP` to `IndexIVFFlat` in FAISS for sublinear ANN search
-  Integrate DuckDB’s `vector` extension for SQL-native ANN
-  Persist FAISS index to disk for faster startup and larger corpora

---

## Summary

This project is an efficient and production-friendly semantic search stack combining:

-  **Embeddings** from pretrained transformer models (via SentenceTransformers)
-  **Data storage** using DuckDB for easy ETL and relational joins
-  **Vector similarity search** via FAISS for low-latency results
-  **Speed-focused architecture** that supports real-time search use cases
-  **Modular design** for switching between FAISS and SQL pipelines

This setup is ideal for building scalable question-answering tools, news recommendation systems, or document retrieval pipelines with strong semantic understanding.

