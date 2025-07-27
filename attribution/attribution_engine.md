# 📘 Phase 3: Attribution Engine (Ablation-Based)

This document provides an in-depth explanation of **Phase 3: Ablation-Based Attribution**, part of the Gist Attribution Engine. This phase is responsible for measuring the influence of individual document **chunks** (paragraphs) on an LLM-generated response, and then rolling up those scores to **document-level** influence. The goal is to go beyond traditional retrieval similarity and apply a **causal method** to score what content truly affected the model's output.

---

##  Purpose

> "If we removed this chunk from the context, would the LLM’s response change?"

This phase answers that question using an **ablation loop** — where we remove one chunk at a time, rerun the LLM, and measure the change in its output.

It enables:

-  Fine-grained content attribution
-  Human-aligned crediting logic
-  Potential for licensing & feedback systems

---

## Files Involved

### `attribution/`

- `run_ablation.py` — Main execution script
- `ablation_utils.py` — Embedding, prompt building, drift scoring
- `cache_manager.py` — LLM and embedding result caching
- `attribution_schema.sql` — SQL schema for storing output (optional)

### `data/`

- `processed/embeddings/embeddings.npy` — Vectorized chunk embeddings
- `processed/embeddings/metadata.pkl` — Metadata (chunk\_id, article\_id, text, etc.)
- `processed/output/attribution_results/*.json` — Output files with scores

---

##  How It Works (Overview)

1. **User query** is embedded
2. **Top-K chunks** are retrieved using FAISS
3. Full context is sent to LLM → generates response
4. Each chunk is **ablated** (removed one at a time)
5. New responses are compared using **cosine drift**
6. Influence scores are computed per chunk
7. Scores are **normalized and rolled up** to document level
8. Output saved as `.json` or optionally inserted into DuckDB

---

##  `run_ablation.py`

###  Command-line usage:

```bash
python attribution/run_ablation.py \
  --query "Why is inflation rising?" \
  --embeddings_path data/processed/embeddings/embeddings.npy \
  --metadata_path data/processed/embeddings/metadata.pkl \
  --top_k 5
```

###  What it does:

- Loads Top-K most relevant chunks using FAISS
- Sends a full prompt to OpenAI's GPT model
- For each chunk, creates a new prompt without that chunk
- Calculates embedding drift: `1 - cosine_similarity(full, ablated)`
- Normalizes and rolls up chunk scores to document-level
- Writes final attribution result to JSON

---

##  `ablation_utils.py`

###  `embed_text(text)`

Uses SentenceTransformer (MiniLM) to generate a normalized embedding for a given string.

###  `build_full_prompt(query, chunks)`

Creates the base prompt with all chunks included:

```text
You are answering a user question using only relevant context below.
Ignore unrelated or unhelpful text.

Question: {query}

Context:
- chunk 1
- chunk 2
...
```

###  `build_ablated_prompt(query, chunks, omit_index)`

Same as above, but omits one chunk. This forms the core of the ablation.

###  `cosine_drift(e_full, e_ablated)`

Computes drift = `1 - cosine_similarity()` between full and ablated embeddings.

###  `rollup_to_documents(chunk_scores, chunk_to_article)`

Groups chunk scores by `article_id` and normalizes them to compute document-level attribution.

---

##  `cache_manager.py`

Avoids repeated LLM and embedding calls by caching them on disk.

- Uses SHA256 of the prompt as filename
- Stores both LLM response and its embedding vector

---

##  Drift Thresholding

Chunks with very low drift (< `0.05`) are considered noise and given zero influence:

```python
if drift < 0.05:
    drift = 0.0
```

This reduces false attribution to irrelevant paragraphs.

---

##  Example Output (Simplified)

```json
{
  "query": "How has Elon Musk influenced U.S. policy during Trump’s second term?",
  "full_response": "Elon Musk has split from the administration after identifying spending cuts via DOGE.",
  "influence_by_chunk": {
    "lewrockwell_21_1": 0.55,
    "lewrockwell_21_0": 0.27,
    "counterpunch_36_2": 0.17,
    "...": "..."
  },
  "influence_by_doc": {
    "lewrockwell_21": 0.83,
    "counterpunch_36": 0.17
  }
}
```

---

##  What Makes This Better Than ProRata

| Feature                 | This Engine    | ProRata (likely)            |
| ----------------------- | -------------- | --------------------------- |
| True LLM Ablation       |  Yes          |  No                        |
| Semantic Drift Scoring  |  Cosine delta |  Retrieval similarity only |
| Chunk-level Attribution |               |                            |
| Document Roll-up        |               |  (document-level only)     |
| Caching                 |  Prompt-based |                            |

---

## Business Impact

-  More honest attribution = better trust and explainability
-  Ability to trace output to specific claims, not just documents
-  Can support licensing, feedback, content flagging, and summarization
-  Enables future monetization by showing influence weight per source

---

##  Next Steps

- Add entity-level influence scoring
- Add visual UI overlays per chunk
- Add token-level or sentence-level ablation

Let me know if you'd like the UI or scoring engine connected to this module.

---

 Attribution engine complete. Your outputs now show *what mattered, where, and how much.*
