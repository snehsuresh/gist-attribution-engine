# 📊 End-to-End Data Pipeline and Preprocessing Summary

This document details the entire journey of our data preparation and enhancement process, from initial ingestion to final output tables used in downstream attribution, retrieval, and summarization systems. The design is inspired by the Gist Attribution architecture (akin to ProRata.ai), but has been significantly enhanced to support fine-grained, entity-aware, session-aware, and feedback-ready processing at scale.

---

## 🏁 Phase 1: Data Ingestion & Normalization

### 🎯 Objective:

Establish a unified, queryable foundation of raw content and simulated user behavior that can support document-level and chunk-level downstream analysis.

### ✅ Key Inputs:

- `articles_export.csv`: Cleaned, deduplicated articles with fields like `id`, `title`, `content`, `published_at`, `tags`
- Simulated user logs: Views, clicks, and queries aligned with the article topics

### 🔨 What We Did:

1. **ETL Setup with PySpark**: We used PySpark to process large volumes of articles and prepare them for chunk-level operations.
2. **Chunking Articles**: Each article was split into paragraphs (based on `\n\n`) to create granular units of meaning, improving interpretability and attribution.
3. **NER with Hugging Face Transformers**:
   - We applied `dslim/bert-base-NER` with aggregation strategy set to `simple`.
   - Entities were extracted for each chunk and stored with their type and model confidence.
4. **Structured Table Creation**:
   - `article_chunks`: (chunk\_id, article\_id, chunk\_index, chunk\_text)
   - `doc_entities`: (article\_id, chunk\_id, entity, entity\_type, relevance\_score)

---

## 🧼 Entity Cleaning & Validation

### 🎯 Objective:

Remove low-quality entities and validate structural assumptions in chunking for clean downstream attribution.

### 🧹 Cleaning Steps:

- Removed tokens that:
  - Start with `##` (subwords from tokenizer)
  - Are shorter than 3 characters
  - Are only numbers or whitespace
  - Are isolated uppercase letters (e.g. `"A"`, `"XY"`)

### ✅ Validation Metrics:

- **41,041** total chunks
- **55,421** raw entities extracted
- **38,483** cleaned entities retained
- 0 chunks with missing `chunk_id`
- Only **2** articles with non-contiguous `chunk_index` — not critical

This confirms the chunking + entity pipeline has integrity and precision.

---

## ✨ Enhancements Incorporated

These features extend beyond ProRata.ai’s typical document-level attribution and make the pipeline more powerful and flexible.

### 1. 📌 Fine-Grained Attribution

> Each article is broken down into paragraphs, allowing attribution to be performed at the **chunk level**.

**Why it matters:**

- Attribution is more interpretable
- Enables paragraph-level influence tracking
- Reduces noise from irrelevant parts of long articles

### 2. 🧠 Entity-Aware Attribution

> Named entities like "Ukraine", "NATO", "Trump" are extracted and linked to chunks.

**Why it matters:**

- Enables **semantic summarization** by entity
- Supports future **concept-based monetization** or filtering
- Lets us answer: *"Which entities influenced this LLM answer the most?"*

### 3. 🔁 Session-Aware User Logs

> Simulated logs now include sessions, queries, and views/clicks across time.

**Why it matters:**

- More realistic modeling of user behavior
- Enables analysis of influence **across multiple related queries**
- Vital for attribution algorithms to reflect actual user journeys

### 4. 💬 Feedback Integration

> The user logs schema supports `feedback_useful`, `feedback_confusing`, etc.

**Why it matters:**

- Allows future use of **human feedback to refine attribution**
- Helps score publisher quality and rank documents beyond ablation
- Makes the product **interactive and learnable**

---

## 📦 Output Tables (in DuckDB)

### `articles`

Raw article metadata and full text.

```sql
id, title, source, published_at, content, tags
```

### `article_chunks`

Paragraph-level splits with ordering.

```sql
chunk_id, article_id, chunk_index, chunk_text
```

### `doc_entities`

Named entities with confidence scores.

```sql
article_id, chunk_id, entity, entity_type, relevance_score
```

### `doc_entities_cleaned`

Cleaned version of above with filtered noise.

### `user_events` (simulated)

```sql
user_id, session_id, query, article_id, chunk_id, event_type, timestamp
```

Includes views, clicks, and feedback types.

---

## 🧭 Why This Design Matters

This pipeline isn’t just “text-in, summary-out.” It’s designed to:

- **Diagnose attribution accuracy**: Which chunk or concept truly influenced an answer?
- **Support monetization**: Credit sources based on influence at the paragraph or entity level
- **Enable transparency**: Show users or auditors *why* a doc mattered
- **Scale ethically**: Plug into future models of licensing, user feedback, and topic tracking

---

## 📈 Next Steps (Optional)

- Embed `article_chunks` using Sentence Transformers
- Perform **chunk-level Top-K vector retrieval**
- Implement **ablation scoring** at the chunk level
- Compute **influence per entity**, not just per document
- Build **summary UIs that highlight top entities and paragraphs**

---

## 🏁 Conclusion

This foundation transforms raw content into a rich, attribution-ready knowledge base. By splitting, tagging, cleaning, and indexing content at a fine-grained level—and simulating real user behavior—we enable far more than document-level attribution. We enable *interpretable, explainable, entity-aware AI*. And that's a meaningful leap beyond existing approaches.

