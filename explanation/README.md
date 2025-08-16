# Phase 4: Explanation and Interpretation Layer

This document covers **Phase 4** of the Gist Attribution Engine, which generates **human-readable, structured explanations** for why each retrieved content chunk influenced an LLM's response. This phase builds directly on the attribution results from Phase 3 and transforms influence scores into interpretable natural language summaries.

---

## Objective

> Move beyond numeric attribution scores and explain _why_ each chunk mattered.

This enables:

- User trust through transparency
- Summaries for audit logs or report export
- Value demonstration to content providers (licensing, ranking)
- Interpretability for researchers or end-users

---

## Files Involved

### `explanation/`

- `run_explanation.py` — Driver script to load attribution and metadata, call LLM, save results
- `explanation_utils.py` — Helper functions to build explanation prompts and call the OpenAI API
- `prompt_templates.py` — Template for instructing the LLM
- `explanation_schema.sql` — Optional SQL schema to persist explanations in DuckDB or SQLite

---

## How It Works

1. Takes output from Phase 3 (attribution JSON)
2. Loads chunk metadata to retrieve full text snippets
3. Applies a threshold (e.g., 0.05) to filter out low-impact chunks
4. Builds a prompt using question, response, and selected chunks
5. Calls OpenAI’s GPT-3.5 with a structured instruction
6. Parses the LLM output into a list of JSON explanations
7. Saves the final output as an explanation JSON

---

## `run_explanation.py`

### Usage:

```bash
python explanation/run_explanation.py \
  --input_path data/processed/output/attribution_results/<query>.json \
  --metadata_path data/processed/embeddings/metadata.pkl \
  --threshold 0.05
```

### What It Does:

- Loads Top-K influence results from attribution
- Filters chunks below the influence threshold
- Retrieves text snippets from metadata
- Calls `build_explanation_prompt()` and `call_llm_for_explanation()`
- Parses output and saves final explanation file to:

```bash
data/processed/output/explanation_results/<query>_explanation.json
```

---

## `explanation_utils.py`

### `build_explanation_prompt()`

Creates a context-rich prompt like:

```text
Question: How has Elon Musk influenced U.S. policy…
LLM Answer: Elon Musk has split from the administration after identifying…

Chunks:
- chunk_id: xyz_01
  influence: 0.56
  text_snippet: "Elon Musk left after proposing DOGE cuts."
```

### `call_llm_for_explanation()`

Calls GPT-3.5 using OpenAI SDK, with caching. Ensures return is a valid JSON array.

---

## `prompt_templates.py`

Defines `EXPLANATION_TEMPLATE` with example output to enforce:

```json
[
  {
    "chunk_id": "id1",
    "influence": 0.56,
    "explanation": "This snippet highlights Musk’s influence on policy via DOGE."
  },
  …
]
```

Final instruction enforces: “return only a JSON array, no markdown, no text.”

---

## `explanation_schema.sql` (Optional)

```sql
CREATE TABLE IF NOT EXISTS explanations (
  query TEXT,
  chunk_id TEXT,
  explanation TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Use this if you want to persist explanations in DuckDB or SQLite.

---

## Final Output Format

Each explanation result is saved as:

```json
{
  "query": "How has Elon Musk…",
  "explained_chunks": [
    {
      "chunk_id": "abc_01",
      "influence": 0.56,
      "text_snippet": "Elon Musk broke from admin…"
    },
    …
  ],
  "explanations": [
    {
      "chunk_id": "abc_01",
      "influence": 0.56,
      "explanation": "This snippet explains Musk’s impact on fiscal policy."
    },
    …
  ]
}
```

---

## What This Enables

| Feature | Benefit |
| --- | --- |
| Chunk-level rationale | Shows _why_ something mattered, not just how much |
| Structured output | Easily rendered in UI or dashboards |
| Transparent LLM reasoning | Useful for audits, licensing, review |
| Scalable to doc/entity level | Can aggregate later if needed |
