
"""
Templates for generating human-readable explanations in Phase 4.
"""
EXPLANATION_TEMPLATE = """
You are an AI assistant that provides concise JSON-formatted explanations for how specific text snippets influenced an answer.

Question: {query}

LLM Answer: {full_response}

Below are the text snippets selected for explanation, each with its influence score and snippet text:
{context}

For each snippet, output a JSON array of objects with:
- "chunk_id": snippet identifier
- "influence": numeric score
- "explanation": one-sentence rationale

Example output:
[
  {{
    "chunk_id": "id1",
    "influence": 0.56,
    "explanation": "This snippet highlighted a key policy change."
  }},
  ...
]
- Now generate the explanations.
+ Now generate *only* the JSON array—no markdown, no extra text, just the array itself.
"""