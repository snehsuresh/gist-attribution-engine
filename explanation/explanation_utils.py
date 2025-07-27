import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from openai import OpenAI
from prompt_templates import EXPLANATION_TEMPLATE
from attribution.cache_manager import load_from_cache, save_to_cache

client = OpenAI()


def build_explanation_prompt(query: str, full_response: str, explained_chunks: list) -> str:
    """
    Construct a prompt for explaining why each selected chunk influenced the answer.
    """
    context_lines = []
    for chunk in explained_chunks:
        context_lines.append(
            f"- chunk_id: {chunk['chunk_id']}\n  influence: {chunk['influence']:.2f}\n  text_snippet: {chunk['text_snippet']}"
        )
    context = "\n".join(context_lines)
    return EXPLANATION_TEMPLATE.format(
        query=query,
        full_response=full_response,
        context=context
    )


def call_llm_for_explanation(prompt: str) -> str:
    """
    Call OpenAI LLM to generate JSON-formatted explanations, with caching.
    """
    cached = load_from_cache(prompt)
    if cached:
        return cached['response']
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise JSON explanations."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0
    )
    text = res.choices[0].message.content.strip()
    save_to_cache(prompt, {'response': text, 'embedding': []})
    return text
