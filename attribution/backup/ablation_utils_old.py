from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# Initialize the embedding model once
_model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text: str) -> list:
    """
    Convert text to a normalized embedding vector.
    """
    vec = _model.encode(
        text,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vec


def build_full_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Build the full prompt with more neutral and selective framing.
    """
    prompt = f"""You are answering a user question using only relevant context below. 
Ignore any unrelated or unhelpful text.

Question: {query}

Context:
    """
    for c in chunks:
        prompt += f"- {c['chunk_text'].strip()}\n"
    return prompt



def build_ablated_prompt(query: str, chunks: List[Dict], omit_index: int) -> str:
    """
    Build prompt without the omitted chunk, keeping framing identical.
    """
    prompt = f"""You are answering a user question using only relevant context below. 
Ignore any unrelated or unhelpful text.

Question: {query}

Context:
"""
    for i, c in enumerate(chunks):
        if i != omit_index:
            prompt += f"- {c['chunk_text'].strip()}\n"
    return prompt



def cosine_drift(e_full, e_ablated) -> float:
    """
    Compute semantic drift: 1 - cosine_similarity.
    """
    sim = cosine_similarity([e_full], [e_ablated])[0][0]
    return 1.0 - sim


def rollup_to_documents(chunk_scores: Dict[str, float], chunk_to_article: Dict[str, str]) -> Dict[str, float]:
    """
    Sum and normalize chunk scores by their article_id.
    """
    doc_scores: Dict[str, float] = {}
    for cid, score in chunk_scores.items():
        aid = chunk_to_article.get(cid)
        if not aid:
            continue
        doc_scores[aid] = doc_scores.get(aid, 0.0) + score
    total = sum(doc_scores.values()) or 1.0
    return {aid: s/total for aid, s in doc_scores.items()}