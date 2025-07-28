from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# Initialize the embedding model once
_model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text: str) -> list:
    """
    Convert text to a normalized embedding vector.
    """
    return _model.encode(text, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)


def build_full_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Build the full prompt with neutral framing.
    """
    prompt = f"You are answering a user question using only relevant context below.\n"
    prompt += f"Ignore unrelated or unhelpful text.\n\nQuestion: {query}\n\nContext:\n"
    for c in chunks:
        prompt += f"- {c['chunk_text'].strip()}\n"
    return prompt


def build_ablated_prompt(query: str, chunks: List[Dict], omit_index: int) -> str:
    """
    Build a prompt omitting one chunk.
    """
    prompt = f"You are answering a user question using only relevant context below.\n"
    prompt += f"Ignore unrelated or unhelpful text.\n\nQuestion: {query}\n\nContext:\n"
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


def jaccard_overlap(text1: str, text2: str) -> float:
    """
    Compute Jaccard overlap between two texts at token level.
    """
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def compute_combined_score(drift: float,
                           faiss_score: float = None,
                           chunk_text: str = None,
                           full_response: str = None,
                           mode: str = 'drift') -> float:
    """
    Compute final score based on selected mode:
    - 'drift': use drift only
    - 'drift_faiss': drift * faiss_score
    - 'drift_overlap': drift + lambda * jaccard_overlap(text, full_response)
    """
    if mode == 'drift':
        return drift
    if mode == 'drift_faiss' and faiss_score is not None:
        return drift * faiss_score
    if mode == 'drift_overlap' and chunk_text and full_response:
        overlap = jaccard_overlap(full_response, chunk_text)
        return drift + 0.1 * overlap
    # fallback
    return drift


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