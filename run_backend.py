from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi.responses import StreamingResponse
import subprocess
import os
import sys
import json
import pandas as pd

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root) 

from utils.helpers import sanitize_filename
from attribution.run_ablation import stream_first_response

app = FastAPI(title="Gist Attribution Pipeline API")

class ProcessRequest(BaseModel):
    query: str = Field(..., description="The user query to process")
    top_k: int = Field(5, ge=1, description="Number of top chunks to retrieve/ablate")
    scoring_mode: str = Field("drift", description="Scoring mode: drift, drift_faiss, or drift_overlap")
    threshold: float = Field(0.05, ge=0.0, le=1.0, description="Minimum influence score to explain")
    filter_articles: Optional[List[str]] = Field(None, description="List of article_id strings to restrict ablation")

@app.post("/embed")
async def embed():
    try:
        subprocess.run(["python", "embedding/embedder.py"], check=True)
        subprocess.run(["python", "embedding/dump_embeddings.py"], check=True)
        return {
            "status": "ok",
            "embeddings_path": "data/processed/embeddings.npy",
            "metadata_path": "data/processed/metadata.pkl"
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

@app.post("/first_response")
async def first_response(req: ProcessRequest):
    """Stream the initial GPT answer token-by-token."""
    return StreamingResponse(
        stream_first_response(
            query=req.query,
            embeddings_path="data/processed/embeddings.npy",
            metadata_path="data/processed/metadata.pkl",
            top_k=req.top_k
        ),
        media_type="text/plain"
    )

@app.post("/process")
async def process(req: ProcessRequest):
    sanitized = sanitize_filename(req.query)
    emb_path = "data/processed/embeddings.npy"
    meta_path = "data/processed/metadata.pkl"
    attr_dir = "data/processed/output/attribution_results"
    exp_dir  = "data/processed/output/explanation_results"
    os.makedirs(attr_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    attr_file = os.path.join(attr_dir, f"{sanitized}.json")
    exp_file  = os.path.join(exp_dir, f"{sanitized}_explanation.json")

    # 1) Ablation
    ablation_cmd = [
        "python", "attribution/run_ablation.py",
        "--query", req.query,
        "--embeddings_path", emb_path,
        "--metadata_path", meta_path,
        "--top_k", str(req.top_k),
        "--scoring_mode", req.scoring_mode
    ]
    if req.filter_articles:
        ablation_cmd += ["--filter_articles", ",".join(req.filter_articles)]
    try:
        subprocess.run(ablation_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Ablation failed: {e}")

    # 2) Explanation
    explanation_cmd = [
        "python", "explanation/run_explanation.py",
        "--input_path", attr_file,
        "--metadata_path", meta_path,
        "--threshold", str(req.threshold)
    ]
    try:
        subprocess.run(explanation_cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")

    # 3) Load results
    try:
        with open(attr_file, 'r') as f:
            attribution = json.load(f)
        with open(exp_file, 'r') as f:
            explanation = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load result files: {e}")

    # 4) Metrics
    top_k_chunks = len(attribution.get("influence_by_chunk", {}))
    explained_chunks = explanation.get("explained_chunks", [])
    chunks_above_threshold = len(explained_chunks)

    doc_items = sorted(attribution.get("influence_by_doc", {}).items(), key=lambda x: x[1], reverse=True)
    top_documents = [{"article_id": doc, "influence": round(score,4)} for doc, score in doc_items[:3]]

    most_chunk = max(explained_chunks, key=lambda x: x.get("influence",0), default={})
    most_influential_chunk = {
        "chunk_id": most_chunk.get("chunk_id"),
        "influence": round(most_chunk.get("influence",0),4),
        "text": most_chunk.get("text_snippet","")[:200]
    } if most_chunk else {}

    metrics = {
        "top_k_chunks": top_k_chunks,
        "chunks_above_threshold": chunks_above_threshold,
        "threshold_used": req.threshold,
        "influence_by_chunk": {k: round(v,4) for k,v in attribution.get("influence_by_chunk",{}).items()},
        "influence_by_doc":   {k: round(v,4) for k,v in attribution.get("influence_by_doc",{}).items()},
        "top_documents": top_documents,
        "most_influential_chunk": most_influential_chunk
    }

    # 5) Enrich with metadata
    meta_df = pd.read_pickle(meta_path)
    chunk_metadata = {
        row.chunk_id: {
            "article_id": row.article_id,
            "title": row.title,
            "source": row.article_id.split("_")[0]
        }
        for _, row in meta_df.iterrows()
    }

    return {
        "status": "ok",
        "query": req.query,
        "metrics": metrics,
        "attribution": attribution,
        "explanation": explanation,
        "chunk_metadata": chunk_metadata
    }

# To run:
# uvicorn run_api:app --reload --host 0.0.0.0 --port 8000