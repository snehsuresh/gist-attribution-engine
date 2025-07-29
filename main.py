# run_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import subprocess
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import sanitize_filename



app = FastAPI(title="Gist Attribution Pipeline API")

# --- Request model for ablation + explanation ---
class ProcessRequest(BaseModel):
    query: str = Field(..., description="The user query to process")
    top_k: int = Field(5, ge=1, description="Number of top chunks to retrieve/ablate")
    scoring_mode: str = Field("drift", description="Scoring mode: drift, drift_faiss, or drift_overlap")
    threshold: float = Field(0.05, ge=0.0, le=1.0, description="Minimum influence score to explain")
    filter_articles: Optional[List[str]] = Field(
        None,
        description="Optional list of article_id strings to restrict ablation"
    )

# --- Embedding endpoint ---
@app.post("/embed")
async def embed():
    try:
        # Step 1: chunk → embeddings in DuckDB
        subprocess.run(
            ["python", "embedding/embedder.py"],
            check=True
        )
        # Step 2: dump to .npy + .pkl
        subprocess.run(
            ["python", "embedding/dump_embeddings.py"],
            check=True
        )

        return {
            "status": "ok",
            "embeddings_path": "data/processed/embeddings.npy",
            "metadata_path": "data/processed/metadata.pkl"
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

# --- Ablation + Explanation endpoint ---
@app.post("/process")
async def process(req: ProcessRequest):
    # sanitize query for filenames
    # sanitized = (
    #     req.query
    #     .lower()
    #     .replace(" ", "_")
    #     .replace("?", "")
    #     .replace("/", "_")
    # )
    sanitized = sanitize_filename(req.query)

    # build common paths
    emb_path = "data/processed/embeddings.npy"
    meta_path = "data/processed/metadata.pkl"
    attr_dir = "data/processed/output/attribution_results"
    exp_dir  = "data/processed/output/explanation_results"
    os.makedirs(attr_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    attr_file = os.path.join(attr_dir, f"{sanitized}.json")
    exp_file  = os.path.join(exp_dir, f"{sanitized}_explanation.json")

    # 1) Run ablation
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

    # 2) Run explanation
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

    return {
        "status": "ok",
        "attribution_result": attr_file,
        "explanation_result": exp_file
    }