# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Gist Attribution Demo", layout="wide")
st.title("Gist Attribution Engine")
st.write("Enter a query to see the LLM answer, attribution metrics, and explanations.")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K Chunks", min_value=1, value=5)
    threshold = st.slider("Influence Threshaold", 0.0, 1.0, 0.05)
    scoring_mode = st.selectbox("Scoring Mode", ["drift", "drift_faiss", "drift_overlap"])

query = st.text_input("Your Query", placeholder="Why did Elon Musk distance himself?")
run = st.button("Run Attribution")

if run and query:
    with st.spinner("Processing query..."):
        payload = {
            "query": query,
            "top_k": top_k,
            "scoring_mode": scoring_mode,
            "threshold": threshold
        }
        try:
            res = requests.post(f"{API_BASE}/process", json=payload)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            st.error(f"API request failed: {e}")
            st.stop()

    # Answer
    st.subheader("Answer")
    st.write(data["attribution"]["full_response"])

    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks Retrieved", data["metrics"]["top_k_chunks"])
    c2.metric("Explained Chunks", data["metrics"]["chunks_above_threshold"])
    c3.metric("Threshold", f"{data['metrics']['threshold_used']}")

    # Top Documents Chart
    st.subheader("Top Documents by Influence")
    docs = []
    for doc_id, inf in data["metrics"]["influence_by_doc"].items():
        # find first metadata entry for this article
        match = next((m for m in data["chunk_metadata"].values() if m["article_id"] == doc_id), {})
        docs.append({
            "title": match.get("title", doc_id),
            "source": match.get("source", "unknown"),
            "influence": inf
        })
    docs_df = pd.DataFrame(docs).sort_values("influence", ascending=False)
    st.table(docs_df)

    # Most Influential Chunk
    st.subheader("⭐ Most Influential Chunk")
    mic = data["metrics"]["most_influential_chunk"]
    st.write(f"**Influence:** {mic['influence']}  ")
    st.write(mic["text"])

    # Explained Chunks
    st.subheader("Explained Chunks")
    for chunk in data["explanation"]["explained_chunks"]:
        meta = data["chunk_metadata"].get(chunk["chunk_id"], {})
        label = f"Chunk #{chunk['chunk_id'].split('_')[-1]} from {meta.get('source','unknown')} (Influence: {chunk['influence']:.4f})"
        with st.expander(label):
            st.write(chunk["text_snippet"])
            expl = next((e for e in data["explanation"]["explanations"] if e["chunk_id"] == chunk["chunk_id"]), {})
            if expl:
                st.markdown(f"**Explanation:** {expl.get('explanation','')}")

    # Document Influence Breakdown
    st.subheader("Document Influence Breakdown")
    doc_rows = []
    for doc_id, inf in data["metrics"]["influence_by_doc"].items():
        match = next((m for m in data["chunk_metadata"].values() if m["article_id"] == doc_id), {})
        doc_rows.append({
            "title": match.get("title", doc_id),
            "source": match.get("source", "unknown"),
            "influence": inf
        })
    doc_df = pd.DataFrame(doc_rows).sort_values("influence", ascending=False)
    st.table(doc_df)

    # Chunk Influence Scores
    st.subheader("🔢 Chunk Influence Scores")
    chunk_rows = []
    for chunk_id, inf in data["metrics"]["influence_by_chunk"].items():
        meta = data["chunk_metadata"].get(chunk_id, {})
        chunk_rows.append({
            "chunk": f"#{chunk_id.split('_')[-1]}",
            "source": meta.get("source", "unknown"),
            "influence": inf
        })
    chunk_df = pd.DataFrame(chunk_rows).sort_values("influence", ascending=False)
    st.table(chunk_df)
