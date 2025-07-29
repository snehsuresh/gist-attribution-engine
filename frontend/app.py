import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Gist Attribution Demo", layout="wide")
st.title("Gist Attribution Engine")
st.write("Enter a query to see the LLM answer, attribution metrics, and explanations.")

with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top K Chunks", min_value=1, value=5)
    threshold = st.slider("Influence Threshold", 0.0, 1.0, 0.05)
    scoring_mode = st.selectbox("Scoring Mode", ["drift", "drift_faiss", "drift_overlap"])

query = st.text_input("Your Query", placeholder="Why did Elon Musk distance himself?")
run = st.button("Run Attribution")

if run and query:
    st.caption(f"Query: *{query}*")
    payload = {
        "query": query,
        "top_k": top_k,
        "scoring_mode": scoring_mode,
        "threshold": threshold
    }

    # 1. Stream first response token-by-token
    response_area = st.empty()
    response_text = ""
    with st.spinner("Getting initial answer..."):
        try:
            res = requests.post(f"{API_BASE}/first_response", json=payload, stream=True)
            for chunk in res.iter_content(chunk_size=None):
                if chunk:
                    text = chunk.decode()
                    response_text += text
                    response_area.markdown(f"**Answer:** {response_text}")
        except Exception as e:
            st.error(f"Streaming failed: {e}")
            st.stop()

    # 2. Inform user and run ablation
    st.info("Got answer. Now running ablation + influence analysis...")
    with st.spinner("Processing ablation and explanations..."):
        try:
            proc_res = requests.post(f"{API_BASE}/process", json=payload)
            proc_res.raise_for_status()
            data = proc_res.json()
        except Exception as e:
            st.error(f"Attribution failed: {e}")
            st.stop()

    # 3. Display metrics
    st.subheader("Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks Retrieved", data["metrics"]["top_k_chunks"])
    c2.metric("Explained Chunks", data["metrics"]["chunks_above_threshold"])
    c3.metric("Threshold", f"{data['metrics']['threshold_used']}")

    # 4. Prepare dataframes
    docs = []
    for doc_id, inf in data["metrics"]["influence_by_doc"].items():
        match = next((m for m in data["chunk_metadata"].values() if m["article_id"] == doc_id), {})
        docs.append({
            "title": match.get("title", doc_id),
            "source": match.get("source", "unknown"),
            "influence": inf
        })
    docs_df = pd.DataFrame(docs).sort_values("influence", ascending=False).reset_index(drop=True)

    chunk_rows = []
    for chunk_id, inf in data["metrics"]["influence_by_chunk"].items():
        meta = data["chunk_metadata"].get(chunk_id, {})
        chunk_rows.append({
            "chunk": f"#{chunk_id.split('_')[-1]}",
            "source": meta.get("source", "unknown"),
            "influence": inf
        })
    chunk_df = pd.DataFrame(chunk_rows).sort_values("influence", ascending=False).reset_index(drop=True)

    # 5. Full-width tables and document chart
    st.subheader("Top Documents by Influence")
    st.table(docs_df)
    fig1 = px.bar(
        docs_df,
        x="influence",
        y="title",
        color="source",
        orientation="h",
        labels={"influence": "Normalized Influence", "title": "Article Title"},
        title="Document Influence"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Chunk Influence Breakdown")
    st.table(chunk_df)

    # 6. Side-by-side chunk bar chart and pie chart
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(
            chunk_df,
            x="chunk",
            y="influence",
            text="influence",
            labels={"chunk": "Chunk", "influence": "Influence"},
            title="Chunk Influence"
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        source_inf = (
            chunk_df.groupby("source")["influence"]
            .sum()
            .reset_index()
            .sort_values(by="influence", ascending=False)
        )
        fig3 = px.pie(
            source_inf,
            values="influence",
            names="source",
            title="Influence by Source"
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.success("Attribution complete!")
