"""
Streamlit UI — AI Book Detection System

Run with:
  streamlit run streamlit_app.py
"""

import os
import io
import json
import tempfile

import httpx
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8001/api/v1")
# On Streamlit Cloud, set API_BASE_URL in secrets.toml or environment variables

st.set_page_config(
    page_title="AI Book Detector",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API Base URL", value=API_BASE)
    download_report = st.checkbox("Download PDF Report", value=False)
    st.markdown("---")
    st.markdown(
        "**How it works**\n"
        "1. Upload a PDF book (≤ 50 MB)\n"
        "2. Text is extracted page-by-page\n"
        "3. Chunked into 300–500 word blocks\n"
        "4. Each chunk is scored with heuristics + LLM\n"
        "5. Results are aggregated into a final score"
    )

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🔍 AI Generated Content Detector")
st.caption("Upload a PDF book (200–350 pages) to analyse how much is AI-generated.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    help="Max 50 MB",
)

if uploaded_file is not None:
    st.info(f"File: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    if st.button("🚀 Analyse Book", type="primary"):
        file_bytes = uploaded_file.read()

        endpoint = f"{api_url}/analyze-book/report" if download_report else f"{api_url}/analyze-book"

        with st.spinner("Analysing… this may take a few minutes for large books."):
            try:
                response = httpx.post(
                    endpoint,
                    files={"file": (uploaded_file.name, file_bytes, "application/pdf")},
                    timeout=600.0,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                st.error(f"API error {exc.response.status_code}: {exc.response.text}")
                st.stop()
            except Exception as exc:
                st.error(f"Connection error: {exc}")
                st.stop()

        # --- PDF report mode ---
        if download_report:
            st.success("Report generated!")
            st.download_button(
                label="⬇️ Download PDF Report",
                data=response.content,
                file_name="ai_detection_report.pdf",
                mime="application/pdf",
            )
            st.stop()

        # --- JSON result mode ---
        result = response.json()

        # -------- KPI row --------
        pct = result["overall_ai_percentage"]
        conf = result["confidence"]
        variance = result["score_variance"]
        stats = result["processing_stats"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            colour = "red" if pct >= 70 else ("orange" if pct >= 40 else "green")
            st.metric("Overall AI Likelihood", f"{pct:.1f}%")
        with col2:
            st.metric("Confidence", conf)
        with col3:
            st.metric("Score Variance", f"{variance:.4f}")
        with col4:
            st.metric("Chunks Analysed", stats["total_chunks"])

        st.markdown("---")

        # -------- Gauge chart --------
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            title={"text": "AI-Generated %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "crimson"},
                "steps": [
                    {"range": [0, 40], "color": "#d5f5e3"},
                    {"range": [40, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": pct,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # -------- Chunk heatmap --------
        st.subheader("📊 Chunk-Level AI Scores")
        chunks = result["chunks"]
        df = pd.DataFrame(chunks)
        df["label"] = df["label"]

        fig_bar = px.bar(
            df,
            x="chunk_id",
            y="ai_score",
            color="ai_score",
            color_continuous_scale=["green", "yellow", "red"],
            range_color=[0, 1],
            labels={"chunk_id": "Chunk ID", "ai_score": "AI Score"},
            hover_data=["page", "word_count", "label", "text_snippet"],
        )
        fig_bar.add_hline(y=0.70, line_dash="dash", line_color="red", annotation_text="High-Risk Threshold")
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

        # -------- Page-level heatmap --------
        st.subheader("🗺️ Page-Level Heatmap")
        page_df = df.groupby("page")["ai_score"].mean().reset_index()
        page_df.columns = ["page", "avg_ai_score"]

        fig_heat = px.density_heatmap(
            page_df,
            x="page",
            y=["avg_ai_score"],
            color_continuous_scale="RdYlGn_r",
            labels={"page": "Page", "avg_ai_score": "Avg AI Score"},
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # -------- High-risk sections --------
        high_risk = result["high_risk_sections"]
        if high_risk:
            st.subheader(f"🚨 High-Risk Sections ({len(high_risk)})")
            for sec in high_risk[:10]:
                with st.expander(f"Page {sec['page']} — Score: {sec['ai_score']:.2f}"):
                    st.write(sec["text_snippet"])
        else:
            st.success("✅ No high-risk sections detected.")

        # -------- Full chunk table --------
        with st.expander("📋 Full Chunk Data"):
            st.dataframe(df[[
                "chunk_id", "page", "word_count", "ai_score",
                "llm_score", "heuristic_score", "label", "text_snippet"
            ]], use_container_width=True)

        # -------- Processing stats --------
        with st.expander("⚙️ Processing Statistics"):
            st.json(stats)

        # -------- Raw JSON --------
        with st.expander("🔧 Raw JSON Response"):
            st.json(result)
