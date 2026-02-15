# ============================================================
# InsightForge AI — Fully Nova Controlled
# Multilingual • OCR • RAG • Executive Intelligence
# ============================================================

import io
import time
import json
import hashlib
import re
from typing import List, Dict, Any, Tuple

from langdetect import detect
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from PIL import Image

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from rag_index import RagIndex
from bedrock_utils import (
    ask_with_evidence,
    extract_fields_json,
    compare_docs,
    DOC_TYPES,
    recommend_rag_settings,
    nova_image_to_text,
    generate_dashboard_insights_dynamic,
    suggest_questions
)

# ============================================================
# PAGE CONFIG + CLEAN UI
# ============================================================

st.set_page_config(page_title="🚀 InsightForge AI", layout="wide")

st.markdown("""
<style>
header {visibility: hidden;}
.block-container {padding-top: 1rem;}
.stApp {
    background: linear-gradient(135deg, #f4f7ff, #eef2ff);
}
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}
.kpi-title {
    font-weight: 600;
    font-size: 0.9rem;
    opacity: 0.7;
}
.kpi-value {
    font-weight: 800;
    font-size: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 InsightForge AI")
st.caption("Multilingual Executive Intelligence Engine powered by Amazon Nova")

# ============================================================
# UTILITIES
# ============================================================

def chunk_text(text: str, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def detect_language(text: str) -> str:
    try:
        if len(text) < 50:
            return "English"
        code = detect(text)
        mapping = {
            "ta": "Tamil",
            "hi": "Hindi",
            "te": "Telugu",
            "ml": "Malayalam",
            "kn": "Kannada",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "zh-cn": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "ru": "Russian",
            "en": "English"
        }
        return mapping.get(code.lower(), "English")
    except:
        return "English"


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text() or ""
        text += t + "\n"
    return text


def extract_text_from_pdf_with_ocr(uploaded_pdf):
    pdf_bytes = uploaded_pdf.getvalue()
    text = extract_text_from_pdf(io.BytesIO(pdf_bytes))

    # If scanned
    if len(text.strip()) < 200:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for i in range(min(5, doc.page_count)):
                pix = doc[i].get_pixmap(dpi=180)
                img_bytes = pix.tobytes("png")
                ocr = nova_image_to_text(img_bytes, image_format="png")
                text += "\n" + ocr
        except:
            pass

    # Clean repeated timestamps (newspaper issue)
    text = re.sub(r"(20\d{2}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s*){2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def make_pdf_report(title: str, sections: List[Tuple[str, str]]) -> bytes:
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Heading1"]), Spacer(1, 0.2 * inch)]
    for h, b in sections:
        story.append(Paragraph(h, styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(b.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))
    path = "/tmp/report.pdf"
    SimpleDocTemplate(path).build(story)
    return open(path, "rb").read()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")
mode = st.sidebar.radio("Mode", ["Single Document", "Compare Two Documents"])

# ============================================================
# SINGLE DOCUMENT MODE
# ============================================================

if mode == "Single Document":

    uploaded_file = st.file_uploader(
        "Upload PDF / Image",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if not uploaded_file:
        st.stop()

    # Extract text
    if uploaded_file.name.lower().endswith(".pdf"):
        with st.spinner("Extracting PDF..."):
            full_text = extract_text_from_pdf_with_ocr(uploaded_file)
    else:
        img_bytes = uploaded_file.getvalue()
        full_text = nova_image_to_text(img_bytes, image_format="png")

    if not full_text:
        st.warning("No readable text found.")
        st.stop()

    # Detect language
    doc_language = detect_language(full_text)
    st.session_state["doc_language"] = doc_language
    st.caption(f"Detected Language: {doc_language}")

    # ============================================================
    # NOVA EXECUTIVE DASHBOARD
    # ============================================================

    st.subheader("📊 Executive Dashboard")

    with st.spinner("Nova analyzing document..."):
        dashboard = generate_dashboard_insights_dynamic(
            full_text,
            output_language=doc_language
        )

    if not dashboard:
        st.warning("Nova could not generate dashboard.")
        st.stop()

    summary = dashboard.get("summary")
    kpis = dashboard.get("kpis")
    charts = dashboard.get("charts")
    risks = dashboard.get("risks")
    actions = dashboard.get("next_actions")

    if summary:
        st.markdown("### Executive Summary")
        st.write(summary)

    if kpis:
        st.markdown("### Key Metrics")
        cols = st.columns(3)
        for i, k in enumerate(kpis[:9]):
            with cols[i % 3]:
                st.markdown(
                    f"<div class='kpi-card'><div class='kpi-title'>{k['label']}</div><div class='kpi-value'>{k['value']}</div></div>",
                    unsafe_allow_html=True
                )

    if charts:
        st.markdown("### Charts")
        for ch in charts:
            df = pd.DataFrame(ch.get("data", []))
            if "x" in df and "y" in df:
                st.bar_chart(df.set_index("x"))

    if risks:
        st.markdown("### Risks")
        for r in risks:
            st.write("•", r)

    if actions:
        st.markdown("### Recommended Actions")
        for a in actions:
            st.write("•", a)

    # PDF Download
    if st.button("Download Executive Report"):
        sections = [
            ("Summary", summary or ""),
            ("Risks", "\n".join(risks or [])),
            ("Actions", "\n".join(actions or []))
        ]
        pdf_bytes = make_pdf_report("InsightForge AI Report", sections)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="Insight_Report.pdf",
            mime="application/pdf"
        )

    st.divider()

    # ============================================================
    # RAG AUTO BUILD
    # ============================================================

    rec = recommend_rag_settings(full_text)
    chunk_size = rec["chunk_size"]
    overlap = rec["overlap"]
    top_k = rec["top_k"]

    chunks = chunk_text(full_text, chunk_size, overlap)

    if "rag" not in st.session_state:
        with st.spinner("Building AI index..."):
            rag = RagIndex(dim=1024)
            rag.add_chunks(chunks)
        st.session_state["rag"] = rag

    # ============================================================
    # NOVA SUGGESTED QUESTIONS (AUTO MULTILINGUAL)
    # ============================================================

    st.subheader("✨ Suggested Questions")

    if "suggested_questions" not in st.session_state:
        with st.spinner("Generating questions..."):
            st.session_state["suggested_questions"] = suggest_questions(
                full_text,
                user_interest=f"Generate questions strictly in {doc_language} language.",
                n=6
            )

    questions = st.session_state["suggested_questions"]

    cols = st.columns(3)
    for i, q in enumerate(questions):
        with cols[i % 3]:
            if st.button(q):
                st.session_state["pending_q"] = q

    st.divider()

    # ============================================================
    # CHAT SECTION
    # ============================================================

    st.subheader("💬 Ask Your Own Question")

    pending = st.session_state.pop("pending_q", None)

    if pending:
        query = pending
    else:
        query = st.text_input("Type your question")

    if query:
        with st.spinner("Thinking..."):
            hits, _ = st.session_state["rag"].search(query, k=top_k)
            ctx = [h.text for h in hits]

            answer = ask_with_evidence(
                f"Answer strictly in {doc_language} language.\n\nQuestion: {query}",
                ctx
            )

        st.markdown("### Answer")
        st.write(answer.get("answer"))

        if answer.get("evidence"):
            with st.expander("Evidence"):
                st.write(answer.get("evidence"))

# ============================================================
# COMPARE MODE
# ============================================================

else:

    st.subheader("🆚 Compare Two PDFs")

    file_a = st.file_uploader("Upload Document A", type=["pdf"])
    file_b = st.file_uploader("Upload Document B", type=["pdf"])

    if not file_a or not file_b:
        st.stop()

    text_a = extract_text_from_pdf_with_ocr(file_a)
    text_b = extract_text_from_pdf_with_ocr(file_b)

    q = st.text_input("Comparison Question")

    if st.button("Compare") and q:
        with st.spinner("Nova comparing documents..."):
            result = compare_docs(q, [text_a], [text_b], "Doc A", "Doc B")

        st.write(result)
