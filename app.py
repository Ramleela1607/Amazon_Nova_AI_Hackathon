import io
import time
import json
import hashlib
import re
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from pypdf import PdfReader
from PIL import Image

from rag_index import RagIndex
from bedrock_utils import (
    ask_with_evidence,
    extract_fields_json,
    compare_docs,
    DOC_TYPES,
    recommend_rag_settings,
    nova_image_to_text,
    suggest_questions,
    generate_dashboard_insights_dynamic,
)

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="🚀 Insight Engine AI", layout="wide")

st.title("🚀 Insight Engine AI")
st.caption("Upload ANY document • Multilingual • Smart Executive Insights")

# ------------------------------------------------------------
# CLEAN OCR NOISE (fix newspaper timestamp repetition)
# ------------------------------------------------------------
def clean_ocr_noise(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    cleaned = []
    seen = set()
    for line in lines:
        ln = line.strip()
        if not ln:
            continue
        if ln in seen:
            continue
        if re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", ln):
            continue
        seen.add(ln)
        cleaned.append(ln)
    return "\n".join(cleaned)

# ------------------------------------------------------------
# FILE EXTRACTION
# ------------------------------------------------------------
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_image_text(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return nova_image_to_text(buf.getvalue(), image_format="png")

def extract_excel_text(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    previews = []
    text_blocks = []
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        previews.append((sheet, df))
        text_blocks.append(df.to_csv(index=False))
    return "\n".join(text_blocks), previews

def extract_docx_text(uploaded_file):
    try:
        import docx
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        return ""

def extract_ppt_text(uploaded_file):
    try:
        from pptx import Presentation
        prs = Presentation(uploaded_file)
        text = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
        return "\n".join(text)
    except:
        return ""

# ------------------------------------------------------------
# DASHBOARD RENDER (ONLY SHOW IF DATA EXISTS)
# ------------------------------------------------------------
def render_dashboard(data: Dict[str, Any]):
    if not data:
        return

    if data.get("summary"):
        st.subheader("Executive Summary")
        st.write(data["summary"])

    if data.get("kpis"):
        st.subheader("Key Metrics")
        cols = st.columns(3)
        for i, k in enumerate(data["kpis"]):
            with cols[i % 3]:
                st.metric(k["label"], k["value"])

    if data.get("derived_insights"):
        st.subheader("Insights")
        for d in data["derived_insights"]:
            st.write("•", d)

    if data.get("charts"):
        st.subheader("Visual Analysis")
        for chart in data["charts"]:
            df = pd.DataFrame(chart["data"])
            if not df.empty:
                st.bar_chart(df.set_index("x"))

    if data.get("risks"):
        st.subheader("Risks")
        for r in data["risks"]:
            st.write("⚠", r)

    if data.get("next_actions"):
        st.subheader("Recommended Actions")
        for a in data["next_actions"]:
            st.write("✓", a)

# ------------------------------------------------------------
# UPLOAD SECTION
# ------------------------------------------------------------
st.subheader("Upload Document")

uploaded_file = st.file_uploader(
    "Upload PDF / Image / Excel / Word / PPT",
    type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "csv", "docx", "pptx"]
)

if not uploaded_file:
    st.stop()

file_type = uploaded_file.name.split(".")[-1].lower()

text = ""
excel_preview = []

with st.spinner("Extracting content..."):
    if file_type == "pdf":
        text = extract_pdf_text(uploaded_file)
    elif file_type in ["png", "jpg", "jpeg"]:
        text = extract_image_text(uploaded_file)
    elif file_type in ["xlsx", "xls", "csv"]:
        text, excel_preview = extract_excel_text(uploaded_file)
    elif file_type == "docx":
        text = extract_docx_text(uploaded_file)
    elif file_type == "pptx":
        text = extract_ppt_text(uploaded_file)

text = clean_ocr_noise(text)

if not text.strip():
    st.warning("No readable text detected.")
    st.stop()

# ------------------------------------------------------------
# SHOW EXTRACTED TEXT DOWNLOAD
# ------------------------------------------------------------
st.download_button(
    "Download Extracted Text",
    text,
    file_name="extracted_text.txt"
)

# ------------------------------------------------------------
# SHOW EXCEL PREVIEW IF EXISTS
# ------------------------------------------------------------
if excel_preview:
    st.subheader("Excel Preview")
    for sheet, df in excel_preview:
        st.write(f"Sheet: {sheet}")
        st.dataframe(df.head(50))

# ------------------------------------------------------------
# GENERATE DASHBOARD (NOVA DECIDES)
# ------------------------------------------------------------
doc_hash = hashlib.md5(text[:5000].encode()).hexdigest()
cache_key = f"dashboard_{doc_hash}"

if cache_key not in st.session_state:
    with st.spinner("Analyzing with Nova..."):
        st.session_state[cache_key] = generate_dashboard_insights_dynamic(text)

dashboard_data = st.session_state[cache_key]

render_dashboard(dashboard_data)

# ------------------------------------------------------------
# SUGGESTED QUESTIONS (AUTO)
# ------------------------------------------------------------
st.subheader("Ask About This Document")

if "suggested" not in st.session_state:
    st.session_state["suggested"] = suggest_questions(text, n=5)

for q in st.session_state["suggested"]:
    if st.button(q):
        st.session_state["user_question"] = q

user_q = st.text_input("Type your question")

if "user_question" in st.session_state:
    user_q = st.session_state.pop("user_question")

if user_q:
    with st.spinner("Generating answer..."):
        answer = ask_with_evidence(user_q, [text])
    st.subheader("Answer")
    st.write(answer.get("answer", "No answer available."))
