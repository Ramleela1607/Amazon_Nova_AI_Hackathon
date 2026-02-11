import os
import json
import streamlit as st
from pypdf import PdfReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from rag_index import RagIndex
from bedrock_utils import (
    ask_with_evidence,
    extract_fields_json,
    detect_doc_type,
    compare_docs,
    DOC_TYPES,
)

st.set_page_config(page_title="Smart Document Copilot", layout="wide")
st.title("📄 Smart Document Copilot")
st.caption("Amazon Nova (Bedrock): Multimodal RAG + Evidence + Compare + PDF Report  #AmazonNova")


def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    texts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        texts.append(f"\n\n--- Page {i+1} ---\n{page_text}")
    return "".join(texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def make_pdf_report(filename: str, title: str, sections: list[tuple[str, str]]) -> bytes:
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(title, styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))

    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))
        # Escape minimal HTML issues
        safe = (body or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))

    path = f"/tmp/{filename}"
    doc = SimpleDocTemplate(path)
    doc.build(story)

    with open(path, "rb") as f:
        return f.read()


# ---------- Sidebar settings ----------
st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk size (chars)", 400, 2000, 1000, 100)
overlap = st.sidebar.slider("Overlap (chars)", 0, 400, 150, 25)
top_k = st.sidebar.slider("Top-K sources", 2, 8, 4, 1)

mode = st.sidebar.radio("Mode", ["Single Document", "Compare Two Documents"])

st.divider()

# ---------- Session state ----------
if "single_rag" not in st.session_state:
    st.session_state.single_rag = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []  # list of dicts


# =======================
# MODE 1: Single Document
# =======================
if mode == "Single Document":
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None:
        st.info("Upload a PDF to begin.")
        st.stop()

    full_text = extract_text_from_pdf(uploaded)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    st.session_state.doc_text = full_text
    st.session_state.doc_chunks = chunks

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Extracted text (preview)")
        st.text_area("Preview", full_text[:6000], height=300)
    with c2:
        st.subheader("Chunks")
        st.write(f"Total chunks: **{len(chunks)}**")
        idx = st.number_input("Preview chunk #", 1, len(chunks), 1)
        st.text_area("Chunk preview", chunks[idx - 1], height=300)

    st.subheader("1) Build Index (Nova embeddings → FAISS)")
    if st.button("🚀 Build Index", type="primary"):
        with st.spinner("Building index..."):
            rag = RagIndex(dim=1024)
            rag.add_chunks(chunks)
            st.session_state.single_rag = rag
        st.success("Index built!")

    if st.session_state.single_rag is None:
        st.warning("Build the index first.")
        st.stop()

    st.divider()

    # -------- Doc type + JSON extraction --------
    st.subheader("2) Extract key fields (JSON)")
    doc_type = st.selectbox("Document type", DOC_TYPES, index=0, help="Choose Auto to detect doc type.")
    if st.button("🧾 Extract key fields as JSON"):
        with st.spinner("Extracting..."):
            out = extract_fields_json(full_text, doc_type=doc_type)
        st.code(out, language="json")

    st.divider()

    # -------- Q&A with evidence --------
    st.subheader("3) Ask questions (with evidence)")
    question = st.text_input("Your question", placeholder="e.g., What are the key results? What is the due date? What are main risks?")

    if st.button("💬 Ask"):
        if not question.strip():
            st.error("Please type a question.")
            st.stop()

        with st.spinner("Retrieving sources..."):
            hits, scores = st.session_state.single_rag.search(question, k=top_k)
            ctx = [h.text for h in hits]

        with st.spinner("Answering with Nova Lite + evidence..."):
            ans = ask_with_evidence(question, ctx)

        st.session_state.qa_log.append({
            "question": question,
            "answer": ans.get("answer", ""),
            "sources_used": ans.get("sources_used", []),
            "evidence": ans.get("evidence", []),
        })

        st.markdown("### ✅ Answer")
        st.write(ans.get("answer", ""))

        # Evidence cards
        st.markdown("### 🧾 Evidence snippets")
        ev = ans.get("evidence", [])
        if not ev:
            st.info("No evidence snippets returned (answer may be 'I don't know').")
        else:
            for item in ev:
                st.write(f"**Source {item.get('source')}**: “{item.get('quote')}”")

        # Sources expanders
        st.markdown("### 🔎 Retrieved sources")
        for i, (h, s) in enumerate(zip(hits, scores), start=1):
            with st.expander(f"Source {i} (chunk #{h.chunk_id}) — score {s:.3f}"):
                st.write(h.text[:2000])

    st.divider()

    # -------- Export PDF report --------
    st.subheader("4) Export report (PDF)")
    if st.button("📄 Generate PDF report"):
        # Include last extraction if you want; here we include Q&A log + summary
        auto_type = detect_doc_type(full_text)
        qa_text = ""
        for i, item in enumerate(st.session_state.qa_log[-10:], start=1):
            qa_text += f"{i}. Q: {item['question']}\nA: {item['answer']}\nSources used: {item.get('sources_used', [])}\n\n"

        sections = [
            ("Document type (auto)", auto_type),
            ("Document summary (first 1200 chars)", full_text[:1200]),
            ("Recent Q&A (up to last 10)", qa_text or "No Q&A yet."),
        ]
        pdf_bytes = make_pdf_report(
            filename="smart_doc_copilot_report.pdf",
            title="Smart Document Copilot Report",
            sections=sections,
        )
        st.download_button(
            "⬇️ Download report",
            data=pdf_bytes,
            file_name="smart_doc_copilot_report.pdf",
            mime="application/pdf",
        )


# ===========================
# MODE 2: Compare Two Documents
# ===========================
else:
    st.subheader("Compare Two Documents")
    colA, colB = st.columns(2)

    with colA:
        up_a = st.file_uploader("Upload PDF (Doc A)", type=["pdf"], key="docA")
    with colB:
        up_b = st.file_uploader("Upload PDF (Doc B)", type=["pdf"], key="docB")

    if up_a is None or up_b is None:
        st.info("Upload both PDFs to compare.")
        st.stop()

    text_a = extract_text_from_pdf(up_a)
    text_b = extract_text_from_pdf(up_b)

    chunks_a = chunk_text(text_a, chunk_size=chunk_size, overlap=overlap)
    chunks_b = chunk_text(text_b, chunk_size=chunk_size, overlap=overlap)

    st.write(f"Doc A chunks: **{len(chunks_a)}** | Doc B chunks: **{len(chunks_b)}**")

    if st.button("🚀 Build indexes for A & B", type="primary"):
        with st.spinner("Building index A..."):
            rag_a = RagIndex(dim=1024)
            rag_a.add_chunks(chunks_a)
        with st.spinner("Building index B..."):
            rag_b = RagIndex(dim=1024)
            rag_b.add_chunks(chunks_b)
        st.session_state.rag_a = rag_a
        st.session_state.rag_b = rag_b
        st.session_state.text_a = text_a
        st.session_state.text_b = text_b
        st.success("Both indexes built!")

    if "rag_a" not in st.session_state or "rag_b" not in st.session_state:
        st.warning("Build indexes first.")
        st.stop()

    st.divider()
    st.subheader("Ask a comparison question")
    q = st.text_input("Comparison question", placeholder="e.g., Which candidate is better for backend role and why?")
    if st.button("🆚 Compare"):
        if not q.strip():
            st.error("Type a comparison question.")
            st.stop()

        with st.spinner("Retrieving from Doc A..."):
            hits_a, _ = st.session_state.rag_a.search(q, k=top_k)
            ctx_a = [h.text for h in hits_a]

        with st.spinner("Retrieving from Doc B..."):
            hits_b, _ = st.session_state.rag_b.search(q, k=top_k)
            ctx_b = [h.text for h in hits_b]

        with st.spinner("Comparing with Nova Lite..."):
            out = compare_docs(q, ctx_a, ctx_b, label_a="Doc A", label_b="Doc B")

        st.markdown("### ✅ Comparison result")
        st.write(out)

    st.divider()
    st.subheader("Export comparison report (PDF)")
    if st.button("📄 Generate comparison PDF"):
        sections = [
            ("Doc A (preview)", st.session_state.text_a[:1200]),
            ("Doc B (preview)", st.session_state.text_b[:1200]),
            ("Comparison note", "Use the Compare button first to generate a comparison output."),
        ]
        pdf_bytes = make_pdf_report(
            filename="smart_doc_copilot_compare_report.pdf",
            title="Smart Document Copilot – Comparison Report",
            sections=sections,
        )
        st.download_button(
            "⬇️ Download comparison report",
            data=pdf_bytes,
            file_name="smart_doc_copilot_compare_report.pdf",
            mime="application/pdf",
        )
