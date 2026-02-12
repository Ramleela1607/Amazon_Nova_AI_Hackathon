import time
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

# ---------- Page ----------
st.set_page_config(page_title="Smart Document Copilot", layout="wide")

# ---------- Premium UI (background + cards) ----------
st.markdown("""
<style>
/* Full app background */
.stApp {
  background: radial-gradient(circle at 20% 10%, rgba(99,102,241,0.25), transparent 35%),
              radial-gradient(circle at 80% 20%, rgba(16,185,129,0.20), transparent 40%),
              linear-gradient(180deg, #0b1020 0%, #070a12 100%);
  color: #e5e7eb;
}

/* Sidebar style */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* Containers spacing */
.block-container { padding-top: 1.2rem; }

/* Metrics as cards */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 14px;
  border-radius: 16px;
}

/* Buttons */
.stButton button {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.06);
}
.stButton button:hover {
  border: 1px solid rgba(99,102,241,0.5);
}

/* Expanders */
div[data-testid="stExpander"] {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
  border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("📄 Smart Document Copilot")
st.markdown(
    "<div style='opacity:0.9'>Amazon Nova on Bedrock • Multimodal RAG • Evidence • Compare • PDF Report &nbsp; <b>#AmazonNova</b></div>",
    unsafe_allow_html=True
)


# ---------- Helpers ----------
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
        safe = (body or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))

    path = f"/tmp/{filename}"
    doc = SimpleDocTemplate(path)
    doc.build(story)

    with open(path, "rb") as f:
        return f.read()


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")
st.sidebar.button("🔄 Reset / New session", on_click=reset_session, use_container_width=True)

mode = st.sidebar.radio("Mode", ["Single Document", "Compare Two Documents"])

chunk_size = st.sidebar.slider("Chunk size (chars)", 400, 2000, 1000, 100)
overlap = st.sidebar.slider("Overlap (chars)", 0, 400, 150, 25)
top_k = st.sidebar.slider("Top-K sources", 2, 8, 4, 1)

st.sidebar.markdown("---")
st.sidebar.caption("✅ Demo tips:\n- Ask about key results\n- Ask limitations\n- Ask future work\n- Show Evidence + Sources\n- Download PDF report")


# ---------- Session init ----------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []


# =======================
# Mode: Single Document
# =======================
if mode == "Single Document":
    uploaded = st.file_uploader("📤 Upload a PDF", type=["pdf"])

    if uploaded is None:
        st.info("Upload a PDF → Build Index → Start chatting. Use **Reset** anytime to start fresh.")
        st.stop()

    full_text = extract_text_from_pdf(uploaded)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    # Premium metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", len(chunks))
    c2.metric("Top-K", top_k)
    c3.metric("Chunk size", chunk_size)
    c4.metric("Mode", "Single")

    with st.expander("📄 Document preview", expanded=False):
        st.text_area("Extracted text (preview)", full_text[:8000], height=240)

    st.subheader("1) Index the document")
    if "single_rag" not in st.session_state:
        st.session_state.single_rag = None

    a, b = st.columns([1, 1])
    with a:
        if st.button("🚀 Build Index (Nova embeddings → FAISS)", type="primary", use_container_width=True):
            with st.spinner("Building index..."):
                rag = RagIndex(dim=1024)
                rag.add_chunks(chunks)
                st.session_state.single_rag = rag
            st.success("Index ready ✅")

    with b:
        st.info("Once indexed, chat below. Answers include Evidence + Sources. Response time is shown at bottom.")

    st.divider()

    st.subheader("2) Extract key fields (JSON)")
    doc_type = st.selectbox("Document type", DOC_TYPES, index=0)
    if st.button("🧾 Extract key fields as JSON"):
        with st.spinner("Extracting..."):
            out = extract_fields_json(full_text, doc_type=doc_type)
        st.code(out, language="json")

    st.divider()

    st.subheader("3) Chat with your document")

    if st.session_state.single_rag is None:
        st.warning("Build the index first to enable chat.")
        st.stop()

    # Render chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask something… (e.g., What are the key results and what do they imply?)")

    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.spinner("Retrieving sources..."):
            hits, scores = st.session_state.single_rag.search(user_q, k=top_k)
            ctx = [h.text for h in hits]

        with st.spinner("Thinking with Nova Lite..."):
            start_time = time.time()
            ans = ask_with_evidence(user_q, ctx)
            response_time = round(time.time() - start_time, 2)

        answer_text = ans.get("answer", "")
        evidence_text = ans.get("evidence", "")

        with st.chat_message("assistant"):
            st.markdown(answer_text)

            if evidence_text.strip():
                with st.expander("📌 Evidence (exact quotes)"):
                    st.markdown(evidence_text)

            st.markdown("**Retrieved sources**")
            for i, (h, s) in enumerate(zip(hits, scores), start=1):
                with st.expander(f"Source {i} • score {s:.3f} • chunk #{h.chunk_id}"):
                    st.write(h.text[:2000])

            st.markdown("---")
            st.caption(f"⏱ Average response time: {response_time} sec")

        st.session_state.chat.append({"role": "assistant", "content": answer_text})
        st.session_state.qa_log.append({
            "question": user_q,
            "answer": answer_text,
            "evidence": evidence_text,
        })

    st.divider()

    st.subheader("4) Export report (PDF)")
    if st.button("📄 Generate PDF report"):
        auto_type = detect_doc_type(full_text)

        qa_text = ""
        for i, item in enumerate(st.session_state.qa_log[-10:], start=1):
            qa_text += f"{i}. Q: {item['question']}\nA: {item['answer']}\n"
            if item.get("evidence"):
                qa_text += f"Evidence:\n{item['evidence']}\n"
            qa_text += "\n"

        sections = [
            ("Document type (auto)", auto_type),
            ("Document preview (first 1200 chars)", full_text[:1200]),
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


# =======================
# Mode: Compare Two Docs
# =======================
else:
    st.subheader("🆚 Compare Two PDFs")

    c1, c2 = st.columns(2)
    with c1:
        up_a = st.file_uploader("Upload PDF (Doc A)", type=["pdf"], key="docA")
    with c2:
        up_b = st.file_uploader("Upload PDF (Doc B)", type=["pdf"], key="docB")

    if up_a is None or up_b is None:
        st.info("Upload both PDFs → Build indexes → Ask a comparison question.")
        st.stop()

    text_a = extract_text_from_pdf(up_a)
    text_b = extract_text_from_pdf(up_b)

    chunks_a = chunk_text(text_a, chunk_size=chunk_size, overlap=overlap)
    chunks_b = chunk_text(text_b, chunk_size=chunk_size, overlap=overlap)

    m1, m2, m3 = st.columns(3)
    m1.metric("Doc A chunks", len(chunks_a))
    m2.metric("Doc B chunks", len(chunks_b))
    m3.metric("Mode", "Compare")

    with st.expander("Doc A preview", expanded=False):
        st.text_area("Doc A (preview)", text_a[:5000], height=220)
    with st.expander("Doc B preview", expanded=False):
        st.text_area("Doc B (preview)", text_b[:5000], height=220)

    if "rag_a" not in st.session_state:
        st.session_state.rag_a = None
    if "rag_b" not in st.session_state:
        st.session_state.rag_b = None

    st.subheader("1) Build indexes for both documents")
    if st.button("🚀 Build indexes A & B", type="primary"):
        with st.spinner("Building index A..."):
            rag_a = RagIndex(dim=1024)
            rag_a.add_chunks(chunks_a)
        with st.spinner("Building index B..."):
            rag_b = RagIndex(dim=1024)
            rag_b.add_chunks(chunks_b)

        st.session_state.rag_a = rag_a
        st.session_state.rag_b = rag_b
        st.success("Both indexes ready ✅")

    if st.session_state.rag_a is None or st.session_state.rag_b is None:
        st.warning("Build indexes first.")
        st.stop()

    st.divider()
    st.subheader("2) Ask a comparison question")

    q = st.text_input("Comparison question", placeholder="e.g., Which document shows stronger results and why?")
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

        st.session_state.last_compare = out
        st.session_state.last_compare_q = q

    st.divider()
    st.subheader("3) Export comparison report (PDF)")

    if st.button("📄 Generate comparison PDF"):
        compare_out = st.session_state.get("last_compare", "Run a comparison first to include output here.")
        compare_q = st.session_state.get("last_compare_q", "(no question yet)")

        sections = [
            ("Comparison question", compare_q),
            ("Comparison output", compare_out),
            ("Doc A preview (first 1200 chars)", text_a[:1200]),
            ("Doc B preview (first 1200 chars)", text_b[:1200]),
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
