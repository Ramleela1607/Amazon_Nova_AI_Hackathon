import time
import streamlit as st
from pypdf import PdfReader
from PIL import Image

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
    textract_image_to_text,
)

# ---------- Page ----------
st.set_page_config(page_title="Smart Document Copilot", layout="wide")

# ---------- Premium UI (light background image) ----------
st.markdown("""
<style>
/* ---------- Light theme + background image ---------- */
.stApp {
  background-image:
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='800'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop stop-color='%23e0f2fe' offset='0'/><stop stop-color='%23fce7f3' offset='0.5'/><stop stop-color='%23ecfccb' offset='1'/></linearGradient></defs><rect width='1200' height='800' fill='url(%23g)'/><path d='M0,520 C220,460 420,610 650,540 C880,470 1020,570 1200,520 L1200,800 L0,800 Z' fill='%23ffffff' fill-opacity='0.55'/><path d='M0,600 C260,540 430,700 700,620 C940,550 1040,660 1200,610' stroke='%2399f6e4' stroke-width='18' stroke-opacity='0.35' fill='none'/><path d='M0,650 C280,590 470,740 760,660 C980,600 1090,700 1200,660' stroke='%23a5b4fc' stroke-width='14' stroke-opacity='0.30' fill='none'/></svg>");
  background-size: cover;
  background-attachment: fixed;
  color: #111827 !important;
}

html, body, [class*="css"]  { color: #111827 !important; }

section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.72) !important;
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(17,24,39,0.08);
}
section[data-testid="stSidebar"] *{ color: #111827 !important; }

.block-container { padding-top: 1.2rem; }

div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(17,24,39,0.10);
  padding: 14px;
  border-radius: 16px;
  box-shadow: 0 8px 22px rgba(17,24,39,0.07);
}

div[data-testid="stExpander"] {
  border-radius: 16px;
  border: 1px solid rgba(17,24,39,0.10);
  background: rgba(255,255,255,0.78);
  box-shadow: 0 8px 22px rgba(17,24,39,0.06);
}

.stButton button {
  border-radius: 14px;
  border: 1px solid rgba(17,24,39,0.12);
  background: rgba(255,255,255,0.85);
  color: #111827 !important;
  font-weight: 600;
}
.stButton button:hover {
  border: 1px solid rgba(99,102,241,0.45);
  box-shadow: 0 8px 20px rgba(99,102,241,0.18);
}

div[data-baseweb="input"] input, textarea {
  background: rgba(255,255,255,0.88) !important;
  color: #111827 !important;
  border-radius: 12px !important;
  border: 1px solid rgba(17,24,39,0.14) !important;
}

div[data-testid="stChatMessage"]{
  background: rgba(255,255,255,0.80);
  border: 1px solid rgba(17,24,39,0.10);
  border-radius: 16px;
  box-shadow: 0 8px 18px rgba(17,24,39,0.05);
}
div[data-testid="stChatMessage"] *{ color: #111827 !important; }

a { color: #1d4ed8 !important; }
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
st.sidebar.caption("✅ Demo tips:\n- Upload PDF OR Image\n- Ask: 'What text is in the image?'\n- Show Evidence + Sources\n- Download PDF report")

# ---------- Session init ----------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "qa_log" not in st.session_state:
    st.session_state.qa_log = []

# =======================
# Mode: Single Document
# =======================
if mode == "Single Document":
    uploaded_pdf = st.file_uploader("📤 Upload a PDF (optional)", type=["pdf"])
    uploaded_img = st.file_uploader("🖼️ Upload an Image (optional)", type=["png", "jpg", "jpeg"])
    user_text = st.text_area("✍️ Paste text / notes (optional)", height=140, placeholder="Paste any text you want the bot to use...")

    if uploaded_pdf is None and uploaded_img is None and not user_text.strip():
        st.info("Upload a PDF or Image or paste text → Build Index → Start chatting.")
        st.stop()

    full_text_parts = []

    # PDF → text
    if uploaded_pdf is not None:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        full_text_parts.append("=== PDF TEXT ===\n" + pdf_text)

    # Image → OCR (Textract) → text
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded image", use_container_width=True)

        with st.spinner("Extracting text from image (Amazon Textract OCR)..."):
            image_bytes = uploaded_img.getvalue()
            ocr_text = textract_image_to_text(image_bytes, region="ap-south-1")

        if ocr_text.strip():
            st.success("Image text extracted ✅")
            with st.expander("🧾 Extracted image text (OCR)", expanded=False):
                st.text_area("OCR text", ocr_text, height=160)

            full_text_parts.append("=== IMAGE OCR TEXT ===\n" + ocr_text)
        else:
            st.warning("No readable text detected in the image.")
            full_text_parts.append(f"=== IMAGE UPLOADED ===\nFilename: {uploaded_img.name}\n(No OCR text found)")

    # User notes
    if user_text.strip():
        full_text_parts.append("=== USER NOTES ===\n" + user_text.strip())

    full_text = "\n\n".join(full_text_parts)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", len(chunks))
    c2.metric("Top-K", top_k)
    c3.metric("Chunk size", chunk_size)
    c4.metric("Mode", "Single")

    with st.expander("📄 Combined text preview", expanded=False):
        st.text_area("Preview", full_text[:8000], height=240)

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
        st.info("Chat below. OCR text from images is included in retrieval. Response time appears at bottom.")

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

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask something… (try: What text is in the image?)")

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
        st.session_state.qa_log.append({"question": user_q, "answer": answer_text, "evidence": evidence_text})

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
            ("Combined text preview (first 1200 chars)", full_text[:1200]),
            ("Recent Q&A (up to last 10)", qa_text or "No Q&A yet."),
        ]
        pdf_bytes = make_pdf_report("smart_doc_copilot_report.pdf", "Smart Document Copilot Report", sections)
        st.download_button("⬇️ Download report", data=pdf_bytes, file_name="smart_doc_copilot_report.pdf", mime="application/pdf")

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
