import io
import time
import hashlib
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
    recommend_rag_settings,
    nova_image_to_text,
    nova_image_insights_brief,   # ✅ NEW: brief insights only
    generate_report_title,
    suggest_questions,
)

# ---------- Page ----------
st.set_page_config(page_title="Smart Document Copilot", layout="wide")

# ---------- Premium UI ----------
st.markdown("""
<style>
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
  background: rgba(255,255,255,0.90);
  color: #111827 !important;
  font-weight: 650;
}
.stButton button:hover {
  border: 1px solid rgba(99,102,241,0.45);
  box-shadow: 0 8px 20px rgba(99,102,241,0.18);
}
div[data-baseweb="input"] input, textarea {
  background: rgba(255,255,255,0.92) !important;
  color: #111827 !important;
  border-radius: 12px !important;
  border: 1px solid rgba(17,24,39,0.14) !important;
}
div[data-testid="stChatMessage"]{
  background: rgba(255,255,255,0.86);
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
        start += max(1, chunk_size - overlap)
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
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    for k in [
        "single_rag", "index_fp",
        "last_doc_fp", "auto_rag_settings",
        "suggest_fp", "suggested_questions",
        "latest_answer", "latest_evidence", "latest_sources",
        "pending_question",
        "compare_fp", "rag_a", "rag_b",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

def build_index_if_needed(full_text: str, chunk_size: int, overlap: int):
    if not full_text or len(full_text.strip()) < 10:
        return

    fp_src = f"{full_text[:25000]}||cs={chunk_size}||ov={overlap}"
    new_fp = hashlib.md5(fp_src.encode("utf-8", errors="ignore")).hexdigest()

    if st.session_state.get("index_fp") == new_fp and isinstance(st.session_state.get("single_rag"), RagIndex):
        return

    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    with st.spinner("🚀 Auto-building index (Nova embeddings → FAISS)..."):
        rag = RagIndex(dim=1024)
        rag.add_chunks(chunks)

    st.session_state["single_rag"] = rag
    st.session_state["index_fp"] = new_fp

# ---------- Session init ----------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Controls")
st.sidebar.button("🔄 Reset / New session", on_click=reset_session, use_container_width=True)

mode = st.sidebar.radio("Mode", ["Single Document", "Compare Two Documents"])

user_interest = st.sidebar.selectbox(
    "User interest",
    ["General", "Finance", "HR/Recruiter", "Legal", "Research/Student", "Operations"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "✅ Demo tips:\n"
    "- Upload a research paper PDF\n"
    "- Upload an invoice image\n"
    "- Click a suggested question\n"
    "- Show Evidence + Sources\n"
    "- Export PDF report\n"
)

# =======================
# Mode: Single Document
# =======================
if mode == "Single Document":
    k = st.session_state.get("uploader_key", 0)

    uploaded_pdf = st.file_uploader("📤 Upload a PDF (optional)", type=["pdf"], key=f"pdf_uploader_{k}")
    uploaded_img = st.file_uploader("🖼️ Upload an Image (optional)", type=["png", "jpg", "jpeg", "webp"], key=f"img_uploader_{k}")
    user_text = st.text_area("✍️ Paste extra text / notes (optional)", height=100)

    if uploaded_pdf is None and uploaded_img is None and not user_text.strip():
        st.info("Upload a PDF or Image (or paste text) → Index builds automatically → Start chatting.")
        st.stop()

    full_text_parts = []
    image_ocr_text = ""
    image_brief_insights = ""

    # PDF
    if uploaded_pdf is not None:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        full_text_parts.append("=== PDF TEXT ===\n" + pdf_text)

    # Image -> internal text + brief insights (UI shows ONLY brief insights)
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded image", use_container_width=True)
    
        # ✅ ADD THIS HERE (normalize to PNG bytes so Bedrock detects correct MIME)
        img_rgb = Image.open(uploaded_img).convert("RGB")
        buf = io.BytesIO()
        img_rgb.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        img_fmt = "png"  # always png after conversion
    
        # Use Nova multimodal (internal retrieval text + brief insights)
        with st.spinner("🔍 Reading image with Nova Lite..."):
            ocr_text = nova_image_to_text(img_bytes, image_format=img_fmt)
    
        with st.spinner("💡 Generating image insights..."):
            insights = nova_image_insights_brief(img_bytes, image_format=img_fmt)
    
        # ✅ IMPORTANT (per your requirement):
        # - Do NOT display extracted image text
        # - Only display the 2-line insights
        if insights.strip():
            st.subheader("🖼️ Image Insights")
            st.markdown(insights)
    
        # Include BOTH into retrieval text (even though OCR is not displayed)
        if ocr_text.strip():
            full_text_parts.append("=== IMAGE TEXT (NOVA) ===\n" + ocr_text)
        if insights.strip():
            full_text_parts.append("=== IMAGE INSIGHTS (NOVA) ===\n" + insights)


    # Notes
    if user_text.strip():
        full_text_parts.append("=== USER NOTES ===\n" + user_text.strip())

    full_text = "\n\n".join(full_text_parts)

    # doc fingerprint
    doc_fp = hashlib.md5(full_text[:20000].encode("utf-8", errors="ignore")).hexdigest()

    # Auto settings per doc
    if st.session_state.get("last_doc_fp") != doc_fp:
        st.session_state["last_doc_fp"] = doc_fp
        with st.spinner("⚙️ Nova is auto-optimizing retrieval settings..."):
            st.session_state["auto_rag_settings"] = recommend_rag_settings(full_text)

        # reset suggestion cache
        for kk in ["suggest_fp", "suggested_questions"]:
            if kk in st.session_state:
                del st.session_state[kk]

        # also force index rebuild
        st.session_state.pop("index_fp", None)

    rec = st.session_state.get("auto_rag_settings", {"chunk_size": 1000, "overlap": 150, "top_k": 4})
    auto_chunk_size, auto_overlap, auto_top_k = rec["chunk_size"], rec["overlap"], rec["top_k"]

    # ✅ Retrieval settings ONLY in sidebar (adjustable, but auto suggested)
    st.sidebar.subheader("🔎 Retrieval settings")
    use_auto = st.sidebar.toggle("Use Nova auto-optimized settings", value=True)

    if use_auto:
        chunk_size, overlap, top_k = auto_chunk_size, auto_overlap, auto_top_k
        st.sidebar.caption(f"Auto suggested: chunk={chunk_size}, overlap={overlap}, top_k={top_k}")
    else:
        chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 2000, int(auto_chunk_size), 50)
        overlap = st.sidebar.slider("Overlap (chars)", 0, 400, int(auto_overlap), 25)
        top_k = st.sidebar.slider("Top-K sources", 2, 8, int(auto_top_k), 1)

    if st.sidebar.button("♻️ Rebuild index now", use_container_width=True):
        st.session_state.pop("index_fp", None)

    # ✅ auto build index
    build_index_if_needed(full_text, chunk_size=chunk_size, overlap=overlap)

    st.divider()

    # Extract fields
    st.subheader("2) Extract key fields (JSON)")
    doc_type = st.selectbox("Document type", DOC_TYPES, index=0)
    if st.button("🧾 Extract key fields as JSON", use_container_width=True):
        with st.spinner("Extracting..."):
            out = extract_fields_json(full_text, doc_type=doc_type)
        st.code(out, language="json")

    st.divider()

    # Suggested questions
    st.markdown("### ✨ Nova-suggested questions (auto from your document)")
    suggest_fp = f"{doc_fp}:{user_interest}"

    if st.session_state.get("suggest_fp") != suggest_fp:
        st.session_state["suggest_fp"] = suggest_fp
        with st.spinner("Generating questions from your document..."):
            st.session_state["suggested_questions"] = suggest_questions(full_text, user_interest=user_interest, n=6)

    qs = st.session_state.get("suggested_questions", [])
    if qs:
        cols = st.columns(3)
        for i, q in enumerate(qs):
            with cols[i % 3]:
                if st.button(q, use_container_width=True, key=f"dynq_{i}"):
                    st.session_state["pending_question"] = q
                    st.rerun()
    else:
        st.warning("Nova couldn’t generate suggestions for this document. Try uploading a richer document.")

    if st.button("🔄 Refresh questions", use_container_width=True):
        with st.spinner("Refreshing questions from your document..."):
            st.session_state["suggested_questions"] = suggest_questions(full_text, user_interest=user_interest, n=6)
        st.rerun()

    st.divider()

    # ✅ Chat section (NO accumulating outputs)
    st.subheader("3) Chat with your document")

    rag = st.session_state.get("single_rag", None)
    if not isinstance(rag, RagIndex):
        st.error("Index not ready. Try clicking **Rebuild index now** in the sidebar, or Reset.")
        st.stop()

    if len(full_text.strip()) < 50:
        st.warning("Not enough text to chat. Upload a PDF or a text-heavy image.")
        st.stop()

    # Ask UI
    st.markdown("#### 💬 Ask a question")
    user_q = None
    if "pending_question" in st.session_state:
        user_q = st.session_state.pop("pending_question")
    else:
        user_q = st.chat_input("Ask something about the document…")

    if user_q:
        user_q = user_q.strip()
        if user_q:
            with st.spinner("Retrieving sources..."):
                hits, scores = rag.search(user_q, k=top_k)
                ctx = [h.text for h in hits]

            with st.spinner("Thinking with Nova Lite..."):
                start_time = time.time()
                ans = ask_with_evidence(user_q, ctx)
                response_time = round(time.time() - start_time, 2)

            answer_text = ans.get("answer", "").strip()
            evidence_text = ans.get("evidence", "").strip()

            st.session_state["latest_answer"] = answer_text if answer_text else "I don't know based on the document."
            st.session_state["latest_evidence"] = evidence_text
            st.session_state["latest_sources"] = list(zip(hits, scores))
            st.session_state["latest_q"] = user_q
            st.session_state["latest_rt"] = response_time

    # ✅ Render ONLY latest answer (replaces previous)
    if st.session_state.get("latest_answer"):
        st.markdown(f"### ✅ Latest Answer")
        st.markdown(f"**Question:** {st.session_state.get('latest_q','')}")
        st.markdown(st.session_state.get("latest_answer", ""))

        ev = st.session_state.get("latest_evidence", "")
        if ev:
            with st.expander("📌 Evidence (exact quotes)"):
                st.markdown(ev)

        sources = st.session_state.get("latest_sources", [])
        if sources:
            with st.expander("📚 Retrieved sources (debug)", expanded=False):
                for i, (h, s) in enumerate(sources, start=1):
                    st.markdown(f"**Source {i} • score {s:.3f} • chunk #{h.chunk_id}**")
                    st.write(h.text[:1200])
                    st.markdown("---")

        st.caption(f"⏱ Average response time: {st.session_state.get('latest_rt', 0)} sec")

    st.divider()

    # Export
    st.subheader("4) Export report (PDF)")
    if st.button("📄 Generate PDF report", use_container_width=True):
        with st.spinner("Creating a title with Nova Lite..."):
            report_title = generate_report_title(full_text)

        auto_type = detect_doc_type(full_text)

        latest_block = ""
        if st.session_state.get("latest_answer"):
            latest_block = (
                f"Question: {st.session_state.get('latest_q','')}\n\n"
                f"Answer:\n{st.session_state.get('latest_answer','')}\n\n"
                f"Evidence:\n{st.session_state.get('latest_evidence','')}\n"
            )

        sections = [
            ("Report title (generated)", report_title),
            ("Document type (auto)", auto_type),
            ("Retrieval settings", f"chunk_size={chunk_size}, overlap={overlap}, top_k={top_k}"),
            ("Document preview (first 1200 chars)", full_text[:1200]),
            ("Latest Q&A", latest_block or "No Q&A yet."),
        ]

        safe_filename = "".join(ch for ch in report_title if ch.isalnum() or ch in (" ", "-", "_")).strip()
        safe_filename = safe_filename.replace(" ", "_") or "Smart_Document_Report"
        file_name = f"{safe_filename}.pdf"

        pdf_bytes = make_pdf_report(file_name, report_title, sections)
        st.download_button(
            "⬇️ Download report",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf",
            use_container_width=True
        )

# =======================
# Mode: Compare Two Docs
# =======================
else:
    st.subheader("🆚 Compare Two PDFs")
    k = st.session_state.get("uploader_key", 0)

    c1, c2 = st.columns(2)
    with c1:
        up_a = st.file_uploader("Upload PDF (Doc A)", type=["pdf"], key=f"docA_{k}")
    with c2:
        up_b = st.file_uploader("Upload PDF (Doc B)", type=["pdf"], key=f"docB_{k}")

    if up_a is None or up_b is None:
        st.info("Upload both PDFs → Ask a comparison question.")
        st.stop()

    text_a = extract_text_from_pdf(up_a)
    text_b = extract_text_from_pdf(up_b)

    with st.spinner("⚙️ Auto-tuning retrieval settings for comparison..."):
        rec = recommend_rag_settings(text_a + "\n\n" + text_b)
    chunk_size, overlap, top_k = rec["chunk_size"], rec["overlap"], rec["top_k"]

    chunks_a = chunk_text(text_a, chunk_size=chunk_size, overlap=overlap)
    chunks_b = chunk_text(text_b, chunk_size=chunk_size, overlap=overlap)

    if "rag_a" not in st.session_state or "rag_b" not in st.session_state:
        with st.spinner("🚀 Auto-building indexes for Doc A & Doc B..."):
            rag_a = RagIndex(dim=1024)
            rag_a.add_chunks(chunks_a)
            rag_b = RagIndex(dim=1024)
            rag_b.add_chunks(chunks_b)
        st.session_state["rag_a"] = rag_a
        st.session_state["rag_b"] = rag_b

    q = st.text_input("Comparison question", placeholder="e.g., Which document shows stronger results and why?")
    if st.button("🆚 Compare", use_container_width=True):
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

