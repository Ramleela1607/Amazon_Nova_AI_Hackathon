import io
import time
import hashlib
import streamlit as st
from pypdf import PdfReader
from PIL import Image
import pandas as pd
import re
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
    nova_image_insights_brief,    
    generate_report_title,
    suggest_questions,
    generate_dashboard_insights,
)

# ---------- Page ----------
st.set_page_config(page_title="Smart Document Copilot", layout="wide")

# ---------- Premium UI ----------
st.markdown("""
<style>
/* ----------------------------
   VIDEO-LIKE ANIMATED BACKGROUND
   ---------------------------- */
.stApp {
  background: radial-gradient(circle at 10% 20%, rgba(224,242,254,0.95), transparent 45%),
              radial-gradient(circle at 90% 10%, rgba(252,231,243,0.92), transparent 45%),
              radial-gradient(circle at 50% 90%, rgba(236,252,203,0.92), transparent 50%),
              linear-gradient(120deg, rgba(255,255,255,0.85), rgba(255,255,255,0.72));
  background-attachment: fixed;
  color: #0f172a !important;
  position: relative;
  overflow-x: hidden;
}
.stApp::before{
  content:"";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;

  background:
    radial-gradient(circle at 15% 35%, rgba(99,102,241,0.22), transparent 40%),
    radial-gradient(circle at 75% 20%, rgba(16,185,129,0.18), transparent 40%),
    radial-gradient(circle at 70% 80%, rgba(236,72,153,0.14), transparent 42%),
    radial-gradient(circle at 25% 85%, rgba(14,165,233,0.16), transparent 45%);
  filter: blur(22px);
  animation: floatBg 14s ease-in-out infinite alternate;
  opacity: 0.9;
}
@keyframes floatBg {
  0%   { transform: translate3d(-20px, -18px, 0) scale(1.02); }
  50%  { transform: translate3d(24px, 16px, 0) scale(1.06); }
  100% { transform: translate3d(-10px, 26px, 0) scale(1.03); }
}

/* Everything above background layer */
.block-container,
section[data-testid="stSidebar"],
header,
footer,
div[data-testid="stAppViewContainer"]{
  position: relative;
  z-index: 1;
}

/* Make ALL text readable */
html, body, [class*="css"], p, span, div {
  color: #0f172a !important;
}

/* Sidebar: frosted glass */
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.78) !important;
  backdrop-filter: blur(14px);
  border-right: 1px solid rgba(15,23,42,0.10);
}
section[data-testid="stSidebar"] *{
  color: #0f172a !important;
}

/* Titles / headers clarity */
h1, h2, h3 {
  color: #0b1220 !important;
  text-shadow: 0 1px 0 rgba(255,255,255,0.65);
}

/* Expanders */
div[data-testid="stExpander"] {
  border-radius: 18px;
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.86);
  box-shadow: 0 10px 24px rgba(2,6,23,0.07);
}

/* Chat bubbles */
div[data-testid="stChatMessage"]{
  background: rgba(255,255,255,0.90);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 18px;
  box-shadow: 0 10px 22px rgba(2,6,23,0.06);
}
div[data-testid="stChatMessage"] *{
  color: #0f172a !important;
}

/* Inputs */
div[data-baseweb="input"] input, textarea {
  background: rgba(255,255,255,0.94) !important;
  color: #0f172a !important;
  border-radius: 14px !important;
  border: 1px solid rgba(15,23,42,0.16) !important;
}

/* Buttons */
.stButton button {
  border-radius: 14px;
  border: 1px solid rgba(15,23,42,0.14);
  background: rgba(255,255,255,0.92);
  color: #0f172a !important;
  font-weight: 700;
  transition: transform 120ms ease, box-shadow 120ms ease;
}
.stButton button:hover {
  transform: translateY(-1px);
  border: 1px solid rgba(99,102,241,0.45);
  box-shadow: 0 10px 26px rgba(99,102,241,0.18);
}

a { color: #1d4ed8 !important; }
.block-container { padding-top: 1.1rem; }
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

def try_parse_number(value: str):
    """
    Extract a usable numeric value from strings like "$12,340.50", "45%", "INR 1000".
    Returns float or None.
    """
    if value is None:
        return None
    s = str(value).strip()
    s = s.replace(",", "")
    m = re.search(r"(-?\d+(\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

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
        "latest_answer", "latest_evidence", "latest_sources", "latest_q", "latest_rt",
        "rag_a", "rag_b",
        "pending_question",
    ]:
        st.session_state.pop(k, None)
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

def run_single_question(user_q: str, top_k: int):
    """Run a single question and store ONLY latest result (no chat accumulation)."""
    rag = st.session_state.get("single_rag", None)
    if not isinstance(rag, RagIndex):
        st.error("Index not ready. Rebuild from sidebar or Reset.")
        st.stop()

    with st.spinner("Retrieving sources..."):
        hits, scores = rag.search(user_q, k=top_k)
        ctx = [h.text for h in hits]

    with st.spinner("Thinking with Nova Lite..."):
        t0 = time.time()
        ans = ask_with_evidence(user_q, ctx)
        rt = round(time.time() - t0, 2)

    st.session_state["latest_q"] = user_q
    st.session_state["latest_answer"] = (ans.get("answer") or "").strip()
    st.session_state["latest_evidence"] = (ans.get("evidence") or "").strip()
    st.session_state["latest_sources"] = list(zip(hits, scores))
    st.session_state["latest_rt"] = rt


# ---------- Session init ----------
st.session_state.setdefault("uploader_key", 0)

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

# =======================
# Mode: Single Document
# =======================
if mode == "Single Document":
    k = st.session_state.get("uploader_key", 0)

    uploaded_pdf = st.file_uploader("📤 Upload a PDF (optional)", type=["pdf"], key=f"pdf_uploader_{k}")
    uploaded_img = st.file_uploader("🖼️ Upload an Image (optional)", type=["png", "jpg", "jpeg", "webp"], key=f"img_uploader_{k}")
    user_text = st.text_area("✍️ Paste extra text / notes (optional)", height=90)

    if uploaded_pdf is None and uploaded_img is None and not user_text.strip():
        st.info("Upload a PDF or Image (or paste text) → Index builds automatically → Start chatting.")
        st.stop()

    full_text_parts = []

    # PDF
    if uploaded_pdf is not None:
        pdf_text = extract_text_from_pdf(uploaded_pdf)
        full_text_parts.append("=== PDF TEXT ===\n" + pdf_text)

    # Image: cache insights + internal OCR
    if uploaded_img is not None:
        img_rgb = Image.open(uploaded_img).convert("RGB")
        buf = io.BytesIO()
        img_rgb.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        img_fp = hashlib.md5(img_bytes[:20000]).hexdigest()
        img_cache_key = f"img_insights:{img_fp}"

        if img_cache_key not in st.session_state:
            with st.spinner("💡 Generating image insights..."):
                st.session_state[img_cache_key] = nova_image_insights_brief(img_bytes, image_format="png")

        insights = st.session_state.get(img_cache_key, "")

        if insights.strip():
            st.subheader("Insights")
            st.markdown(insights.replace("\n", "  \n"))

        # OCR for retrieval only (do not show)
        try:
            ocr_text = nova_image_to_text(img_bytes, image_format="png")
        except Exception:
            ocr_text = ""

        if ocr_text.strip():
            full_text_parts.append("=== IMAGE TEXT ===\n" + ocr_text)
        if insights.strip():
            full_text_parts.append("=== IMAGE INSIGHTS ===\n" + insights)

    # Notes
    if user_text.strip():
        full_text_parts.append("=== USER NOTES ===\n" + user_text.strip())

    full_text = "\n\n".join(full_text_parts)
    # ========================
    # 📊 Executive Dashboard
    # ========================
    st.subheader("📊 Executive Dashboard")
    dash_key = f"dashboard:{doc_fp}"
    
    # Refresh button (forces re-run for dashboard only)
    dcol1, dcol2 = st.columns([1, 5])
    with dcol1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.session_state.pop(dash_key, None)
    
    if dash_key not in st.session_state:
        with st.spinner("Analyzing document for dashboard insights..."):
            st.session_state[dash_key] = generate_dashboard_insights(full_text)
    
    dashboard = st.session_state.get(dash_key, {}) or {}
    
    summary = dashboard.get("summary", "")
    doc_type_guess = dashboard.get("doc_type_guess", "generic")
    risk_score = int(dashboard.get("risk_score", 0) or 0)
    key_numbers = dashboard.get("key_numbers", []) or []
    key_dates = dashboard.get("key_dates", []) or []
    risks = dashboard.get("risks", []) or []
    next_actions = dashboard.get("next_actions", []) or []
    
    # --- Top Row: Type + Risk
    c1, c2, c3 = st.columns(3)
    c1.metric("Doc Type (Nova)", str(doc_type_guess))
    c2.metric("Risk Score", f"{risk_score}/100")
    
    # Risk meter
    with c3:
        st.caption("Risk Meter")
        st.progress(min(max(risk_score, 0), 100))
    
    # --- Summary
    if summary.strip():
        st.markdown("### 🧾 Executive Summary")
        st.markdown(summary)
    
    # --- Key Numbers → Chart
    numeric_rows = []
    for it in key_numbers:
        label = (it or {}).get("label", "").strip()
        val = (it or {}).get("value", "")
        num = try_parse_number(val)
        if label and num is not None:
            numeric_rows.append({"label": label, "value": num, "raw": str(val)})
    
    if numeric_rows:
        st.markdown("### 📈 Key Numbers (Chart)")
        df = pd.DataFrame(numeric_rows).sort_values("value", ascending=False)
        st.bar_chart(df.set_index("label")[["value"]])
    
        with st.expander("Numbers (raw)", expanded=False):
            st.dataframe(df[["label", "raw"]], use_container_width=True)
    else:
        if key_numbers:
            st.markdown("### 📌 Key Numbers")
            for it in key_numbers[:6]:
                st.markdown(f"- **{it.get('label','Metric')}**: {it.get('value','-')}")
        else:
            st.caption("No key numeric metrics detected for charting.")
    
    # --- Key Dates Timeline
    if key_dates:
        st.markdown("### 🗓️ Key Dates Timeline")
        df_dates = pd.DataFrame(key_dates)
        df_dates = df_dates.rename(columns={"label": "Event", "value": "Date"})
        st.dataframe(df_dates, use_container_width=True)
    
    # --- Risks + Actions
    if risks:
        st.markdown("### ⚠️ Risks")
        for r in risks[:6]:
            st.markdown(f"- {r}")
    
    if next_actions:
        st.markdown("### ✅ Next Actions")
        for a in next_actions[:6]:
            st.markdown(f"- {a}")
    
    st.divider()


    doc_fp = hashlib.md5(full_text[:20000].encode("utf-8", errors="ignore")).hexdigest()

    # Auto settings per doc
    if st.session_state.get("last_doc_fp") != doc_fp:
        st.session_state["last_doc_fp"] = doc_fp
        with st.spinner("⚙️ Nova is auto-optimizing retrieval settings..."):
            st.session_state["auto_rag_settings"] = recommend_rag_settings(full_text)

        st.session_state.pop("suggest_fp", None)
        st.session_state.pop("suggested_questions", None)
        st.session_state.pop("index_fp", None)

    rec = st.session_state.get("auto_rag_settings", {"chunk_size": 1000, "overlap": 150, "top_k": 4})
    auto_chunk_size, auto_overlap, auto_top_k = rec["chunk_size"], rec["overlap"], rec["top_k"]

    # Sidebar retrieval settings (auto + adjustable)
    st.sidebar.subheader("🔎 Retrieval settings")
    use_auto = st.sidebar.toggle("Use Nova auto-optimized settings", value=True)

    if use_auto:
        chunk_size, overlap, top_k = auto_chunk_size, auto_overlap, auto_top_k
        st.sidebar.caption(f"Auto: chunk={chunk_size}, overlap={overlap}, top_k={top_k}")
    else:
        chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 2000, int(auto_chunk_size), 50)
        overlap = st.sidebar.slider("Overlap (chars)", 0, 400, int(auto_overlap), 25)
        top_k = st.sidebar.slider("Top-K sources", 2, 8, int(auto_top_k), 1)

    if st.sidebar.button("♻️ Rebuild index now", use_container_width=True):
        st.session_state.pop("index_fp", None)

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

    # Suggested questions (auto)
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
                if st.button(q, use_container_width=True, key=f"dynq_{doc_fp}_{i}"):
                    # auto-run immediately
                    st.session_state["pending_question"] = q
                    st.rerun()
    else:
        st.caption("Suggestions unavailable for this upload.")

    if st.button("🔄 Refresh questions", use_container_width=True):
        with st.spinner("Refreshing questions..."):
            st.session_state["suggested_questions"] = suggest_questions(full_text, user_interest=user_interest, n=6)
        st.rerun()

    st.divider()

    # Chat section (manual + auto-run from suggestions)
    st.subheader("3) Chat with your document")
    st.markdown("#### 💬 Ask a question")

    # If suggestion clicked, it runs automatically
    pending = st.session_state.pop("pending_question", "")
    if pending:
        run_single_question(pending, top_k=top_k)

    # Manual input (stays inline here, NOT bottom)
    with st.form("ask_form", clear_on_submit=True):
        q_text = st.text_input(
            "Type your question",
            placeholder="e.g., What is the total amount and due date?",
            label_visibility="collapsed",
        )
        ask_clicked = st.form_submit_button("Ask", use_container_width=True)

    if ask_clicked and q_text.strip():
        run_single_question(q_text.strip(), top_k=top_k)

    # Show ONLY latest output
    if st.session_state.get("latest_answer"):
        st.markdown("### ✅ Latest Answer")
        st.markdown(f"**Question:** {st.session_state.get('latest_q','')}")
        st.markdown(st.session_state["latest_answer"] or "I don't know based on the document.")

        ev = st.session_state.get("latest_evidence", "")
        if ev:
            with st.expander("📌 Evidence"):
                st.markdown(ev)

        st.caption(f"⏱ Response time: {st.session_state.get('latest_rt', 0)} sec")

    st.divider()

    # Export PDF
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




