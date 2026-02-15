import io
import time
import json
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional

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
    detect_doc_type,
    compare_docs,
    DOC_TYPES,
    recommend_rag_settings,
    nova_image_to_text,
    nova_image_insights_brief,
    generate_report_title,
    suggest_questions,
    generate_dashboard_insights_dynamic,  # AI dashboard (may fail JSON; hybrid fallback handles)
)

# ============================================================
# Page + Premium UI
# ============================================================
st.set_page_config(page_title="Smart Document Copilot", layout="wide")

st.markdown(
    """
<style>
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
.block-container,
section[data-testid="stSidebar"],
header,
footer,
div[data-testid="stAppViewContainer"]{
  position: relative;
  z-index: 1;
}
html, body, [class*="css"], p, span, div { color: #0f172a !important; }
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.78) !important;
  backdrop-filter: blur(14px);
  border-right: 1px solid rgba(15,23,42,0.10);
}
section[data-testid="stSidebar"] *{ color: #0f172a !important; }
h1, h2, h3 { color: #0b1220 !important; text-shadow: 0 1px 0 rgba(255,255,255,0.65); }
div[data-testid="stExpander"] {
  border-radius: 18px;
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.86);
  box-shadow: 0 10px 24px rgba(2,6,23,0.07);
}
div[data-testid="stChatMessage"]{
  background: rgba(255,255,255,0.90);
  border: 1px solid rgba(15,23,42,0.10);
  border-radius: 18px;
  box-shadow: 0 10px 22px rgba(2,6,23,0.06);
}
div[data-testid="stChatMessage"] *{ color: #0f172a !important; }
div[data-baseweb="input"] input, textarea {
  background: rgba(255,255,255,0.94) !important;
  color: #0f172a !important;
  border-radius: 14px !important;
  border: 1px solid rgba(15,23,42,0.16) !important;
}
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
""",
    unsafe_allow_html=True,
)

st.title("📄 Smart Document Copilot")
st.markdown(
    "<div style='opacity:0.9'>Amazon Nova on Bedrock • Multimodal RAG • Evidence • Compare • PDF Report &nbsp; <b>#AmazonNova</b></div>",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers
# ============================================================

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        start += max(1, chunk_size - overlap)
    return chunks

def normalize_image_to_png_bytes(uploaded_img) -> bytes:
    img = Image.open(uploaded_img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def extract_text_from_pdf_basic(file_like) -> str:
    """Digital PDFs: pypdf extraction."""
    reader = PdfReader(file_like)
    texts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            texts.append(f"\n\n--- Page {i+1} ---\n{t}")
    return "".join(texts)

def pdf_pages_to_png_bytes(pdf_bytes: bytes, max_pages: int = 6) -> List[bytes]:
    """Render PDF pages to PNG via PyMuPDF if installed (scanned PDFs)."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i in range(min(max_pages, doc.page_count)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=180)
        pages.append(pix.tobytes("png"))
    return pages

def extract_text_from_pdf_with_ocr(uploaded_pdf, max_ocr_pages: int = 6) -> str:
    """
    Hybrid PDF:
    1) pypdf
    2) if too short -> render pages -> nova_image_to_text OCR
    Cached by PDF hash.
    """
    pdf_bytes = uploaded_pdf.getvalue()
    cache_key = "pdf_text:" + sha1_bytes(pdf_bytes[:300000])

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    basic_text = extract_text_from_pdf_basic(io.BytesIO(pdf_bytes))
    basic_len = len((basic_text or "").strip())
    combined = (basic_text or "").strip()

    if basic_len < 250:
        ocr_parts = []
        page_pngs = pdf_pages_to_png_bytes(pdf_bytes, max_pages=max_ocr_pages)
        if page_pngs:
            with st.spinner(f"🧠 OCR on scanned/table PDF (first {len(page_pngs)} pages)..."):
                for idx, png_bytes in enumerate(page_pngs, start=1):
                    try:
                        t = nova_image_to_text(png_bytes, image_format="png")
                    except Exception:
                        t = ""
                    if t.strip():
                        ocr_parts.append(f"\n\n--- OCR Page {idx} ---\n{t.strip()}")
        if ocr_parts:
            combined = (combined + "\n\n" + "\n".join(ocr_parts)).strip()

    st.session_state[cache_key] = combined
    return combined

def extract_text_from_word(uploaded_docx) -> str:
    try:
        from docx import Document
    except Exception:
        return ""
    b = uploaded_docx.getvalue()
    cache_key = "docx_text:" + sha1_bytes(b[:300000])
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    doc = Document(io.BytesIO(b))
    parts = []
    for p in doc.paragraphs:
        if p.text and p.text.strip():
            parts.append(p.text.strip())
    text = "\n".join(parts).strip()
    st.session_state[cache_key] = text
    return text

def extract_text_from_pptx(uploaded_pptx) -> str:
    try:
        from pptx import Presentation
    except Exception:
        return ""
    b = uploaded_pptx.getvalue()
    cache_key = "pptx_text:" + sha1_bytes(b[:300000])
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    prs = Presentation(io.BytesIO(b))
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text and shape.text.strip():
                parts.append(shape.text.strip())
    text = "\n\n".join(parts).strip()
    st.session_state[cache_key] = text
    return text

def read_excel_to_tables(uploaded_xls) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """
    Returns:
      - extracted_text (string summary)
      - tables dict: sheet_name -> dataframe
    """
    b = uploaded_xls.getvalue()
    cache_key = "excel_tables:" + sha1_bytes(b[:300000])
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    tables: Dict[str, pd.DataFrame] = {}
    text_parts = []
    try:
        xls = pd.ExcelFile(io.BytesIO(b))
        for name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=name)
                if df is not None and not df.empty:
                    tables[name] = df
                    text_parts.append(f"--- Sheet: {name} ---")
                    text_parts.append(df.head(30).to_csv(index=False))
            except Exception:
                continue
    except Exception:
        # Try CSV fallback if user uploaded .csv with excel uploader by mistake
        try:
            df = pd.read_csv(io.BytesIO(b))
            tables["Sheet1"] = df
            text_parts.append("--- Sheet: Sheet1 ---")
            text_parts.append(df.head(30).to_csv(index=False))
        except Exception:
            pass

    extracted_text = "\n".join(text_parts).strip()
    st.session_state[cache_key] = (extracted_text, tables)
    return extracted_text, tables

def extract_dates_with_events(text: str, max_items: int = 120) -> List[Dict[str, str]]:
    if not text or not str(text).strip():
        return []
    t = str(text)
    t = t.replace("\r", "\n")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"(\d{1,2})\s*\n\s*([A-Za-z]{3,9})\s*\n\s*(\d{2,4})", r"\1 \2 \3", t)
    t = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t, flags=re.IGNORECASE)

    months = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
    p_range = rf"\b({months}\s+\d{{4}})\s*-\s*(Present|{months}\s+\d{{4}})\b"
    p_iso = r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
    p_slash = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    p_mdy = rf"\b{months}\s+\d{{1,2}},?\s+\d{{2,4}}\b"
    p_dmy = rf"\b\d{{1,2}}\s+{months},?\s+\d{{2,4}}\b"
    p_my = rf"\b{months}\s+\d{{4}}\b"

    date_re = re.compile("|".join([p_range, p_iso, p_slash, p_mdy, p_dmy, p_my]), re.IGNORECASE)

    results: List[Dict[str, str]] = []
    seen = set()
    for line in t.split("\n"):
        ln = line.strip()
        if not ln:
            continue
        for m in date_re.finditer(ln):
            date_str = m.group(0).strip()
            event = ln[:m.start()].strip() or ln[m.end():].strip()
            event = re.sub(r"\s+", " ", event).strip(" -:•;|")
            if not event:
                event = "Date mentioned"
            words = event.split()
            if len(words) > 16:
                event = " ".join(words[-16:])
            key = (event.lower(), date_str.lower())
            if key in seen:
                continue
            seen.add(key)
            results.append({"label": event, "value": date_str})
            if len(results) >= max_items:
                break
        if len(results) >= max_items:
            break
    return results

def try_parse_number(value: str):
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def local_mine_metrics(text: str, max_items: int = 80) -> Dict[str, Any]:
    out = {"kpis": [], "charts": [], "table_preview": [], "derived_insights": []}
    if not text or not str(text).strip():
        return out

    t = str(text)
    t = t.replace("\r", "\n").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    metric_candidates = []
    for line in t.splitlines():
        ln = line.strip()
        if len(ln) < 4:
            continue
        m = re.search(r"^(.{2,60}?)\s*[:\-]\s*([^\n]{1,50})$", ln)
        if m:
            metric_candidates.append((m.group(1).strip(), m.group(2).strip()))

    num_re = re.compile(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")
    loose = []
    for line in t.splitlines():
        ln = line.strip()
        if not ln:
            continue
        nums = list(num_re.finditer(ln))
        if not nums:
            continue
        if len(nums) >= 2 and len(ln) <= 180:
            out["table_preview"].append({"row": ln})
        for mm in nums[:3]:
            val = mm.group(1)
            left = re.sub(r"\s+", " ", ln[:mm.start()].strip())
            words = left.split()
            label = " ".join(words[-6:]) if words else "Number"
            loose.append((label, val))

    combined = metric_candidates[:max_items] + loose[:max_items]

    numeric_items = []
    for label, val in combined:
        num = try_parse_number(val)
        if num is None:
            continue
        numeric_items.append((label, num, val))

    numeric_sorted = sorted(numeric_items, key=lambda x: abs(x[1]), reverse=True)

    for label, _num, raw in numeric_sorted[:9]:
        out["kpis"].append({"label": label[:35], "value": raw, "note": ""})

    nums_only = [x[1] for x in numeric_items]
    if len(nums_only) >= 3:
        mn, mx = min(nums_only), max(nums_only)
        avg = sum(nums_only) / len(nums_only)
        out["derived_insights"].append(f"Detected {len(nums_only)} numeric values. Min={mn:g}, Max={mx:g}, Avg={avg:g}.")
        if mx != 0 and abs(mx) > 10 * max(1e-9, abs(avg)):
            out["derived_insights"].append("Some values are much larger than average (possible totals/outliers).")

    if numeric_sorted:
        data = [{"x": lab[:28], "y": float(num)} for lab, num, _raw in numeric_sorted[:12]]
        out["charts"].append({"title": "Top Numeric Values (Auto)", "type": "bar", "data": data})

    out["table_preview"] = out["table_preview"][:8]
    return out

def merge_ai_and_local(ai: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(ai or {})
    if not isinstance(merged.get("kpis"), list) or not merged.get("kpis"):
        merged["kpis"] = local.get("kpis", [])
    if not isinstance(merged.get("charts"), list) or not merged.get("charts"):
        merged["charts"] = local.get("charts", [])
    if not isinstance(merged.get("table_preview"), list) or not merged.get("table_preview"):
        merged["table_preview"] = local.get("table_preview", [])
    if not isinstance(merged.get("derived_insights"), list) or not merged.get("derived_insights"):
        merged["derived_insights"] = local.get("derived_insights", [])

    if not merged.get("summary") or "could not be generated" in str(merged.get("summary", "")).lower():
        if local.get("kpis") or local.get("charts"):
            merged["summary"] = "Auto dashboard generated from detected numbers/tables in the document."
    merged.setdefault("doc_type_guess", "generic")
    merged.setdefault("risk_score", 0)
    merged.setdefault("risks", [])
    merged.setdefault("next_actions", [])
    return merged

def make_pdf_report(filename: str, title: str, sections: List[Tuple[str, str]]) -> bytes:
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Heading1"]), Spacer(1, 0.2 * inch)]
    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))
        safe = (body or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))
    path = f"/tmp/{filename}"
    SimpleDocTemplate(path).build(story)
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
    # Clear caches
    for kk in list(st.session_state.keys()):
        if str(kk).startswith(("dashboard:", "img_insights:", "dates:", "pdf_text:", "excel_tables:", "docx_text:", "pptx_text:")):
            st.session_state.pop(kk, None)
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

def show_dashboard_debug(local_dash: Dict[str, Any], ai_dash: Dict[str, Any]):
    st.markdown("### 🧪 Debug: What the dashboard used")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Local mined KPIs**")
        st.json(local_dash.get("kpis", [])[:12])
        st.markdown("**Local charts data**")
        st.json(local_dash.get("charts", [])[:2])
    with c2:
        st.markdown("**AI raw output (cached)**")
        st.json(ai_dash)

# Session init
st.session_state.setdefault("uploader_key", 0)

# ============================================================
# Sidebar (Removed User Interest)
# ============================================================
st.sidebar.header("⚙️ Controls")
st.sidebar.button("🔄 Reset / New session", on_click=reset_session, use_container_width=True)
mode = st.sidebar.radio("Mode", ["Single Document", "Compare Two Documents"])
st.sidebar.markdown("---")

st.sidebar.subheader("🧠 AI Controls")
use_ai_dashboard = st.sidebar.toggle("AI Dashboard (Nova)", value=True)
lazy_ai = st.sidebar.toggle("Lazy AI (faster first load)", value=True)
show_ai_inputs = st.sidebar.toggle("Show what Nova used (debug)", value=False)

st.sidebar.markdown("---")
# ============================================================
# Mode: Single Document
# ============================================================
if mode == "Single Document":
    k = st.session_state.get("uploader_key", 0)

    # ---- Mutually exclusive uploads ----
    # We keep one "active upload type" based on what user selected.
    # Streamlit file_uploader supports `disabled=...` so we disable others once one is chosen.
    st.subheader("1) Upload a document")

    # Determine current selections from session_state (best-effort)
    # If keys don't exist yet, default to None.
    pdf_key = f"pdf_uploader_{k}"
    img_key = f"img_uploader_{k}"
    excel_key = f"excel_uploader_{k}"
    docx_key = f"docx_uploader_{k}"
    pptx_key = f"pptx_uploader_{k}"

    existing_pdf = st.session_state.get(pdf_key)
    existing_img = st.session_state.get(img_key)
    existing_excel = st.session_state.get(excel_key)
    existing_docx = st.session_state.get(docx_key)
    existing_pptx = st.session_state.get(pptx_key)

    any_other_selected = lambda key: any([
        (key != "pdf" and existing_pdf is not None),
        (key != "img" and existing_img is not None),
        (key != "excel" and existing_excel is not None),
        (key != "docx" and existing_docx is not None),
        (key != "pptx" and existing_pptx is not None),
    ])

    colA, colB = st.columns(2)
    with colA:
        uploaded_pdf = st.file_uploader(
            "📤 Upload PDF",
            type=["pdf"],
            key=pdf_key,
            disabled=any_other_selected("pdf"),
        )
        uploaded_img = st.file_uploader(
            "🖼️ Upload Image (PNG/JPG/WebP)",
            type=["png", "jpg", "jpeg", "webp"],
            key=img_key,
            disabled=any_other_selected("img"),
        )
    with colB:
        uploaded_excel = st.file_uploader(
            "📊 Upload Excel/CSV",
            type=["xlsx", "xls", "csv"],
            key=excel_key,
            disabled=any_other_selected("excel"),
        )
        uploaded_docx = st.file_uploader(
            "📝 Upload Word (DOCX)",
            type=["docx"],
            key=docx_key,
            disabled=any_other_selected("docx"),
        )
        uploaded_pptx = st.file_uploader(
            "📽️ Upload PowerPoint (PPTX)",
            type=["pptx"],
            key=pptx_key,
            disabled=any_other_selected("pptx"),
        )

    user_text = st.text_area("✍️ Paste extra text / notes (optional)", height=90)

    if (
        uploaded_pdf is None
        and uploaded_img is None
        and uploaded_excel is None
        and uploaded_docx is None
        and uploaded_pptx is None
        and not user_text.strip()
    ):
        st.info("Upload ONE document type (PDF/Image/Excel/Word/PPT) or paste text → Dashboard + RAG + Q&A.")
        st.stop()

    # ============================================================
    # Extract text (fast for Excel/Word/PPT, hybrid for PDF/Image)
    # ============================================================
    full_text_parts: List[str] = []
    file_fingerprint_parts: List[str] = []

    excel_tables: Dict[str, pd.DataFrame] = {}

    # PDF
    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()
        file_fingerprint_parts.append("pdf:" + sha1_bytes(pdf_bytes[:300000]))
        pdf_text = extract_text_from_pdf_with_ocr(uploaded_pdf, max_ocr_pages=6)
        if pdf_text.strip():
            full_text_parts.append("=== PDF TEXT ===\n" + pdf_text)

    # Image
    if uploaded_img is not None:
        img_bytes = normalize_image_to_png_bytes(uploaded_img)
        file_fingerprint_parts.append("img:" + sha1_bytes(img_bytes[:200000]))

        img_fp = hashlib.md5(img_bytes[:20000]).hexdigest()
        img_cache_key = f"img_insights:{img_fp}"
        if img_cache_key not in st.session_state:
            with st.spinner("💡 Generating image insights..."):
                try:
                    st.session_state[img_cache_key] = nova_image_insights_brief(img_bytes, image_format="png")
                except Exception:
                    st.session_state[img_cache_key] = ""

        insights = st.session_state.get(img_cache_key, "")
        if insights.strip():
            st.markdown("### 🖼️ Image Insights")
            st.markdown(insights.replace("\n", "  \n"))

        with st.spinner("🔤 OCR image..."):
            try:
                ocr_text = nova_image_to_text(img_bytes, image_format="png")
            except Exception:
                ocr_text = ""

        if ocr_text.strip():
            full_text_parts.append("=== IMAGE TEXT ===\n" + ocr_text)
        if insights.strip():
            full_text_parts.append("=== IMAGE INSIGHTS ===\n" + insights)

    # Excel/CSV
    if uploaded_excel is not None:
        x_bytes = uploaded_excel.getvalue()
        file_fingerprint_parts.append("excel:" + sha1_bytes(x_bytes[:300000]))
        excel_text, excel_tables = read_excel_to_tables(uploaded_excel)
        if excel_tables:
            st.markdown("### 📊 Excel Preview")
            sheet_names = list(excel_tables.keys())
            sel = st.selectbox("Select sheet", sheet_names, index=0)
            df_sel = excel_tables.get(sel)
            if df_sel is not None:
                st.dataframe(df_sel, use_container_width=True, height=340)
        else:
            st.warning("Could not read Excel/CSV content (file may be empty or unsupported).")

        if excel_text.strip():
            full_text_parts.append("=== EXCEL TEXT (preview CSV) ===\n" + excel_text)

    # Word
    if uploaded_docx is not None:
        d_bytes = uploaded_docx.getvalue()
        file_fingerprint_parts.append("docx:" + sha1_bytes(d_bytes[:300000]))
        docx_text = extract_text_from_word(uploaded_docx)
        if docx_text.strip():
            full_text_parts.append("=== DOCX TEXT ===\n" + docx_text)

    # PPT
    if uploaded_pptx is not None:
        p_bytes = uploaded_pptx.getvalue()
        file_fingerprint_parts.append("pptx:" + sha1_bytes(p_bytes[:300000]))
        pptx_text = extract_text_from_pptx(uploaded_pptx)
        if pptx_text.strip():
            full_text_parts.append("=== PPTX TEXT ===\n" + pptx_text)

    # Notes
    if user_text.strip():
        full_text_parts.append("=== USER NOTES ===\n" + user_text.strip())

    full_text = "\n\n".join(full_text_parts).strip()

    # Download extracted text
    if full_text.strip():
        st.download_button(
            "⬇️ Download extracted text (TXT)",
            data=full_text.encode("utf-8", errors="ignore"),
            file_name="extracted_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("🔎 Debug: extracted text length", expanded=False):
        st.write("Characters in full_text:", len(full_text))
        st.write("Preview:", (full_text[:900] + "...") if len(full_text) > 900 else full_text)
        st.write("File fingerprint parts:", file_fingerprint_parts)

    # IMPORTANT: do NOT stop app for Excel; show dashboard fallback even if short
    if len(full_text.strip()) < 30:
        st.warning("Low/empty extracted text. If this is Excel, the table preview above is the primary signal.")

    # ============================================================
    # Fingerprint (fixes “wrong until refresh”)
    # Use BOTH file bytes hash + extracted text hash
    # ============================================================
    file_fp = "|".join(file_fingerprint_parts) if file_fingerprint_parts else "nofile"
    text_fp = hashlib.md5(full_text[:20000].encode("utf-8", errors="ignore")).hexdigest()
    doc_fp = hashlib.md5((file_fp + "::" + text_fp).encode("utf-8", errors="ignore")).hexdigest()

    # ============================================================
    # Dates (cached)
    # ============================================================
    dates_key = f"dates:{doc_fp}"
    if dates_key not in st.session_state:
        st.session_state[dates_key] = extract_dates_with_events(full_text, max_items=120)
    local_dates = st.session_state.get(dates_key, [])

    # ============================================================
    # 📊 Executive Dashboard (Hybrid + Lazy AI)
    # ============================================================
    st.subheader("📊 Executive Dashboard")

    dash_key = f"dashboard:{doc_fp}"

    dcol1, dcol2 = st.columns([1, 5])
    with dcol1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.session_state.pop(dash_key, None)
            st.rerun()

    # Local deterministic mining always
    local_dash = local_mine_metrics(full_text)

    run_ai_now = False
    if use_ai_dashboard:
        if not lazy_ai:
            run_ai_now = True
        else:
            run_ai_now = st.button("⚡ Run AI Dashboard (Nova)", use_container_width=True)

    # Compute AI dashboard only if requested, cache it
    if use_ai_dashboard and run_ai_now and dash_key not in st.session_state:
        with st.spinner("🧠 Nova generating AI dashboard..."):
            try:
                ai_dash = generate_dashboard_insights_dynamic(full_text)
            except Exception:
                ai_dash = {
                    "doc_type_guess": "generic",
                    "summary": "Dashboard could not be generated (AI error). Using local fallback.",
                    "kpis": [],
                    "derived_insights": [],
                    "charts": [],
                    "table_preview": [],
                    "risk_score": 0,
                    "risks": [],
                    "next_actions": [],
                }
            st.session_state[dash_key] = ai_dash

    ai_cached = st.session_state.get(dash_key, {}) if use_ai_dashboard else {}
    dashboard = merge_ai_and_local(ai_cached, local_dash)

    summary = dashboard.get("summary", "") or ""
    doc_type_guess = dashboard.get("doc_type_guess", "generic")
    risk_score = int(dashboard.get("risk_score", 0) or 0)

    kpis = dashboard.get("kpis", []) or []
    derived = dashboard.get("derived_insights", []) or []
    charts = dashboard.get("charts", []) or []
    table_preview = dashboard.get("table_preview", []) or []
    risks = dashboard.get("risks", []) or []
    next_actions = dashboard.get("next_actions", []) or []

    c1, c2, c3 = st.columns(3)
    c1.metric("Doc Type (Detected)", str(doc_type_guess))
    c2.metric("Risk Score", f"{risk_score}/100")
    with c3:
        st.caption("Risk Meter")
        st.progress(min(max(risk_score, 0), 100))

    if summary.strip():
        st.markdown("### 🧾 Executive Summary")
        st.markdown(summary)
    else:
        st.caption("Summary not available (AI not run or low extracted text).")

    if kpis:
        st.markdown("### 🔑 KPIs")
        cols = st.columns(3)
        for i, kpi in enumerate(kpis[:9]):
            with cols[i % 3]:
                st.metric(
                    str(kpi.get("label", "KPI"))[:40],
                    str(kpi.get("value", "-")),
                    (str(kpi.get("note", ""))[:40] if kpi.get("note") else None),
                )

    if derived:
        st.markdown("### ✨ Derived Insights")
        for d in derived[:10]:
            st.markdown(f"- {d}")

    if charts:
        st.markdown("### 📈 Auto Charts")
        for ch in charts[:4]:
            title = ch.get("title", "Chart")
            ctype = ch.get("type", "bar")
            data = ch.get("data", []) or []

            st.markdown(f"**{title}**")
            if not data:
                st.caption("No chart data available.")
                continue

            dfc = pd.DataFrame(data)
            if "x" in dfc.columns and "y" in dfc.columns:
                dfc["x"] = dfc["x"].astype(str)
                dfc["y"] = pd.to_numeric(dfc["y"], errors="coerce").fillna(0)
                if ctype == "line":
                    st.line_chart(dfc.set_index("x")[["y"]])
                else:
                    st.bar_chart(dfc.set_index("x")[["y"]])
            else:
                st.dataframe(dfc, use_container_width=True)

    # show any mined table lines
    if table_preview:
        st.markdown("### 🧩 Table Preview (Mined)")
        st.dataframe(pd.DataFrame(table_preview), use_container_width=True)

    # Key dates
    st.markdown("### 🗓️ Key Dates Timeline (AI Structured)")
    if local_dates:
        df_dates = pd.DataFrame(local_dates).rename(columns={"label": "Event", "value": "Date"})
        st.dataframe(df_dates, use_container_width=True)
    else:
        st.caption("No dates detected (often means extracted text is empty or dates are image-only).")

    if risks:
        st.markdown("### ⚠️ Risks")
        for r in risks[:8]:
            st.markdown(f"- {r}")

    if next_actions:
        st.markdown("### ✅ Next Actions")
        for a in next_actions[:8]:
            st.markdown(f"- {a}")

    if show_ai_inputs:
        show_dashboard_debug(local_dash, ai_cached)

    st.divider()

    # ============================================================
    # Auto RAG settings + Index
    # ============================================================
    if st.session_state.get("last_doc_fp") != doc_fp:
        st.session_state["last_doc_fp"] = doc_fp
        with st.spinner("⚙️ Nova is auto-optimizing retrieval settings..."):
            st.session_state["auto_rag_settings"] = recommend_rag_settings(full_text)
        st.session_state.pop("suggest_fp", None)
        st.session_state.pop("suggested_questions", None)
        st.session_state.pop("index_fp", None)

    rec = st.session_state.get("auto_rag_settings", {"chunk_size": 1000, "overlap": 150, "top_k": 4})
    auto_chunk_size, auto_overlap, auto_top_k = rec["chunk_size"], rec["overlap"], rec["top_k"]

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

    # ============================================================
    # Extract fields
    # ============================================================
    st.subheader("2) Extract key fields (JSON)")
    doc_type = st.selectbox("Document type", DOC_TYPES, index=0)
    if st.button("🧾 Extract key fields as JSON", use_container_width=True):
        with st.spinner("Extracting..."):
            out = extract_fields_json(full_text, doc_type=doc_type)
        st.code(out, language="json")

    st.divider()

    # ============================================================
    # Suggested Questions (LAZY so it doesn't block Excel)
    # ============================================================
    st.markdown("### ✨ Nova-suggested questions")
    if st.button("💡 Generate suggested questions (Nova)", use_container_width=True):
        with st.spinner("Generating questions..."):
            st.session_state["suggested_questions"] = suggest_questions(full_text, user_interest="General", n=6)

    qs = st.session_state.get("suggested_questions", [])
    if qs:
        cols = st.columns(3)
        for i, q in enumerate(qs):
            with cols[i % 3]:
                if st.button(q, use_container_width=True, key=f"dynq_{doc_fp}_{i}"):
                    st.session_state["pending_question"] = q
                    st.rerun()

    st.divider()

    # ============================================================
    # Chat
    # ============================================================
    st.subheader("3) Chat with your document")
    st.markdown("#### 💬 Ask a question")

    pending = st.session_state.pop("pending_question", "")
    if pending:
        run_single_question(pending, top_k=top_k)

    with st.form("ask_form", clear_on_submit=True):
        q_text = st.text_input(
            "Type your question",
            placeholder="e.g., What is the total amount and due date?",
            label_visibility="collapsed",
        )
        ask_clicked = st.form_submit_button("Ask", use_container_width=True)

    if ask_clicked and q_text.strip():
        run_single_question(q_text.strip(), top_k=top_k)

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

    # ============================================================
    # Export PDF report
    # ============================================================
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
            use_container_width=True,
        )

# ============================================================
# Mode: Compare Two Docs (PDFs)
# ============================================================
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

    text_a = extract_text_from_pdf_with_ocr(up_a, max_ocr_pages=4)
    text_b = extract_text_from_pdf_with_ocr(up_b, max_ocr_pages=4)

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
