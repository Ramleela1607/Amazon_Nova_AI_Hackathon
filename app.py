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
    generate_dashboard_insights_dynamic,  # AI dashboard (may fail JSON -> we handle hybrid fallback)
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
# Helpers (Text, OCR, Dates, Dashboard Hybrid Mining)
# ============================================================

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


def extract_text_from_pdf_basic(file_like) -> str:
    """Digital PDFs: try pypdf extraction."""
    reader = PdfReader(file_like)
    texts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            texts.append(f"\n\n--- Page {i+1} ---\n{t}")
    return "".join(texts)


def pdf_pages_to_png_bytes(pdf_bytes: bytes, max_pages: int = 6) -> List[bytes]:
    """
    Render PDF pages to PNG bytes using PyMuPDF if available.
    If PyMuPDF isn't installed, returns [].
    """
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
    Hybrid PDF text:
    1) pypdf (digital text)
    2) if too short -> OCR first pages (render->image->nova_image_to_text)
    Cached by pdf hash.
    """
    pdf_bytes = uploaded_pdf.getvalue()
    cache_key = "pdf_text:" + hashlib.md5(pdf_bytes[:200000]).hexdigest()

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    basic_text = extract_text_from_pdf_basic(io.BytesIO(pdf_bytes))
    basic_len = len((basic_text or "").strip())

    combined = basic_text or ""

    # OCR fallback if likely scanned or image-based PDF
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


def normalize_image_to_png_bytes(uploaded_img) -> bytes:
    img = Image.open(uploaded_img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_dates_with_events(text: str, max_items: int = 120) -> List[Dict[str, str]]:
    """
    Robust date extraction for PDF/OCR text including ranges and month-year (resumes too).
    Returns: [{"label": event, "value": date_or_range}, ...]
    """
    if not text or not str(text).strip():
        return []

    t = str(text)
    t = t.replace("\r", "\n")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)

    # Fix OCR split: "12\nMay\n2024"
    t = re.sub(r"(\d{1,2})\s*\n\s*([A-Za-z]{3,9})\s*\n\s*(\d{2,4})", r"\1 \2 \3", t)

    # Remove ordinals: 12th -> 12
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

            # event = closest phrase before date on same line
            event = ln[:m.start()].strip()
            if not event:
                event = ln[m.end():].strip()

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
    """Extract float from noisy currency/percent strings."""
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
    """
    Deterministic mining:
    - Finds label:value metrics
    - Finds table-ish lines with numbers
    - Produces fallback KPIs + chart candidates
    """
    out = {
        "kpis": [],               # [{"label","value","note"}]
        "charts": [],             # [{"title","type","data":[{"x","y"}]}]
        "table_preview": [],      # list of dict rows (best-effort)
        "derived_insights": [],   # strings
    }

    if not text or not str(text).strip():
        return out

    t = str(text)
    t = t.replace("\r", "\n")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # 1) label : value patterns
    metric_candidates = []
    for line in t.splitlines():
        ln = line.strip()
        if len(ln) < 4:
            continue
        m = re.search(r"^(.{2,60}?)\s*[:\-]\s*([^\n]{1,50})$", ln)
        if m:
            label = m.group(1).strip()
            val = m.group(2).strip()
            metric_candidates.append((label, val))

    # 2) numbers with nearby words (table-ish)
    num_re = re.compile(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")
    loose = []
    for line in t.splitlines():
        ln = line.strip()
        if not ln:
            continue
        nums = list(num_re.finditer(ln))
        if not nums:
            continue
        # Build a small "row preview" if line looks like a table row
        if len(nums) >= 2 and len(ln) <= 180:
            out["table_preview"].append({"row": ln})
            if len(out["table_preview"]) >= 8:
                pass

        for mm in nums[:3]:
            val = mm.group(1)
            left = ln[:mm.start()].strip()
            left = re.sub(r"\s+", " ", left)
            words = left.split()
            label = " ".join(words[-6:]) if words else "Number"
            loose.append((label, val))

    # Combine
    combined = metric_candidates[:max_items] + loose[:max_items]

    # Build KPI list (top 9 numeric values)
    numeric_items = []
    for label, val in combined:
        num = try_parse_number(val)
        if num is None:
            continue
        numeric_items.append((label, num, val))

    numeric_items_sorted = sorted(numeric_items, key=lambda x: abs(x[1]), reverse=True)
    for label, num, raw in numeric_items_sorted[:9]:
        out["kpis"].append({"label": label[:35], "value": raw, "note": ""})

    # Derived insights (basic stats)
    nums_only = [x[1] for x in numeric_items]
    if len(nums_only) >= 3:
        mn, mx = min(nums_only), max(nums_only)
        avg = sum(nums_only) / len(nums_only)
        out["derived_insights"].append(f"Detected {len(nums_only)} numeric values. Min={mn:g}, Max={mx:g}, Avg={avg:g}.")
        # Outlier-ish hint
        if mx != 0 and abs(mx) > 10 * max(1e-9, abs(avg)):
            out["derived_insights"].append("Some values are much larger than average (possible totals/outliers).")

    # Chart candidate: bar chart of top values
    if numeric_items_sorted:
        data = []
        for label, num, _raw in numeric_items_sorted[:12]:
            data.append({"x": label[:28], "y": float(num)})
        out["charts"].append({"title": "Top Numeric Values (Auto)", "type": "bar", "data": data})

    # Keep previews small
    out["table_preview"] = out["table_preview"][:8]

    return out


def merge_ai_and_local(ai: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    """Hybrid merge: AI if available, else deterministic local fallback."""
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
        # Keep AI summary if good; else build minimal fallback
        if local.get("kpis") or local.get("charts"):
            merged["summary"] = "Auto dashboard generated from detected numbers/tables in the document."
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
        if str(kk).startswith(("dashboard:", "img_insights:", "dates:", "pdf_text:")):
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


# Session init
st.session_state.setdefault("uploader_key", 0)

# Sidebar
st.sidebar.header("⚙️ Controls")
st.sidebar.button("🔄 Reset / New session", on_click=reset_session, use_container_width=True)
mode = st.sidebar.radio("Mode", ["Single Document", "Compare Two Documents"])

user_interest = st.sidebar.selectbox(
    "User interest",
    ["General", "Finance", "HR/Recruiter", "Legal", "Research/Student", "Operations"],
    index=0,
)

st.sidebar.markdown("---")

# ============================================================
# Mode: Single Document
# ============================================================
if mode == "Single Document":
    k = st.session_state.get("uploader_key", 0)

    uploaded_pdf = st.file_uploader("📤 Upload a PDF (optional)", type=["pdf"], key=f"pdf_uploader_{k}")
    uploaded_img = st.file_uploader("🖼️ Upload an Image (optional)", type=["png", "jpg", "jpeg", "webp"], key=f"img_uploader_{k}")
    user_text = st.text_area("✍️ Paste extra text / notes (optional)", height=90)

    if uploaded_pdf is None and uploaded_img is None and not user_text.strip():
        st.info("Upload a PDF or Image (or paste text) → Index builds automatically → Start chatting.")
        st.stop()

    full_text_parts = []

    # PDF (with OCR fallback)
    if uploaded_pdf is not None:
        pdf_text = extract_text_from_pdf_with_ocr(uploaded_pdf, max_ocr_pages=6)
        if pdf_text.strip():
            full_text_parts.append("=== PDF TEXT ===\n" + pdf_text)

    # Image (OCR + cached brief insights)
    if uploaded_img is not None:
        img_bytes = normalize_image_to_png_bytes(uploaded_img)
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
            st.subheader("Insights")
            st.markdown(insights.replace("\n", "  \n"))

        # OCR text for retrieval + dates + dashboard mining
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

    full_text = "\n\n".join(full_text_parts).strip()

    # Debug: helps when OCR is empty
    with st.expander("🔎 Debug: extracted text length", expanded=False):
        st.write("Characters in full_text:", len(full_text))
        st.write("Preview:", (full_text[:900] + "...") if len(full_text) > 900 else full_text)

    # fingerprint (MUST be before caching keys)
    doc_fp = hashlib.md5(full_text[:20000].encode("utf-8", errors="ignore")).hexdigest()

    # ============================================================
    # Dates (cached)
    # ============================================================
    dates_key = f"dates:{doc_fp}"
    if dates_key not in st.session_state:
        st.session_state[dates_key] = extract_dates_with_events(full_text, max_items=120)
    local_dates = st.session_state.get(dates_key, [])

    # ============================================================
    # 📊 Executive Dashboard (HYBRID)
    # ============================================================
    st.subheader("📊 Executive Dashboard")
    dash_key = f"dashboard:{doc_fp}"

    dcol1, _ = st.columns([1, 5])
    with dcol1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.session_state.pop(dash_key, None)
            st.rerun()

    # Local deterministic mining always (fast + stable)
    local_dash = local_mine_metrics(full_text)

    # AI dashboard (cached); if it fails -> we still show local fallback
    if dash_key not in st.session_state:
        with st.spinner("Analyzing document for dashboard insights..."):
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

    dashboard = merge_ai_and_local(st.session_state.get(dash_key, {}), local_dash)

    summary = dashboard.get("summary", "")
    doc_type_guess = dashboard.get("doc_type_guess", "generic")
    risk_score = int(dashboard.get("risk_score", 0) or 0)

    kpis = dashboard.get("kpis", []) or []
    derived = dashboard.get("derived_insights", []) or []
    charts = dashboard.get("charts", []) or []
    table_preview = dashboard.get("table_preview", []) or []
    risks = dashboard.get("risks", []) or []
    next_actions = dashboard.get("next_actions", []) or []

    c1, c2, c3 = st.columns(3)
    c1.metric("Doc Type (Nova)", str(doc_type_guess))
    c2.metric("Risk Score", f"{risk_score}/100")
    with c3:
        st.caption("Risk Meter")
        st.progress(min(max(risk_score, 0), 100))

    if summary.strip():
        st.markdown("### 🧾 Executive Summary")
        st.markdown(summary)
    else:
        st.caption("No summary generated.")

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

    if table_preview:
        st.markdown("### 🧩 Table Preview")
        st.dataframe(pd.DataFrame(table_preview), use_container_width=True)

    st.markdown("### 🗓️ Key Dates Timeline (AI Structured)")
    if local_dates:
        df_dates = pd.DataFrame(local_dates).rename(columns={"label": "Event", "value": "Date"})
        st.dataframe(df_dates, use_container_width=True)
    else:
        st.caption("No dates detected in the document. (Often means extracted text is empty)")

    if risks:
        st.markdown("### ⚠️ Risks")
        for r in risks[:8]:
            st.markdown(f"- {r}")

    if next_actions:
        st.markdown("### ✅ Next Actions")
        for a in next_actions[:8]:
            st.markdown(f"- {a}")

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
    # Suggested questions
    # ============================================================
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
                    st.session_state["pending_question"] = q
                    st.rerun()  # IMPORTANT: trigger Q immediately
    else:
        st.caption("Suggestions unavailable for this upload.")

    if st.button("🔄 Refresh questions", use_container_width=True):
        with st.spinner("Refreshing questions..."):
            st.session_state["suggested_questions"] = suggest_questions(full_text, user_interest=user_interest, n=6)
        st.rerun()

    st.divider()

    # ============================================================
    # Chat (manual + auto-run from suggested)
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
    # Export PDF
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
# Mode: Compare Two Docs
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
