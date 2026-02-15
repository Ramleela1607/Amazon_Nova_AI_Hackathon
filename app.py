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
    generate_dashboard_insights_dynamic,  # AI dashboard (may return invalid JSON; we hybrid-fallback)
)

# Optional nicer charts
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    px = None
    PLOTLY_OK = False


# ============================================================
# Page + DARK NEON UI
# ============================================================
st.set_page_config(page_title="Smart Document Copilot", layout="wide")

st.markdown(
    """
<style>
:root{
  --bg0:#060716;
  --bg1:#090a1f;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.10);
  --txt:#eaf2ff;
  --muted: rgba(234,242,255,0.70);
  --neon1:#7c3aed;
  --neon2:#22d3ee;
  --neon3:#a3ff12;
  --warn:#fb7185;
}

.stApp{
  background:
    radial-gradient(900px circle at 15% 20%, rgba(124,58,237,0.22), transparent 60%),
    radial-gradient(900px circle at 85% 15%, rgba(34,211,238,0.18), transparent 58%),
    radial-gradient(1000px circle at 70% 85%, rgba(163,255,18,0.10), transparent 62%),
    linear-gradient(135deg, var(--bg0), var(--bg1));
  color: var(--txt) !important;
}

.block-container{ padding-top: 1rem; }

html, body, [class*="css"], p, span, div, label {
  color: var(--txt) !important;
}

section[data-testid="stSidebar"]{
  background: rgba(0,0,0,0.35) !important;
  backdrop-filter: blur(14px);
  border-right: 1px solid var(--stroke);
}
section[data-testid="stSidebar"] *{
  color: var(--txt) !important;
}

h1,h2,h3{
  color: var(--txt) !important;
  text-shadow: 0 0 18px rgba(124,58,237,0.20);
}

a{ color: var(--neon2) !important; }

div[data-testid="stExpander"]{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 18px;
}

div[data-testid="stChatMessage"]{
  background: var(--card2);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}

div[data-baseweb="input"] input, textarea{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 14px !important;
}

.stButton button{
  background: linear-gradient(135deg, rgba(124,58,237,0.28), rgba(34,211,238,0.16)) !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 14px !important;
  color: var(--txt) !important;
  font-weight: 800 !important;
  box-shadow: 0 10px 28px rgba(124,58,237,0.14);
  transition: transform 120ms ease, box-shadow 120ms ease;
}
.stButton button:hover{
  transform: translateY(-1px);
  box-shadow: 0 14px 38px rgba(34,211,238,0.16);
}

[data-testid="stMetricValue"]{
  text-shadow: 0 0 18px rgba(34,211,238,0.18);
}

hr{
  border-color: rgba(255,255,255,0.12) !important;
}

.neon-card{
  background: var(--card2);
  border: 1px solid rgba(34,211,238,0.18);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 14px 34px rgba(0,0,0,0.36);
}

.neon-pill{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(124,58,237,0.30);
  background: rgba(124,58,237,0.10);
  color: var(--txt);
  font-weight: 700;
  margin-right: 6px;
}

.small-muted{ color: var(--muted) !important; font-size: 0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚡📄 Smart Document Copilot")
st.markdown(
    "<span class='neon-pill'>Amazon Nova</span>"
    "<span class='neon-pill'>Bedrock</span>"
    "<span class='neon-pill'>Multimodal RAG</span>"
    "<span class='neon-pill'>Evidence</span>"
    "<span class='neon-pill'>Dashboard</span>"
    "<div class='small-muted'>Hackathon mode • Neon UI • Works for PDF / Image / Excel / CSV</div>",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers: chunking, OCR, dates, Excel parsing, local mining
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
    reader = PdfReader(file_like)
    texts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            texts.append(f"\n\n--- Page {i+1} ---\n{t}")
    return "".join(texts)


def pdf_pages_to_png_bytes(pdf_bytes: bytes, max_pages: int = 6) -> List[bytes]:
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


def extract_text_from_pdf_with_ocr(uploaded_pdf, max_ocr_pages: int = 6) -> Tuple[str, str]:
    """
    Returns: (combined_text, file_fingerprint)
    fingerprint is based on file bytes -> prevents “first run wrong, refresh correct”
    """
    pdf_bytes = uploaded_pdf.getvalue()
    fp = hashlib.md5(pdf_bytes).hexdigest()
    cache_key = f"pdf_text:{fp}:{max_ocr_pages}"

    if cache_key in st.session_state:
        return st.session_state[cache_key], fp

    basic_text = extract_text_from_pdf_basic(io.BytesIO(pdf_bytes))
    combined = (basic_text or "").strip()

    if len(combined) < 250:
        ocr_parts = []
        page_pngs = pdf_pages_to_png_bytes(pdf_bytes, max_pages=max_ocr_pages)
        if page_pngs:
            with st.spinner(f"🧠 OCR PDF (first {len(page_pngs)} pages)..."):
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
    return combined, fp


def normalize_image_to_png_bytes(uploaded_img) -> bytes:
    img = Image.open(uploaded_img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_text_from_image_with_ocr(uploaded_img) -> Tuple[str, str, str]:
    """
    Returns: (ocr_text, insights_brief, file_fingerprint)
    """
    img_bytes = normalize_image_to_png_bytes(uploaded_img)
    fp = hashlib.md5(img_bytes).hexdigest()

    ins_key = f"img_insights:{fp}"
    if ins_key not in st.session_state:
        with st.spinner("💡 Generating image insights..."):
            try:
                st.session_state[ins_key] = nova_image_insights_brief(img_bytes, image_format="png")
            except Exception:
                st.session_state[ins_key] = ""

    try:
        ocr_text = nova_image_to_text(img_bytes, image_format="png")
    except Exception:
        ocr_text = ""

    return (ocr_text or "").strip(), (st.session_state.get(ins_key, "") or "").strip(), fp


def read_excel_or_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], str, str]:
    """
    Returns: (df, extracted_text, file_fingerprint)
    extracted_text is a compact textual summary for dashboard/RAG.
    """
    b = uploaded_file.getvalue()
    fp = hashlib.md5(b).hexdigest()
    name = (uploaded_file.name or "").lower()

    df = None
    err = None

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(b))
        else:
            # xlsx/xlsm/xlsb/ods etc -> try read_excel
            df = pd.read_excel(io.BytesIO(b), engine="openpyxl")
    except Exception as e:
        err = e

    if df is None or df.empty:
        # try a safer fallback: maybe CSV with weird delimiter
        try:
            df = pd.read_csv(io.BytesIO(b), sep=None, engine="python")
        except Exception:
            df = None

    if df is None or df.empty:
        # return empty
        txt = ""
        if err:
            txt = f"Could not read Excel/CSV content. Error: {str(err)}"
        return None, txt, fp

    # Build compact text summary
    preview_rows = min(20, len(df))
    head = df.head(preview_rows)

    stats_lines = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        for c in numeric_cols[:10]:
            col = pd.to_numeric(df[c], errors="coerce")
            stats_lines.append(
                f"{c}: count={int(col.notna().sum())}, min={col.min(skipna=True)}, max={col.max(skipna=True)}, mean={col.mean(skipna=True)}"
            )

    txt = (
        "=== EXCEL/CSV TABLE PREVIEW ===\n"
        + head.to_csv(index=False)
        + "\n\n=== BASIC STATS ===\n"
        + ("\n".join(stats_lines) if stats_lines else "No numeric columns detected.")
    )
    return df, txt, fp


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


def local_mine_metrics(text: str, max_items: int = 120) -> Dict[str, Any]:
    out = {"kpis": [], "charts": [], "table_preview": [], "derived_insights": []}
    if not text or not str(text).strip():
        return out

    t = str(text)
    t = t.replace("\r", "\n")
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # 1) label:value patterns
    metric_candidates = []
    for line in t.splitlines():
        ln = line.strip()
        if len(ln) < 4:
            continue
        m = re.search(r"^(.{2,70}?)\s*[:\-]\s*([^\n]{1,60})$", ln)
        if m:
            metric_candidates.append((m.group(1).strip(), m.group(2).strip()))

    # 2) numeric signals
    num_re = re.compile(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")
    loose = []
    for line in t.splitlines():
        ln = line.strip()
        if not ln:
            continue
        nums = list(num_re.finditer(ln))
        if not nums:
            continue

        # table-like preview (keep a few rows)
        if len(nums) >= 2 and len(ln) <= 200 and len(out["table_preview"]) < 10:
            out["table_preview"].append({"row": ln})

        for mm in nums[:3]:
            val = mm.group(1)
            left = ln[:mm.start()].strip()
            left = re.sub(r"\s+", " ", left)
            words = left.split()
            label = " ".join(words[-6:]) if words else "Number"
            loose.append((label, val))

    combined = (metric_candidates[:max_items] + loose[:max_items])

    numeric_items = []
    for label, val in combined:
        num = try_parse_number(val)
        if num is None:
            continue
        # try to avoid garbage labels
        clean_label = re.sub(r"[^A-Za-z0-9 %$/._-]+", " ", label).strip()
        if len(clean_label) < 2:
            clean_label = "Number"
        numeric_items.append((clean_label, float(num), val))

    numeric_items_sorted = sorted(numeric_items, key=lambda x: abs(x[1]), reverse=True)

    # KPIs (top 9)
    for label, _num, raw in numeric_items_sorted[:9]:
        out["kpis"].append({"label": label[:40], "value": raw, "note": ""})

    nums_only = [x[1] for x in numeric_items_sorted]
    if len(nums_only) >= 3:
        mn, mx = min(nums_only), max(nums_only)
        avg = sum(nums_only) / len(nums_only)
        out["derived_insights"].append(f"Detected {len(nums_only)} numeric values. Min={mn:g}, Max={mx:g}, Avg={avg:g}.")
        # quick outlier note
        if mx != 0 and abs(mx) > 10 * max(1e-9, abs(avg)):
            out["derived_insights"].append("Some values are much larger than average (likely totals/outliers).")

    if numeric_items_sorted:
        data = [{"x": lab[:28], "y": float(num)} for lab, num, _raw in numeric_items_sorted[:12]]
        out["charts"].append({"title": "Top Numeric Values (Auto)", "type": "bar", "data": data})

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
            merged["summary"] = "Auto dashboard generated from detected signals in the document."
    if "risk_score" not in merged:
        merged["risk_score"] = 0
    if "doc_type_guess" not in merged:
        merged["doc_type_guess"] = "generic"
    if "risks" not in merged:
        merged["risks"] = []
    if "next_actions" not in merged:
        merged["next_actions"] = []
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


def reset_session_inplace():
    # no st.rerun() here (avoid callback no-op warnings)
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

    for kk in list(st.session_state.keys()):
        if str(kk).startswith(("dashboard:", "img_insights:", "dates:", "pdf_text:", "excel_df:", "excel_text:")):
            st.session_state.pop(kk, None)


def build_index_if_needed(full_text: str, chunk_size: int, overlap: int):
    if not full_text or len(full_text.strip()) < 10:
        return

    fp_src = f"{full_text[:25000]}||cs={chunk_size}||ov={overlap}"
    new_fp = hashlib.md5(fp_src.encode("utf-8", errors="ignore")).hexdigest()

    if st.session_state.get("index_fp") == new_fp and isinstance(st.session_state.get("single_rag"), RagIndex):
        return

    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    with st.spinner("⚡ Building index (Nova embeddings → FAISS)..."):
        rag = RagIndex(dim=1024)
        rag.add_chunks(chunks)

    st.session_state["single_rag"] = rag
    st.session_state["index_fp"] = new_fp


def run_single_question(user_q: str, top_k: int):
    rag = st.session_state.get("single_rag", None)
    if not isinstance(rag, RagIndex):
        st.error("Index not ready. Upload a document and build index.")
        st.stop()

    with st.spinner("🔎 Retrieving sources..."):
        hits, scores = rag.search(user_q, k=top_k)
        ctx = [h.text for h in hits]

    with st.spinner("🧠 Thinking with Nova Lite..."):
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

# ============================================================
# SIDEBAR (NO user_interest)
# ============================================================
st.sidebar.header("⚙️ Controls")
if st.sidebar.button("🔄 Reset / New session", use_container_width=True):
    reset_session_inplace()
    st.rerun()

mode = st.sidebar.radio("Mode", ["Single Document", "Compare Two Documents"])

st.sidebar.markdown("---")

# ============================================================
# SINGLE DOCUMENT MODE (PDF / IMAGE / EXCEL / CSV)
# ============================================================
if mode == "Single Document":
    k = st.session_state.get("uploader_key", 0)

    st.sidebar.subheader("📥 Upload")
    input_type = st.sidebar.radio("Choose input type", ["PDF", "Image", "Excel/CSV"], horizontal=False)

    uploaded_pdf = None
    uploaded_img = None
    uploaded_xls = None

    if input_type == "PDF":
        uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"], key=f"pdf_{k}")
    elif input_type == "Image":
        uploaded_img = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"], key=f"img_{k}")
    else:
        uploaded_xls = st.sidebar.file_uploader("Upload Excel / CSV", type=["xlsx", "xlsm", "xls", "csv"], key=f"xls_{k}")

    user_text = st.sidebar.text_area("✍️ Optional notes", height=90, key=f"notes_{k}")

    if uploaded_pdf is None and uploaded_img is None and uploaded_xls is None and not (user_text or "").strip():
        st.info("Upload a file (PDF / Image / Excel/CSV) or paste notes → we build Dashboard + Timeline + RAG Chat.")
        st.stop()

    # ============================================================
    # EXTRACT / OCR / READ TABLES
    # ============================================================
    full_text_parts: List[str] = []
    extracted_df: Optional[pd.DataFrame] = None
    src_fp = "notes_only"

    if uploaded_pdf is not None:
        pdf_text, src_fp = extract_text_from_pdf_with_ocr(uploaded_pdf, max_ocr_pages=6)
        if pdf_text.strip():
            full_text_parts.append("=== PDF TEXT ===\n" + pdf_text)

    if uploaded_img is not None:
        ocr_text, insights, src_fp = extract_text_from_image_with_ocr(uploaded_img)
        if insights:
            st.markdown("<div class='neon-card'><b>🖼️ Image Insights</b><br/>" + insights.replace("\n","<br/>") + "</div>", unsafe_allow_html=True)
        if ocr_text.strip():
            full_text_parts.append("=== IMAGE TEXT ===\n" + ocr_text)
        if insights.strip():
            full_text_parts.append("=== IMAGE INSIGHTS ===\n" + insights)

    if uploaded_xls is not None:
        df, txt, src_fp = read_excel_or_csv(uploaded_xls)
        extracted_df = df
        if df is not None and not df.empty:
            full_text_parts.append(txt)
        else:
            # show the message inside the app (instead of hanging)
            st.warning("Could not read Excel/CSV content (file may be empty or unsupported).")

    if (user_text or "").strip():
        full_text_parts.append("=== USER NOTES ===\n" + user_text.strip())

    full_text = "\n\n".join([p for p in full_text_parts if p and p.strip()]).strip()

    # fingerprint MUST include true source fp (bytes hash)
    doc_fp = hashlib.md5((src_fp + "||" + full_text[:50000]).encode("utf-8", errors="ignore")).hexdigest()

    # ============================================================
    # MAIN TABS (Option A)
    # ============================================================
    tab_dash, tab_timeline, tab_chat, tab_export, tab_debug = st.tabs(
        ["📊 Dashboard", "🗓️ Timeline", "💬 Chat", "⬇️ Export", "🧪 Debug"]
    )

    # ============================================================
    # TAB: DEBUG
    # ============================================================
    with tab_debug:
        st.markdown("<div class='neon-card'><b>🔎 Extraction Debug</b></div>", unsafe_allow_html=True)
        st.write("Input type:", input_type)
        st.write("Source fingerprint:", src_fp)
        st.write("Doc fingerprint:", doc_fp)
        st.write("Extracted text length:", len(full_text))
        if extracted_df is not None:
            st.write("Excel rows/cols:", extracted_df.shape)
        st.text_area("Preview (first 1500 chars)", value=(full_text[:1500] if full_text else ""), height=220)

    # ============================================================
    # DATES (cached)
    # ============================================================
    dates_key = f"dates:{doc_fp}"
    if dates_key not in st.session_state:
        st.session_state[dates_key] = extract_dates_with_events(full_text, max_items=150)
    local_dates = st.session_state.get(dates_key, [])

    # ============================================================
    # DASHBOARD (HYBRID, cached, refreshable)
    # ============================================================
    dash_key = f"dashboard:{doc_fp}"

    with tab_dash:
        st.markdown("<div class='neon-card'><b>📊 Executive Dashboard</b><br/><span class='small-muted'>Hybrid: Nova AI + deterministic fallback (stable for demos)</span></div>", unsafe_allow_html=True)
        cbtn1, cbtn2, _ = st.columns([1, 1, 6])
        with cbtn1:
            refresh_dash = st.button("🔁 Refresh Dashboard", use_container_width=True)
        with cbtn2:
            rebuild_all = st.button("♻️ Re-run Extraction + Dashboard", use_container_width=True)

        if rebuild_all:
            # clear extraction + dashboard caches for this file
            for kk in list(st.session_state.keys()):
                if str(kk).startswith(("dashboard:", "dates:", "pdf_text:", "excel_df:", "excel_text:")):
                    st.session_state.pop(kk, None)
            st.rerun()

        if refresh_dash:
            st.session_state.pop(dash_key, None)
            st.rerun()

        # always compute local mining (fast)
        local_dash = local_mine_metrics(full_text)

        # AI dashboard cached
        if dash_key not in st.session_state:
            with st.spinner("⚡ Nova is building your dashboard..."):
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

        summary = dashboard.get("summary", "") or ""
        doc_type_guess = dashboard.get("doc_type_guess", "generic")
        risk_score = int(dashboard.get("risk_score", 0) or 0)

        kpis = dashboard.get("kpis", []) or []
        derived = dashboard.get("derived_insights", []) or []
        charts = dashboard.get("charts", []) or []
        table_preview = dashboard.get("table_preview", []) or []
        risks = dashboard.get("risks", []) or []
        next_actions = dashboard.get("next_actions", []) or []

        m1, m2, m3 = st.columns(3)
        m1.metric("Doc Type (Nova)", str(doc_type_guess))
        m2.metric("Risk Score", f"{risk_score}/100")
        with m3:
            st.caption("Risk Meter")
            st.progress(min(max(risk_score, 0), 100))

        if summary.strip():
            st.markdown("### 🧾 Executive Summary")
            st.write(summary)

        # Excel preview table (if uploaded)
        if extracted_df is not None and not extracted_df.empty:
            st.markdown("### 📎 Excel Preview")
            st.dataframe(extracted_df.head(50), use_container_width=True)

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

                    if PLOTLY_OK:
                        if ctype == "line":
                            fig = px.line(dfc, x="x", y="y")
                        else:
                            fig = px.bar(dfc, x="x", y="y")
                        fig.update_layout(
                            template="plotly_dark",
                            margin=dict(l=10, r=10, t=40, b=10),
                            height=320,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        if ctype == "line":
                            st.line_chart(dfc.set_index("x")[["y"]])
                        else:
                            st.bar_chart(dfc.set_index("x")[["y"]])
                else:
                    st.dataframe(dfc, use_container_width=True)

        if table_preview:
            st.markdown("### 🧩 Table-ish Preview")
            st.dataframe(pd.DataFrame(table_preview), use_container_width=True)

        if risks:
            st.markdown("### ⚠️ Risks")
            for r in risks[:8]:
                st.markdown(f"- {r}")

        if next_actions:
            st.markdown("### ✅ Next Actions")
            for a in next_actions[:8]:
                st.markdown(f"- {a}")

    # ============================================================
    # TAB: TIMELINE
    # ============================================================
    with tab_timeline:
        st.markdown("<div class='neon-card'><b>🗓️ Key Dates Timeline</b><br/><span class='small-muted'>Extracted from PDF / OCR / Table text</span></div>", unsafe_allow_html=True)
        if local_dates:
            df_dates = pd.DataFrame(local_dates).rename(columns={"label": "Event", "value": "Date"})
            st.dataframe(df_dates, use_container_width=True, height=520)
        else:
            st.warning("No dates detected. If this is a scanned PDF/image, use clearer scan or ensure OCR is working.")

    # ============================================================
    # Build RAG index (once per doc_fp + settings)
    # ============================================================
    if st.session_state.get("last_doc_fp") != doc_fp:
        st.session_state["last_doc_fp"] = doc_fp
        with st.spinner("⚙️ Auto-optimizing retrieval settings..."):
            st.session_state["auto_rag_settings"] = recommend_rag_settings(full_text)
        st.session_state.pop("suggest_fp", None)
        st.session_state.pop("suggested_questions", None)
        st.session_state.pop("index_fp", None)

    rec = st.session_state.get("auto_rag_settings", {"chunk_size": 1000, "overlap": 150, "top_k": 4})
    auto_chunk_size, auto_overlap, auto_top_k = rec["chunk_size"], rec["overlap"], rec["top_k"]

    st.sidebar.subheader("🔎 Retrieval settings")
    use_auto = st.sidebar.toggle("Use auto settings", value=True)

    if use_auto:
        chunk_size, overlap, top_k = auto_chunk_size, auto_overlap, auto_top_k
        st.sidebar.caption(f"Auto: chunk={chunk_size}, overlap={overlap}, top_k={top_k}")
    else:
        chunk_size = st.sidebar.slider("Chunk size", 300, 2000, int(auto_chunk_size), 50)
        overlap = st.sidebar.slider("Overlap", 0, 400, int(auto_overlap), 25)
        top_k = st.sidebar.slider("Top-K sources", 2, 8, int(auto_top_k), 1)

    if st.sidebar.button("♻️ Rebuild index now", use_container_width=True):
        st.session_state.pop("index_fp", None)

    build_index_if_needed(full_text, chunk_size=chunk_size, overlap=overlap)

    # ============================================================
    # Suggested questions (NO user_interest)
    # ============================================================
    suggest_fp = f"{doc_fp}"
    if st.session_state.get("suggest_fp") != suggest_fp:
        st.session_state["suggest_fp"] = suggest_fp
        with st.spinner("✨ Generating Nova-suggested questions..."):
            st.session_state["suggested_questions"] = suggest_questions(full_text, user_interest="General", n=6)

    qs = st.session_state.get("suggested_questions", [])

    # ============================================================
    # TAB: CHAT
    # ============================================================
    with tab_chat:
        st.markdown("<div class='neon-card'><b>💬 Chat with your document</b><br/><span class='small-muted'>RAG + evidence</span></div>", unsafe_allow_html=True)

        st.markdown("### ✨ Nova-suggested questions")
        if qs:
            cols = st.columns(3)
            for i, q in enumerate(qs):
                with cols[i % 3]:
                    if st.button(q, use_container_width=True, key=f"dynq_{doc_fp}_{i}"):
                        st.session_state["pending_question"] = q
                        st.rerun()
        else:
            st.caption("No suggested questions available.")

        if st.button("🔄 Refresh questions", use_container_width=True):
            with st.spinner("Refreshing questions..."):
                st.session_state["suggested_questions"] = suggest_questions(full_text, user_interest="General", n=6)
            st.rerun()

        st.markdown("---")
        pending = st.session_state.pop("pending_question", "")
        if pending:
            run_single_question(pending, top_k=top_k)

        with st.form("ask_form", clear_on_submit=True):
            q_text = st.text_input("Ask", placeholder="e.g., What are the key metrics and deadlines?", label_visibility="collapsed")
            ask_clicked = st.form_submit_button("Ask Nova", use_container_width=True)

        if ask_clicked and q_text.strip():
            run_single_question(q_text.strip(), top_k=top_k)

        if st.session_state.get("latest_answer"):
            st.markdown("### ✅ Latest Answer")
            st.markdown(f"**Question:** {st.session_state.get('latest_q','')}")
            st.write(st.session_state["latest_answer"] or "I don't know based on the document.")

            ev = st.session_state.get("latest_evidence", "")
            if ev:
                with st.expander("📌 Evidence"):
                    st.markdown(ev)

            st.caption(f"⏱ Response time: {st.session_state.get('latest_rt', 0)} sec")

    # ============================================================
    # TAB: EXPORT (downloads + report)
    # ============================================================
    with tab_export:
        st.markdown("<div class='neon-card'><b>⬇️ Download Center</b><br/><span class='small-muted'>Export extracted text + dashboard + timeline + PDF report</span></div>", unsafe_allow_html=True)

        # Download extracted text
        if full_text.strip():
            st.download_button(
                "⬇️ Download extracted text (.txt)",
                data=full_text.encode("utf-8", errors="ignore"),
                file_name="extracted_text.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.warning("Extracted text is empty. If this is scanned, OCR may not be reading it.")

        # Download dashboard JSON
        dash_obj = st.session_state.get(dash_key, {})
        st.download_button(
            "⬇️ Download dashboard JSON",
            data=json.dumps(dash_obj, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="dashboard.json",
            mime="application/json",
            use_container_width=True,
        )

        # Download dates CSV
        if local_dates:
            df_dates = pd.DataFrame(local_dates).rename(columns={"label": "Event", "value": "Date"})
            st.download_button(
                "⬇️ Download dates timeline (.csv)",
                data=df_dates.to_csv(index=False).encode("utf-8"),
                file_name="dates_timeline.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### 📄 PDF Report")
        if st.button("🧾 Generate PDF report", use_container_width=True):
            with st.spinner("Creating report title with Nova..."):
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
                ("Executive summary", (summary or "")),
                ("Derived insights", "\n".join(derived[:12]) if derived else "None"),
                ("Key Dates", "\n".join([f"- {d['value']}: {d['label']}" for d in local_dates[:30]]) if local_dates else "None"),
                ("Latest Q&A", latest_block or "No Q&A yet."),
                ("Extracted text preview (first 1200 chars)", full_text[:1200]),
            ]

            safe_filename = "".join(ch for ch in report_title if ch.isalnum() or ch in (" ", "-", "_")).strip()
            safe_filename = safe_filename.replace(" ", "_") or "Smart_Document_Report"
            file_name = f"{safe_filename}.pdf"

            pdf_bytes = make_pdf_report(file_name, report_title, sections)
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
                use_container_width=True,
            )


# ============================================================
# COMPARE TWO DOCS MODE
# ============================================================
else:
    st.subheader("🆚 Compare Two PDFs (Hybrid OCR)")
    k = st.session_state.get("uploader_key", 0)

    c1, c2 = st.columns(2)
    with c1:
        up_a = st.file_uploader("Upload PDF (Doc A)", type=["pdf"], key=f"docA_{k}")
    with c2:
        up_b = st.file_uploader("Upload PDF (Doc B)", type=["pdf"], key=f"docB_{k}")

    if up_a is None or up_b is None:
        st.info("Upload both PDFs → Ask a comparison question.")
        st.stop()

    text_a, _ = extract_text_from_pdf_with_ocr(up_a, max_ocr_pages=4)
    text_b, _ = extract_text_from_pdf_with_ocr(up_b, max_ocr_pages=4)

    with st.spinner("⚙️ Auto-tuning retrieval settings for comparison..."):
        rec = recommend_rag_settings(text_a + "\n\n" + text_b)

    chunk_size, overlap, top_k = rec["chunk_size"], rec["overlap"], rec["top_k"]

    chunks_a = chunk_text(text_a, chunk_size=chunk_size, overlap=overlap)
    chunks_b = chunk_text(text_b, chunk_size=chunk_size, overlap=overlap)

    if "rag_a" not in st.session_state or "rag_b" not in st.session_state:
        with st.spinner("⚡ Building indexes for Doc A & Doc B..."):
            rag_a = RagIndex(dim=1024)
            rag_a.add_chunks(chunks_a)
            rag_b = RagIndex(dim=1024)
            rag_b.add_chunks(chunks_b)
        st.session_state["rag_a"] = rag_a
        st.session_state["rag_b"] = rag_b

    q = st.text_input("Comparison question", placeholder="e.g., Which document has higher totals and key deadlines?")
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
