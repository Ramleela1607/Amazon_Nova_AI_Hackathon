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
    generate_dashboard_insights_dynamic,
)

# ============================================================
# UI
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
}
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.78) !important;
  backdrop-filter: blur(14px);
}
div[data-testid="stExpander"], div[data-testid="stChatMessage"] {
  border-radius: 18px;
  border: 1px solid rgba(15,23,42,0.10);
  background: rgba(255,255,255,0.86);
}
.stButton button { border-radius: 14px; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📄 Smart Document Copilot")
st.markdown(
    "<div style='opacity:0.9'>Amazon Nova on Bedrock • Multimodal RAG • Evidence • Compare • PDF Report</div>",
    unsafe_allow_html=True,
)

# ============================================================
# Reset (FIXES YOUR ERROR)
# ============================================================
def reset_session():
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1

    # wipe known keys
    wipe = [
        "single_rag", "index_fp", "auto_rag_settings", "last_doc_id",
        "suggested_questions", "pending_question",
        "latest_q", "latest_answer", "latest_evidence", "latest_sources", "latest_rt",
        "rag_a", "rag_b",
    ]
    for k in wipe:
        st.session_state.pop(k, None)

    # wipe caches
    for k in list(st.session_state.keys()):
        if str(k).startswith(("pdf_text:", "img_insights:", "dashboard:", "dates:", "csv:", "excel:")):
            st.session_state.pop(k, None)

    # reset dashboard nonce map
    st.session_state["dash_nonce"] = {}

    st.rerun()

# ============================================================
# Small utils
# ============================================================
def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def md5_text(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8", errors="ignore")).hexdigest()

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

# ============================================================
# PDF extraction (hybrid text + OCR fallback)
# ============================================================
def extract_text_from_pdf_basic(file_like) -> str:
    reader = PdfReader(file_like)
    texts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            texts.append(f"\n\n--- Page {i+1} ---\n{t}")
    return "".join(texts).strip()

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

def extract_text_from_pdf_with_ocr(pdf_bytes: bytes, max_ocr_pages: int = 6) -> str:
    cache_key = "pdf_text:" + sha1_bytes(pdf_bytes[:400000])
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    basic = extract_text_from_pdf_basic(io.BytesIO(pdf_bytes))
    combined = (basic or "").strip()

    if len(combined) < 250:
        page_pngs = pdf_pages_to_png_bytes(pdf_bytes, max_pages=max_ocr_pages)
        ocr_parts = []
        if page_pngs:
            for idx, png in enumerate(page_pngs, start=1):
                try:
                    t = nova_image_to_text(png, image_format="png")
                except Exception:
                    t = ""
                if t.strip():
                    ocr_parts.append(f"\n\n--- OCR Page {idx} ---\n{t.strip()}")
        if ocr_parts:
            combined = (combined + "\n\n" + "\n".join(ocr_parts)).strip()

    st.session_state[cache_key] = combined
    return combined

# ============================================================
# Image extraction
# ============================================================
def normalize_image_to_png_bytes(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def extract_text_from_image(img_bytes: bytes) -> Tuple[str, str]:
    png_bytes = normalize_image_to_png_bytes(img_bytes)
    img_fp = hashlib.md5(png_bytes[:20000]).hexdigest()

    insights_key = f"img_insights:{img_fp}"
    if insights_key not in st.session_state:
        try:
            st.session_state[insights_key] = nova_image_insights_brief(png_bytes, image_format="png") or ""
        except Exception:
            st.session_state[insights_key] = ""

    insights = st.session_state.get(insights_key, "")

    try:
        ocr = nova_image_to_text(png_bytes, image_format="png") or ""
    except Exception:
        ocr = ""

    return ocr.strip(), insights.strip()

# ============================================================
# Excel/CSV extraction (robust)
# ============================================================
def read_csv_bytes(csv_bytes: bytes) -> Tuple[str, Optional[pd.DataFrame]]:
    cache_key = "csv:" + sha1_bytes(csv_bytes[:400000])
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    df = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes), encoding=enc)
            if df is not None and not df.empty:
                break
        except Exception:
            df = None

    if df is None or df.empty:
        st.session_state[cache_key] = ("", None)
        return "", None

    df = df.dropna(how="all")
    text = df.head(40).to_csv(index=False)

    st.session_state[cache_key] = (text.strip(), df)
    return text.strip(), df

def read_excel_bytes(excel_bytes: bytes, filename: str) -> Tuple[str, Dict[str, pd.DataFrame], str]:
    cache_key = "excel:" + sha1_bytes(excel_bytes[:400000])
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    warning = ""
    tables: Dict[str, pd.DataFrame] = {}
    text_parts: List[str] = []

    ext = (filename or "").lower().split(".")[-1].strip()

    if ext in ["xlsx", "xlsm"]:
        try:
            xls = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
            for sh in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sh, engine="openpyxl")
                    df = df.dropna(how="all")
                    if df is not None and not df.empty:
                        tables[sh] = df
                        text_parts.append(f"--- Sheet: {sh} ---")
                        text_parts.append(df.head(30).to_csv(index=False))
                except Exception:
                    continue
        except Exception as e:
            warning = f"Excel read failed (.xlsx). Error: {e}"

    elif ext == "xls":
        warning = "Legacy .xls detected. Please save as .xlsx (recommended) OR install `xlrd` and enable xls support."
    else:
        warning = "Unsupported Excel extension. Upload .xlsx/.xlsm (recommended)."

    extracted_text = "\n".join(text_parts).strip()
    result = (extracted_text, tables, warning)
    st.session_state[cache_key] = result
    return result

# ============================================================
# Dates
# ============================================================
def extract_dates_with_events(text: str, max_items: int = 120) -> List[Dict[str, str]]:
    if not text or not str(text).strip():
        return []

    t = str(text)
    t = t.replace("\r", "\n").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", t, flags=re.IGNORECASE)

    months = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
    pats = [
        rf"\b({months}\s+\d{{4}})\s*-\s*(Present|{months}\s+\d{{4}})\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        rf"\b{months}\s+\d{{1,2}},?\s+\d{{2,4}}\b",
        rf"\b\d{{1,2}}\s+{months},?\s+\d{{2,4}}\b",
        rf"\b{months}\s+\d{{4}}\b",
    ]
    date_re = re.compile("|".join(pats), re.IGNORECASE)

    results = []
    seen = set()

    for line in t.split("\n"):
        ln = line.strip()
        if not ln:
            continue
        for m in date_re.finditer(ln):
            ds = m.group(0).strip()
            ev = (ln[:m.start()].strip() or ln[m.end():].strip())
            ev = re.sub(r"\s+", " ", ev).strip(" -:•;|")
            if not ev:
                ev = "Date mentioned"

            key = (ev.lower(), ds.lower())
            if key in seen:
                continue
            seen.add(key)

            results.append({"label": ev[:70], "value": ds})
            if len(results) >= max_items:
                break
        if len(results) >= max_items:
            break

    return results

# ============================================================
# Dashboard (local mining + AI merge)
# ============================================================
def local_dashboard_from_text(text: str, max_items: int = 120) -> Dict[str, Any]:
    """
    Stronger deterministic mining for OCR/PDF:
    - Prefers explicit metrics (Label: Value) and (Label 4,250)
    - Ignores years/dates and small junk integers
    - Dedupes values
    - Produces cleaner KPI labels + chart
    """
    out = {"kpis": [], "charts": [], "table_preview": [], "derived_insights": []}
    if not text or not str(text).strip():
        return out

    t = str(text).replace("\r", "\n").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Join broken lines lightly to catch "Users Onboarded\n4,250"
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    joined = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        # if line has no number but next line is mainly a number -> join
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if (not re.search(r"\d", ln)) and re.fullmatch(r"[₹$€]?\s*[\d,]+(\.\d+)?%?", nxt):
                joined.append(f"{ln} {nxt}")
                i += 2
                continue
            if (not re.search(r"\d", ln)) and re.search(r"^[\d,]+(\.\d+)?%?$", nxt):
                joined.append(f"{ln} {nxt}")
                i += 2
                continue
            # common OCR: "Users Onboarded" then "4,250"
            if (not re.search(r"\d", ln)) and re.fullmatch(r"[\d,]+(\.\d+)?", nxt):
                joined.append(f"{ln}: {nxt}")
                i += 2
                continue
        joined.append(ln)
        i += 1

    # Helpers
    months = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
    date_like = re.compile(rf"({months}\s+\d{{1,2}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\b\d{{4}}\b)", re.IGNORECASE)

    def is_year(n: float) -> bool:
        return 1900 <= n <= 2100 and float(int(n)) == float(n)

    def bad_label(lbl: str) -> bool:
        s = (lbl or "").strip().lower()
        if not s or s in {"number", "value", "metric"}:
            return True
        if len(s) < 3:
            return True
        # avoid date-driven labels
        if any(x in s for x in ["date", "uat", "window", "kickoff", "go-live", "timeline"]):
            return True
        return False

    # Extract candidates with priority
    candidates = []

    # 1) Strong: Label: Value
    for ln in joined:
        m = re.search(r"^(.{2,60}?)\s*[:\-]\s*([₹$€]?\s*[\d,]+(?:\.\d+)?%?)\s*$", ln)
        if m:
            label = m.group(1).strip()
            val = m.group(2).strip()
            candidates.append((label, val, 3))

    # 2) Medium: "Label 4,250" (no colon)
    for ln in joined:
        m = re.search(r"^(.{3,60}?)\s+([₹$€]?\s*[\d,]+(?:\.\d+)?%?)\s*$", ln)
        if m:
            label = m.group(1).strip()
            val = m.group(2).strip()
            # avoid things that look like dates
            if date_like.search(ln):
                continue
            candidates.append((label, val, 2))

    # 3) Weak: any numbers with a nearby label
    num_re = re.compile(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")
    for ln in joined:
        nums = list(num_re.finditer(ln))
        if not nums:
            continue

        # table-ish preview
        if len(nums) >= 2 and len(ln) <= 220:
            out["table_preview"].append({"row": ln})

        for mm in nums[:2]:
            raw = mm.group(1)
            left = re.sub(r"\s+", " ", ln[:mm.start()].strip())
            label = " ".join(left.split()[-6:]) if left else "Number"
            candidates.append((label, raw, 1))

    # Convert to numeric, filter junk, dedupe
    numeric = []
    seen = set()

    for label, raw, prio in candidates[:max_items * 3]:
        num = try_parse_number(raw)
        if num is None:
            continue

        # Filters
        if is_year(num):
            continue
        if abs(num) < 10:  # tiny integers usually junk in OCR
            continue
        if date_like.search(label):
            continue
        if bad_label(label):
            continue

        key = (label.lower().strip(), round(float(num), 6))
        if key in seen:
            continue
        seen.add(key)

        numeric.append((prio, label, float(num), raw))

    # Pick best KPIs: sort by priority then magnitude
    numeric_sorted = sorted(numeric, key=lambda x: (x[0], abs(x[2])), reverse=True)

    for prio, label, num, raw in numeric_sorted[:9]:
        out["kpis"].append({"label": label[:40], "value": raw, "note": ""})

    # Derived insights
    vals = [x[2] for x in numeric_sorted]
    if len(vals) >= 3:
        mn, mx = min(vals), max(vals)
        avg = sum(vals) / len(vals)
        out["derived_insights"].append(f"Detected {len(vals)} numeric metrics (filtered). Min={mn:g}, Max={mx:g}, Avg={avg:g}.")

    # Chart from KPIs
    if numeric_sorted:
        data = [{"x": lab[:28], "y": float(v)} for _p, lab, v, _raw in numeric_sorted[:12]]
        out["charts"].append({"title": "Top Metrics (Filtered)", "type": "bar", "data": data})

    out["table_preview"] = out["table_preview"][:8]
    return out

def local_dashboard_from_excel_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    out = {"kpis": [], "charts": [], "table_preview": [], "derived_insights": []}
    if not tables:
        return out

    first_sheet = next(iter(tables.keys()))
    df0 = tables[first_sheet]
    out["table_preview"].append({"sheet": first_sheet, "rows": int(df0.shape[0]), "cols": int(df0.shape[1])})

    numeric_points = []
    for sh, df in list(tables.items())[:6]:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for c in num_cols[:8]:
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            if series.empty:
                continue
            numeric_points.append((f"{sh}:{c} sum", float(series.sum())))
            numeric_points.append((f"{sh}:{c} avg", float(series.mean())))
            numeric_points.append((f"{sh}:{c} max", float(series.max())))

    numeric_points = sorted(numeric_points, key=lambda x: abs(x[1]), reverse=True)

    for label, val in numeric_points[:9]:
        out["kpis"].append({"label": label[:40], "value": f"{val:,.2f}", "note": ""})

    if numeric_points:
        chart_data = [{"x": lbl[:28], "y": float(v)} for lbl, v in numeric_points[:12]]
        out["charts"].append({"title": "Top Excel Metrics (Auto)", "type": "bar", "data": chart_data})

    out["derived_insights"].append(f"Parsed {len(tables)} sheets and computed numeric summaries.")
    return out

def salvage_json(txt: str) -> Optional[Dict[str, Any]]:
    if not txt:
        return None
    t = txt.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(t[s:e+1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None

def merge_ai_and_local(ai: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(ai or {})
    merged.setdefault("doc_type_guess", "generic")
    merged.setdefault("summary", "")
    merged.setdefault("risk_score", 0)
    merged.setdefault("risks", [])
    merged.setdefault("next_actions", [])

    if not isinstance(merged.get("kpis"), list) or not merged.get("kpis"):
        merged["kpis"] = local.get("kpis", [])
    if not isinstance(merged.get("charts"), list) or not merged.get("charts"):
        merged["charts"] = local.get("charts", [])
    if not isinstance(merged.get("table_preview"), list) or not merged.get("table_preview"):
        merged["table_preview"] = local.get("table_preview", [])
    if not isinstance(merged.get("derived_insights"), list) or not merged.get("derived_insights"):
        merged["derived_insights"] = local.get("derived_insights", [])

    if (not merged.get("summary")) and (local.get("kpis") or local.get("charts")):
        merged["summary"] = "Auto dashboard generated from detected signals in the document."
    return merged

# ============================================================
# Report
# ============================================================
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

# ============================================================
# Session init + Sidebar
# ============================================================
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("dash_nonce", {})

st.sidebar.header("⚙️ Controls")
st.sidebar.button("🔄 Reset / New session", on_click=reset_session, use_container_width=True)

mode = st.sidebar.radio("Mode", ["Single Document", "Compare Two Documents"])
st.sidebar.markdown("---")

st.sidebar.subheader("🧠 Dashboard Controls")
use_ai_dashboard = st.sidebar.toggle("Use AI Dashboard (Nova)", value=True)
lazy_ai = st.sidebar.toggle("Lazy AI (fast first render)", value=True)
show_debug = st.sidebar.toggle("Show debug (text preview)", value=False)

st.sidebar.markdown("---")

# ============================================================
# RAG Index helpers
# ============================================================
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
        st.error("Index not ready yet.")
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

# ============================================================
# Mode: Single Document
# ============================================================
if mode == "Single Document":
    k = st.session_state.get("uploader_key", 0)

    st.subheader("1) Upload a document (PDF / Image / Excel / CSV / Word / PPT / TXT)")
    uploaded = st.file_uploader(
        "Upload file",
        type=["pdf", "png", "jpg", "jpeg", "webp", "xlsx", "xlsm", "xls", "csv", "docx", "pptx", "txt"],
        key=f"any_uploader_{k}",
    )
    user_text = st.text_area("✍️ Paste extra text / notes (optional)", height=90)

    if uploaded is None and not user_text.strip():
        st.info("Upload a file or paste text to generate dashboard + chat.")
        st.stop()

    file_bytes = b""
    filename = ""
    ext = ""

    if uploaded is not None:
        file_bytes = uploaded.getvalue() or b""
        filename = uploaded.name or ""
        ext = filename.lower().split(".")[-1].strip()

    file_hash = sha1_bytes(file_bytes[:600000]) if file_bytes else "nofile"

    extracted_text = ""
    excel_tables: Dict[str, pd.DataFrame] = {}
    img_insights = ""
    csv_df: Optional[pd.DataFrame] = None

    t0 = time.time()

    if uploaded is not None and file_bytes:
        if ext == "pdf":
            with st.spinner("Extracting PDF text/OCR..."):
                extracted_text = extract_text_from_pdf_with_ocr(file_bytes, max_ocr_pages=6)

        elif ext in ["png", "jpg", "jpeg", "webp"]:
            with st.spinner("Extracting image OCR + insights..."):
                extracted_text, img_insights = extract_text_from_image(file_bytes)

        elif ext in ["xlsx", "xlsm", "xls"]:
            with st.spinner("Reading Excel sheets..."):
                extracted_text, excel_tables, warn = read_excel_bytes(file_bytes, filename)
            if warn:
                st.warning(warn)
            if not excel_tables and not extracted_text.strip():
                st.error("Could not read Excel content. Try saving as .xlsx (not .xls) and re-upload.")
                st.stop()

        elif ext == "csv":
            with st.spinner("Reading CSV..."):
                extracted_text, csv_df = read_csv_bytes(file_bytes)
            if csv_df is None:
                st.error("Could not read CSV content. Try saving as UTF-8 and re-upload.")
                st.stop()

        elif ext == "docx":
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_bytes))
                extracted_text = "\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()]).strip()
            except Exception:
                st.error("DOCX read failed. Ensure file is valid.")
                extracted_text = ""

        elif ext == "pptx":
            try:
                from pptx import Presentation
                prs = Presentation(io.BytesIO(file_bytes))
                parts = []
                for si, slide in enumerate(prs.slides, start=1):
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text and shape.text.strip():
                            parts.append(f"[Slide {si}] {shape.text.strip()}")
                extracted_text = "\n".join(parts).strip()
            except Exception:
                st.error("PPTX read failed. Export PPT as PDF and upload.")
                extracted_text = ""

        elif ext == "txt":
            extracted_text = file_bytes.decode("utf-8", errors="ignore").strip()

    extract_secs = round(time.time() - t0, 2)

    parts = []
    if extracted_text.strip():
        parts.append("=== EXTRACTED TEXT ===\n" + extracted_text.strip())
    if img_insights.strip():
        parts.append("=== IMAGE INSIGHTS ===\n" + img_insights.strip())
    if user_text.strip():
        parts.append("=== USER NOTES ===\n" + user_text.strip())

    full_text = "\n\n".join(parts).strip()

    # Excel preview
    if excel_tables:
        st.markdown("### 📊 Excel Preview")
        sheet_names = list(excel_tables.keys())
        sel = st.selectbox("Sheet", sheet_names, index=0)
        st.dataframe(excel_tables[sel], use_container_width=True, height=360)

    if csv_df is not None:
        st.markdown("### 📄 CSV Preview")
        st.dataframe(csv_df.head(300), use_container_width=True, height=360)

    if img_insights.strip():
        st.markdown("### 💡 Image Insights")
        st.markdown(img_insights.replace("\n", "  \n"))

    # Download extracted text
    st.markdown("### ⬇️ Download extracted text")
    st.download_button(
        "Download extracted text (.txt)",
        data=(full_text or "").encode("utf-8", errors="ignore"),
        file_name="extracted_text.txt",
        mime="text/plain",
        use_container_width=True,
    )

    if show_debug:
        with st.expander("🔎 Debug", expanded=False):
            st.write("Filename:", filename)
            st.write("Ext:", ext)
            st.write("File hash:", file_hash)
            st.write("Extract seconds:", extract_secs)
            st.write("Extracted text length:", len(full_text))
            st.write((full_text[:1400] + "...") if len(full_text) > 1400 else full_text)

    # ============================
    # Stable doc_id + refresh nonce
    # ============================
    text_hash = md5_text(full_text[:30000])
    doc_id = f"{file_hash}:{text_hash}"

    st.session_state["dash_nonce"].setdefault(doc_id, 0)
    nonce = st.session_state["dash_nonce"][doc_id]

    dash_key = f"dashboard:{doc_id}:n={nonce}"
    dates_key = f"dates:{doc_id}:n={nonce}"

    # Dates cache
    if dates_key not in st.session_state:
        st.session_state[dates_key] = extract_dates_with_events(full_text, max_items=120)
    local_dates = st.session_state.get(dates_key, [])

    # ============================
    # Executive Dashboard
    # ============================
    st.subheader("📊 Executive Dashboard")

    cbtn1, cbtn2 = st.columns([1, 6])
    with cbtn1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.session_state["dash_nonce"][doc_id] += 1
            st.rerun()

    # Local dashboard always
    if excel_tables:
        local_dash = local_dashboard_from_excel_tables(excel_tables)
    else:
        local_dash = local_dashboard_from_text(full_text)

    # AI dashboard (optional)
    run_ai_now = False
    if use_ai_dashboard:
        run_ai_now = True if not lazy_ai else st.button("⚡ Run AI Dashboard (Nova)", use_container_width=True)

    if dash_key not in st.session_state:
        ai_dash = {}
        if use_ai_dashboard and run_ai_now and full_text.strip():
            with st.spinner("🧠 Nova generating AI dashboard..."):
                try:
                    raw = generate_dashboard_insights_dynamic(full_text)
                    if isinstance(raw, dict):
                        ai_dash = raw
                    else:
                        ai_dash = salvage_json(str(raw)) or {}
                except Exception:
                    ai_dash = {}
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

    top1, top2, top3 = st.columns(3)
    top1.metric("Doc Type", str(doc_type_guess))
    top2.metric("Risk Score", f"{risk_score}/100")
    with top3:
        st.caption("Risk Meter")
        st.progress(min(max(risk_score, 0), 100))

    if summary.strip():
        st.markdown("### 🧾 Executive Summary")
        st.markdown(summary)

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
        for d in derived[:12]:
            st.markdown(f"- {d}")

    if charts:
        st.markdown("### 📈 Auto Charts")
        for ch in charts[:4]:
            title = ch.get("title", "Chart")
            ctype = ch.get("type", "bar")
            data = ch.get("data", []) or []

            st.markdown(f"**{title}**")
            if not data:
                st.caption("No chart data.")
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

    st.markdown("### 🗓️ Key Dates Timeline")
    if local_dates:
        st.dataframe(pd.DataFrame(local_dates).rename(columns={"label": "Event", "value": "Date"}), use_container_width=True)
    else:
        st.caption("No dates detected in extracted text.")

    if risks:
        st.markdown("### ⚠️ Risks")
        for r in risks[:10]:
            st.markdown(f"- {r}")

    if next_actions:
        st.markdown("### ✅ Next Actions")
        for a in next_actions[:10]:
            st.markdown(f"- {a}")

    st.divider()

    # ============================
    # Auto RAG settings + Index
    # ============================
    if st.session_state.get("last_doc_id") != doc_id:
        st.session_state["last_doc_id"] = doc_id
        with st.spinner("⚙️ Auto-optimizing retrieval settings..."):
            st.session_state["auto_rag_settings"] = recommend_rag_settings(full_text)
        st.session_state.pop("index_fp", None)

    rec = st.session_state.get("auto_rag_settings", {"chunk_size": 1000, "overlap": 150, "top_k": 4})
    chunk_size, overlap, top_k = rec["chunk_size"], rec["overlap"], rec["top_k"]

    build_index_if_needed(full_text, chunk_size=chunk_size, overlap=overlap)

    # ============================
    # Extract fields JSON
    # ============================
    st.subheader("2) Extract key fields (JSON)")
    doc_type = st.selectbox("Document type", DOC_TYPES, index=0)
    if st.button("🧾 Extract key fields as JSON", use_container_width=True):
        with st.spinner("Extracting..."):
            out = extract_fields_json(full_text, doc_type=doc_type)
        st.code(out, language="json")

    st.divider()

    # ============================
    # Suggested questions (manual)
    # ============================
    st.markdown("### ✨ Nova-suggested questions")
    if st.button("💡 Generate suggested questions", use_container_width=True):
        with st.spinner("Generating questions..."):
            st.session_state["suggested_questions"] = suggest_questions(full_text, user_interest="General", n=6)

    qs = st.session_state.get("suggested_questions", [])
    if qs:
        cols = st.columns(3)
        for i, q in enumerate(qs):
            with cols[i % 3]:
                if st.button(q, use_container_width=True, key=f"dynq_{doc_id}_{i}"):
                    st.session_state["pending_question"] = q
                    st.rerun()

    st.divider()

    # ============================
    # Chat
    # ============================
    st.subheader("3) Chat with your document")
    pending = st.session_state.pop("pending_question", "")
    if pending:
        run_single_question(pending, top_k=top_k)

    with st.form("ask_form", clear_on_submit=True):
        q_text = st.text_input("Ask", placeholder="e.g., What are the totals and dates?", label_visibility="collapsed")
        ask_clicked = st.form_submit_button("Ask", use_container_width=True)

    if ask_clicked and q_text.strip():
        run_single_question(q_text.strip(), top_k=top_k)

    if st.session_state.get("latest_answer"):
        st.markdown("### ✅ Latest Answer")
        st.markdown(f"**Question:** {st.session_state.get('latest_q','')}")
        st.markdown(st.session_state["latest_answer"] or "I don't know based on the document.")
        if st.session_state.get("latest_evidence"):
            with st.expander("📌 Evidence"):
                st.markdown(st.session_state["latest_evidence"])
        st.caption(f"⏱ Response time: {st.session_state.get('latest_rt', 0)} sec")

# ============================================================
# Mode: Compare Two Docs (PDF)
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

    a_bytes = up_a.getvalue() or b""
    b_bytes = up_b.getvalue() or b""

    text_a = extract_text_from_pdf_with_ocr(a_bytes, max_ocr_pages=4)
    text_b = extract_text_from_pdf_with_ocr(b_bytes, max_ocr_pages=4)

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

    q = st.text_input("Comparison question", placeholder="e.g., Which doc shows stronger results and why?")
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


