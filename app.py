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
    generate_dashboard_insights_dynamic,  # AI dashboard (may output invalid JSON; we salvage)
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

    # OCR if too short
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
    """
    Returns (ocr_text, insights)
    """
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
    # Try multiple encodings
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes), encoding=enc)
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        return "", None

    df = df.dropna(how="all")
    text = df.head(40).to_csv(index=False)
    return text.strip(), df

def read_excel_bytes(excel_bytes: bytes, filename: str) -> Tuple[str, Dict[str, pd.DataFrame], str]:
    """
    Returns:
      extracted_text, tables_by_sheet, warning_message
    """
    warning = ""
    tables: Dict[str, pd.DataFrame] = {}
    text_parts: List[str] = []

    ext = (filename or "").lower().split(".")[-1].strip()

    # XLSX path
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
            warning = f"Excel read failed (xlsx). Error: {e}"

    # XLS path (needs xlrd)
    elif ext == "xls":
        try:
            # xlrd is required for .xls
            xls = pd.ExcelFile(io.BytesIO(excel_bytes), engine="xlrd")
            for sh in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sh, engine="xlrd")
                    df = df.dropna(how="all")
                    if df is not None and not df.empty:
                        tables[sh] = df
                        text_parts.append(f"--- Sheet: {sh} ---")
                        text_parts.append(df.head(30).to_csv(index=False))
                except Exception:
                    continue
        except Exception:
            warning = "This looks like a legacy .xls file. Install `xlrd` or save as .xlsx and re-upload."

    else:
        warning = "Unsupported Excel extension. Please upload .xlsx/.xlsm or .xls (requires xlrd)."

    extracted_text = "\n".join(text_parts).strip()
    return extracted_text, tables, warning

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
# Dashboard (local + AI + salvage JSON)
# ============================================================

def local_dashboard_from_text(text: str, max_items: int = 80) -> Dict[str, Any]:
    out = {"kpis": [], "charts": [], "table_preview": [], "derived_insights": []}
    if not text or not str(text).strip():
        return out

    t = str(text).replace("\r", "\n").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # label:value lines
    metric_candidates = []
    for line in t.splitlines():
        ln = line.strip()
        if len(ln) < 4:
            continue
        m = re.search(r"^(.{2,60}?)\s*[:\-]\s*([^\n]{1,60})$", ln)
        if m:
            metric_candidates.append((m.group(1).strip(), m.group(2).strip()))

    # loose numbers
    num_re = re.compile(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")
    loose = []
    for line in t.splitlines():
        ln = line.strip()
        if not ln:
            continue
        nums = list(num_re.finditer(ln))
        if not nums:
            continue
        if len(nums) >= 2 and len(ln) <= 200:
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
        out["kpis"].append({"label": label[:40], "value": raw, "note": ""})

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

def local_dashboard_from_excel_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Better dashboard for Excel: compute summary stats from numeric columns.
    """
    out = {"kpis": [], "charts": [], "table_preview": [], "derived_insights": []}
    if not tables:
        return out

    # Table preview: first sheet head
    first_sheet = next(iter(tables.keys()))
    df0 = tables[first_sheet]
    out["table_preview"].append({"sheet": first_sheet, "rows": int(df0.shape[0]), "cols": int(df0.shape[1])})

    # Numeric summaries
    numeric_points = []
    for sh, df in list(tables.items())[:6]:
        df = df.copy()
        # coerce numeric columns
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="ignore")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for c in num_cols[:8]:
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            if series.empty:
                continue
            ssum = float(series.sum())
            savg = float(series.mean())
            smax = float(series.max())
            numeric_points.append((f"{sh}:{c} sum", ssum))
            numeric_points.append((f"{sh}:{c} avg", savg))
            numeric_points.append((f"{sh}:{c} max", smax))

    numeric_points = sorted(numeric_points, key=lambda x: abs(x[1]), reverse=True)

    for label, val in numeric_points[:9]:
        out["kpis"].append({"label": label[:40], "value": f"{val:,.2f}", "note": ""})

    if numeric_points:
        chart_data = [{"x": lbl[:28], "y": float(v)} for lbl, v in numeric_points[:12]]
        out["charts"].append({"title": "Top Excel Metrics (Auto)", "type": "bar", "data": chart_data})

    out["derived_insights"].append(f"Parsed {len(tables)} sheets. Computed summaries from numeric columns.")
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
    # salvage between first { and last }
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
    if not merged.get("doc_type_guess"):
        merged["doc_type_guess"] = "generic"
    if merged.get("risk_score") is None:
        merged["risk_score"] = 0
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

    if not merged.get("summary") or "could not" in str(merged.get("summary", "")).lower():
        if local.get("kpis") or local.get("charts"):
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
# State + Sidebar
# ============================================================
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("dash_nonce", {})  # per-doc refresh nonce

st.sidebar.header("⚙️ Controls")
st.sidebar.button("🔄 Reset / New session", on_click=lambda: reset_session(), use_container_width=True)

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

    # ------------------------------------------------------------
    # Extract file bytes + extension
    # ------------------------------------------------------------
    file_bytes = b""
    filename = ""
    ext = ""

    if uploaded is not None:
        file_bytes = uploaded.getvalue() or b""
        filename = uploaded.name or ""
        ext = filename.lower().split(".")[-1].strip()

    file_hash = sha1_bytes(file_bytes[:600000]) if file_bytes else "nofile"

    # ------------------------------------------------------------
    # Extract text / tables
    # ------------------------------------------------------------
    extracted_text = ""
    excel_tables: Dict[str, pd.DataFrame] = {}
    img_insights = ""
    csv_df: Optional[pd.DataFrame] = None

    t_start = time.time()

    if uploaded is not None and file_bytes:
        if ext == "pdf":
            with st.spinner("Extracting PDF text/OCR..."):
                extracted_text = extract_text_from_pdf_with_ocr(file_bytes, max_ocr_pages=6)

        elif ext in ["png", "jpg", "jpeg", "webp"]:
            with st.spinner("Extracting image OCR + insights..."):
                extracted_text, img_insights = extract_text_from_image(file_bytes)

        elif ext in ["xlsx", "xlsm", "xls"]:
            # Excel should NOT say OCR; it should parse immediately
            with st.spinner("Reading Excel sheets..."):
                extracted_text, excel_tables, warn = read_excel_bytes(file_bytes, filename)
            if warn:
                st.warning(warn)

        elif ext == "csv":
            with st.spinner("Reading CSV..."):
                extracted_text, csv_df = read_csv_bytes(file_bytes)
                if csv_df is None:
                    st.warning("CSV read failed. Try saving again with UTF-8 encoding.")

        elif ext == "docx":
            # word parsing
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_bytes))
                extracted_text = "\n".join([p.text for p in doc.paragraphs if p.text and p.text.strip()]).strip()
            except Exception:
                extracted_text = ""
                st.warning("DOCX read failed. Ensure `python-docx` is installed and file is valid.")

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
                extracted_text = ""
                st.warning("PPTX read failed. Ensure `python-pptx` is installed or export PPT as PDF and upload.")

        elif ext == "txt":
            try:
                extracted_text = file_bytes.decode("utf-8", errors="ignore").strip()
            except Exception:
                extracted_text = ""

    # Merge all text sources
    parts = []
    if extracted_text.strip():
        parts.append("=== EXTRACTED TEXT ===\n" + extracted_text.strip())
    if img_insights.strip():
        parts.append("=== IMAGE INSIGHTS ===\n" + img_insights.strip())
    if user_text.strip():
        parts.append("=== USER NOTES ===\n" + user_text.strip())

    full_text = "\n\n".join(parts).strip()
    extract_secs = round(time.time() - t_start, 2)

    # ------------------------------------------------------------
    # Previews (Excel / CSV / Image)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Download extracted text
    # ------------------------------------------------------------
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
            st.write("File hash:", file_hash)
            st.write("Extract seconds:", extract_secs)
            st.write("Extracted text length:", len(full_text))
            st.write((full_text[:1400] + "...") if len(full_text) > 1400 else full_text)

    # ------------------------------------------------------------
    # ✅ Cache key that fixes “wrong first dashboard” + refresh not updating
    # key uses: file_hash + extracted_text_hash + per-doc nonce
    # ------------------------------------------------------------
    text_hash = md5_text(full_text[:30000])
    doc_id = f"{file_hash}:{text_hash}"

    st.session_state["dash_nonce"].setdefault(doc_id, 0)
    nonce = st.session_state["dash_nonce"][doc_id]

    dash_key = f"dashboard:{doc_id}:n={nonce}"
    dates_key = f"dates:{doc_id}:n={nonce}"

    # ------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------
    if dates_key not in st.session_state:
        st.session_state[dates_key] = extract_dates_with_events(full_text, max_items=120)
    local_dates = st.session_state.get(dates_key, [])

    # ------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------
    st.subheader("📊 Executive Dashboard")

    cbtn1, cbtn2 = st.columns([1, 6])
    with cbtn1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            # increment nonce so cache key changes => forces recompute
            st.session_state["dash_nonce"][doc_id] += 1
            st.rerun()

    # Local dashboard should always work
    if excel_tables:
        local_dash = local_dashboard_from_excel_tables(excel_tables)
    else:
        local_dash = local_dashboard_from_text(full_text)

    # AI dashboard (optional + safe)
    run_ai_now = False
    if use_ai_dashboard:
        run_ai_now = True if not lazy_ai else st.button("⚡ Run AI Dashboard (Nova)", use_container_width=True)

    if dash_key not in st.session_state:
        ai_dash = {}
        if use_ai_dashboard and run_ai_now and full_text.strip():
            with st.spinner("🧠 Nova generating AI dashboard..."):
                try:
                    raw = generate_dashboard_insights_dynamic(full_text)
                    # raw might already be dict; if not, salvage
                    if isinstance(raw, dict):
                        ai_dash = raw
                    else:
                        ai_dash = salvage_json(str(raw)) or {}
                except Exception:
                    ai_dash = {}
        st.session_state[dash_key] = ai_dash

    ai_cached = st.session_state.get(dash_key, {}) or {}
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

    # ------------------------------------------------------------
    # RAG settings + index
    # ------------------------------------------------------------
    if st.session_state.get("last_doc_id") != doc_id:
        st.session_state["last_doc_id"] = doc_id
        with st.spinner("⚙️ Auto-optimizing retrieval settings..."):
            st.session_state["auto_rag_settings"] = recommend_rag_settings(full_text)
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
        chunk_size = st.sidebar.slider("Chunk size (chars)", 300, 2000, int(auto_chunk_size), 50)
        overlap = st.sidebar.slider("Overlap (chars)", 0, 400, int(auto_overlap), 25)
        top_k = st.sidebar.slider("Top-K sources", 2, 8, int(auto_top_k), 1)

    if st.sidebar.button("♻️ Rebuild index now", use_container_width=True):
        st.session_state.pop("index_fp", None)

    build_index_if_needed(full_text, chunk_size=chunk_size, overlap=overlap)

    st.divider()

    # ------------------------------------------------------------
    # Extract fields JSON
    # ------------------------------------------------------------
    st.subheader("2) Extract key fields (JSON)")
    doc_type = st.selectbox("Document type", DOC_TYPES, index=0)
    if st.button("🧾 Extract key fields as JSON", use_container_width=True):
        with st.spinner("Extracting..."):
            out = extract_fields_json(full_text, doc_type=doc_type)
        st.code(out, language="json")

    st.divider()

    # ------------------------------------------------------------
    # Suggested questions (manual trigger so it doesn't slow Excel)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------
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

    st.divider()

    # ------------------------------------------------------------
    # Export PDF report
    # ------------------------------------------------------------
    st.subheader("4) Export report (PDF)")
    if st.button("📄 Generate PDF report", use_container_width=True):
        with st.spinner("Creating a title..."):
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
            ("Extracted text preview (first 1200 chars)", full_text[:1200]),
            ("Latest Q&A", latest_block or "No Q&A yet."),
        ]

        safe_filename = "".join(ch for ch in report_title if ch.isalnum() or ch in (" ", "-", "_")).strip()
        safe_filename = safe_filename.replace(" ", "_") or "Smart_Document_Report"
        file_name = f"{safe_filename}.pdf"

        pdf_bytes = make_pdf_report(file_name, report_title, sections)
        st.download_button("⬇️ Download report", data=pdf_bytes, file_name=file_name, mime="application/pdf", use_container_width=True)

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
