import os
import streamlit as st
from pypdf import PdfReader
import json
from rag_index import RagIndex
from bedrock_utils import ask_nova_lite, extract_fields_json

st.set_page_config(page_title="Smart Document Copilot", layout="wide")
st.title("📄 Smart Document Copilot")
st.caption("Upload a PDF → build an index (Nova embeddings) → ask questions (Nova 2 Lite + sources)")

# --- Helpers ---
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

# --- UI: Region ---
region = st.text_input("Bedrock region", value="ap-south-1", help="Use the same region where Nova 2 Lite appears in Model Catalog.")

# --- Upload ---
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

full_text = extract_text_from_pdf(uploaded)

colA, colB = st.columns(2)

with colA:
    st.subheader("Extracted text (preview)")
    st.text_area("Preview", full_text[:6000], height=350)

with colB:
    st.subheader("Chunking settings")
    chunk_size = st.slider("Chunk size (chars)", 400, 2000, 1000, 100)
    overlap = st.slider("Overlap (chars)", 0, 400, 150, 25)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    st.write(f"Total chunks: **{len(chunks)}**")

    sample_id = st.number_input("Preview chunk #", min_value=1, max_value=len(chunks), value=1, step=1)
    st.text_area("Chunk preview", chunks[sample_id - 1], height=350)

st.divider()

# --- Index build / load ---
INDEX_PREFIX = "doc_index"

if "rag" not in st.session_state:
    st.session_state.rag = None

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Build / Load Index")

    if st.button("🚀 Build Index (Nova embeddings → FAISS)", use_container_width=True):
        with st.spinner("Embedding chunks with Nova and building FAISS index..."):
            rag = RagIndex(dim=1024)
            rag.add_chunks(chunks)
            rag.save(INDEX_PREFIX)
            st.session_state.rag = rag
        st.success("Index built and saved!")

with right:
    st.subheader("2) Load Existing Index")

    if st.button("📂 Load Index from disk", use_container_width=True):
        if os.path.exists(f"{INDEX_PREFIX}.faiss") and os.path.exists(f"{INDEX_PREFIX}.meta.json"):
            st.session_state.rag = RagIndex.load(INDEX_PREFIX, dim=1024)
            st.success("Index loaded!")
        else:
            st.error("No saved index found. Build the index first.")

st.divider()

# --- Chat Q&A ---
st.subheader("3) Ask questions about the document")
st.subheader("4) Extract key fields (JSON)")
if st.button("🧾 Extract key fields"):
    with st.spinner("Extracting..."):
        out = extract_fields_json(full_text, doc_type="auto")
    st.code(out, language="json")


if st.session_state.rag is None:
    st.warning("Build or load an index first.")
    st.stop()

question = st.text_input("Your question", placeholder="e.g., What are the main points? What is the total amount? What are the key requirements?")
top_k = st.slider("How many sources to retrieve", 2, 8, 4, 1)

if st.button("💬 Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.error("Please type a question.")
        st.stop()

    with st.spinner("Retrieving relevant sources..."):
        hits, scores = st.session_state.rag.search(question, k=top_k)

    context_chunks = [h.text for h in hits]

    with st.spinner("Asking Nova 2 Lite..."):
        answer = ask_nova_lite(question, context_chunks=context_chunks)
    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 🔎 Retrieved sources")
    for i, (h, s) in enumerate(zip(hits, scores), start=1):
        with st.expander(f"Source {i} (chunk #{h.chunk_id}) — score {s:.3f}"):
            st.write(h.text[:2000])
