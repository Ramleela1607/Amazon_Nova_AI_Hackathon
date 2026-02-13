import json
from typing import List, Dict, Any

import boto3

# Regions
GEN_REGION = "ap-south-1"   # Nova Lite inference profile is here for you
EMBED_REGION = "us-east-1"  # embeddings supported here (your earlier setup)

# IMPORTANT: use your inference profile ID (from your script output)
# You confirmed this exists:
#   ID: apac.amazon.nova-lite-v1:0
NOVA_LITE_MODEL_ID = "apac.amazon.nova-lite-v1:0"

# Multimodal embeddings model ID (works from EMBED_REGION)
NOVA_MM_EMBED_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"


def get_bedrock_runtime(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def _find_first_float_list(obj):
    """Walk nested dict/list to find first list of floats (embedding vector)."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) for x in obj):
            return obj
        for item in obj:
            got = _find_first_float_list(item)
            if got is not None:
                return got
    elif isinstance(obj, dict):
        for v in obj.values():
            got = _find_first_float_list(v)
            if got is not None:
                return got
    return None


def embed_text(text: str, dim: int = 1024) -> List[float]:
    """Create an embedding for text via Nova Multimodal Embeddings."""
    brt = get_bedrock_runtime(EMBED_REGION)

    body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": dim,
            "text": {"truncationMode": "END", "value": text},
        },
    }

    resp = brt.invoke_model(
        modelId=NOVA_MM_EMBED_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    out = json.loads(resp["body"].read())
    vec = _find_first_float_list(out)
    if vec is None:
        raise ValueError(f"Could not find embedding vector in response keys: {list(out.keys())}")
    return vec
    
def textract_image_to_text(image_bytes: bytes, region: str = "ap-south-1") -> str:
    """
    Extracts text from an image using Amazon Textract.
    Works great on Streamlit Cloud (no system OCR installs needed).
    """
    textract = boto3.client("textract", region_name=region)
    resp = textract.detect_document_text(Document={"Bytes": image_bytes})

    lines = []
    for block in resp.get("Blocks", []):
        if block.get("BlockType") == "LINE" and block.get("Text"):
            lines.append(block["Text"])
    return "\n".join(lines).strip()


# ---------------- Doc type + Extraction ----------------

DOC_TYPES = ["auto", "resume", "invoice", "contract", "research_paper", "generic"]

EXTRACTION_SCHEMAS = {
    "resume": {
        "fields": [
            "name", "email", "phone", "location",
            "skills", "years_experience", "latest_role",
            "education", "certifications"
        ]
    },
    "invoice": {
        "fields": [
            "vendor", "invoice_number", "invoice_date",
            "due_date", "total_amount", "currency", "line_items"
        ]
    },
    "contract": {
        "fields": [
            "parties", "effective_date", "end_date",
            "termination_clause", "payment_terms", "governing_law",
            "key_obligations", "risks"
        ]
    },
    "research_paper": {
        "fields": [
            "title", "authors", "abstract_summary",
            "methodology", "metrics", "key_results",
            "limitations", "future_work"
        ]
    },
    "generic": {
        "fields": ["summary", "key_points", "dates", "numbers", "entities"]
    },
}


def detect_doc_type(doc_text: str) -> str:
    brt = get_bedrock_runtime(GEN_REGION)
    prompt = f"""
Classify the document into exactly ONE type from:
resume, invoice, contract, research_paper, generic

Return ONLY the type word (no extra text).

Document excerpt:
{doc_text[:8000]}
"""
    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    t = resp["output"]["message"]["content"][0]["text"].strip().lower()

    for allowed in ["resume", "invoice", "contract", "research_paper", "generic"]:
        if allowed in t:
            return allowed
    return "generic"


def extract_fields_json(doc_text: str, doc_type: str = "auto") -> str:
    brt = get_bedrock_runtime(GEN_REGION)

    if doc_type == "auto":
        doc_type = detect_doc_type(doc_text)

    schema = EXTRACTION_SCHEMAS.get(doc_type, EXTRACTION_SCHEMAS["generic"])
    fields = schema["fields"]

    prompt = f"""
You are an information extraction assistant.

Extract fields for doc_type="{doc_type}" and output VALID JSON only.

JSON format:
{{
  "doc_type": "{doc_type}",
  "fields": {{ ... }},
  "confidence": 0.0,
  "notes": ["..."]
}}

Fields to extract:
{fields}

Rules:
- Output JSON only (no backticks, no extra commentary).
- If a field is missing, set it to null.
- For lists, use arrays.
- confidence between 0 and 1.

Document:
{doc_text[:14000]}
"""
    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return resp["output"]["message"]["content"][0]["text"]


# ---------------- Q&A with short+insight + evidence ----------------

def ask_with_evidence(question: str, context_chunks: List[str]) -> dict:
    brt = get_bedrock_runtime(GEN_REGION)

    sources_block = "\n\n".join(
        [f"[Source {i+1}]\n{c}" for i, c in enumerate(context_chunks)]
    )

    prompt = f"""
You are Smart Document Copilot.

Answer the question using ONLY the provided sources.

Answer style:
- 3–4 concise sentences.
- Include key numbers or metrics if available.
- Add one useful insight (implication or takeaway).
- Do NOT mention response time or latency.
- Be professional and clear.
- If information is not found, say:
  "I don't know based on the document."

After the answer, add a section exactly like this:

Evidence:
- short exact quote
- short exact quote

Evidence rules:
- Provide 1–3 short exact quotes from the sources.
- Keep quotes short and relevant.

SOURCES:
{sources_block}

QUESTION:
{question}
"""

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )

    text = resp["output"]["message"]["content"][0]["text"]

    # Split answer and evidence cleanly
    if "Evidence:" in text:
        answer_part, evidence_part = text.split("Evidence:", 1)
    else:
        answer_part = text
        evidence_part = ""

    return {
        "answer": answer_part.strip(),
        "evidence": evidence_part.strip()
    }


# ---------------- Compare two docs ----------------

def compare_docs(question: str, ctx_a: List[str], ctx_b: List[str], label_a="Doc A", label_b="Doc B") -> str:
    brt = get_bedrock_runtime(GEN_REGION)

    a_block = "\n\n".join([f"[A{i+1}]\n{c}" for i, c in enumerate(ctx_a)])
    b_block = "\n\n".join([f"[B{i+1}]\n{c}" for i, c in enumerate(ctx_b)])

    prompt = f"""
You are a careful comparer.
Compare {label_a} vs {label_b} using ONLY the sources.
Cite evidence with IDs like A1, A2, B1, B3.

Output format:
- Summary (2-4 lines)
- Key differences (bullets)
- Recommendation (1-2 lines)
- Citations used (list of A#/B#)

If not enough info, say so.

QUESTION:
{question}

{label_a} SOURCES:
{a_block}

{label_b} SOURCES:
{b_block}
"""
    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return resp["output"]["message"]["content"][0]["text"]


