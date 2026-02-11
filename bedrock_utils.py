import json
import boto3
import re
from typing import List, Dict, Any

# Regions (as you already set)
GEN_REGION = "ap-south-1"      # Nova Lite profile is here for you
EMBED_REGION = "us-east-1"     # embeddings supported here

# Use the inference profile ID you discovered
NOVA_LITE_MODEL_ID = "apac.amazon.nova-lite-v1:0"
NOVA_MM_EMBED_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"


def get_bedrock_runtime(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def _find_first_float_list(obj):
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


def embed_text(text: str, dim: int = 1024):
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


# ---------- NEW: Doc type detection + extraction schemas ----------

DOC_TYPES = ["auto", "resume", "invoice", "contract", "research_paper", "generic"]

EXTRACTION_SCHEMAS = {
    "resume": {
        "fields": ["name", "email", "phone", "location", "skills", "years_experience", "latest_role", "education", "certifications"],
    },
    "invoice": {
        "fields": ["vendor", "invoice_number", "invoice_date", "due_date", "total_amount", "currency", "line_items"],
    },
    "contract": {
        "fields": ["parties", "effective_date", "end_date", "termination_clause", "payment_terms", "governing_law", "key_obligations", "risks"],
    },
    "research_paper": {
        "fields": ["title", "authors", "abstract_summary", "methodology", "metrics", "key_results", "limitations", "future_work"],
    },
    "generic": {
        "fields": ["summary", "key_points", "dates", "numbers", "entities"],
    },
}


def detect_doc_type(doc_text: str) -> str:
    brt = get_bedrock_runtime(GEN_REGION)
    prompt = f"""
Classify the document into exactly one type from this list:
resume, invoice, contract, research_paper, generic

Return only the type word.

Document (excerpt):
{doc_text[:8000]}
"""
    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    t = resp["output"]["message"]["content"][0]["text"].strip().lower()
    # sanitize
    for allowed in ["resume", "invoice", "contract", "research_paper", "generic"]:
        if allowed in t:
            return allowed
    return "generic"


def extract_fields_json(doc_text: str, doc_type: str = "auto") -> str:
    brt = get_bedrock_runtime(GEN_REGION)

    if doc_type == "auto":
        doc_type = detect_doc_type(doc_text)

    schema = EXTRACTION_SCHEMAS.get(doc_type, EXTRACTION_SCHEMAS["generic"])

    prompt = f"""
You are an information extraction assistant.
Extract the fields for doc_type="{doc_type}" and output VALID JSON only.

JSON format:
{{
  "doc_type": "{doc_type}",
  "fields": {{ ... }},
  "confidence": 0.0,
  "notes": ["..."]
}}

Fields to extract:
{schema["fields"]}

Rules:
- Output JSON only (no backticks).
- If a field is missing, set it to null.
- Keep lists as arrays.

Document:
{doc_text[:14000]}
"""
    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return resp["output"]["message"]["content"][0]["text"]


# ---------- UPDATED: Ask + Evidence snippets ----------

def ask_with_evidence(question: str, context_chunks: List[str]) -> Dict[str, Any]:
    """
    Returns dict:
    {
      "answer": "...",
      "sources_used": [1,3],
      "evidence": [{"source":1,"quote":"..."}, ...]
    }
    """
    brt = get_bedrock_runtime(GEN_REGION)

    sources_block = "\n\n".join([f"[Source {i+1}]\n{c}" for i, c in enumerate(context_chunks)])

    prompt = f"""
You are Smart Document Copilot.
Answer using ONLY the sources.

Return VALID JSON only in this schema:
{{
  "answer": "string",
  "sources_used": [1,2],
  "evidence": [
    {{"source": 1, "quote": "short exact snippet from Source 1"}},
    {{"source": 2, "quote": "short exact snippet from Source 2"}}
  ]
}}

Rules:
- evidence quotes must be copied EXACTLY from the sources (short).
- If not found, set answer to "I don't know based on the document.", sources_used=[], evidence=[].

SOURCES:
{sources_block}

QUESTION:
{question}
"""
    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    txt = resp["output"]["message"]["content"][0]["text"]

    # best-effort JSON parse
    try:
        return json.loads(txt)
    except Exception:
        # fallback: return raw
        return {"answer": txt, "sources_used": [], "evidence": []}


# ---------- NEW: Compare two documents ----------

def compare_docs(question: str, ctx_a: List[str], ctx_b: List[str], label_a="Doc A", label_b="Doc B") -> str:
    brt = get_bedrock_runtime(GEN_REGION)

    a_block = "\n\n".join([f"[A{i+1}]\n{c}" for i, c in enumerate(ctx_a)])
    b_block = "\n\n".join([f"[B{i+1}]\n{c}" for i, c in enumerate(ctx_b)])

    prompt = f"""
You are a careful comparer.
Compare {label_a} vs {label_b} using ONLY the sources.
Cite evidence with IDs like A1, A2, B1, B3.

Output format:
- Summary
- Key differences (bullets)
- Recommendation (if applicable)
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
