import json
from typing import List, Dict
import boto3

# Regions
GEN_REGION = "ap-south-1"   # Nova Lite inference profile region
EMBED_REGION = "us-east-1"  # embeddings supported here

# Use your inference profile ID (you confirmed this exists)
# Example: apac.amazon.nova-lite-v1:0
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


# ---------------- Image -> Text / Insights (Nova Lite multimodal) ----------------

def _normalize_img_format(fmt: str) -> str:
    fmt = (fmt or "").lower().strip(". ")
    if fmt in ("jpg", "jpe"):
        return "jpeg"
    if fmt not in ("png", "jpeg", "webp"):
        # Nova supports common formats; fallback to jpeg
        return "jpeg"
    return fmt


def nova_image_to_text(image_bytes: bytes, image_format: str = "png") -> str:
    """
    Uses Nova Lite multimodal to extract all visible text from the image.
    This replaces Textract for your hackathon/demo.
    """
    brt = get_bedrock_runtime(GEN_REGION)
    image_format = _normalize_img_format(image_format)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": image_format,
                        "source": {"bytes": image_bytes},
                    }
                },
                {
                    "text": (
                        "Extract ALL visible text from this image.\n"
                        "Rules:\n"
                        "- Output ONLY the extracted text.\n"
                        "- Preserve line breaks.\n"
                        "- If there is no text, output exactly: NO_TEXT_FOUND"
                    )
                },
            ],
        }
    ]

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": 700, "temperature": 0.0, "topP": 0.9},
    )

    txt = resp["output"]["message"]["content"][0]["text"].strip()
    if "NO_TEXT_FOUND" in txt.upper():
        return ""
    return txt


def nova_image_insights(image_bytes: bytes, image_format: str = "png") -> str:
    """
    Uses Nova Lite multimodal to generate short insights from the image.
    Good for making demo look "smart" even if image has minimal text.
    """
    brt = get_bedrock_runtime(GEN_REGION)
    image_format = _normalize_img_format(image_format)

    messages = [
        {
            "role": "user",
            "content": [
                {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                {
                    "text": (
                        "Give:\n"
                        "1) A 1-line description of the image\n"
                        "2) 3 bullet key insights (short)\n"
                        "3) If you see numbers/dates, list them\n"
                        "Be concise."
                    )
                },
            ],
        }
    ]

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=messages,
        inferenceConfig={"maxTokens": 350, "temperature": 0.2, "topP": 0.9},
    )
    return resp["output"]["message"]["content"][0]["text"].strip()

def generate_report_title(doc_text: str) -> str:
    """
    Creates a short, human-friendly report title from the document content.
    Returns plain text (no quotes/backticks).
    """
    brt = get_bedrock_runtime(GEN_REGION)

    prompt = f"""
You are a helpful assistant. Create a short report title based on the content.

Rules:
- 4 to 8 words
- Title Case
- No quotes, no punctuation at the end
- No emojis
- If content is unclear, output: Smart Document Report

Content excerpt:
{doc_text[:6000]}
"""
    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 40, "temperature": 0.2, "topP": 0.9},
    )

    title = resp["output"]["message"]["content"][0]["text"].strip()

    # Safety cleanup
    title = title.replace('"', "").replace("`", "").strip()
    if not title:
        return "Smart Document Report"

    # Keep it short
    words = title.split()
    if len(words) > 10:
        title = " ".join(words[:10])

    return title


# ---------------- Doc type + Extraction ----------------

DOC_TYPES = ["auto", "resume", "invoice", "contract", "research_paper", "generic"]

EXTRACTION_SCHEMAS = {
    "resume": {
        "fields": ["name", "email", "phone", "location", "skills", "years_experience", "latest_role", "education", "certifications"]
    },
    "invoice": {
        "fields": ["vendor", "invoice_number", "invoice_date", "due_date", "total_amount", "currency", "line_items"]
    },
    "contract": {
        "fields": ["parties", "effective_date", "end_date", "termination_clause", "payment_terms", "governing_law", "key_obligations", "risks"]
    },
    "research_paper": {
        "fields": ["title", "authors", "abstract_summary", "methodology", "metrics", "key_results", "limitations", "future_work"]
    },
    "generic": {"fields": ["summary", "key_points", "dates", "numbers", "entities"]},
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

def ask_with_evidence(question: str, context_chunks: List[str]) -> Dict[str, str]:
    brt = get_bedrock_runtime(GEN_REGION)

    sources_block = "\n\n".join([f"[Source {i+1}]\n{c}" for i, c in enumerate(context_chunks)])

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

After the answer, add:

Evidence:
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

    if "Evidence:" in text:
        answer_part, evidence_part = text.split("Evidence:", 1)
    else:
        answer_part, evidence_part = text, ""

    return {"answer": answer_part.strip(), "evidence": evidence_part.strip()}


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

