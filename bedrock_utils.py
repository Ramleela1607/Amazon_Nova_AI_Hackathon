import re
import json
from typing import List, Dict, Any
import boto3

# -------------------------
# Regions / Model IDs
# -------------------------
GEN_REGION = "ap-south-1"   # Nova Lite inference profile region
EMBED_REGION = "us-east-1"  # embeddings supported region

# Nova Lite inference profile ID
NOVA_LITE_MODEL_ID = "apac.amazon.nova-lite-v1:0"

# Embeddings model ID
NOVA_MM_EMBED_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"


def get_bedrock_runtime(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # remove ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_array(s: str) -> str:
    """
    Return the first JSON array substring found in s, e.g. [ ... ].
    Raises ValueError if not found.
    """
    s = _strip_code_fences(s)
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        raise ValueError("No JSON array found")
    return m.group(0)



# -------------------------
# Embeddings (for FAISS)
# -------------------------
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


def _img_block(image_bytes: bytes, image_format: str) -> Dict[str, Any]:
    """
    Build a Bedrock multimodal image content block.

    IMPORTANT:
    - Provide RAW bytes (not base64, not decoded text).
    - Ensure format is correct: png/jpeg/webp
    """
    fmt = (image_format or "png").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in ("png", "jpeg", "webp"):
        fmt = "png"

    return {"image": {"format": fmt, "source": {"bytes": image_bytes}}}


# -------------------------
# Multimodal (Image -> Text / Insights) using Nova Lite
# -------------------------
def nova_image_to_text(image_bytes: bytes, image_format: str = "png") -> str:
    """
    Extract readable text from an image using Nova Lite.
    NOTE: Your app should use this ONLY internally for retrieval.
    """
    brt = get_bedrock_runtime(GEN_REGION)

    msg = [
        {
            "role": "user",
            "content": [
                {"text": "Extract ALL readable text from this image. Return only the extracted text."},
                _img_block(image_bytes, image_format),
            ],
        }
    ]

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=msg,
        inferenceConfig={"maxTokens": 800, "temperature": 0.2, "topP": 0.9},
    )
    return resp["output"]["message"]["content"][0]["text"].strip()


def nova_image_insights_brief(image_bytes: bytes, image_format: str = "png") -> str:
    """
    Return STRICTLY 2 lines:

    What it is:
    One useful insight (implication / issue / next step):
    """

    brt = get_bedrock_runtime(GEN_REGION)

    prompt = """
You are Smart Document Copilot.

Carefully analyze the image.

Respond in EXACTLY this format (2 lines total):

What it is: <document type + 1-2 specific visible identifiers such as invoice number, total amount, due date, company name>
One useful insight (implication / issue / next step): <clear, practical, risk-aware insight>

STRICT RULES:
- MUST include at least one EXACT visible value (number, amount, date, or ID).
- If it is an invoice, include the total amount.
- If financial details exist, mention verification or payment risk.
- Keep each line under 30 words.
- No bullets.
- No extra lines.
- No extra commentary.
- Do NOT invent data.
- If text is unclear, say what is confidently visible.
""".strip()

    msg = [
        {
            "role": "user",
            "content": [
                {"text": prompt},
                _img_block(image_bytes, image_format),
            ],
        }
    ]

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=msg,
        inferenceConfig={
            "maxTokens": 150,
            "temperature": 0.15,   # Lower = more factual
            "topP": 0.9
        },
    )

    return resp["output"]["message"]["content"][0]["text"].strip()



# -------------------------
# Doc types + Extraction
# -------------------------
DOC_TYPES = ["auto", "resume", "invoice", "contract", "research_paper", "generic"]

EXTRACTION_SCHEMAS = {
    "resume": {"fields": ["name", "email", "phone", "location", "skills", "years_experience", "latest_role", "education", "certifications"]},
    "invoice": {"fields": ["vendor", "invoice_number", "invoice_date", "due_date", "total_amount", "currency", "line_items", "payment_terms"]},
    "contract": {"fields": ["parties", "effective_date", "end_date", "termination_clause", "payment_terms", "governing_law", "key_obligations", "risks"]},
    "research_paper": {"fields": ["title", "authors", "abstract_summary", "methodology", "metrics", "key_results", "limitations", "future_work"]},
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
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 20, "temperature": 0.2, "topP": 0.9},
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
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 900, "temperature": 0.2, "topP": 0.9},
    )
    return resp["output"]["message"]["content"][0]["text"]


# -------------------------
# Q&A with Evidence (no JSON output)
# -------------------------
def ask_with_evidence(question: str, context_chunks: List[str]) -> dict:
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
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 700, "temperature": 0.2, "topP": 0.9},
    )

    text = resp["output"]["message"]["content"][0]["text"]

    if "Evidence:" in text:
        answer_part, evidence_part = text.split("Evidence:", 1)
    else:
        answer_part, evidence_part = text, ""

    return {"answer": answer_part.strip(), "evidence": evidence_part.strip()}


# -------------------------
# Compare docs
# -------------------------
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
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 900, "temperature": 0.2, "topP": 0.9},
    )
    return resp["output"]["message"]["content"][0]["text"]


# -------------------------
# Nova Auto RAG tuning
# -------------------------
def recommend_rag_settings(doc_text: str) -> dict:
    brt = get_bedrock_runtime(GEN_REGION)

    prompt = f"""
You are optimizing RAG settings for a document QA app.

Choose values for:
- chunk_size: integer 600 to 1600
- overlap: integer 50 to 250
- top_k: integer 3 to 6

Heuristics:
- Long/technical docs → chunk_size 1100-1500, overlap 150-220, top_k 4-6
- Short docs → chunk_size 700-1000, overlap 80-150, top_k 3-4
- Avoid too small chunks.
- Keep overlap moderate.

Return ONLY valid JSON like:
{{"chunk_size": 1200, "overlap": 180, "top_k": 4}}

Document excerpt:
{doc_text[:8000]}
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 120, "temperature": 0.3, "topP": 0.9},
    )
    txt = resp["output"]["message"]["content"][0]["text"].strip()

    try:
        out = json.loads(txt)
        chunk_size = int(out.get("chunk_size", 1000))
        overlap = int(out.get("overlap", 150))
        top_k = int(out.get("top_k", 4))
    except Exception:
        chunk_size, overlap, top_k = 1000, 150, 4

    chunk_size = max(600, min(1600, chunk_size))
    overlap = max(50, min(250, overlap))
    top_k = max(3, min(6, top_k))

    return {"chunk_size": chunk_size, "overlap": overlap, "top_k": top_k}


# -------------------------
# Nova Title Generation (PDF)
# -------------------------
def generate_report_title(doc_text: str) -> str:
    brt = get_bedrock_runtime(GEN_REGION)

    prompt = f"""
Create a short report title based on the content.

Rules:
- 4 to 8 words
- Title Case
- No quotes, no emojis
- No punctuation at the end
- If unclear: Smart Document Report

Content excerpt:
{doc_text[:6000]}
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 40, "temperature": 0.3, "topP": 0.9},
    )

    title = resp["output"]["message"]["content"][0]["text"].strip()
    title = title.replace('"', "").replace("`", "").strip()

    if not title:
        return "Smart Document Report"

    words = title.split()
    if len(words) > 10:
        title = " ".join(words[:10])

    return title


# -------------------------
# Suggested questions (persona-based)
# -------------------------
def suggest_questions(doc_text: str, user_interest: str = "General", n: int = 6) -> List[str]:
    brt = get_bedrock_runtime(GEN_REGION)

    doc_excerpt = (doc_text or "")[:9000]

    def _call_model(strict: bool) -> str:
        if strict:
            prompt = f"""
You are a smart document copilot.

Return ONLY a valid JSON array of {n} strings.
No markdown. No code fences. No explanation.

Rules:
- Each question <= 12 words.
- Make questions specific to this document.
- Mix: summary, key numbers, risks, actions, missing info check.
- Personalized for role: {user_interest}

EXAMPLE OUTPUT:
["What is the document about?","What key numbers are mentioned?"]

Document excerpt:
{doc_excerpt}
""".strip()
        else:
            prompt = f"""
You are a smart document copilot.

Generate {n} useful questions a user might ask about this document,
personalized for the user's interest/role: {user_interest}

Rules:
- Return ONLY valid JSON array of strings.
- Each question must be <= 12 words.
- No numbering, no bullets.
- Make questions specific to the document.
- Mix: summary, key numbers, risks, actions, and one "missing info" check.

Document excerpt:
{doc_excerpt}
""".strip()

        resp = brt.converse(
            modelId=NOVA_LITE_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 320, "temperature": 0.3, "topP": 0.9},
        )
        return resp["output"]["message"]["content"][0]["text"].strip()

    def _parse_questions(txt: str) -> List[str]:
        txt = _strip_code_fences(txt)

        # Normalize smart quotes → ASCII quotes
        txt = txt.replace("“", '"').replace("”", '"').replace("’", "'")

        # Extract the first JSON array from the text
        arr_txt = _extract_json_array(txt)

        # Remove trailing commas inside arrays (common model slip)
        arr_txt = re.sub(r",\s*([\]\}])", r"\1", arr_txt)

        arr = json.loads(arr_txt)
        if not isinstance(arr, list):
            return []

        out: List[str] = []
        for q in arr:
            if isinstance(q, str):
                q = q.strip()
                if not q:
                    continue
                # enforce <= 12 words softly
                words = q.split()
                if len(words) > 12:
                    q = " ".join(words[:12])
                out.append(q)

        # de-dup while preserving order
        dedup = []
        seen = set()
        for q in out:
            if q.lower() not in seen:
                dedup.append(q)
                seen.add(q.lower())

        return dedup[:n]

    # --- Attempt 1 (normal) ---
    try:
        txt = _call_model(strict=False)
        qs = _parse_questions(txt)
        if qs:
            return qs
    except Exception:
        pass

    # --- Attempt 2 (stricter) ---
    try:
        txt = _call_model(strict=True)
        qs = _parse_questions(txt)
        if qs:
            return qs
    except Exception:
        pass

    # --- Final fallback: always return something doc-aware ---
    # try to grab a few "entities" from text to make questions feel specific
    entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,2}\b", doc_excerpt)
    entities = [e.strip() for e in entities if e.strip()]
    entities = list(dict.fromkeys(entities))[:3]  # unique, max 3

    hint = f" ({entities[0]})" if entities else ""
    role = (user_interest or "General").lower()

    if "finance" in role:
        fallback = [
            f"What total amounts or fees appear{hint}?",
            "What payment terms or due dates are stated?",
            "Any missing invoice fields or inconsistencies?",
            "What risks exist in payment details?",
            "What actions should finance take next?",
            "What numbers should be verified immediately?",
        ]
    elif "legal" in role:
        fallback = [
            f"What obligations are stated{hint}?",
            "Any deadlines, term dates, or notices mentioned?",
            "What risks or liabilities are implied?",
            "What clauses look missing or unclear?",
            "What should be reviewed before signing?",
            "What follow-up questions should we ask?",
        ]
    elif "hr" in role or "recruit" in role:
        fallback = [
            f"What is the candidate/profile summary{hint}?",
            "What key skills and experience are mentioned?",
            "Any gaps, inconsistencies, or missing info?",
            "What role fit is implied and why?",
            "What questions should we ask next?",
            "What dates/timelines should we verify?",
        ]
    elif "research" in role or "student" in role:
        fallback = [
            f"What is the main objective{hint}?",
            "What methodology or approach is described?",
            "What key results or metrics are reported?",
            "What limitations or assumptions are stated?",
            "What future work or next steps are suggested?",
            "What important details seem missing?",
        ]
    else:
        fallback = [
            f"What is this document about{hint}?",
            "What are the key points or sections?",
            "What key numbers, dates, or IDs appear?",
            "What risks, issues, or constraints are mentioned?",
            "What actions or next steps are recommended?",
            "What important information is missing?",
        ]

    return fallback[:n]
