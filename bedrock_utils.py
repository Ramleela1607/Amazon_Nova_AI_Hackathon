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


import re

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_object(s: str) -> str:
    """
    Returns the first JSON object substring found in s: { ... }
    """
    s = _strip_code_fences(s)
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("No JSON object found")
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

    excerpt = (doc_text or "").strip()
    excerpt = excerpt[:12000] if len(excerpt) > 12000 else excerpt

    prompt = f"""
You are a smart document copilot.

Generate {n} useful questions a user might ask about this document,
personalized for the user's interest/role: {user_interest}.

OUTPUT RULES (VERY IMPORTANT):
- Return ONLY a valid JSON array of strings like: ["q1","q2",...]
- No extra text, no markdown, no code fences.
- Each question must be <= 12 words.
- Make questions specific to the document.
- Mix: summary, key numbers, risks, actions, and one missing-info check.

Document excerpt:
{excerpt}
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 320, "temperature": 0.3, "topP": 0.9},
    )

    txt = (resp["output"]["message"]["content"][0]["text"] or "").strip()

    # 1) strict JSON
    try:
        arr = json.loads(_strip_code_fences(txt))
        if isinstance(arr, list):
            out = [q.strip() for q in arr if isinstance(q, str) and q.strip()]
            # de-dup
            seen = set()
            cleaned = []
            for q in out:
                k = q.lower()
                if k not in seen:
                    seen.add(k)
                    cleaned.append(q)
            return cleaned[:n]
    except Exception:
        pass

    # 2) try extracting first JSON array from messy output
    try:
        json_arr_str = _extract_json_array(txt)  # uses your helper
        arr = json.loads(json_arr_str)
        if isinstance(arr, list):
            out = [q.strip() for q in arr if isinstance(q, str) and q.strip()]
            seen = set()
            cleaned = []
            for q in out:
                k = q.lower()
                if k not in seen:
                    seen.add(k)
                    cleaned.append(q)
            return cleaned[:n]
    except Exception:
        pass

    # 3) fallback: parse as lines if model didn't return JSON
    lines = []
    for line in _strip_code_fences(txt).splitlines():
        line = line.strip().lstrip("-•").strip()
        if not line:
            continue
        # remove numbering like "1) " or "1. "
        if len(line) > 2 and line[0].isdigit() and line[1] in [")", "."]:
            line = line[2:].strip()
        if line:
            lines.append(line)

    seen = set()
    cleaned = []
    for q in lines:
        k = q.lower()
        if k not in seen:
            seen.add(k)
            cleaned.append(q)
        if len(cleaned) >= n:
            break

    return cleaned

# -------------------------
# Nova Executive Dashboard Insights
# -------------------------
def generate_dashboard_insights(doc_text: str) -> dict:
    """
    Generate structured dashboard insights using Nova.
    Returns strict JSON with:
    - summary
    - doc_type_guess
    - risk_score
    - key_numbers [{label,value}]
    - key_dates [{label,value}]
    - risks [str]
    - next_actions [str]
    """

    brt = get_bedrock_runtime(GEN_REGION)

    excerpt = (doc_text or "").strip()
    excerpt = excerpt[:15000]

    prompt = f"""
You are an executive document intelligence system.

Analyze the document and return STRICT JSON only.

Required JSON format:

{{
  "summary": "2-4 sentence executive summary",
  "doc_type_guess": "invoice | contract | resume | research_paper | generic",
  "risk_score": 0-100,
  "key_numbers": [
    {{"label": "Metric name", "value": "exact visible value"}}
  ],
  "key_dates": [
    {{"label": "Event name", "value": "exact date string"}}
  ],
  "risks": ["short risk statement"],
  "next_actions": ["short action recommendation"]
}}

Rules:
- Output VALID JSON only.
- Do not include markdown.
- Do not include commentary.
- key_dates must pair a real event with a real visible date.
- risk_score must reflect financial, legal, deadline, or compliance risk.
- If something does not exist, return empty list.
- Do NOT hallucinate.

Document:
{excerpt}
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 800, "temperature": 0.2, "topP": 0.9},
    )

    raw = resp["output"]["message"]["content"][0]["text"].strip()

    # Clean possible formatting issues
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        cleaned = raw[start:end+1]
        return json.loads(cleaned)
    except Exception:
        return {
            "summary": "",
            "doc_type_guess": "generic",
            "risk_score": 0,
            "key_numbers": [],
            "key_dates": [],
            "risks": [],
            "next_actions": [],
        }

import re
import json
from typing import Dict, Any, List

def generate_dashboard_insights_dynamic(doc_text: str, max_metrics: int = 40) -> Dict[str, Any]:
    """
    Dynamic dashboard builder:
    - Works for ANY PDF/image text (numbers, tables, lists)
    - Returns strict JSON for charts + insights
    """

    brt = get_bedrock_runtime(GEN_REGION)

    # --- Local mining (deterministic) ---
    text = (doc_text or "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Extract "label: value" style metrics (very common in invoices/reports)
    metric_candidates = []
    for line in text.splitlines():
        ln = line.strip()
        if not ln or len(ln) < 4:
            continue
        # pattern: Label ... 1234.56 (or $1,234 / 45% / INR 5000)
        m = re.search(r"(.{3,60}?)\s*[:\-–—]\s*([^\n]{1,40})$", ln)
        if m:
            label = m.group(1).strip()
            val = m.group(2).strip()
            metric_candidates.append({"label": label, "value": val})

    # Also grab “numbers with nearby words” (fallback for tables/OCR)
    # Example: "Total 4950", "Due Date 14/02/2023", etc.
    loose_candidates = []
    num_re = re.compile(r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)")
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        nums = list(num_re.finditer(ln))
        if not nums:
            continue
        for mm in nums[:3]:
            num = mm.group(1)
            left = ln[:mm.start()].strip()
            left = re.sub(r"\s+", " ", left)
            # take last few words as label
            words = left.split()
            label = " ".join(words[-6:]) if words else "Number"
            loose_candidates.append({"label": label, "value": num})

    # Compact the input (so Nova gets signals not full doc)
    metric_candidates = metric_candidates[:max_metrics]
    loose_candidates = loose_candidates[:max_metrics]

    # Try to pass a small excerpt too (for summary/doc type)
    excerpt = text[:9000] if len(text) > 9000 else text

    prompt = f"""
You are an Executive Dashboard generator for ANY document.

You will receive:
1) Raw excerpt (may contain tables)
2) Candidate metrics (label/value)
3) Loose numeric signals (label/value)

Your job:
- Return ONLY valid JSON (no markdown/backticks)
- Detect what kind of document this is
- Produce a dashboard with KPIs, derived calculations, and chart-ready data

Return JSON with EXACT keys:
{{
  "doc_type_guess": "generic",
  "summary": "string",
  "kpis": [{{"label":"", "value":"", "note":""}}],
  "derived_insights": ["string"],
  "charts": [
    {{
      "title": "string",
      "type": "bar|line",
      "x": "string",
      "y": "string",
      "data": [{{"x":"", "y": 0}}]
    }}
  ],
  "table_preview": [{{"col1":"", "col2":"", "col3":""}}], 
  "risk_score": 0,
  "risks": ["string"],
  "next_actions": ["string"]
}}

Rules:
- Use ONLY what is supported by text/signals (no invention).
- If you detect a table-like structure, summarize it and put a small preview in table_preview.
- KPIs: select the most important 6–10 values.
- Derived calculations MUST be dynamic:
  - compute min/max/avg when there are multiple numbers
  - compute deltas / % change when pairs/series exist
  - flag anomalies (outliers / sudden jumps) if patterns exist
- Charts MUST be dynamic:
  - If many numeric items exist -> bar chart of top values
  - If series (like stages/months/years) exist -> line chart
- risk_score 0–100 based on risks detected.

RAW EXCERPT:
{excerpt}

CANDIDATE METRICS (label/value):
{json.dumps(metric_candidates, ensure_ascii=False)}

LOOSE NUMERIC SIGNALS (label/value):
{json.dumps(loose_candidates, ensure_ascii=False)}
""".strip()

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1000, "temperature": 0.15, "topP": 0.9},
    )

    txt = (resp["output"]["message"]["content"][0]["text"] or "").strip()

    # Robust JSON parse
    try:
        out = json.loads(txt)
        if isinstance(out, dict):
            return out
    except Exception:
        # Try to salvage if extra text around JSON
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                out = json.loads(txt[start:end+1])
                if isinstance(out, dict):
                    return out
        except Exception:
            pass

    # fallback
    return {
        "doc_type_guess": "generic",
        "summary": "Dashboard could not be generated for this upload.",
        "kpis": [],
        "derived_insights": [],
        "charts": [],
        "table_preview": [],
        "risk_score": 0,
        "risks": [],
        "next_actions": [],
    }
