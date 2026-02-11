import json
import boto3

GEN_REGION = "ap-south-1"
EMBED_REGION = "us-east-1"

NOVA_LITE_MODEL_ID = "apac.amazon.nova-lite-v1:0"
NOVA_MM_EMBED_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"

def get_bedrock_runtime(region: str):
    return boto3.client("bedrock-runtime", region_name=region)

def _find_first_float_list(obj):
    """
    Walk a nested dict/list and return the first list that looks like an embedding (list of numbers).
    """
    if isinstance(obj, list):
        # embedding is a list of floats
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
            "text": {"truncationMode": "END", "value": text}
        }
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
    
def extract_fields_json(doc_text: str, doc_type: str = "auto"):
    brt = get_bedrock_runtime(GEN_REGION)

    prompt = f"""
You are an information extraction assistant.
Detect document type if doc_type is "auto".
Extract key fields as VALID JSON only.

doc_type: {doc_type}

Return JSON with:
- doc_type
- fields (object of extracted fields)
- confidence (0-1)
- notes (array)

Document:
{doc_text[:12000]}
"""

    resp = brt.converse(
        modelId=NOVA_LITE_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    return resp["output"]["message"]["content"][0]["text"]
    

def ask_nova_lite(question: str, context_chunks: list[str]):
    brt = get_bedrock_runtime(GEN_REGION)

    context_block = "\n\n".join(
        [f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = f"""You are Smart Document Copilot.
Answer the user's question using ONLY the sources provided.
If the answer is not in the sources, say: "I don't know based on the document."

SOURCES:
{context_block}

QUESTION:
{question}

Return:
1) Answer
2) Sources used: list the source numbers you relied on (e.g., 1,3)
"""

    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
    }

    response = brt.invoke_model(
        modelId=NOVA_LITE_MODEL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]
