import boto3

REGION = "ap-south-1"  # your normal region

bedrock = boto3.client("bedrock", region_name=REGION)  # control plane (NOT bedrock-runtime)

resp = bedrock.list_inference_profiles(maxResults=100)

for p in resp.get("inferenceProfileSummaries", []):
    name = p.get("inferenceProfileName", "")
    pid = p.get("inferenceProfileId", "")
    arn = p.get("inferenceProfileArn", "")
    desc = p.get("description", "")
    if "Nova" in name or "nova" in name or "Nova" in desc or "nova" in desc:
        print("NAME:", name)
        print("ID:", pid)
        print("ARN:", arn)
        print("-" * 60)
