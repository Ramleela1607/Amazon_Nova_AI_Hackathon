import boto3

# Try a few regions because inference profiles can be listed from different regions depending on setup
REGIONS_TO_TRY = ["ap-south-1", "us-east-1", "us-west-2", "eu-west-1","ap-south-2"]

KEYWORDS = ["nova", "nova-2", "nova 2", "lite"]

def matches(p):
    hay = " ".join([
        str(p.get("inferenceProfileName", "")),
        str(p.get("description", "")),
        str(p.get("inferenceProfileId", "")),
        str(p.get("inferenceProfileArn", "")),
        str(p.get("models", "")),
    ]).lower()
    return any(k in hay for k in KEYWORDS)

for region in REGIONS_TO_TRY:
    print(f"\n=== Region: {region} ===")
    try:
        bedrock = boto3.client("bedrock", region_name=region)  # control-plane
        resp = bedrock.list_inference_profiles(maxResults=100)
        profiles = resp.get("inferenceProfileSummaries", [])
        found = 0

        for p in profiles:
            if matches(p):
                found += 1
                print("NAME:", p.get("inferenceProfileName"))
                print("ID:  ", p.get("inferenceProfileId"))
                print("ARN: ", p.get("inferenceProfileArn"))
                print("TYPE:", p.get("type"))  # SYSTEM_DEFINED or APPLICATION
                print("-" * 60)

        if found == 0:
            print("No Nova-like profiles found here.")
    except Exception as e:
        print("Error:", repr(e))
