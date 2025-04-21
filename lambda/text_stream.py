import os
import json
import uuid
import boto3
from datetime import datetime
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch

# --- Environment variables ---
S3_BUCKET = os.environ.get("S3_BUCKET", "humanai-document-store")
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST")
STM_INDEX = os.environ.get("STM_INDEX", "humanai-stm")
VECTOR_DIM = int(os.environ.get("VECTOR_DIM", "768"))

# --- Init clients ---
s3 = boto3.client("s3")

opensearch = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": 443}],
    use_ssl=True,
    verify_certs=True
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Ensure index exists ---
def ensure_index(index_name):
    if not opensearch.indices.exists(index_name):
        opensearch.indices.create(
            index=index_name,
            body={
                "settings": {
                    "index": {
                        "knn": True
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {"type": "knn_vector", "dimension": VECTOR_DIM},
                        "timestamp": {"type": "date"},
                        "priority": {"type": "float"},
                        "emotion": {"type": "float"},
                        "tags": {"type": "keyword"},
                        "important": {"type": "boolean"}
                    }
                }
            }
        )

ensure_index(STM_INDEX)

# --- Simulated meta-cognitive importance check ---
def is_important(text: str):
    return "remember" in text.lower() or "important" in text.lower()

# --- Lambda Handler ---
def lambda_handler(event, context):
    # Input from S3 OR raw event
    if "s3_key" in event:
        s3_key = event["s3_key"]
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        raw_text = obj["Body"].read().decode("utf-8")
    elif "text" in event:
        raw_text = event["text"]
    else:
        return {"statusCode": 400, "body": "No text provided."}

    print(f"📝 Processing text: {raw_text[:80]}...")

    # --- Embedding ---
    embedding = model.encode(raw_text).tolist()

    # --- Metadata ---
    priority = 1.0
    emotion = 0.2
    tags = ["text"]
    important = is_important(raw_text)

    doc = {
        "content": raw_text,
        "embedding": embedding,
        "timestamp": datetime.utcnow(),
        "priority": priority,
        "emotion": emotion,
        "tags": tags,
        "important": important
    }

    doc_id = str(uuid.uuid4())
    opensearch.index(index=STM_INDEX, id=doc_id, body=doc)

    print(f"✅ Indexed document {doc_id} into {STM_INDEX}")
    return {
        "statusCode": 200,
        "body": f"Indexed text ({'important' if important else 'normal'})"
    }
# This Lambda function processes text input, generates embeddings using a pre-trained model,
# and indexes the data into an OpenSearch index with metadata for priority, emotion, tags, and importance.
# It also checks for meta-cognitive importance based on keywords in the text.
# The function can handle input from S3 or directly from the event payload.