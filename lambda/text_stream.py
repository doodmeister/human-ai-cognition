import json
import boto3
import os
import uuid
import random
import time
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURE LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# ENVIRONMENT VALIDATION
# -----------------------------
def validate_env_vars():
    required_vars = ['AWS_REGION', 'OPENSEARCH_HOST', 'STM_INDEX', 'META_INDEX']
    for var in required_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required environment variable: {var}")

validate_env_vars()

# -----------------------------
# AWS/OPENSEARCH INITIALIZATION
# -----------------------------
region = os.environ['AWS_REGION']
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
host = os.environ['OPENSEARCH_HOST']
stm_index = os.environ.get('STM_INDEX', 'humanai-stm')
vector_dim = int(os.environ.get('VECTOR_DIM', 768))
meta_index = os.environ.get('META_INDEX', 'humanai-meta')

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
BOOST_AMOUNT = float(os.getenv("BOOST_AMOUNT", 0.2))
DREAM_TRIGGER_THRESHOLD = int(os.getenv("DREAM_TRIGGER_THRESHOLD", 5))

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30
)

# -----------------------------
# LAZY EMBEDDING MODEL LOADING
# -----------------------------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Loading SentenceTransformer model...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

# -----------------------------
# HANDLER FUNCTION
# -----------------------------
def lambda_handler(event, context):
    text_input = event.get('text')
    if not text_input:
        logger.warning("No text provided.")
        return {'statusCode': 400, 'body': json.dumps('No text provided.')}

    now = datetime.utcnow()
    memory_doc = process_text_to_memory(text_input, now)

    index_into_stm(memory_doc)

    reheat_count = memory_reheat(memory_doc['embedding'])

    if reheat_count >= DREAM_TRIGGER_THRESHOLD:
        trigger_dream_needed(now)

    return {
        'statusCode': 200,
        'body': json.dumps(f'Text processed and stored into STM. Reheated {reheat_count} memories.')
    }

# -----------------------------
# MEMORY PROCESSING
# -----------------------------
def process_text_to_memory(text, now):
    embedding = generate_embedding(text)
    attention_score = 1.0

    salience = compute_salience(text)
    emotions = simulate_emotion_tags(text)
    tags = extract_tags(text)

    doc = {
        "content": text,
        "embedding": embedding,
        "timestamp": now.isoformat(),
        "priority": salience,
        "attention_score": attention_score,
        "times_accessed": 0,
        "important": determine_importance(text),
        "emotion_joy": emotions['joy'],
        "emotion_fear": emotions['fear'],
        "emotion_surprise": emotions['surprise'],
        "emotion_anger": emotions['anger'],
        "emotion_sadness": emotions['sadness'],
        "emotion_trust": emotions['trust'],
        "emotion_disgust": emotions['disgust'],
        "emotion_anticipation": emotions['anticipation'],
        "tags": tags,
        "last_decay_check": now.isoformat()
    }
    return doc

def index_into_stm(doc):
    try:
        client.index(index=stm_index, id=str(uuid.uuid4()), body=doc)
    except Exception as e:
        logger.error(f"Failed to index document into STM: {e}")
        raise

# -----------------------------
# ATTRIBUTE SIMULATION
# -----------------------------
def compute_salience(text):
    base_salience = min(1.0, len(text) / 300)
    keyword_bonus = 0.1 if any(keyword in text.lower() for keyword in ['important', 'urgent', 'goal']) else 0.0
    random_bias = random.uniform(0.0, 0.05)
    return min(1.0, base_salience + keyword_bonus + random_bias)

def simulate_emotion_tags(text):
    lower_text = text.lower()

    emotions = {
        'joy': 0.6 if any(word in lower_text for word in ['happy', 'joy', 'love']) else random.uniform(0.0, 0.3),
        'fear': 0.6 if any(word in lower_text for word in ['fear', 'danger', 'worry', 'scared']) else random.uniform(0.0, 0.3),
        'surprise': 0.6 if any(word in lower_text for word in ['surprise', 'unexpected', 'shock']) else random.uniform(0.0, 0.3),
        'anger': 0.6 if any(word in lower_text for word in ['angry', 'rage', 'mad', 'furious']) else random.uniform(0.0, 0.3),
        'sadness': 0.6 if any(word in lower_text for word in ['sad', 'cry', 'upset', 'depressed']) else random.uniform(0.0, 0.3),
        'trust': 0.6 if any(word in lower_text for word in ['trust', 'rely', 'faith']) else random.uniform(0.0, 0.3),
        'disgust': 0.6 if any(word in lower_text for word in ['disgust', 'gross', 'nausea', 'repulsed']) else random.uniform(0.0, 0.3),
        'anticipation': 0.6 if any(word in lower_text for word in ['expect', 'await', 'anticipate', 'soon']) else random.uniform(0.0, 0.3),
    }

    return emotions

def determine_importance(text):
    important_keywords = ['must', 'need', 'critical', 'urgent']
    return any(word in text.lower() for word in important_keywords)

def extract_tags(text):
    basic_tags = []
    if 'project' in text.lower():
        basic_tags.append('project')
    if 'meeting' in text.lower():
        basic_tags.append('meeting')
    if 'deadline' in text.lower():
        basic_tags.append('deadline')
    return basic_tags

def generate_embedding(text):
    model = get_embedder()
    return model.encode(text).tolist()

# -----------------------------
# MEMORY REHEATING + DECAY
# -----------------------------
def memory_reheat(new_embedding, similarity_threshold=SIMILARITY_THRESHOLD, boost_amount=BOOST_AMOUNT):
    try:
        query = {
            "size": 100,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": new_embedding}
                    }
                }
            }
        }
        response = client.search(index=stm_index, body=query)
        if 'hits' not in response:
            logger.warning("No hits found in memory reheating query.")
            return 0
    except Exception as e:
        logger.error(f"Failed to query STM for memory reheating: {e}")
        return 0

    reheated = 0
    try:
        hits = response['hits']['hits']

        for hit in hits:
            score = hit['_score'] - 1.0
            if score >= similarity_threshold:
                source = hit['_source']
                new_attention = min(1.0, source.get('attention_score', 0.5) + boost_amount)
                doc_id = hit['_id']

                last_check = datetime.fromisoformat(source.get('last_decay_check', datetime.utcnow().isoformat()))
                elapsed_hours = (datetime.utcnow() - last_check).total_seconds() / 3600
                decay_factor = max(0.0, 1.0 - 0.01 * elapsed_hours)  # 1% decay per hour

                updated_attention = new_attention * decay_factor

                client.update(
                    index=stm_index,
                    id=doc_id,
                    body={"doc": {
                        "attention_score": updated_attention,
                        "times_accessed": source.get('times_accessed', 0) + 1,
                        "last_decay_check": datetime.utcnow().isoformat()
                    }}
                )
                reheated += 1
        logger.info(f"✅ Memory reheating complete: {reheated} memories boosted/decayed.")
        return reheated
    except Exception as e:
        logger.error(f"Error during reheating update: {e}")
        return 0

# -----------------------------
# DREAM TRIGGER EVENT
# -----------------------------
def trigger_dream_needed(now):
    meta_doc = {
        "event": "dream_needed",
        "timestamp": now.isoformat(),
        "reason": "excessive_memory_reheat"
    }
    client.index(index=meta_index, id=str(uuid.uuid4()), body=meta_doc)
    logger.info("🌙 Dream Needed flag triggered.")
