import json
import boto3
import os
import uuid
import random
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import numpy as np

# AWS/OpenSearch initialization
region = os.environ['AWS_REGION']
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
host = os.environ['OPENSEARCH_HOST']
stm_index = os.environ.get('STM_INDEX', 'humanai-stm')
vector_dim = int(os.environ.get('VECTOR_DIM', 768))
meta_index = os.environ.get('META_INDEX', 'humanai-meta')  # New meta index for dream_needed flag

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

def lambda_handler(event, context):
    text_input = event.get('text')
    if not text_input:
        return {'statusCode': 400, 'body': json.dumps('No text provided.')}

    now = datetime.utcnow()
    memory_doc = process_text_to_memory(text_input, now)

    index_into_stm(memory_doc)

    reheat_count = memory_reheat(memory_doc['embedding'])

    if reheat_count >= 5:
        trigger_dream_needed(now)

    return {
        'statusCode': 200,
        'body': json.dumps(f'Text processed and stored into STM. Reheated {reheat_count} memories.')
    }

def process_text_to_memory(text, now):
    embedding = fake_text_embedding(text)
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
        "tags": tags
    }
    return doc

def index_into_stm(doc):
    client.index(index=stm_index, id=str(uuid.uuid4()), body=doc)

def compute_salience(text):
    base_salience = min(1.0, len(text) / 300)
    keyword_bonus = 0.1 if any(keyword in text.lower() for keyword in ['important', 'urgent', 'goal']) else 0.0
    random_bias = random.uniform(0.0, 0.05)
    return min(1.0, base_salience + keyword_bonus + random_bias)

def simulate_emotion_tags(text):
    joy = 0.6 if 'happy' in text.lower() else random.uniform(0.0, 0.3)
    fear = 0.6 if 'fear' in text.lower() or 'danger' in text.lower() else random.uniform(0.0, 0.3)
    surprise = 0.6 if 'surprised' in text.lower() else random.uniform(0.0, 0.3)
    return {'joy': joy, 'fear': fear, 'surprise': surprise}

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

def fake_text_embedding(text):
    return np.random.randn(vector_dim).tolist()

def memory_reheat(new_embedding, similarity_threshold=0.7, boost_amount=0.2):
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
    hits = response['hits']['hits']

    reheated = 0
    for hit in hits:
        score = hit['_score'] - 1.0
        if score >= similarity_threshold:
            doc_id = hit['_id']
            source = hit['_source']
            new_attention = min(1.0, source.get('attention_score', 0.5) + boost_amount)
            client.update(index=stm_index, id=doc_id, body={
                "doc": {
                    "attention_score": new_attention,
                    "times_accessed": source.get('times_accessed', 0) + 1
                }
            })
            reheated += 1

    print(f"✅ Memory reheating: {reheated} memories boosted.")
    return reheated

def trigger_dream_needed(now):
    meta_doc = {
        "event": "dream_needed",
        "timestamp": now.isoformat(),
        "reason": "excessive_memory_reheat"
    }
    client.index(index=meta_index, id=str(uuid.uuid4()), body=meta_doc)
    print("🌙 Dream Needed flag triggered due to memory reheating surge.")
