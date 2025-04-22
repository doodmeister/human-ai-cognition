import json
import boto3
import os
import uuid
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import numpy as np
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA

from metacognition.meta_feedback import MetaFeedbackManager

# AWS/OpenSearch initialization
region = os.environ['AWS_REGION']
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key,
                   credentials.secret_key,
                   region,
                   service,
                   session_token=credentials.token)

host = os.environ['OPENSEARCH_HOST']
stm_index = os.environ.get('STM_INDEX', 'humanai-stm')
ltm_index = os.environ.get('LTM_INDEX', 'humanai-ltm')
vector_dim = int(os.environ.get('VECTOR_DIM', 768))

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

meta_manager = MetaFeedbackManager()

def lambda_handler(event, context):
    stm_entries = fetch_all_stm_entries()
    if not stm_entries:
        return {'statusCode': 200, 'body': json.dumps('No STM entries to process.')}

    now = datetime.utcnow()
    embeddings = np.array([entry['_source']['embedding'] for entry in stm_entries])
    timestamps = [entry['_source'].get('timestamp') for entry in stm_entries]
    times_accessed = np.array([entry['_source'].get('times_accessed', 0) for entry in stm_entries])
    priority = np.array([entry['_source'].get('priority', 1.0) for entry in stm_entries])
    importance = np.array([1.0 if entry['_source'].get('important') else 0.0 for entry in stm_entries])

    emotion_features = ['emotion_joy', 'emotion_fear', 'emotion_surprise']
    emotions = np.array([
        np.mean([entry['_source'].get(feat, 0.0) for feat in emotion_features])
        for entry in stm_entries
    ])

    decay = np.array([
        np.exp(-0.001 * max((now - datetime.fromisoformat(ts)).total_seconds(), 0))
        if ts else 1.0 for ts in timestamps
    ])

    salience = decay * (
        0.4 * priority +
        0.3 * emotions +
        0.2 * importance +
        0.1 * np.log1p(times_accessed)
    )

    clusterer = HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(embeddings)

    selected_entries = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_salience = salience[cluster_indices]
        top_idx = cluster_indices[np.argmax(cluster_salience)]
        selected_entries.append(stm_entries[top_idx]['_source'])

    for entry in selected_entries:
        index_into_ltm(entry)

    avg_salience_selected = float(np.mean([
        salience[i] for i, entry in enumerate(stm_entries) if entry['_source'] in selected_entries
    ]))

    meta_manager.send_feedback(
        event='dream_consolidation',
        count=len(selected_entries),
        avg_salience=avg_salience_selected
    )

    log_visualization_data(embeddings, labels)

    return {
        'statusCode': 200,
        'body': json.dumps(f'{len(selected_entries)} entries consolidated into LTM.')
    }

def fetch_all_stm_entries():
    query = {
        "query": {"match_all": {}},
        "size": 1000
    }
    response = client.search(index=stm_index, body=query)
    return response['hits']['hits']

def index_into_ltm(entry):
    doc = {
        "content": entry['content'],
        "embedding": entry['embedding'],
        "timestamp": datetime.utcnow().isoformat(),
        "priority": entry.get('priority', 1.0),
        "emotion_vector": {
            "joy": entry.get('emotion_joy', 0.0),
            "fear": entry.get('emotion_fear', 0.0),
            "surprise": entry.get('emotion_surprise', 0.0)
        },
        "tags": entry.get('tags', []),
        "important": entry.get('important', False),
        "times_accessed": entry.get('times_accessed', 0),
        "memory_type": "episodic" if 'contextual_tags' in entry else "semantic"
    }
    client.index(index=ltm_index, id=str(uuid.uuid4()), body=doc)

def log_visualization_data(embeddings, labels):
    pca_proj = PCA(n_components=2).fit_transform(embeddings)
    vis_data = [
        {"x": float(x), "y": float(y), "label": int(lbl)}
        for (x, y), lbl in zip(pca_proj, labels)
    ]
    with open("/tmp/cluster_vis_log.json", "w") as f:
        json.dump(vis_data, f)
    # Consider uploading to S3 or pushing to OpenSearch for visualization