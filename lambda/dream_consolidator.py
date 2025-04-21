import json
import boto3
import os
import uuid
from datetime import datetime
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from sklearn.cluster import KMeans
import numpy as np

# Initialize AWS credentials and OpenSearch client
region = os.environ['AWS_REGION']
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key,
                   credentials.secret_key,
                   region,
                   service,
                   session_token=credentials.token)

host = os.environ['OPENSEARCH_HOST']  # e.g., 'search-domain.region.es.amazonaws.com'
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

def lambda_handler(event, context):
    # Fetch all STM entries
    stm_entries = fetch_all_stm_entries()
    if not stm_entries:
        print("No STM entries found.")
        return {
            'statusCode': 200,
            'body': json.dumps('No STM entries to process.')
        }

    # Extract embeddings and metadata
    embeddings = np.array([entry['_source']['embedding'] for entry in stm_entries])
    priorities = np.array([entry['_source'].get('priority', 1.0) for entry in stm_entries])
    emotions = np.array([entry['_source'].get('emotion', 0.0) for entry in stm_entries])
    importance = np.array([1.0 if entry['_source'].get('important') else 0.0 for entry in stm_entries])
    access_counts = np.array([entry['_source'].get('times_accessed', 0) for entry in stm_entries])

    # Compute salience scores
    salience = (
        0.4 * priorities +
        0.3 * emotions +
        0.2 * importance +
        0.1 * np.log1p(access_counts)
    )

    # Perform clustering
    num_clusters = min(5, len(embeddings))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(embeddings)

    # Select representative entries from each cluster
    selected_entries = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_salience = salience[cluster_indices]
        top_index = cluster_indices[np.argmax(cluster_salience)]
        selected_entries.append(stm_entries[top_index]['_source'])

    # Index selected entries into LTM
    for entry in selected_entries:
        index_into_ltm(entry)

    return {
        'statusCode': 200,
        'body': json.dumps(f'{len(selected_entries)} entries consolidated into LTM.')
    }

def fetch_all_stm_entries():
    query = {
        "query": {
            "match_all": {}
        },
        "size": 1000  # Adjust as needed
    }
    response = client.search(index=stm_index, body=query)
    return response['hits']['hits']

def index_into_ltm(entry):
    doc = {
        "content": entry['content'],
        "embedding": entry['embedding'],
        "timestamp": datetime.utcnow().isoformat(),
        "priority": entry.get('priority', 1.0),
        "emotion": entry.get('emotion', 0.0),
        "tags": entry.get('tags', []),
        "important": entry.get('important', False),
        "times_accessed": entry.get('times_accessed', 0)
    }
    client.index(index=ltm_index, id=str(uuid.uuid4()), body=doc)

