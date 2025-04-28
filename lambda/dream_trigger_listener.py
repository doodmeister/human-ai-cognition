import json
import boto3
import os
import uuid
from datetime import datetime, timedelta
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import requests

# AWS/OpenSearch initialization
region = os.environ['AWS_REGION']
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
host = os.environ['OPENSEARCH_HOST']
meta_index = os.environ.get('META_INDEX', 'humanai-meta')
dream_trigger_url = os.environ['DREAM_LAMBDA_URL']  # API Gateway URL to call Dream State

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

def lambda_handler(event, context):
    # Search for any recent dream_needed flags in the past 10 minutes
    now = datetime.utcnow()
    start_time = (now - timedelta(minutes=10)).isoformat()

    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"event": "dream_needed"}},
                    {"range": {"timestamp": {"gte": start_time}}}
                ]
            }
        },
        "size": 1
    }

    response = client.search(index=meta_index, body=query)
    hits = response['hits']['hits']

    if hits:
        print("🌙 Dream trigger detected! Initiating Dream State...")
        trigger_dream_cycle()
        return {
            'statusCode': 200,
            'body': json.dumps('Dream cycle triggered.')
        }
    else:
        print("No recent dream_needed flags detected.")
        return {
            'statusCode': 200,
            'body': json.dumps('No dream trigger found.')
        }

def trigger_dream_cycle():
    try:
        response = requests.post(dream_trigger_url)
        print(f"Dream State Lambda invoked, status code: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Failed to invoke Dream State: {e}")