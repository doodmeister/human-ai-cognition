"""
Dream Consolidator Lambda Function

This script consolidates short-term memories (STM) into long-term memory (LTM)
using clustering and salience-based selection. Integrates with AWS OpenSearch,
supports memory decay and sweeping, enforces age limits, and provides robust
retry logic, dynamic configuration overrides, and clean shutdown.
Author: doodmeister
Date: 2025-04-28
"""

import os
import json
import uuid
import time
import logging
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import List, Dict, Any

import boto3
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import numpy as np
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA

# --- Configuration (can be overridden via `event['configuration']`) ---
AWS_REGION        = os.getenv('AWS_REGION', 'us-east-1')
OPENSEARCH_HOST   = os.getenv('OPENSEARCH_HOST')
STM_INDEX         = os.getenv('STM_INDEX', 'humanai-stm')
LTM_INDEX         = os.getenv('LTM_INDEX', 'humanai-ltm')
VECTOR_DIM        = int(os.getenv('VECTOR_DIM', '768'))
FORGET_THRESHOLD  = float(os.getenv('FORGET_THRESHOLD', '0.05'))
BASE_DECAY_RATE   = float(os.getenv('BASE_DECAY_RATE', '0.01'))
MIN_CLUSTER_SIZE  = int(os.getenv('MIN_CLUSTER_SIZE', '3'))
SCROLL_SIZE       = int(os.getenv('SCROLL_SIZE', '1000'))
MAX_TOTAL_ENTRIES = int(os.getenv('MAX_TOTAL_ENTRIES', '10000'))
MAX_STM_AGE_HOURS = int(os.getenv('MAX_STM_AGE_HOURS', '72'))
VISUALIZATION_BUCKET = os.getenv('VISUALIZATION_BUCKET')

SALIENCE_WEIGHTS = {
    "priority":       0.4,
    "emotions":       0.3,
    "importance":     0.2,
    "times_accessed": 0.1
}

if not OPENSEARCH_HOST:
    raise EnvironmentError("Missing required environment variable: OPENSEARCH_HOST")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Retry Decorator ---
def retry(exceptions, tries=3, delay=1, backoff=2):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    logger.warning("Retrying %s due to %s; sleeping %s s", f.__name__, e, mdelay)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return wrapped
    return decorator

# --- OpenSearch Client ---
class OpenSearchClient:
    def __init__(self):
        self._client: OpenSearch = None

    def _get_credentials(self) -> Dict[str, str]:
        secret_id = os.getenv('OS_CREDENTIAL_SECRET', 'MySecretId')
        try:
            sm = boto3.client('secretsmanager', region_name=AWS_REGION)
            secret = sm.get_secret_value(SecretId=secret_id)
            return json.loads(secret['SecretString'])
        except ClientError:
            logger.warning("SecretsManager lookup failed; using default AWS credentials")
            sess = boto3.Session()
            creds = sess.get_credentials().get_frozen_credentials()
            return {
                'access_key':  creds.access_key,
                'secret_key':  creds.secret_key,
                'session_token': creds.token
            }

    def get_client(self) -> OpenSearch:
        if self._client is None:
            creds = self._get_credentials()
            awsauth = AWS4Auth(
                creds['access_key'],
                creds['secret_key'],
                AWS_REGION,
                'es',
                session_token=creds.get('session_token')
            )
            self._client = OpenSearch(
                hosts=[{'host': OPENSEARCH_HOST, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=30
            )
        return self._client

    def cleanup(self):
        if self._client:
            try:
                self._client.transport.close()
            except Exception as e:
                logger.warning("Error closing OpenSearch client: %s", e)
            finally:
                self._client = None

opensearch = OpenSearchClient()

# --- Helper Functions ---
def validate_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Merge event['configuration'] overrides into our config dict."""
    cfg = {}
    overrides = event.get('configuration', {})
    for key in ('MIN_CLUSTER_SIZE','FORGET_THRESHOLD','BASE_DECAY_RATE'):
        if key in overrides:
            try:
                cfg[key] = type(globals()[key])(overrides[key])
            except Exception:
                logger.warning("Invalid override for %s, ignoring", key)
    return cfg

def fetch_all_stm_entries() -> List[Dict[str, Any]]:
    """Fetch STM entries via scroll API."""
    client = opensearch.get_client()
    try:
        resp = client.search(
            index=STM_INDEX,
            body={"query": {"match_all": {}}},
            scroll="2m",
            size=SCROLL_SIZE
        )
        sid  = resp.get('_scroll_id')
        hits = resp['hits']['hits']
        all_hits = hits.copy()
        while hits and len(all_hits) < MAX_TOTAL_ENTRIES:
            resp = client.scroll(scroll_id=sid, scroll="2m")
            hits = resp['hits']['hits']
            all_hits.extend(hits)
        return all_hits[:MAX_TOTAL_ENTRIES]
    except Exception as e:
        logger.error("Error fetching STM entries: %s", e)
        return []

@retry(Exception, tries=3)
def index_into_ltm(source: Dict[str, Any]) -> None:
    """Index a single memory entry into LTM, with retry."""
    client = opensearch.get_client()
    doc = {
        "content":         source.get('content'),
        "embedding":       source.get('embedding'),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "priority":        source.get('priority', 1.0),
        "emotion_vector": {
            "joy":      source.get('emotion_joy', 0.0),
            "fear":     source.get('emotion_fear', 0.0),
            "surprise": source.get('emotion_surprise', 0.0)
        },
        "tags":            source.get('tags', []),
        "important":       source.get('important', False),
        "times_accessed":  source.get('times_accessed', 0),
        "memory_type":     "episodic" if source.get('contextual_tags') else "semantic"
    }
    client.index(index=LTM_INDEX, id=str(uuid.uuid4()), body=doc)

def enforce_stm_age_limit(client: OpenSearch, max_age_hours: int) -> int:
    """Delete STM entries older than `max_age_hours`."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    try:
        resp = client.delete_by_query(
            index=STM_INDEX,
            body={"query": {"range": {"timestamp": {"lt": cutoff.isoformat()}}}}
        )
        deleted = resp.get('deleted', 0)
        logger.info("Enforced STM age limit: removed %d docs older than %d hours.", deleted, max_age_hours)
        return deleted
    except Exception as e:
        logger.error("Failed to enforce STM age limit: %s", e)
        return 0

def sweep_and_decay_memories(forget_threshold: float, base_decay_rate: float) -> Dict[str,int]:
    """Scroll through STM, decay attention, and delete or update each."""
    client = opensearch.get_client()
    decayed = deleted = 0
    try:
        # use scroll here too
        resp = client.search(index=STM_INDEX, body={"query":{"match_all":{}}}, scroll="2m", size=SCROLL_SIZE)
        sid, hits = resp.get('_scroll_id'), resp['hits']['hits']
        all_hits = hits.copy()
        while hits:
            resp = client.scroll(scroll_id=sid, scroll="2m")
            hits = resp['hits']['hits']
            all_hits.extend(hits)
        for hit in all_hits:
            src = hit['_source']
            doc_id = hit['_id']
            curr = src.get('attention_score', 0.5)
            last = src.get('last_decay_check', datetime.now(timezone.utc).isoformat())
            elapsed_h = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()/3600
            emotions = [src.get(k,0.0) for k in
                ('emotion_joy','emotion_fear','emotion_surprise','emotion_anger',
                 'emotion_sadness','emotion_trust','emotion_disgust','emotion_anticipation')]
            max_em = max(emotions) if emotions else 0.0
            rate = base_decay_rate * (2.0 if max_em < 0.2 else 0.5 if max_em > 0.5 else 1.0)
            new_attn = max(0.0, curr * max(0.0, 1 - elapsed_h * rate))
            try:
                if new_attn < forget_threshold:
                    client.delete(index=STM_INDEX, id=doc_id)
                    deleted += 1
                else:
                    client.update(
                        index=STM_INDEX,
                        id=doc_id,
                        body={"doc":{
                            "attention_score": new_attn,
                            "last_decay_check": datetime.now(timezone.utc).isoformat()
                        }}
                    )
                    decayed += 1
            except Exception as e:
                logger.error("Error updating/deleting memory %s: %s", doc_id, e)
        logger.info("Sweeper completed: %d decayed, %d deleted.", decayed, deleted)
    except Exception as e:
        logger.error("Sweeper failed: %s", e)
    return {"decayed": decayed, "deleted": deleted}

def save_visualization_data(vis_data: List[Dict[str, Any]], timestamp: str) -> None:
    """Persist cluster viz JSON to S3."""
    if not VISUALIZATION_BUCKET:
        return
    try:
        s3 = boto3.client('s3')
        key = f"dream_vis/cluster_{timestamp}.json"
        s3.put_object(Bucket=VISUALIZATION_BUCKET,
                      Key=key,
                      Body=json.dumps(vis_data),
                      ContentType='application/json')
        logger.info("Saved visualization data to s3://%s/%s", VISUALIZATION_BUCKET, key)
    except Exception as e:
        logger.warning("Failed to save visualization: %s", e)

def log_consolidation_metrics(initial: int, clusters: int, consolidated: int, sweep: Dict[str,int], aged_out: int) -> None:
    """Emit CloudWatch metrics for consolidation run."""
    try:
        cw = boto3.client('cloudwatch')
        ts = datetime.now(timezone.utc)
        data = [
            {'MetricName':'InitialMemories','Value':initial,'Unit':'Count'},
            {'MetricName':'Clusters','Value':clusters,'Unit':'Count'},
            {'MetricName':'Consolidated','Value':consolidated,'Unit':'Count'},
            {'MetricName':'Decayed','Value':sweep['decayed'],'Unit':'Count'},
            {'MetricName':'Deleted','Value':sweep['deleted'],'Unit':'Count'},
            {'MetricName':'AgedOut','Value':aged_out,'Unit':'Count'},
        ]
        cw.put_metric_data(Namespace='DreamConsolidation',
                           MetricData=[{**m,'Timestamp':ts} for m in data])
    except Exception as e:
        logger.warning("Metrics emission failed: %s", e)

# --- Lambda Handler ---
def lambda_handler(event, context) -> Dict[str,Any]:
    logger.info("Dream consolidation started.")
    # 1) Apply any config overrides
    overrides = validate_event(event)
    cluster_size     = overrides.get('MIN_CLUSTER_SIZE', MIN_CLUSTER_SIZE)
    forget_thresh    = overrides.get('FORGET_THRESHOLD', FORGET_THRESHOLD)
    base_decay_rate  = overrides.get('BASE_DECAY_RATE', BASE_DECAY_RATE)

    # 2) Enforce age-limit before anything else
    client = opensearch.get_client()
    aged_out = enforce_stm_age_limit(client, MAX_STM_AGE_HOURS)

    # 3) Fetch and filter embeddings
    stm_hits = fetch_all_stm_entries()
    logger.info("Fetched %d STM entries.", len(stm_hits))
    if not stm_hits:
        opensearch.cleanup()
        return {"statusCode":200,"body":json.dumps("No STM entries to process.")}

    valid_sources, embeddings = [], []
    for hit in stm_hits:
        emb = hit['_source'].get('embedding')
        if isinstance(emb, list) and len(emb)==VECTOR_DIM:
            valid_sources.append(hit['_source'])
            embeddings.append(emb)
        else:
            logger.warning("Skipping invalid embedding for doc %s", hit.get('_id'))
    if not valid_sources:
        opensearch.cleanup()
        return {"statusCode":200,"body":json.dumps("No valid embeddings found.")}

    embeddings = np.array(embeddings)
    now = datetime.now(timezone.utc)

    # 4) Compute salience
    times_accessed = np.array([s.get('times_accessed',0) for s in valid_sources])
    priority       = np.array([s.get('priority',1.0) for s in valid_sources])
    importance     = np.array([1.0 if s.get('important') else 0.0 for s in valid_sources])
    emotion_feats  = ['emotion_joy','emotion_fear','emotion_surprise']
    emotions       = np.array([
        np.mean([s.get(f,0.0) for f in emotion_feats])
        for s in valid_sources
    ])
    timestamps     = [s.get('timestamp') for s in valid_sources]
    decay = np.array([
        np.exp(-0.001 * max((now - datetime.fromisoformat(ts)).total_seconds(), 0))
        if ts else 1.0
        for ts in timestamps
    ])
    salience = decay * (
        SALIENCE_WEIGHTS["priority"]       * priority +
        SALIENCE_WEIGHTS["emotions"]       * emotions +
        SALIENCE_WEIGHTS["importance"]     * importance +
        SALIENCE_WEIGHTS["times_accessed"] * np.log1p(times_accessed)
    )

    # 5) Cluster & pick representatives
    clusterer = HDBSCAN(min_cluster_size=cluster_size)
    labels    = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info("Formed %d clusters.", n_clusters)

    selected = []
    for cid in set(labels):
        if cid == -1:
            continue
        idxs = np.where(labels==cid)[0]
        best = idxs[np.argmax(salience[idxs])]
        selected.append(valid_sources[best])

    # 6) Index into LTM
    for src in selected:
        index_into_ltm(src)
    logger.info("Indexed %d entries into LTM.", len(selected))

    # 7) Visualization save
    try:
        proj = PCA(n_components=2).fit_transform(embeddings)
        vis = [{"x":float(x),"y":float(y),"label":int(lbl)}
               for (x,y),lbl in zip(proj, labels)]
        save_visualization_data(vis, now.isoformat())
    except Exception as e:
        logger.warning("Visualization step failed: %s", e)

    # 8) Sweep & decay
    sweep_results = sweep_and_decay_memories(forget_thresh, base_decay_rate)

    # 9) Emit metrics & cleanup
    log_consolidation_metrics(
        initial=len(stm_hits),
        clusters=n_clusters,
        consolidated=len(selected),
        sweep=sweep_results,
        aged_out=aged_out
    )
    opensearch.cleanup()

    return {
        "statusCode": 200,
        "body": json.dumps({
            "consolidated": len(selected),
            "swept":       sweep_results,
            "aged_out":    aged_out
        })
    }
