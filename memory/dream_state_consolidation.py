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
import torch

from metacognition.meta_feedback import MetaFeedbackManager
from model.dpad_transformer import DPADRNN, DPADTrainer
from memory.procedural_memory import ProceduralMemory

# AWS/OpenSearch initialization
region = os.environ['AWS_REGION']
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
host = os.environ['OPENSEARCH_HOST']
stm_index = os.environ.get('STM_INDEX', 'humanai-stm')
ltm_index = os.environ.get('LTM_INDEX', 'humanai-ltm')
vector_dim = int(os.environ.get('VECTOR_DIM', 768))
s3_bucket = os.environ.get('S3_MODEL_BUCKET')

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

meta_manager = MetaFeedbackManager()
procedural_memory = ProceduralMemory()

def upload_file_to_s3(local_path, bucket, key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket, key)
        print(f"✅ Uploaded {local_path} to s3://{bucket}/{key}")
    except Exception as e:
        print(f"⚠️ S3 upload failed: {e}")

def lambda_handler(event, context):
    stm_entries = fetch_all_stm_entries()
    if not stm_entries:
        return {'statusCode': 200, 'body': json.dumps('No STM entries to process.')}

    now = datetime.utcnow()
    embeddings, salience_scores, replay_data = compute_salience_and_replay_data(stm_entries, now)
    labels = cluster_memories(embeddings)

    selected_entries, selected_replay_data = select_high_salience_memories(labels, salience_scores, stm_entries, replay_data)

    losses_behavior, losses_residual = retrain_dpad(selected_replay_data, embeddings)

    for entry in selected_entries:
        index_into_ltm(entry)

    avg_salience_selected = float(np.mean([
        salience_scores[i] for i, entry in enumerate(stm_entries) if entry['_source'] in selected_entries
    ]))

    meta_manager.send_feedback(
        event='dream_consolidation',
        count=len(selected_entries),
        avg_salience=avg_salience_selected
    )

    log_visualization_data(embeddings, labels)

    export_procedural_memories()

    return {
        'statusCode': 200,
        'body': json.dumps(f'{len(selected_entries)} entries consolidated into LTM and retrained into DPAD. Procedural memories exported.')
    }

def fetch_all_stm_entries():
    query = {"query": {"match_all": {}}, "size": 1000}
    response = client.search(index=stm_index, body=query)
    return response['hits']['hits']

def compute_salience_and_replay_data(stm_entries, now):
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

    replay_data = []
    for entry in stm_entries:
        embedding = np.array(entry['_source']['embedding'])
        behavior = np.sum(embedding)
        replay_data.append((embedding, behavior))

    # Add procedural memory handling
    procedural_entries = []
    for entry in stm_entries:
        if entry['_source'].get('memory_type') == 'procedural':
            steps = entry['_source'].get('steps', [])
            name = entry['_source'].get('name')
            if steps and name:
                try:
                    procedural_memory.add_procedure(name, steps)
                    procedural_entries.append(entry)
                except Exception as e:
                    print(f"Failed to add procedural memory: {e}")

    # Update salience calculation for procedural memories
    for entry in procedural_entries:
        idx = stm_entries.index(entry)
        # Increase salience for procedural memories that have been successfully executed
        if entry['_source'].get('execution_success', False):
            salience[idx] *= 1.2

    return embeddings, salience, replay_data

def cluster_memories(embeddings):
    clusterer = HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(embeddings)
    return labels

def select_high_salience_memories(labels, salience, stm_entries, replay_data):
    selected_entries = []
    selected_replay = []
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_salience = salience[cluster_indices]
        top_idx = cluster_indices[np.argmax(cluster_salience)]

        selected_entries.append(stm_entries[top_idx]['_source'])
        selected_replay.append(replay_data[top_idx])

    return selected_entries, selected_replay

def retrain_dpad(replay_data, embeddings, device="cpu", epochs_behavior=5, epochs_residual=5):
    if not replay_data:
        print("No STM memories selected for retraining.")
        return [], []

    print(f"Starting DPAD retraining on {len(replay_data)} replay memories...")

    model = DPADRNN(
        input_size=768,
        hidden_size=128,
        output_size=1,
        nonlinear_input=True,
        nonlinear_recurrence=True,
        nonlinear_behavior_readout=True,
        nonlinear_reconstruction=True,
        use_layernorm=True,
        dropout=0.1,
    )

    try:
        model.load("/tmp/dpad_model.pth", device=device)
        print("Loaded existing DPAD model.")
    except Exception:
        print("No existing model found. Training a fresh DPAD model.")

    inputs = torch.stack([torch.tensor(x, dtype=torch.float32) for (x, _) in replay_data])
    behaviors = torch.stack([torch.tensor(y, dtype=torch.float32) for (_, y) in replay_data])

    dataset = torch.utils.data.TensorDataset(inputs, behaviors, inputs)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    trainer = DPADTrainer(model, device=device)

    optimizer_behavior = torch.optim.Adam(
        list(model.input_map.parameters()) +
        list(model.rnn_behavior.parameters()) +
        list(model.behavior_readout.parameters()), lr=1e-4)

    optimizer_residual = torch.optim.Adam(
        list(model.rnn_residual.parameters()) +
        list(model.reconstruction_head.parameters()), lr=1e-4)

    behavior_losses = trainer.train_behavior_phase(data_loader, optimizer_behavior, torch.nn.MSELoss(), epochs=epochs_behavior)
    residual_losses = trainer.train_residual_phase(data_loader, optimizer_residual, torch.nn.MSELoss(), epochs=epochs_residual)

    model.save("/tmp/dpad_model.pth")
    print("✅ DPAD retraining complete and model saved.")

    if s3_bucket:
        timestamp = datetime.utcnow().isoformat()
        s3_key_model = f"dpad_models/dpad_model_{timestamp}.pth"
        upload_file_to_s3("/tmp/dpad_model.pth", s3_bucket, s3_key_model)

        log_dream_retraining_metrics(behavior_losses, residual_losses, embeddings, timestamp)
    else:
        print("⚠️ No S3 bucket configured; skipping model upload.")

    return behavior_losses, residual_losses

def log_dream_retraining_metrics(losses_behavior, losses_residual, replay_embeddings, timestamp, output_dir="/tmp/dream_logs"):
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/behavior_loss.json", "w") as f:
        json.dump(losses_behavior, f)
    with open(f"{output_dir}/residual_loss.json", "w") as f:
        json.dump(losses_residual, f)

    pca_proj = PCA(n_components=2).fit_transform(replay_embeddings)
    replay_vis = [{"x": float(x), "y": float(y)} for x, y in pca_proj]
    with open(f"{output_dir}/replay_embeddings_pca.json", "w") as f:
        json.dump(replay_vis, f)

    if s3_bucket:
        upload_file_to_s3(f"{output_dir}/behavior_loss.json", s3_bucket, f"dpad_logs/behavior_loss_{timestamp}.json")
        upload_file_to_s3(f"{output_dir}/residual_loss.json", s3_bucket, f"dpad_logs/residual_loss_{timestamp}.json")
        upload_file_to_s3(f"{output_dir}/replay_embeddings_pca.json", s3_bucket, f"dpad_logs/replay_pca_{timestamp}.json")

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
        "memory_type": entry.get('memory_type', 'semantic')
    }
    
    # Add procedural-specific fields
    if entry.get('memory_type') == 'procedural':
        doc.update({
            "steps": entry.get('steps', []),
            "name": entry.get('name'),
            "execution_success": entry.get('execution_success', False),
            "execution_count": entry.get('execution_count', 0)
        })
    
    client.index(index=ltm_index, id=str(uuid.uuid4()), body=doc)

def log_visualization_data(embeddings, labels):
    pca_proj = PCA(n_components=2).fit_transform(embeddings)
    vis_data = [
        {"x": float(x), "y": float(y), "label": int(lbl)}
        for (x, y), lbl in zip(pca_proj, labels)
    ]
    with open("/tmp/cluster_vis_log.json", "w") as f:
        json.dump(vis_data, f)

def export_procedural_memories():
    """Export consolidated procedural memories to a JSON file."""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    local_path = f"/tmp/procedural_memories_{timestamp}.json"
    
    try:
        success = procedural_memory.export_procedures(local_path)
        if success:
            if s3_bucket:
                s3_key = f"procedural_memories/backup_{timestamp}.json"
                upload_file_to_s3(local_path, s3_bucket, s3_key)
                print(f"✅ Exported procedural memories to S3")
            return True
        return False
    except Exception as e:
        print(f"Failed to export procedural memories: {e}")
        return False