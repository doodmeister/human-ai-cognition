"""
Video Stream Processing Lambda

This Lambda downloads a video from S3, validates it, streams frames
through a CNN in batches, and uploads the JSON results back to S3.

Author: doodmeister
Date: 2025-04-28
"""

import os
import json
import tempfile
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import List, Dict, Any

import boto3
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from botocore.exceptions import ClientError

# --- Helpers ---
def get_env_var(name: str, default: Any = None, required: bool = False) -> Any:
    val = os.getenv(name, default)
    if required and val is None:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return val

def retry(exceptions, tries=3, delay=1, backoff=2):
    """Simple retry decorator with exponential backoff."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    logger.warning(
                        "Retry %s due to %s, sleeping %s s",
                        f.__name__, e, mdelay
                    )
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return wrapped
    return decorator

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
S3_BUCKET          = get_env_var("S3_BUCKET", required=True)
MODEL_PATH         = get_env_var("MODEL_PATH", "/opt/models/cnn_video.pt")
FRAME_RATE         = int(get_env_var("FRAME_RATE", "1"))        # frames per second
MAX_VIDEO_SIZE_MB  = float(get_env_var("MAX_VIDEO_SIZE_MB", "500"))
BATCH_SIZE         = int(get_env_var("BATCH_SIZE", "8"))
SUPPORTED_FORMATS  = tuple(get_env_var("SUPPORTED_FORMATS", ".mp4,.avi,.mov").split(","))

# --- AWS Clients ---
s3 = boto3.client("s3")

# --- Retry-Wrapped S3 Operations ---
@retry((ClientError,), tries=3, delay=1)
def download_from_s3(bucket: str, key: str, local_path: str):
    return s3.download_file(bucket, key, local_path)

@retry((ClientError,), tries=3, delay=1)
def upload_to_s3(bucket: str, key: str, body: bytes, content_type: str):
    return s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)

# --- Model Definition & Loading ---
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

def load_model(path: str) -> torch.nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    model = SimpleCNN()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    raise

# --- Transform Pipeline ---
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# --- Metrics Logging ---
def log_metrics(video_key: str, frames: int, elapsed: float, success: bool):
    metrics = {
        "timestamp": time.time(),
        "video_key": video_key,
        "frames_processed": frames,
        "processing_time_s": elapsed,
        "success": success
    }
    logger.info("Metrics: %s", json.dumps(metrics))

# --- Lambda Handler ---
def lambda_handler(event, context):
    video_key = event.get("video_key")
    if not video_key:
        return {"statusCode": 400, "body": json.dumps({"error": "Missing video_key"})}

    ext = os.path.splitext(video_key)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        return {"statusCode": 400, "body": json.dumps({"error": "Unsupported format"})}

    start = time.time()
    frames_processed = 0

    try:
        # 1) Pre-check size via HEAD
        head = s3.head_object(Bucket=S3_BUCKET, Key=video_key)
        size_mb = head["ContentLength"] / (1024**2)
        if size_mb > MAX_VIDEO_SIZE_MB:
            raise ValueError(f"Video too large: {size_mb:.1f}MB > {MAX_VIDEO_SIZE_MB}MB")

        # 2) Download
        with tempfile.TemporaryDirectory() as tmp:
            local_video = os.path.join(tmp, f"input{ext}")
            download_from_s3(S3_BUCKET, video_key, local_video)

            # 3) Validate with OpenCV
            cap = cv2.VideoCapture(local_video)
            if not cap.isOpened():
                raise RuntimeError("Cannot open video")
            fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            logger.info("Video opened: fps=%s, frames=%s", fps, total_frames)

            # 4) Stream & batch inference
            interval = max(int(fps / FRAME_RATE), 1)
            batch_tensors = []
            batch_indices = []
            results: List[Dict[str,int]] = []
            idx = 0

            success, frame = cap.read()
            while success:
                if idx % interval == 0:
                    img = frame.copy()
                    tensor = transform(img).unsqueeze(0)  # (1,3,128,128)
                    batch_tensors.append(tensor)
                    batch_indices.append(idx)

                    # run batch when full
                    if len(batch_tensors) >= BATCH_SIZE:
                        batch = torch.cat(batch_tensors, dim=0)
                        with torch.no_grad():
                            logits = model(batch)
                            preds = logits.argmax(dim=1).cpu().numpy()
                        for i, p in zip(batch_indices, preds):
                            results.append({"frame": i, "prediction": int(p)})
                        batch_tensors.clear()
                        batch_indices.clear()

                idx += 1
                success, frame = cap.read()

            # flush remaining batch
            if batch_tensors:
                batch = torch.cat(batch_tensors, dim=0)
                with torch.no_grad():
                    logits = model(batch)
                    preds = logits.argmax(dim=1).cpu().numpy()
                for i, p in zip(batch_indices, preds):
                    results.append({"frame": i, "prediction": int(p)})

            cap.release()
            frames_processed = len(results)
            logger.info("Processed %d frames", frames_processed)

            # 5) Upload results
            base = os.path.splitext(video_key)[0]
            result_key = f"{base}_cnn_results.json"
            body = json.dumps(results).encode("utf-8")
            upload_to_s3(S3_BUCKET, result_key, body, "application/json")
            logger.info("Results uploaded to %s", result_key)

        # 6) Log metrics & return
        elapsed = time.time() - start
        log_metrics(video_key, frames_processed, elapsed, success=True)
        return {"statusCode": 200,
                "body": f"Processed {frames_processed} frames from {video_key}"}

    except Exception as e:
        logger.error("Error processing %s: %s", video_key, e, exc_info=True)
        elapsed = time.time() - start
        log_metrics(video_key or "unknown", frames_processed, elapsed, success=False)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
