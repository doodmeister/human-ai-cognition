import json
import os
import tempfile
import boto3
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Any

# Add near top of file
import os

# Configuration
S3_BUCKET = os.environ['S3_BUCKET']
MODEL_PATH = os.environ.get('MODEL_PATH', '/opt/models/cnn_video.pt')
FRAME_RATE = int(os.environ.get('FRAME_RATE', '1'))
SUPPORTED_FORMATS = ('.mp4', '.avi', '.mov')

s3 = boto3.client("s3")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define a simple CNN (must match the one saved to MODEL_PATH)
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 2)  # Example: binary classification (e.g., object present or not)
        )

    def forward(self, x):
        return self.net(x)

def load_model(model_path):
    """Safe model loading with error handling."""
    try:
        model = SimpleCNN()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

# Initialize model (once per cold start)
model = load_model(MODEL_PATH)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor()
])


def validate_video_file(video_path):
    """Validate video file before processing."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Unable to open video file: {video_path}")
    
    # Get basic video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return {'fps': fps, 'frame_count': frame_count}


def check_video_size(video_path: str, max_size_mb: int = 500) -> None:
    """Verify video file size is within Lambda limits."""
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValueError(f"Video file too large: {file_size_mb:.1f}MB (max {max_size_mb}MB)")


def extract_frames(video_path: str, frame_rate: int = 1) -> List[np.ndarray]:
    """
    Extract frames from video at specified frame rate.
    
    Args:
        video_path: Path to video file
        frame_rate: Number of frames to extract per second
        
    Returns:
        List of numpy arrays containing frame data
    """
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate)

    frames = []
    count = 0
    success, image = vidcap.read()
    while success:
        if count % interval == 0:
            frames.append(image)
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    return frames


def log_metrics(video_key, frame_count, processing_time, success=True):
    """Log processing metrics."""
    metrics = {
        'timestamp': time.time(),
        'video_key': video_key,
        'frames_processed': frame_count,
        'processing_time_seconds': processing_time,
        'success': success
    }
    logger.info(f"Processing metrics: {json.dumps(metrics)}")


def lambda_handler(event, context):
    try:
        # Input validation
        if 'video_key' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing video_key in event'})
            }
        
        video_key = event['video_key']
        if not video_key.lower().endswith(SUPPORTED_FORMATS):
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Unsupported video format'})
            }

        print(f"📥 Processing video: {video_key}")

        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, "video.mp4")
            s3.download_file(S3_BUCKET, video_key, local_path)

            # Validate video file
            video_info = validate_video_file(local_path)
            print(f"🎥 Video info: {video_info}")

            # Check video size
            check_video_size(local_path)

            frames = extract_frames(local_path, FRAME_RATE)
            print(f"🖼️ Extracted {len(frames)} frames")

            results = []
            for i, frame in enumerate(frames):
                input_tensor = transform(frame).unsqueeze(0)  # (1, 3, 128, 128)
                with torch.no_grad():
                    logits = model(input_tensor)
                    prediction = torch.argmax(logits, dim=1).item()
                results.append({
                    "frame": i,
                    "prediction": prediction
                })

        # Upload results as JSON to S3
        results_key = video_key.replace(".mp4", "_cnn_results.json")
        result_data = json.dumps(results)
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=results_key,
            Body=result_data,
            ContentType="application/json"
        )

        processing_time = time.time() - start_time
        log_metrics(video_key, len(frames), processing_time, success=True)

        print(f"✅ Uploaded results to: {results_key}")
        return {
            "statusCode": 200,
            "body": f"Processed {len(results)} frames from {video_key}"
        }

    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        log_metrics(video_key, 0, 0, success=False)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# This Lambda function processes a video file, extracts frames, runs a CNN model on each frame,
# and uploads the results to S3 as a JSON file.
# Ensure you have the necessary IAM permissions for S3 access and Lambda execution.
# Make sure to package the model and dependencies correctly for Lambda deployment.
