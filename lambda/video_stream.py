import json
import os
import tempfile
import boto3
import torch
import torchvision.transforms as T
import cv2
import numpy as np

# Load from environment
S3_BUCKET = os.environ["S3_BUCKET"]
MODEL_PATH = "/opt/models/cnn_video.pt"  # Path in Lambda layer or .zip bundle

s3 = boto3.client("s3")

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

# Initialize model (once per cold start)
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor()
])


def extract_frames(video_path, frame_rate=1):
    """
    Extracts 1 frame per second from the video.
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


def lambda_handler(event, context):
    # Assume event includes video file key
    video_key = event.get("video_key", "sample.mp4")
    print(f"📥 Processing video: {video_key}")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "video.mp4")
        s3.download_file(S3_BUCKET, video_key, local_path)

        frames = extract_frames(local_path)
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

    print(f"✅ Uploaded results to: {results_key}")
    return {
        "statusCode": 200,
        "body": f"Processed {len(results)} frames from {video_key}"
    }
# This Lambda function processes a video file, extracts frames, runs a CNN model on each frame,
# and uploads the results to S3 as a JSON file.
# Ensure you have the necessary IAM permissions for S3 access and Lambda execution.
# Make sure to package the model and dependencies correctly for Lambda deployment.
