import boto3
import requests
import os
from datetime import datetime

# AWS Setup
s3 = boto3.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET", "cognition-inputs")

# Weather API setup
NWS_API_URL = "https://api.weather.gov/gridpoints/ILX/57,73/forecast"

def fetch_weather_summary():
    """Fetch a short weather summary string from NWS API."""
    try:
        headers = {"User-Agent": "HumanAI-Cognition/1.0"}
        response = requests.get(NWS_API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        forecast = response.json()["properties"]["periods"][0]
        return f"Weather Update: {forecast['name']} - {forecast['shortForecast']} at {forecast['temperature']} {forecast['temperatureUnit']}"
    except Exception as e:
        print(f"⚠️ Failed to fetch weather data: {e}")
        return None

def upload_to_s3(text, filename):
    """Upload string data to S3."""
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=f"sensor_data/{filename}",
            Body=text.encode("utf-8"),
            ContentType="text/plain"
        )
        print(f"✅ Uploaded {filename} to {S3_BUCKET}")
        return True
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def lambda_handler(event, context):
    """AWS Lambda entry point."""
    print("📡 Polling weather sensor...")

    summary = fetch_weather_summary()
    if summary:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        filename = f"weather_{timestamp}.txt"
        success = upload_to_s3(summary, filename)
        return {
            "statusCode": 200 if success else 500,
            "message": "Sensor data uploaded." if success else "Upload failed."
        }

    return {
        "statusCode": 500,
        "message": "Failed to retrieve weather data."
    }
