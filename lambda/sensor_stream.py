import boto3
import requests
import time
import os
from datetime import datetime

# AWS S3 setup
s3 = boto3.client('s3')
S3_BUCKET = os.getenv('S3_BUCKET', 'cognition-inputs')

# NWS Weather API endpoint
NWS_API_URL = 'https://api.weather.gov/gridpoints/ILX/57,73/forecast'  # Example: Clinton, IL region

# Interval for polling sensor data (in seconds)
SENSOR_INTERVAL = 600  # every 10 minutes

def fetch_weather_data():
    """Fetch live weather data from the NWS API."""
    try:
        headers = {'User-Agent': 'HumanAI-Cognition/1.0'}
        response = requests.get(NWS_API_URL, headers=headers)
        if response.status_code == 200:
            forecast = response.json()['properties']['periods'][0]  # Get the current forecast period
            summary = f"Weather Update: {forecast['name']} - {forecast['shortForecast']} at {forecast['temperature']} {forecast['temperatureUnit']}"
            return summary.encode('utf-8')
        else:
            print(f"Failed to fetch weather data: {response.status_code}")
    except Exception as e:
        print(f"Error fetching weather data: {e}")
    return None

def upload_sensor_data_to_s3(data, filename):
    """Upload sensor data summary to S3."""
    s3.put_object(Bucket=S3_BUCKET, Key=f"sensor_data/{filename}", Body=data)
    print(f"Uploaded sensor data {filename} to bucket {S3_BUCKET}")

if __name__ == '__main__':
    while True:
        print("Fetching sensor data (weather)...")
        data = fetch_weather_data()
        if data:
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
            filename = f"weather_{timestamp}.txt"
            upload_sensor_data_to_s3(data, filename)
        else:
            print("No data retrieved.")

        print(f"Waiting for {SENSOR_INTERVAL} seconds...")
        time.sleep(SENSOR_INTERVAL)
