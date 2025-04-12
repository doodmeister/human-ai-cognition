
import boto3
import os

s3 = boto3.client('s3')
bucket_name = "humanai-document-store"  # Update if different

def upload_file(file_path):
    key = os.path.basename(file_path)
    s3.upload_file(file_path, bucket_name, key)
    print(f"Uploaded {file_path} to s3://{bucket_name}/{key}")

# Example usage
upload_file("test_files/sample_doc.txt")
upload_file("test_files/sample_audio.mp3")
