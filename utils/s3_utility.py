import boto3
import os

def upload_file_to_s3(local_path, bucket_name, s3_key):
    """
    Upload a local file to S3 bucket.

    Args:
        local_path (str): Path to the file on local filesystem.
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): Destination path inside the S3 bucket.
    """
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"✅ Successfully uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")
