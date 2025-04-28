import json
import boto3
import urllib.parse
from typing import Dict, Any
from botocore.exceptions import ClientError

# Configuration
SUPPORTED_FORMATS = {
    'document': ('.pdf', '.png', '.jpg'),
    'audio': ('.mp3', '.wav'),
    'video': ('.mp4',),
    'text': ('.txt',)
}
MAX_FILE_SIZE = 5_242_880  # 5MB
DEFAULT_LANGUAGE = 'en-US'

class FileProcessor:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.textract = boto3.client('textract')
        self.transcribe = boto3.client('transcribe')
        self.comprehend = boto3.client('comprehend')

    def process_document(self, bucket: str, key: str) -> Dict[str, Any]:
        try:
            response = self.textract.start_document_text_detection(
                DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}}
            )
            return {'status': 'started', 'job_id': response['JobId']}
        except ClientError as e:
            return {'status': 'error', 'message': str(e)}

    def process_media(self, bucket: str, key: str) -> Dict[str, Any]:
        try:
            job_name = f"transcribe_{key.replace('/', '_').replace('.', '_')}"
            file_uri = f"s3://{bucket}/{key}"
            response = self.transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': file_uri},
                MediaFormat=key.split('.')[-1],
                LanguageCode=DEFAULT_LANGUAGE,
                OutputBucketName=bucket
            )
            return {'status': 'started', 'job_name': job_name}
        except ClientError as e:
            return {'status': 'error', 'message': str(e)}

    def process_text(self, bucket: str, key: str) -> Dict[str, Any]:
        try:
            content = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
            response = self.comprehend.detect_sentiment(
                Text=content[:5000],  # Comprehend has a 5000 byte limit
                LanguageCode=DEFAULT_LANGUAGE
            )
            return {'status': 'completed', 'sentiment': response['Sentiment']}
        except ClientError as e:
            return {'status': 'error', 'message': str(e)}

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for processing various file types
    """
    try:
        processor = FileProcessor()
        record = event['Records'][0]['s3']
        bucket = record['bucket']['name']
        key = urllib.parse.unquote_plus(record['object']['key'])
        
        print(f"Processing file: s3://{bucket}/{key}")
        
        # Check file size
        file_size = record['object'].get('size', 0)
        if file_size > MAX_FILE_SIZE:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'File too large'})
            }

        # Process based on file extension
        file_ext = key[key.rfind('.'):].lower()
        
        if file_ext in SUPPORTED_FORMATS['document']:
            result = processor.process_document(bucket, key)
        elif file_ext in SUPPORTED_FORMATS['audio'] + SUPPORTED_FORMATS['video']:
            result = processor.process_media(bucket, key)
        elif file_ext in SUPPORTED_FORMATS['text']:
            result = processor.process_text(bucket, key)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Unsupported file type'})
            }

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Processing complete',
                'file': key,
                'result': result
            })
        }

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
