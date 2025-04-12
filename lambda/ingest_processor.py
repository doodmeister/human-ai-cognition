
import json
import boto3
import urllib.parse

s3 = boto3.client('s3')
textract = boto3.client('textract')
transcribe = boto3.client('transcribe')
comprehend = boto3.client('comprehend')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key    = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
    print(f"Processing file: s3://{bucket}/{key}")

    if key.endswith('.pdf') or key.endswith('.png') or key.endswith('.jpg'):
        response = textract.start_document_text_detection(DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}})
        print("Started Textract job:", response['JobId'])
    elif key.endswith('.mp3') or key.endswith('.wav') or key.endswith('.mp4'):
        job_name = key.replace('/', '_').replace('.', '_')
        file_uri = f"s3://{bucket}/{key}"
        response = transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_uri},
            MediaFormat=key.split('.')[-1],
            LanguageCode='en-US',
            OutputBucketName=bucket
        )
        print("Started Transcribe job:", response['TranscriptionJob']['TranscriptionJobName'])
    elif key.endswith('.txt'):
        content = s3.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
        response = comprehend.detect_sentiment(Text=content, LanguageCode='en')
        print("Comprehend sentiment:", response['Sentiment'])
    else:
        print("Unsupported file type.")
        
    return {"statusCode": 200, "body": json.dumps("Processing complete")}
