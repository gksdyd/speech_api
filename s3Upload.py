from fastapi import UploadFile
import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime

client_s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("CREDENTIALS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("CREDENTIALS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

async def upload_wav_to_s3(file: UploadFile, file_bytes: bytes) -> bool:
    key = f"image/10/{datetime.now():%Y/%m/%d}/{file.filename}"

    try:
        # S3에 업로드
        client_s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=file_bytes,
            ContentType=file.content_type
        )
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"S3 업로드 실패: {e}")
        return False