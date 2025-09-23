from fastapi import UploadFile
import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

client_s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("CREDENTIALS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("CREDENTIALS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)

REGION_NAME = region_name=os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

async def upload_wav_to_s3(file: UploadFile, file_bytes: bytes, uuid: str, path_name: str, debug) -> str | None:
    key = f"{path_name}/10/{datetime.now():%Y/%m/%d}/{uuid}"

    try:
        # S3에 업로드
        client_s3.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=file_bytes,
            ContentType=file.content_type
        )

        file_url = f"https://{BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{key}"
        return file_url
    except (BotoCoreError, ClientError) as e:
        if debug:
            print(f"S3 업로드 실패: {e}")
        return None