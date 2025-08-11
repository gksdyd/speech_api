from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse

from s3Upload import upload_wav_to_s3
from database import save_db_process
from uuid import uuid4

from path import save_file
import os

from pyannote_util import separate_user

app = FastAPI()

@app.post("/speechApi/")
async def upload_audio( lnrd_status_cd: int = Form(...), lnrd_type_ct: int = Form(...), lnrd_title: str = Form(...), ifmm_seq: str = Form(...), file: UploadFile = File(...)):
    audio_bytes = await file.read()     # 바이트로 읽기

    audio_size = len(audio_bytes) / (1024 * 1024)
    print(f"파일 크기: {audio_size:.2f} MB")

    tmp_path = save_file(audio_bytes)
    if tmp_path is None:
        return PlainTextResponse("파일 저장 실패", status_code=504)

    # S3 업로드
    uuid = str(uuid4()) + "." + file.content_type.split("/")[-1]
    file_url = await upload_wav_to_s3(file, audio_bytes, uuid)
    if file_url is None:
        return PlainTextResponse("S3 업로드 실패", status_code=501)

    result_seperate = await separate_user(tmp_path)
    os.remove(tmp_path)

    save_db_process(file_url, file, uuid, len(audio_bytes), lnrd_status_cd , lnrd_type_ct , lnrd_title , ifmm_seq, result_seperate)
    return None