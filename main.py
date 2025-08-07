from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse, JSONResponse

from s3Upload import upload_wav_to_s3
from database import study_usr_inst , insert
from uuid import uuid4

from langTrans import trans_text

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

    ## S3 업로드
    uuid = str(uuid4()) + "." + file.content_type.split("/")[-1]
    file_url = await upload_wav_to_s3(file, audio_bytes, uuid)
    if file_url is None:
        return PlainTextResponse("S3 업로드 실패", status_code=501)

    db_result = await insert(file_url, file, uuid, len(audio_bytes))
    if db_result == 1:
        return PlainTextResponse("DB 저장 실패", status_code=503)

    study_usr_inst_rt = await study_usr_inst(lnrd_status_cd , lnrd_type_ct , lnrd_title , ifmm_seq)
    if study_usr_inst_rt == 1:
        return PlainTextResponse("DB 저장 실패", status_code=503)

    separate_text = separate_user(tmp_path)
    if type(separate_text[0]) == str:
        if separate_text[1] is None:
            return PlainTextResponse("음성을 인식할 수 없습니다.", status_code=400)
        else:
            return PlainTextResponse(f"Google STT 요청 실패: {separate_text[1]}", status_code=500)
    os.remove(tmp_path)

    text_list = await trans_text(separate_text)
    if len(text_list) == 0:
        return PlainTextResponse("번역 실패", status_code=502)

    return JSONResponse(content=text_list)