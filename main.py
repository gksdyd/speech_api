from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import PlainTextResponse, JSONResponse

from s3Upload import upload_wav_to_s3
from database import studyUsrInst , insert
from uuid import uuid4

from langTrans import trans_text

from path import save_file
import os

from pyannote_util import separate_user

app = FastAPI()

@app.post("/speechApi/")
async def upload_audio( lnrdStatusCd: int = Form(...), lnrdTypeCt: int = Form(...), lnrdTitle: str = Form(...), ifmmSeq: str = Form(...), file: UploadFile = File(...)):
    audio_bytes = await file.read()     # 바이트로 읽기

    tmp_path = save_file(audio_bytes)
    if tmp_path is None:
        return PlainTextResponse("파일 저장 실패", status_code=504)

    # S3 업로드
    uuid = str(uuid4()) + "." + file.content_type.split("/")[-1]
    file_url = await upload_wav_to_s3(file, audio_bytes, uuid)
    if file_url is None:
        return PlainTextResponse("S3 업로드 실패", status_code=501)

    db_result = await insert(file_url, file, uuid, len(audio_bytes))
    if db_result == 1:
        return PlainTextResponse("DB 저장 실패", status_code=503)

    studyUsrInstRt = await studyUsrInst(lnrdStatusCd , lnrdTypeCt , lnrdTitle , ifmmSeq)
    if studyUsrInstRt == 1:
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