from fastapi import FastAPI, UploadFile, File, Form

from s3Upload import upload_wav_to_s3
from database import save_db_process, insert_db_lnrd_recoding, update_db_lnrd_recoding_for_empty_contents
from uuid import uuid4

from path import save_file

from pyannote_util import separate_user

from debug import *

app = FastAPI()

@app.post("/speechApi/")
async def upload_audio( lnrd_status_cd: int = Form(...), lnrd_type_cd: int = Form(...), lnrd_title: str = Form(...), ifmm_seq: str = Form(...), file: UploadFile = File(...)):
    audio_bytes = await file.read()     # 바이트로 읽기

    tmp_path = save_file(audio_bytes)
    if tmp_path is None:
        print("파일 저장 실패")
        return -1

    lnrd_run_time = output_audio_len(tmp_path)
    size = output_file_size(tmp_path)

    foreign_key = insert_db_lnrd_recoding(lnrd_status_cd, lnrd_type_cd, lnrd_title, ifmm_seq, lnrd_run_time)

    # S3 업로드
    uuid = str(uuid4()) + "." + file.filename.split(".")[-1]
    file_url = await upload_wav_to_s3(file, audio_bytes, uuid)
    if file_url is None:
        print("S3 업로드 실패")
        os.remove(tmp_path)
        return -1

    result_seperate = await separate_user(tmp_path)
    if result_seperate == -1:
        update_db_lnrd_recoding_for_empty_contents(ifmm_seq, foreign_key)
        print("No Contents")
        os.remove(tmp_path)
        return -1

    os.remove(tmp_path)

    save_db_process(file_url, file, uuid, size, ifmm_seq, result_seperate, foreign_key)
    print("Success")
    return 0
