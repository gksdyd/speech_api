from fastapi import FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse
import speech_recognition as sr
from io import BytesIO

from s3Upload import upload_wav_to_s3

app = FastAPI()

@app.post("/speechApi/")
async def upload_audio(file: UploadFile = File(...)):
    audio_bytes = await file.read()     # 바이트로 읽기
    audio_file = BytesIO(audio_bytes)   # 메모리에 저장

    recognizer = sr.Recognizer()        # STT 객체 생성

    try:
        with sr.AudioFile(audio_file) as source:    # 음성 읽기
            audio = recognizer.record(source)       # 음성 추출

        text = recognizer.recognize_google(audio, language="ko-KR")     # 한국어로 인식

        # S3 업로드
        success = await upload_wav_to_s3(file, audio_bytes)

        if not success:
            return PlainTextResponse("S3 업로드 실패", status_code=501)

        return PlainTextResponse(text)
    except sr.UnknownValueError:
        return PlainTextResponse("음성을 인식할 수 없습니다.", status_code=400)
    except sr.RequestError as e:
        return PlainTextResponse(f"Google STT 요청 실패: {e}", status_code=500)