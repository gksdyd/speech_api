import tempfile

def save_file(audio_bytes: bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            return tmp.name
    except Exception as e:
        print(f"파일 저장 오류: {e}")
        return None