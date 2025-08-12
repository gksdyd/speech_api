import os
from pydub import AudioSegment

def output_file_size(path: str):
    file_size_bytes = os.path.getsize(path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"파일 크기: {file_size_mb:.2f} MB")

    return file_size_bytes

def output_audio_len(path: str):
    audio = AudioSegment.from_file(path, format="wav")
    duration_ms = len(audio)
    minutes = int(duration_ms / 1000 // 60)
    seconds = int(duration_ms / 1000 % 60)

    print(f"총 녹음 시간: ({minutes}분 {seconds}초)")

    return duration_ms / 1000