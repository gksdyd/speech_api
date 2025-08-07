import speech_recognition as sr
from pydub import AudioSegment

recognizer = sr.Recognizer()        # STT 객체 생성

def audio_extract(path: str):
    # sound = AudioSegment.from_file(path)
    # print("길이 (초):", len(sound) / 1000)
    # print("평균 음량 (dBFS):", sound.dBFS)

    try:
        with sr.AudioFile(path) as source:  # 음성 읽기
            audio = recognizer.record(source)  # 음성 추출

        result = recognizer.recognize_google(audio, language="ko-KR")  # 한국어로 인식
        return result
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return -1
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return -1

def preprocess_segment(segment: AudioSegment) -> AudioSegment:
    """
    Pydub AudioSegment를 받아서 전처리 (채널, 샘플레이트, 비트폭, 필터링 등) 후 반환
    """
    processed = segment.set_channels(1)        # mono
    processed = processed.set_frame_rate(16000)  # 16kHz
    processed = processed.set_sample_width(2)    # 16-bit (2 bytes)
    processed = processed.low_pass_filter(6000)  # 6kHz 이하만 통과

    return processed