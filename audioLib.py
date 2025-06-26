import speech_recognition as sr

recognizer = sr.Recognizer()        # STT 객체 생성

def audio_extract(path: str):
    try:
        with sr.AudioFile(path) as source:  # 음성 읽기
            audio = recognizer.record(source)  # 음성 추출
        result = recognizer.recognize_google(audio, language="ko-KR")  # 한국어로 인식
        print(result)
        return result
    except sr.UnknownValueError:
        return ["fail", None]
    except sr.RequestError as e:
        return ["fail", e]