from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
from audioLib import audio_extract, preprocess_segment
import os
from dotenv import load_dotenv
from langTrans import trans_text

load_dotenv()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HUGGING_FACE_KEY")
)

pipeline.to(torch.device("cpu"))

async def separate_user(path: str):
    diarization = pipeline(path)

    tracks = list(diarization.itertracks(yield_label=True))

    result_seperate = []

    for idx, (turn, _, speaker) in enumerate(tracks):
        result_contents = []

        start = turn.start
        end = turn.end
        lnsc_speaker_cd = int(speaker[-1])

        # 1. 오리지널 오디오에서 해당 구간 자르기
        segment = AudioSegment.from_file(path, format="wav")[start * 1000:end * 1000]

        # 2. segment 전처리 적용
        segment = preprocess_segment(segment)

        # 3. 전처리된 segment 저장
        filename = f"speaker{lnsc_speaker_cd}.wav"
        segment.export(filename, format="wav")

        lnsc_contents = audio_extract(filename)
        os.remove(filename)

        if lnsc_contents == -1:
            print("failed to extract text")

        lnsc_contents_eng = await trans_text(lnsc_contents)
        if lnsc_contents_eng == -1:
            print("failed to trans text")
            continue

        result_contents.append(lnsc_speaker_cd)
        result_contents.append(lnsc_contents)
        result_contents.append(lnsc_contents_eng)
        result_seperate.append(result_contents)

    print(str(tracks) + "@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    if len(result_seperate) == 0:
        print(f"음성 추출 실패!!")
        return -1

    for contents in result_seperate:
        print(f"{contents[0]} : {contents[1]} / {contents[2]}")

    return result_seperate