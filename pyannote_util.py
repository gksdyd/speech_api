from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
from audioLib import audio_extract, preprocess_segment
import os
from dotenv import load_dotenv
from langTrans import trans_text
from database import script_usr_inst
from debug import output_file_size, output_audio_len

load_dotenv()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HUGGING_FACE_KEY")
)

pipeline.to(torch.device("cpu"))

async def separate_user(path: str, lnrd_seq: str):
    diarization = pipeline(path)

    tracks = list(diarization.itertracks(yield_label=True))

    # user = 0
    # start = 0
    # end = 0
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
            continue

        lnsc_contents_eng = await trans_text(lnsc_contents)
        if lnsc_contents_eng == -1:
            print("failed to trans text")
            continue

        result_contents.append(lnsc_speaker_cd)
        result_contents.append(lnsc_contents)
        result_contents.append(lnsc_contents_eng)
        result_seperate.append(result_contents)
        await script_usr_inst(lnsc_contents, lnsc_contents_eng, lnsc_speaker_cd, lnrd_seq)

    output_file_size(path)
    output_audio_len(path)

    print(str(tracks) + "@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    for contents in result_seperate:
        print(f"{contents[0]} : {contents[1]} / {contents[2]}")

    return 0


    #     # print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    #     if idx == 0:        # 초기화
    #         start = turn.start
    #         end = turn.end
    #         user = int(speaker[-1])
    #     elif len(tracks) - 1 == idx:    # 마지막 화자 분리 부분 (마지막 화자 분리는 비교할 다음 대상이 없기에 여기서 음성 추출 및 변환을 할지 판단)
    #         if user != int(speaker[-2:]):   # 화자가 다를 경우, 음성 추출 및 변환 진행 / 화자가 같을 경우, end 부분 초기화
    #             # if end - start > 1:
    #
    #             # 1. 오리지널 오디오에서 해당 구간 자르기
    #             segment = AudioSegment.from_file(path, format="wav")[start * 1000:end * 1000]
    #
    #             # 2. segment 전처리 적용
    #             segment = preprocess_segment(segment)
    #
    #             # 3. 전처리된 segment 저장
    #             filename = f"speaker{user}.wav"
    #             segment.export(filename, format="wav")
    #
    #             text = audio_extract(filename)
    #             os.remove(filename)
    #
    #             if text == -1:
    #                 print("failed to extract text")
    #                 return -1
    #             print(text)
    #
    #             temp = await trans_text(text)
    #             if temp == -1:
    #                 print("failed to trans text")
    #                 return -1
    #             print(temp)
    #
    #             await script_usr_inst(text, temp, user, lnrd_seq)
    #             # if study_usr_inst_rt == -1:
    #             #     return PlainTextResponse("DB 저장 실패", status_code=503)
    #
    #             result.append([user, text])
    #             start = turn.start
    #             user = int(speaker[-1])
    #         end = turn.end
    #
    #     if (user != int(speaker[-2:])) or (len(tracks) - 1 == idx):     # 화자가 다를 경우, 음성 추출 및 변환 진행 / 화자가 같을 경우, end 부분 초기화
    #         # if end - start > 1:
    #
    #         # 1. 오리지널 오디오에서 해당 구간 자르기
    #         segment = AudioSegment.from_file(path, format="wav")[start * 1000:end * 1000]
    #
    #         # 2. segment 전처리 적용
    #         segment = preprocess_segment(segment)
    #
    #         # 3. 전처리된 segment 저장
    #         filename = f"speaker{user}.wav"
    #         segment.export(filename, format="wav")
    #
    #         text = audio_extract(filename)
    #         os.remove(filename)
    #
    #         if text == -1:
    #             print("failed to extract text")
    #             return -1
    #         print(text)
    #
    #         temp = await trans_text(text)
    #         if temp == -1:
    #             print("failed to trans text")
    #             return -1
    #         print(temp)
    #
    #         await script_usr_inst(text, temp, user, lnrd_seq)
    #
    #         result.append([user, text])
    #         start = turn.start
    #         user = int(speaker[-1])
    #     end = turn.end
    #
    # return result