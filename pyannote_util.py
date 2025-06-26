from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
from audioLib import audio_extract
import os

def separate_user(path: str):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HUGGING_FACE_KEY"))

    pipeline.to(torch.device("cpu"))

    diarization = pipeline(path, num_speakers=None)

    user = 0
    start = 0
    result = []
    audio = AudioSegment.from_file(path, format="wav")
    tracks = list(diarization.itertracks(yield_label=True))
    for idx, (turn, _, speaker) in enumerate(tracks):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        end = turn.end
        if (user != int(speaker[-1])) or (len(tracks) - 1 == idx):
            if end - start > 1:
                segment = audio[start*1000:end*1000]
                segment.export(f"speaker{user}.wav", format="wav")
                text = audio_extract(os.getcwd() + f"\speaker{user}.wav")

                if type(text) == list:
                    return text

                result.append([user, text])
                os.remove(os.getcwd() + f"\speaker{user}.wav")
            start = turn.start
            user = int(speaker[-1])

    return result