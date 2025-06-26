from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
from audioLib import audio_extract
import os
from pathlib import Path

def separate_user(path: str):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HUGGING_FACE_KEY"))

    pipeline.to(torch.device("cpu"))

    diarization = pipeline(path, num_speakers=None)

    user = 0
    start = 0
    end = 0
    result = []
    audio = AudioSegment.from_file(path, format="wav")
    tracks = list(diarization.itertracks(yield_label=True))
    for idx, (turn, _, speaker) in enumerate(tracks):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        if idx == 0:
            start = turn.start
            end = turn.end
            user = int(speaker[-1])
        elif len(tracks) - 1 == idx:
            if user != int(speaker[-2:]):
                if end - start > 1:
                    segment = audio[start*1000:end*1000]
                    segment.export(f"speaker{user}.wav", format="wav")
                    file_path = Path.cwd() / f"speaker{user}.wav"
                    text = audio_extract(str(file_path))

                    if type(text) == list:
                        return text

                    result.append([user, text])
                    file_path.unlink()
                start = turn.start
                user = int(speaker[-1])
            end = turn.end

        if (user != int(speaker[-2:])) or (len(tracks) - 1 == idx):
            if end - start > 1:
                segment = audio[start*1000:end*1000]
                segment.export(f"speaker{user}.wav", format="wav")
                file_path = Path.cwd() / f"speaker{user}.wav"
                text = audio_extract(str(file_path))

                if type(text) == list:
                    return text

                result.append([user, text])
                file_path.unlink()
            start = turn.start
            end = turn.end
            user = int(speaker[-1])
        else:
            end = turn.end

    return result