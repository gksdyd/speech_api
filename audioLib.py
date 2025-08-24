import speech_recognition as sr
from pydub import AudioSegment
import math
import tempfile
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, resample_poly

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
    try:
        processed = processed.low_pass_filter(6000)  # 6kHz 이하만 통과
    except IndexError:
        return processed

    return processed

def _butter_band(x, sr, low_hz=None, high_hz=None, order=4):
    ny = 0.5 * sr
    if low_hz and high_hz:
        b, a = butter(order, [low_hz/ny, high_hz/ny], btype='band')
    elif low_hz:
        b, a = butter(order, low_hz/ny, btype='highpass')
    elif high_hz:
        b, a = butter(order, high_hz/ny, btype='lowpass')
    else:
        return x
    return filtfilt(b, a, x)

def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64)**2) + 1e-12))

def _peak(x):
    return float(np.max(np.abs(x.astype(np.float64))) + 1e-12)

def _pick_noise(y, sr, frac=0.15):
    """저에너지 구간을 노이즈로 가정해서 추출 (상위 모델 없이도 안전한 휴리스틱)"""
    if len(y) < sr:  # 1초 미만이면 초반부 사용
        return y[:max(1, int(0.5*sr))]
    # 짧은 프레임 RMS로 에너지 랭킹
    win = max(256, int(0.02*sr))  # 20ms
    hop = win // 2
    frames = []
    idxs = []
    for i in range(0, len(y)-win+1, hop):
        frame = y[i:i+win]
        frames.append(_rms(frame))
        idxs.append((i, i+win))
    frames = np.asarray(frames)
    k = max(1, int(len(frames) * frac))
    low_idx = np.argsort(frames)[:k]  # 저에너지 상위 k개
    # 이어붙여서 노이즈 샘플 구성
    parts = [y[s:e] for (s, e) in (idxs[j] for j in low_idx)]
    noise = np.concatenate(parts) if parts else y[:max(1, int(0.5*sr))]
    return noise

def clean_wav(path: str, target_sr=16000, target_dbfs=-20.0,
              band=(70, 7030), use_denoise=True) -> str:
    """
    모노/16k/S16, (선택)밴드 제한, 안전 노멀라이즈, 조건부 노이즈리덕션.
    - HPF~LPF는 완만하게: 70–7800Hz(권장). 상한은 필요 없으면 None.
    - 노멀라이즈는 RMS 타겟 + 피크 -1dBFS 클램프.
    - NR은 저에너지 구간에서 노이즈 추정.
    """
    wav, sr = sf.read(path, dtype='float32', always_2d=True)  # (N, ch)
    # 스테레오→모노 (RMS 가중 평균: 좌우 상쇄 최소화)
    if wav.shape[1] == 1:
        x = wav[:, 0]
    else:
        # 채널별 RMS로 가중치
        ch_rms = np.array([_rms(wav[:, c]) for c in range(wav.shape[1])])
        w = ch_rms / (np.sum(ch_rms) + 1e-12)
        x = np.sum(wav * w[None, :], axis=1).astype(np.float32)

    # resample -> 16k (polyphase, 정수비 안전화)
    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        x = resample_poly(x, int(target_sr//g), int(sr//g)).astype(np.float32)
        sr = target_sr

    # band (zero-phase로 위상 왜곡 방지)
    if band is not None:
        low, high = band
        if low is None and high is None:
            pass
        else:
            x = _butter_band(x, sr, low_hz=low, high_hz=high)

    # RMS normalize to target_dbfs (과도한 boost 제한)
    rms = _rms(x)
    cur_dbfs = 20*np.log10(rms + 1e-12)
    gain_db = float(target_dbfs - cur_dbfs)
    # 너무 큰 이득은 제한 (예: +15 dB 이상이면 +15로 클램프)
    gain_db = np.clip(gain_db, -30.0, 15.0)
    x = (x * (10**(gain_db/20))).astype(np.float32)

    # 피크 세이프티: -1 dBFS 이내로 소프트 클램프
    peak = _peak(x)
    peak_dbfs = 20*np.log10(peak + 1e-12)
    headroom_db = -1.0  # -1 dBFS
    if peak_dbfs > headroom_db:
        scale = 10**((headroom_db - peak_dbfs)/20)
        x = (x * scale).astype(np.float32)

    # 조건부 노이즈 리덕션
    if use_denoise:
        try:
            import noisereduce as nr
            noise = _pick_noise(x, sr, frac=0.15)
            # SNR 추정이 너무 나쁘지 않다면 살짝만 줄이거나 skip
            # 여기서는 기본 파라미터로, 과한 노이즈 추정 방지 위해 stationary=False
            x = nr.reduce_noise(y=x, sr=sr, y_noise=noise, stationary=False, prop_decrease=0.9, n_fft=1024, hop_length=256)
        except Exception:
            pass

    # 최종 피크 재확인 (NR 후 레벨 튈 수 있어 -1 dBFS 재보정)
    peak = _peak(x)
    peak_dbfs = 20*np.log10(peak + 1e-12)
    if peak_dbfs > -1.0:
        scale = 10**((-1.0 - peak_dbfs)/20)
        x = (x * scale).astype(np.float32)

    tmp = tempfile.mkstemp(suffix=".wav")[1]
    # float32 데이터를 PCM_16로 저장 → soundfile이 내부에서 양자화함
    sf.write(tmp, x, sr, subtype='PCM_16')
    return tmp
