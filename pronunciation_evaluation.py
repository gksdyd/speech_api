import math, re, numpy as np
from pydub import AudioSegment, silence
import parselmouth
from dataclasses import dataclass

# ---------- 유틸: SNR, Pause, Syllable ----------
def estimate_snr_db(audio_seg: AudioSegment, frame_ms=25):
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    if audio_seg.channels > 1:
        samples = samples.reshape((-1, audio_seg.channels)).mean(axis=1)
    # 정규화
    samples /= (1 << (8 * audio_seg.sample_width - 1))
    n = max(1, int(audio_seg.frame_rate * frame_ms / 1000))
    if len(samples) < n * 5:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(samples, n)[::n]
    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    noise = np.percentile(rms, 10)
    signal = np.percentile(rms, 90)
    if noise <= 0:
        return None
    return 20 * math.log10(signal / noise)

HANGUL_RX = re.compile(r"[가-힣]")

def count_korean_syllables(text: str) -> int:
    return len(HANGUL_RX.findall(text or ""))

def pause_metrics(audio_seg: AudioSegment, silence_db_drop=16, min_sil_ms=180):
    sils = silence.detect_silence(
        audio_seg,
        min_silence_len=min_sil_ms,
        silence_thresh=audio_seg.dBFS - silence_db_drop,
    )  # [[s,e]...]
    dur = audio_seg.duration_seconds
    if dur <= 0:
        return 0, 0.0, 0.0
    total_sil = sum(e - s for s, e in sils) / 1000.0
    pause_ratio = total_sil / dur
    mean_pause = (total_sil / len(sils)) if sils else 0.0
    return len(sils), pause_ratio, mean_pause

# ---------- 음높이/억양 ----------
@dataclass
class PitchStats:
    f0_median: float
    f0_p05: float
    f0_p95: float
    f0_range: float
    f0_sd: float
    pdq: float
    voiced_ratio: float
    slope_hz_per_s: float

def pitch_stats(wav_path: str, fmin=75, fmax=500) -> PitchStats:
    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch_cc(pitch_floor=fmin, pitch_ceiling=fmax)
    # time step으로 F0 궤적 가져오기
    xs = pitch.xs()
    f0 = pitch.selected_array['frequency']  # 0은 무성
    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]
    if len(voiced) < 5:
        # 유성이 너무 적으면 억양 판단 곤란
        return PitchStats(0,0,0,0,0,0,0,0)

    f0_med = float(np.median(voiced))
    f0_p05 = float(np.percentile(voiced, 5))
    f0_p95 = float(np.percentile(voiced, 95))
    f0_rng = f0_p95 - f0_p05
    f0_sd  = float(np.std(voiced))
    pdq = (f0_p95 - f0_p05) / (f0_med + 1e-9)  # Pitch Dynamism Quotient
    voiced_ratio = float(voiced_mask.mean())

    # F0 기울기(전반 상승/하강 경향)
    t_voiced = xs[voiced_mask]
    if len(t_voiced) >= 2:
        A = np.vstack([t_voiced, np.ones_like(t_voiced)]).T
        slope, _ = np.linalg.lstsq(A, voiced, rcond=None)[0]  # Hz per second
    else:
        slope = 0.0

    return PitchStats(f0_med, f0_p05, f0_p95, f0_rng, float(f0_sd), float(pdq), voiced_ratio, float(slope))

# ---------- 발성 안정성 (Jitter/Shimmer) ----------
def voice_stability(wav_path: str):
    try:
        snd = parselmouth.Sound(wav_path)
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = parselmouth.praat.call([snd, pp], "Get jitter (local)", 0,0,75,500,1.3,1.6)  # fraction
        shimmer_local = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0,0,75,500,1.3,1.6,0.03,0.45)  # fraction
        return float(jitter_local * 100), float(shimmer_local * 100)  # %
    except Exception:
        return None, None

# ---------- 메인: 발음/토네이션 평가 ----------
def evaluate_pronunciation_and_intonation(wav_path: str, recognized_text: str | None = None):
    seg = AudioSegment.from_file(wav_path)
    dur = seg.duration_seconds

    # 1) 발음/명료도 측면
    snr_db = estimate_snr_db(seg)                    # 20dB↑ 양호
    pauses, pause_ratio, mean_pause = pause_metrics(seg)
    syllables = count_korean_syllables(recognized_text or "")
    speaking_rate = (syllables / dur) if dur > 0 else 0.0
    articulation_rate = (syllables / (dur * (1 - pause_ratio))) if (dur > 0 and pause_ratio < 0.95) else 0.0
    jitter_pct, shimmer_pct = voice_stability(wav_path)

    # 2) 억양/프로소디 측면
    p = pitch_stats(wav_path)
    # 해석 기준(경험칙, 한국어 일반 대화 가이드라인 근사)
    # - speaking_rate: 3.0~5.0 음절/s 자연
    # - pause_ratio: 0.15~0.40 자연
    # - jitter: <1% 좋음, 1~2% 보통, >2% 불안정
    # - shimmer: <3% 좋음, 3~6% 보통, >6% 불안정
    # - PDQ: 0.2~0.6 자연스러운 억양, 너무 낮으면 단조, 너무 높으면 과장/요동

    # 3) 정규화해서 점수화(0~100)
    def clamp01(x): return max(0.0, min(1.0, x))
    # 발음 점수 구성: SNR(40%), pause(15%), articulation(15%), jitter(15%), shimmer(15%)
    snr_score = clamp01(( (snr_db or 0) - 10) / (30 - 10))   # 10dB=0, 30dB=1
    pause_score = 1.0 - abs((pause_ratio or 0) - 0.27) / 0.27  # 0.27 근방이 best
    pause_score = clamp01(pause_score)
    ar_score = clamp01(((articulation_rate or 0) - 2.5) / (5.5 - 2.5))  # 2.5~5.5 good
    jit = (jitter_pct if jitter_pct is not None else 3.0)
    shi = (shimmer_pct if shimmer_pct is not None else 6.0)
    jitter_score = clamp01((2.0 - min(jit, 2.0)) / 2.0)      # 0~2% 범위
    shimmer_score = clamp01((6.0 - min(shi, 6.0)) / 6.0)     # 0~6% 범위

    pronunciation_score = 100 * (
        0.40*snr_score + 0.15*pause_score + 0.15*ar_score + 0.15*jitter_score + 0.15*shimmer_score
    )

    # 억양 점수 구성: PDQ(50%), F0 range(25%), F0 SD(15%), voiced_ratio(10%)
    pdq = p.pdq or 0.0
    # PDQ 0.2~0.6을 자연 구간으로 맵핑
    if pdq <= 0:
        pdq_score = 0.0
    elif pdq < 0.2:
        pdq_score = pdq / 0.2 * 0.6   # 너무 단조 → 0~0.6
    elif pdq <= 0.6:
        pdq_score = 0.6 + (pdq - 0.2) / 0.4 * 0.4  # 0.6~1.0
    else:
        pdq_score = max(0.0, 1.0 - (pdq - 0.6))    # 과도한 요동 감점

    # F0 range: 60~300 Hz를 대화 억양 유효 범위로 근사
    f0_range_score = clamp01((min(max(p.f0_range, 60), 300) - 60) / (300 - 60))
    f0_sd_score = clamp01(min(p.f0_sd, 80) / 80)  # 변동성
    voiced_score = clamp01((p.voiced_ratio - 0.5) / (0.9 - 0.5))  # 0.5~0.9 권장

    intonation_score = 100 * (
        0.50*pdq_score + 0.25*f0_range_score + 0.15*f0_sd_score + 0.10*voiced_score
    )

    return {
        "duration_sec": round(dur, 2),
        "recognized_text": recognized_text,
        # 발음/명료도
        "pronunciation": {
            "score_0_100": round(pronunciation_score, 1),
            "snr_db": None if snr_db is None else round(snr_db, 1),
            "pause_count": pauses,
            "pause_ratio": round(pause_ratio, 3),
            "mean_pause_sec": round(mean_pause, 2),
            "speaking_rate_syll_per_sec": round(syllables / dur, 2) if dur > 0 else 0.0,
            "articulation_rate_syll_per_sec": round(articulation_rate, 2),
            "jitter_percent": None if jitter_pct is None else round(jitter_pct, 2),
            "shimmer_percent": None if shimmer_pct is None else round(shimmer_pct, 2),
        },
        # 토네이션/억양
        "intonation": {
            "score_0_100": round(intonation_score, 1),
            "f0_median_hz": round(p.f0_median, 1),
            "f0_range_hz": round(p.f0_range, 1),
            "f0_sd_hz": round(p.f0_sd, 1),
            "pdq": round(p.pdq, 3),
            "voiced_ratio": round(p.voiced_ratio, 3),
            "f0_slope_hz_per_s": round(p.slope_hz_per_s, 2),
        }
    }