import math, numpy as np
from pydub import AudioSegment, silence
import parselmouth
from dataclasses import dataclass
from pronouncing_pyphen import count_english_syllables_in_text
import jiwer

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

# ---------- (accuracy/completeness) ----------
def acc_comp_scores(reference_text: str, recognized_text: str):
    out = jiwer.process_words(reference_text or "", recognized_text or "")
    N = out.hits + out.substitutions + out.deletions  # ref 단어 수

    # 특수 케이스 처리
    hyp_has_text = bool((recognized_text or "").strip())
    if N == 0:
        # 참조와 가설 모두 비었으면 완전 일치로 간주
        if not hyp_has_text:
            return 100.0, 100.0
        # 참조는 없는데 말한 건 있음 → 정확/완전성 0으로
        return 0.0, 0.0

    accuracy = round(100 * (out.hits / N), 1)
    completeness = round(100 * (1 - (out.deletions / N)), 1)
    return accuracy, completeness

# ---------- 메인: 발음/토네이션 평가 ----------
def evaluate_pronunciation_and_intonation(wav_path: str, recognized_text: str, reference_text: str):
    seg = AudioSegment.from_file(wav_path)
    dur = seg.duration_seconds

    # 1) 발음/명료도 측면
    snr_db = estimate_snr_db(seg)                    # 20dB↑ 양호
    pauses, pause_ratio, mean_pause = pause_metrics(seg)
    syllables = count_english_syllables_in_text(recognized_text or "") if recognized_text else 0
    speaking_rate = (syllables / dur) if dur > 0 else 0.0
    articulation_rate = (syllables / (dur * (1 - pause_ratio))) if (dur > 0 and pause_ratio < 0.95) else 0.0
    jitter_pct, shimmer_pct = voice_stability(wav_path)

    # 2) 억양/프로소디 측면
    p = pitch_stats(wav_path)
    # 해석 기준(경험칙, 영어 일반 대화 가이드라인 근사)
    # - speaking_rate: 4.5~6.5 음절/s 자연
    # - pause_ratio: 0.20~0.40 자연
    # - jitter: <1% 좋음, 1~2% 보통, >2% 불안정
    # - shimmer: <3% 좋음, 3~6% 보통, >6% 불안정
    # - PDQ: 0.2~0.6 자연스러운 억양, 너무 낮으면 단조, 너무 높으면 과장/요동

    def hz_to_st(hz, ref):
        # 반음 = 12 * log2(hz/ref)
        if hz is None or hz <= 0 or ref is None or ref <= 0:
            return 0.0
        return 12.0 * np.log2(hz / ref)

    # 성별 독립적 평가는 '중앙값 대비' 반음으로
    f0_median = max(p.f0_median or 0.0, 1e-6)
    f0_range_st = hz_to_st((p.f0_median + (p.f0_range or 0.0)), f0_median) - hz_to_st(p.f0_median, f0_median)
    f0_sd_st = hz_to_st((p.f0_median + (p.f0_sd or 0.0)), f0_median) - hz_to_st(p.f0_median, f0_median)

    # 3) 정규화해서 점수화(0~100)
    def clamp01(x):
        return max(0.0, min(1.0, x))

    # 발음 점수 구성: SNR(40%), pause(15%), articulation(15%), jitter(15%), shimmer(15%)
    snr_score = clamp01(( (snr_db or 0) - 10) / (30 - 10))   # 10dB=0, 30dB=1
    pause_score = clamp01(1.0 - abs((pause_ratio or 0) - 0.30) / 0.30)
    ar_score = clamp01(((articulation_rate or 0) - 4.5) / (7.5 - 4.5))  # 4.5~7.5 good
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

    # F0 Range (semitones): 4~12 st 자연, <3 단조, >14 과장
    if f0_range_st <= 0:
        f0_range_score = 0.0
    elif f0_range_st < 4.0:
        f0_range_score = clamp01(f0_range_st / 4.0 * 0.6)  # 0~0.6
    elif f0_range_st <= 12.0:
        f0_range_score = 0.6 + (f0_range_st - 4.0) / (12.0 - 4.0) * 0.4  # 0.6~1.0
    else:
        # 12~14까지는 유지, 14↑는 과장 감점
        over = max(0.0, f0_range_st - 14.0)
        f0_range_score = clamp01(1.0 - over / 6.0)  # 20st 넘어가면 0 근접

    # F0 SD (semitones): 1.0~3.0 st 권장
    if f0_sd_st <= 0:
        f0_sd_score = 0.0
    elif f0_sd_st < 1.0:
        f0_sd_score = clamp01(f0_sd_st / 1.0 * 0.6)
    elif f0_sd_st <= 3.0:
        f0_sd_score = 0.6 + (f0_sd_st - 1.0) / (3.0 - 1.0) * 0.4
    else:
        f0_sd_score = clamp01(1.0 - (f0_sd_st - 3.0) / 5.0)

    # Voiced ratio: 0.55~0.90 권장
    voiced_score = clamp01(((p.voiced_ratio or 0.0) - 0.55) / (0.90 - 0.55))

    intonation_score = 100 * (
        0.50*pdq_score + 0.25*f0_range_score + 0.15*f0_sd_score + 0.10*voiced_score
    )

    # (1) Fluency 점수 (네 지표로 계산)
    def fluency_score_fn(pause_ratio, mean_pause, speaking_rate, articulation_rate):
        # 영어 대화 근사: pause 중심 0.30, speaking 4.5~6.5, articulation 4.5~7.5, mean_pause 중심 0.4s
        pause_s = clamp01(1 - abs((pause_ratio or 0) - 0.30) / 0.30)
        spk_s = clamp01(((speaking_rate or 0) - 4.5) / (6.5 - 4.5))
        art_s = clamp01(((articulation_rate or 0) - 4.5) / (7.5 - 4.5))
        mp_s = clamp01(1 - abs((mean_pause or 0) - 0.40) / 0.40)
        return round(100 * (0.45 * pause_s + 0.30 * spk_s + 0.15 * art_s + 0.10 * mp_s), 1)

    fluency = fluency_score_fn(pause_ratio, mean_pause, speaking_rate, articulation_rate)

    # (2) Accuracy / Completeness
    accuracy, completeness = acc_comp_scores(reference_text, recognized_text)

    # (3) Azure 스타일 종합 PronScore (옵션)
    parts = []
    parts.append(round(intonation_score, 1))
    parts.append(fluency)
    parts.append(accuracy)
    parts.append(completeness)

    s = sorted(parts)
    pronscore = round((0.4 * s[0] + 0.2 * s[1] + 0.2 * s[2] + 0.2 * s[3]), 1)

    return {
        "duration_sec": round(dur, 2),
        "recognized_text": recognized_text,
        "pronunciation": {
            "score_0_100": round(pronunciation_score, 1),
            "snr_db": None if snr_db is None else round(snr_db, 1),
            "pause_count": pauses,
            "pause_ratio": round(pause_ratio or 0.0, 3),
            "mean_pause_sec": round(mean_pause or 0.0, 2),
            "speaking_rate_syll_per_sec": round(speaking_rate, 2),
            "articulation_rate_syll_per_sec": round(articulation_rate or 0.0, 2),
            "jitter_percent": None if jitter_pct is None else round(jitter_pct, 2),
            "shimmer_percent": None if shimmer_pct is None else round(shimmer_pct, 2),
        },
        "intonation": {
            "score_0_100": round(intonation_score, 1),
            "f0_median_hz": round(p.f0_median or 0.0, 1),
            "f0_range_hz": round(p.f0_range or 0.0, 1),
            "f0_sd_hz": round(p.f0_sd or 0.0, 1),
            "f0_range_semitones": round(f0_range_st, 2),
            "f0_sd_semitones": round(f0_sd_st, 2),
            "pdq": round(p.pdq or 0.0, 3),
            "voiced_ratio": round(p.voiced_ratio or 0.0, 3),
            "f0_slope_hz_per_s": round(p.slope_hz_per_s or 0.0, 2),
        },
        "assessment": {
            "accuracy_0_100": accuracy,            # ← 추가
            "fluency_0_100":  fluency,             # ← 추가
            "completeness_0_100": completeness,    # (참조문장 있을 때만)
            "prosody_0_100": round(intonation_score, 1),
            "pronscore_0_100": pronscore
        }
    }

def text_pronunciation(recognized_text: str, reference_text: str):
    accuracy, completeness = acc_comp_scores(reference_text, recognized_text)

    # (3) Azure 스타일 종합 PronScore (옵션)
    parts = []
    parts.append(accuracy)
    parts.append(completeness)

    s = sorted(parts)
    pronscore = round((0.6 * s[0] + 0.4 * s[1]), 1)

    return {
        "recognized_text": recognized_text,
        "assessment": {
            "accuracy_0_100": accuracy,            # ← 추가
            "completeness_0_100": completeness,    # (참조문장 있을 때만)
            "pronscore_0_100": pronscore
        }
    }