from pyannote.audio import Pipeline, Inference
import torch
from pydub import AudioSegment
from audioLib import audio_extract, preprocess_segment, clean_wav
import os
from dotenv import load_dotenv
from langTrans import trans_text, add_punctuation
import numpy as np
from pyannote.core import Segment
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from typing import List
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from voice_gender_classifier import ECAPA_gender

load_dotenv()

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HUGGING_FACE_KEY")
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.to(device).eval()

pipeline.segmentation.onset = 0.60        # 발화 시작 민감도 낮춤
pipeline.segmentation.offset = 0.62       # 발화 종료 민감도 낮춤
pipeline.segmentation.min_duration_on = 0.25   # 최소 발화 길이(초)
pipeline.segmentation.min_duration_off = 0.18  # 발화 사이 최소 간격

async def separate_user(path: str):
    # 0) 전처리된 파일 생성
    path = clean_wav(path, use_denoise=True)

    # 1) diarization
    diarization = pipeline(path, max_speakers=5)
    tracks = list(diarization.itertracks(yield_label=True))
    tracks.sort(key=lambda t: t[0].start)
    segments_all = [turn for (turn, _, _) in tracks]

    if len(segments_all) == 0:
        print(f"데이터 존재하지 않음")
        return -1

    # 2) 겹침 정리(전부 버리지 말고 겹침만 제거)
    segments = merge_segments_overlap_only(segments_all)

    med_dur = np.median([s.end - s.start for s in segments]) if segments else 1.2
    dur = float(np.clip(1.2 * med_dur, 0.8, 1.5))  # 세그 길이에 맞춰 0.8~1.5s로 조정
    step = dur / 2

    local_embedder = Inference(
        "pyannote/embedding",
        window="sliding",
        duration=dur,
        step=step,
        use_auth_token=os.getenv("HUGGING_FACE_KEY"),
        device=device,
    )

    # 3) 임베딩 추출 (파일당 동일 임베더 & 동일 샘플레이트)
    E = local_embedder({"audio": path})
    X = np.vstack([embed_segment(E, seg) for seg in segments])
    # L2 → (조건부)PCA-화이트닝 → L2
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # 화이트닝은 샘플 충분할 때만
    if X.shape[0] >= 20 and X.shape[0] > 2:  # n>=20 권장
        Xw = pca_whiten(X, var_keep=0.95)
        Xw = np.nan_to_num(Xw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if Xw.ndim == 2 and Xw.shape[1] >= 2:
            X = Xw
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # 4) 거리행렬 + K 자동화 (threshold sweep)
    D = cosine_distances(X)
    n = D.shape[0]
    tri = D[np.triu_indices_from(D, k=1)] if n >= 2 else np.array([0.0])
    p25, p50, p85 = np.percentile(tri, [25, 50, 85])

    if n >= 4:
        grid = np.linspace(p25, p85, 13)  # 예: [0.56, ..., 0.86]
        raw_labels, sil, chosen_th = best_labels_by_threshold(D, X, thresholds=grid)
        # 스윕 실패 → K=2..min(5,n-1) 탐색
        if raw_labels is None:
            if tri.size == 0 or (np.median(tri) < 0.55 and tri.max() <= 0.65):
                raw_labels = np.zeros(n, dtype=int)
                sil, chosen_th = -1.0, None
            else:
                cand = []
                for k_fixed in range(2, min(5, n - 1) + 1):
                    lab = AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=k_fixed).fit_predict(D)
                    if _valid_for_sil(lab):
                        s = silhouette_score(X, lab, metric="cosine")
                    else:
                        # 보조 점수: inter/intra 대략치 (간단 버전)
                        s = - (np.mean([D[lab == u][:, lab == u][np.triu_indices(np.sum(lab == u), 1)].mean()
                                        for u in np.unique(lab) if np.sum(lab == u) >= 2])
                               - tri.mean())
                    cand.append((lab, s, f"fixed-{k_fixed}"))
                # 점수 최대 채택
                raw_labels, sil, chosen_th = max(cand, key=lambda t: t[1]) if cand else (np.zeros(n, int), -1.0, None)

    elif n == 3:
        if tri.size == 0 or (np.median(tri) < 0.55 and tri.max() <= 0.65):
            raw_labels = np.zeros(n, dtype=int)
            sil, chosen_th = -1.0, None
        elif np.min(tri) > 0.78:
            raw_labels = np.arange(n, dtype=int)  # [0,1,2]
            sil, chosen_th = -1.0, None
        else:
            ac = AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=2)
            raw_labels = ac.fit_predict(D)
            sil = silhouette_score(X, raw_labels, metric="cosine") if _valid_for_sil(raw_labels) else -1.0
            chosen_th = None

    elif n == 2:
        # 파라미터화 된 기준
        raw_labels = np.array([0, 1], dtype=int) if D[0, 1] > 0.65 else np.array([0, 0], dtype=int)
        sil, chosen_th = -1.0, None

    else:
        raw_labels = np.array([0], dtype=int);
        sil, chosen_th = -1.0, None

    labels = to_zero_based(raw_labels)

    # 5) 통계 로깅
    print(f"chosen_threshold={chosen_th}, K={len(set(labels))}, silhouette(cos)={sil:.3f}")
    if n >= 2:
        print(f"min={tri.min():.3f} p25={p25:.3f} p50={p50:.3f} p85={p85:.3f} max={tri.max():.3f}")
        np.set_printoptions(precision=6, suppress=False)
        print("D sample (first 5x5):\n", D[:5, :5])

    # 6) 오디오 로드 1회(메모리) + 안전 슬라이스
    base = AudioSegment.from_file(path, format="wav")
    result_seperate = []

    for seg, label in zip(segments, labels):
        lnsc_speaker_cd = int(label) + 1009
        padded = safe_slice(base, seg.start - 0.40, seg.end + 0.40)

        segment = preprocess_segment(padded)

        filename = f"speaker{lnsc_speaker_cd}.wav"
        segment.export(filename, format="wav")

        with torch.no_grad():
            gender = model.predict(filename, device=device)
            if gender == "male":
                gender_cd = 36
            else:
                gender_cd = 37

        lnsc_contents = audio_extract(filename, "ko-KR")

        os.remove(filename)

        if lnsc_contents == -1:
            print("failed to extract text")
            continue

        lnsc_contents_eng = await trans_text(lnsc_contents)
        if lnsc_contents_eng == -1:
            print("failed to trans text")
            continue

        result_seperate.append([lnsc_speaker_cd, lnsc_contents, lnsc_contents_eng, gender_cd])

    print(tracks)

    if len(result_seperate) == 0:
        print(f"음성 추출 실패!!")
        return -1

    for contents in result_seperate:
        print(f"{contents[0]} : {contents[1]} / {contents[2]} / {gender}")

    return result_seperate

def pronunciation_evaluation_user(path: str):
    # 0) 전처리된 파일 생성
    path = clean_wav(path, use_denoise=True)

    lnsc_contents_eng = audio_extract(path, "en-US")
    if lnsc_contents_eng == -1:
        print("failed to extract text")
        return ""
    sentences = add_punctuation(lnsc_contents_eng)
    full_text = " ".join(sentences)
    return full_text

def embed_segment(E, seg: Segment, pad: float = 0.10, min_frames: int = 3):
    """
    E : embedder({"audio": path}) 결과 (SlidingWindowFeature, shape: (T, D))
    seg: pyannote.core.Segment
    pad: 크롭 실패/너무 짧을 때 양쪽으로 추가할 초
    """
    def _crop(s: Segment):
        part = E.crop(s)  # (t, D) 슬라이딩 윈도우 임베딩
        A = np.asarray(getattr(part, "data", part), dtype=np.float64)  # 호환용
        return A

    A = _crop(seg)

    # 프레임이 너무 적거나 빈 경우 소폭 확장해서 재시도
    if A.size == 0 or A.shape[0] < min_frames:
        seg_pad = Segment(max(0.0, seg.start - pad), seg.end + pad)
        A = _crop(seg_pad)

    # 그래도 실패하면 단일 프레임 폴백
    if A.size == 0:
        A = np.asarray(E.data, dtype=np.float64)[:1]

    T = A.shape[0]
    # 프레임별 L2 정규화
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

    # 중앙 가중 풀링 (Hamming)
    if T > 1:
        w = np.hamming(T).astype(np.float64)
        w = w / (w.sum() + 1e-12)
        v = (A * w[:, None]).sum(axis=0)
    else:
        v = A[0]

    # 안전장치 + 최종 L2
    v = np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    v /= (np.linalg.norm(v) + 1e-12)
    return v

def to_zero_based(labels_iterable):
    """라벨을 등장 순서대로 0,1,2…로 리맵"""
    mapping = {}
    out = []
    for lab in labels_iterable:
        if lab not in mapping:
            mapping[lab] = len(mapping)
        out.append(mapping[lab])
    return np.array(out, dtype=int)

def pca_whiten(X: np.ndarray, var_keep: float = 0.95) -> np.ndarray:
    # 샘플 적으면 화이트닝 생략 (20개 기준은 경험적)
    if len(X) < 20:
        return X
    pca = PCA(whiten=True, svd_solver="auto", random_state=0)
    Xw = pca.fit_transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var_keep)) + 1
    return Xw[:, :max(2, k)]

def safe_slice(audio: AudioSegment, start_s: float, end_s: float) -> AudioSegment:
    start_ms = max(0, int(round(start_s * 1000)))
    end_ms = min(len(audio), int(round(end_s * 1000)))
    if end_ms <= start_ms:
        end_ms = min(len(audio), start_ms + 50)  # 최소 50ms 확보
    seg = audio[start_ms:end_ms]
    # 너무 딱딱 끊기는 걸 방지 (미세 fade)
    try:
        seg = seg.fade_in(25).fade_out(25)
    except:
        pass
    return seg

def best_labels_by_threshold(D: np.ndarray, X_for_sil: np.ndarray, thresholds=(0.56, 0.54, 0.52, 0.50, 0.48, 0.46, 0.44)):
    """
    Agglomerative with precomputed distances + distance_threshold 스윕.
    실루엣 최대 & 유효성 체크하여 최종 라벨과 K 반환.
    """
    best = (None, -1.0, None)  # (labels, sil, th)
    n = D.shape[0]
    for th in thresholds:
        try:
            ac = AgglomerativeClustering(
                metric="precomputed", linkage="average",
                distance_threshold=th, n_clusters=None
            )
            labels = ac.fit_predict(D)
            k = len(set(labels))
            # 유효성: 1 < K <= min(n, 8), 각 클러스터 2샘플 이상 권장
            if k <= 1 or k > min(n, 8):
                continue
            # 각 클러스터 최소 2개
            sizes = [np.sum(labels == c) for c in range(k)]
            if min(sizes) < 2:
                continue
            sil = silhouette_score(X_for_sil, labels, metric="cosine")
            if sil > best[1]:
                best = (labels, sil, th)
        except Exception:
            continue
    return best  # (labels or None, sil, th)

def _valid_for_sil(lab):
    uniq = np.unique(lab)
    if len(uniq) < 2: return False
    # 싱글톤 1개까지 허용 (너무 박하게 걸면 스윕이 전부 탈락)
    singletons = sum((np.sum(lab == u) < 2) for u in uniq)
    return singletons <= 1

def merge_segments_overlap_only(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: (s.start, s.end))
    out, cur = [], segs[0]
    for nxt in segs[1:]:
        if nxt.start <= cur.end:  # overlap/abut
            cur = Segment(min(cur.start, nxt.start), max(cur.end, nxt.end))
        else:
            out.append(cur)
            cur = nxt
    out.append(cur)
    return out