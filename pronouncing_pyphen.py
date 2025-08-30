import re
from functools import lru_cache

# 1) 선택 의존성: pronouncing (CMU dict)
try:
    import pronouncing  # CMU Pronouncing Dictionary
except Exception:
    pronouncing = None

# 2) 선택 의존성: pyphen (hyphenation)
try:
    import pyphen
    _pyphen_dic = pyphen.Pyphen(lang="en_US")
except Exception:
    _pyphen_dic = None

_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.I)
_WORD_RX = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # don't → don't, it's → it's

@lru_cache(maxsize=100_000)
def _cmu_syllables(word: str) -> int | None:
    """CMU 사전 기반 음절 수 (없으면 None)."""
    if not pronouncing:
        return None
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return None
    # CMU: 스트레스 숫자(0/1/2)가 달린 모음 수 = 음절 수
    return max(sum(ch.isdigit() for ch in ph) for ph in phones)

@lru_cache(maxsize=100_000)
def _pyphen_syllables(word: str) -> int | None:
    """Pyphen 하이픈 분절 기반 음절 수 (없으면 None)."""
    if not _pyphen_dic:
        return None
    hyph = _pyphen_dic.inserted(word)
    if not hyph:
        return None
    # 분절 조각 개수 = 음절 근사
    return max(1, hyph.count("-") + 1)

def _heuristic_syllables(word: str) -> int:
    """가벼운 휴리스틱(최소 1 보장)."""
    w = word.lower()
    # 약어/스펠링 읽기(AI, USA 등)는 글자 수 근사(보수적으로 1).
    if len(w) <= 2:
        return 1
    groups = _VOWEL_GROUPS.findall(w)
    count = len(groups)

    # 무음 e: cake → 1, make → 1 (단, 'le'/'ye' 등은 제외)
    if w.endswith("e") and not w.endswith(("le", "ye")) and count > 1:
        count -= 1

    # 몇몇 일반 예외 보정
    if w.endswith(("tion", "sion")) and count > 1:
        count -= 1  # nation, vision 등은 과대계산 방지
    if w.endswith("le") and len(w) > 2 and w[-3] not in "aeiouy":
        count += 1  # table, little 등 자음+le 패턴

    return max(1, count)

@lru_cache(maxsize=100_000)
def count_english_syllables(word: str) -> int:
    """
    영어 단어 한 개의 음절 수를 최대 정확도로 추정:
      1) CMU 발음사전 → 2) Pyphen 하이픈 → 3) 휴리스틱
    """
    if not word:
        return 0
    w = word.strip().lower()
    # 1) CMU
    n = _cmu_syllables(w)
    if n:
        return n
    # 2) Pyphen
    n = _pyphen_syllables(w)
    if n:
        return n
    # 3) Heuristic
    return _heuristic_syllables(w)

def count_english_syllables_in_text(text: str) -> int:
    """
    문장/문단 전체의 영어 음절 수 합계.
    아포스트로피 포함 단어만 집계(숫자/기호 제외).
    """
    if not text:
        return 0
    words = _WORD_RX.findall(text)
    return sum(count_english_syllables(w) for w in words)
