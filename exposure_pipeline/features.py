# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import (Dict, List, Sequence, Tuple)

import numpy as np

"""features.py — 칼럼 프로파일링 · 구조 특징 · 약라벨 · 부트스트랩 학습셋"""

from .config import (CATEGORIES, LEGAL_CODES, NEGATIVE_CODE, REGEX_KEYS,
                     REGEX_PATTERNS, SENSITIVE_CODES)

# RF/GPT에 넘길 보조 신호. 라벨을 확정하는 규칙이 아니라 "feature"일 뿐이다.
# 특히 정치/건강 같은 단어가 의견, 평가 문항에 등장하는 오탐을 구분하기 위해
# 의견 문맥을 별도 토큰으로 함께 제공한다.
OPINION_CONTEXT_TERMS = (
    "자긍심", "예상", "전망", "평가", "인식", "관심도", "만족도",
    "중요도", "찬성", "반대", "생각", "의견", "태도", "신뢰",
    "필요", "필요성", "영향력", "전망", "기대", "공정성", "격차",
    "느끼는 정도", "생각하는 정도", "원인", "미칠 영향", "미치는 영향",
    "제도에 대한", "정책에 대한", "지도층에 대한",
    "opinion", "perception", "attitude", "trust", "fairness", "policy",
)

# ===  features.py

"""
가명 데이터셋(테이블) → 컬럼 단위 문서/피처 변환.

분류 단위를 '컬럼'으로 잡은 이유
  - 참조 논문도 PII '라벨(=컬럼 유형)' 식별을 문제로 정의한다.
  - 노출 위험도(EMD, 엔트로피, coverage, PMI)는 모두 속성 분포 위에서
    계산되므로 속성 단위 판정이 downstream과 자연스럽게 맞물린다.
  - 셀 단위 판정이 필요하면 Preprocessor를 값에 직접 적용해 확장하면 된다.
"""


@dataclass
class ColumnProfile:
    name: str
    values: List[str]
    n_rows: int
    n_unique: int
    dtype: str = "object"
    struct: Dict[str, float] = field(default_factory=dict)

    @property
    def unique_ratio(self) -> float:
        return self.n_unique / max(self.n_rows, 1)


def _split_name(name: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"[_\-.]+", " ", s)
    return s.strip()


def profile_dataframe(df, sample_size: int = 300, seed: int = 42) -> List[ColumnProfile]:
    rng = random.Random(seed)
    profiles = []
    for col in df.columns:
        s = df[col]
        vals = [str(v) for v in s.dropna().tolist()]
        sampled = vals if len(vals) <= sample_size else rng.sample(vals, sample_size)
        p = ColumnProfile(
            name=str(col),
            values=sampled,
            n_rows=len(s),
            n_unique=int(s.nunique(dropna=True)),
            dtype=str(s.dtype),
        )
        p.struct = structural_features(p)
        profiles.append(p)
    return profiles


def profile_signal_tokens(p: ColumnProfile) -> List[str]:
    """앞단에서 관찰 가능한 약한 신호를 RF/LLM용 토큰으로 변환한다.

    주의: 이 함수는 카테고리를 *확정*하지 않는다. 키워드/정규식/구조 신호를
    텍스트 feature로만 노출해 RF가 다른 증거와 함께 학습하도록 한다.
    """
    name = _split_name(p.name).lower()
    compact = re.sub(r"\s+", "", name)
    tokens: List[str] = []

    # 1) 이름 기반 카테고리 후보. 직접 라벨링하지 않고 후보 토큰만 제공.
    for code, cat in CATEGORIES.items():
        if code == NEGATIVE_CODE:
            continue
        hit = False
        for kw in cat.keywords:
            k = str(kw).lower().strip()
            if not k:
                continue
            if k in name or re.sub(r"\s+", "", k) in compact:
                hit = True
                break
        if hit:
            tokens.append(f"__NAME_HINT_{code}__")

    # 2) '정치/건강/자녀' 같은 주제 언급과 응답자 속성을 구분하는 반대 신호.
    if any(term in name for term in OPINION_CONTEXT_TERMS):
        tokens.append("__OPINION_CONTEXT__")

    # 3) 값 정규식 히트. '실제 민감 카테고리 검증기'에 연결된 강한 패턴만 사용.
    #    date/hugo/pseudo_id 같은 일반, 헐거운 패턴은 여기서 라벨 힌트로 쓰지 않는다.
    from .config import VALUE_VALIDATORS
    for code, keys in VALUE_VALIDATORS.items():
        rates = [float(p.struct.get(f"re_{key}", 0.0) or 0.0) for key in keys]
        best = max(rates, default=0.0)
        if best >= 0.60:
            tokens.append(f"__VALUE_HINT_{code}_STRONG__")
        elif best >= 0.10:
            tokens.append(f"__VALUE_HINT_{code}_WEAK__")

    # 4) 구조적 모양. 숫자 금액/코드/ID류를 텍스트 모델에서도 구별하게 한다.
    if p.struct.get("is_numeric_dtype", 0.0) >= 0.5:
        tokens.append("__NUMERIC_DTYPE__")
    ur = p.unique_ratio
    if ur >= 0.95:
        tokens.append("__VERY_HIGH_UNIQUE__")
    elif ur <= 0.05:
        tokens.append("__VERY_LOW_UNIQUE__")
    if p.struct.get("digit_ratio", 0.0) >= 0.80:
        tokens.append("__DIGIT_HEAVY__")

    return tokens


def profile_signal_summary(p: ColumnProfile) -> str:
    """GPT prompt에 붙이기 쉬운 사람이 읽을 수 있는 보조신호 요약."""
    toks = profile_signal_tokens(p)
    return ", ".join(toks) if toks else "없음"


def profile_to_text(p: ColumnProfile, max_values: int = 60) -> str:
    """컬럼명(가중 반복) + 대표 값 + 앞단 보조신호를 한 문서로.

    컬럼명은 라벨 신호가 가장 강하므로 3회 반복한다. 보조신호는 정답 라벨이
    아니라 RF가 미탐/오탐 경계에서 참고하는 feature다.
    """
    head = " ".join([_split_name(p.name)] * 3)
    body = " ".join(p.values[:max_values])
    signals = " ".join(profile_signal_tokens(p))
    return f"{head} {signals} {body}".strip()



def sanitize_rf_text(text: str) -> str:
    """RF 학습/추론용 텍스트에서 카테고리 정답을 암시하는 규칙 토큰을 제거한다.

    labels_review의 corrected_label이 RF에 들어가더라도 __NAME_HINT_HEALTH__ 같은
    앞단 규칙 토큰이 남아 있으면 모델이 사람 교정보다 기존 키워드 규칙을 다시
    암기할 수 있다. RF에는 컬럼명/대표값/일반 구조 신호만 남긴다.
    """
    s = str(text)
    s = re.sub(r"__NAME_HINT_[A-Z0-9_]+__", " ", s)
    s = re.sub(r"__VALUE_HINT_[A-Z0-9_]+_(?:STRONG|WEAK)__", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def profile_to_rf_text(p: ColumnProfile, max_values: int = 60) -> str:
    """RandomForest 전용 문서.

    카테고리명 자체가 들어간 NAME/VALUE_HINT 토큰은 제외하고, 의견문맥/숫자형/
    고유도처럼 라벨과 독립적인 구조 신호만 제공한다.
    """
    head = " ".join([_split_name(p.name)] * 6)
    body = " ".join(p.values[:max_values])
    generic = []
    name = _split_name(p.name).lower()
    if any(term in name for term in OPINION_CONTEXT_TERMS):
        generic.append("__OPINION_CONTEXT__")
    if p.struct.get("is_numeric_dtype", 0.0) >= 0.5:
        generic.append("__NUMERIC_DTYPE__")
    ur = p.unique_ratio
    if ur >= 0.95:
        generic.append("__VERY_HIGH_UNIQUE__")
    elif ur <= 0.05:
        generic.append("__VERY_LOW_UNIQUE__")
    if p.struct.get("digit_ratio", 0.0) >= 0.80:
        generic.append("__DIGIT_HEAVY__")
    return f"{head} {' '.join(generic)} {body}".strip()

def structural_features(p: ColumnProfile) -> Dict[str, float]:
    vals = p.values or [""]
    lens = [len(v) for v in vals]
    digit_ratio = np.mean([sum(ch.isdigit() for ch in v) / max(len(v), 1) for v in vals])
    hangul_ratio = np.mean(
        [sum("\uac00" <= ch <= "\ud7a3" for ch in v) / max(len(v), 1) for v in vals]
    )
    feats: Dict[str, float] = {
        "unique_ratio": p.unique_ratio,
        "mean_len": float(np.mean(lens)),
        "std_len": float(np.std(lens)),
        "digit_ratio": float(digit_ratio),
        "hangul_ratio": float(hangul_ratio),
        "is_numeric_dtype": float("int" in p.dtype or "float" in p.dtype),
        "cardinality_log": math.log1p(p.n_unique),
        "entropy_norm": _value_entropy(vals),
    }
    # 논문의 regular expression matching → 구조 피처로 편입
    for key in REGEX_KEYS:
        pat = REGEX_PATTERNS[key]
        hit = sum(1 for v in vals if pat.search(v))
        feats[f"re_{key}"] = hit / max(len(vals), 1)
    return feats


def _value_entropy(vals: Sequence[str]) -> float:
    c = Counter(vals)
    n = sum(c.values())
    if n <= 1:
        return 0.0
    h = -sum((k / n) * math.log(k / n) for k in c.values())
    return h / math.log(n)


def struct_matrix(profiles: Sequence[ColumnProfile]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted(profiles[0].struct.keys()) if profiles else []
    mat = np.array([[p.struct.get(k, 0.0) for k in keys] for p in profiles],
                   dtype=np.float64)
    return mat, keys


# ---------------------------------------------------------------------------
# 약지도 라벨링 (정답셋이 없을 때의 부트스트랩)
# ---------------------------------------------------------------------------
def weak_label(p: ColumnProfile, preprocessor=None) -> Tuple[str, float]:
    """컬럼명 + 값에서 카테고리 키워드 히트를 세어 라벨을 추정.

    ⚠️ 이건 시드일 뿐이다. 실제 운영 라벨로 쓰지 말고, 여기서 나온
    라벨을 사람이 검수해 정답셋을 만들고 그걸로 다시 학습할 것.
    """
    text = (_split_name(p.name) + " " + " ".join(p.values[:80])).lower()
    scores: Dict[str, float] = {}
    for code, cat in CATEGORIES.items():
        if code == NEGATIVE_CODE:
            continue
        hit = sum(1.0 for kw in cat.keywords if kw.lower() in text)
        name_hit = sum(2.0 for kw in cat.keywords
                       if kw.lower() in _split_name(p.name).lower())
        if hit or name_hit:
            scores[code] = hit + name_hit
    if not scores:
        return NEGATIVE_CODE, 0.0
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    return best, scores[best] / total


def bootstrap_training_set(n_per_cat: int = 40, seed: int = 42
                           ) -> Tuple[List[str], List[str]]:
    """카테고리 키워드·프로토타입 문장을 조합해 초기 학습 코퍼스를 합성.

    데이터가 아니라 '스키마 어휘'를 합성하는 것이므로 SMOTE와 달리
    가명 레코드를 만들어내지 않는다(3번 요구사항과 충돌하지 않음).
    정답셋이 확보되면 이 함수 대신 실제 라벨로 학습할 것.
    """
    rng = random.Random(seed)
    texts, labels = [], []
    for code, cat in CATEGORIES.items():
        kws = list(cat.keywords) or [cat.name_ko]
        protos = list(cat.prototypes) or [cat.name_ko]
        for _ in range(n_per_cat):
            k = rng.sample(kws, k=min(len(kws), rng.randint(1, 2)))
            # 키워드 3회 반복은 RF가 합성 템플릿을 암기하게 만들어 제거한다.
            head = " ".join(k)
            body = rng.choice(protos)
            form = rng.choice([
                "{head}", "{head} field", "column {head}",
                "{head} value", "{head} {body}",
            ])
            texts.append(form.format(head=head, body=body))
            labels.append(code)
    return texts, labels


def to_binary(labels: Sequence[str]) -> np.ndarray:
    """민감(1)/비민감(0)"""
    return np.array([1 if l in SENSITIVE_CODES else 0 for l in labels], dtype=int)


def to_legal(labels: Sequence[str]) -> np.ndarray:
    """법정(1)/그 외(0)"""
    return np.array([1 if l in LEGAL_CODES else 0 for l in labels], dtype=int)



