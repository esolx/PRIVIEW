# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import (Dict, List, Optional, Sequence, Tuple)

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

"""sbert_sim.py — SBERT/TF-IDF 프로토타입 유사도 (2단계 구제)"""

from .config import CATEGORIES, LEGAL_CODES, NEGATIVE_CODE
from .threshold import tune_threshold

# ===  sbert_sim.py

"""
Sentence-BERT 코사인 유사도 기반 카테고리 판정.

SBERT (Reimers & Gurevych, 2019) 근거
  - §3 Model: pooling 기본값은 MEAN. CLS/MAX보다 안정적이며 회귀 목적함수
    학습 시 MAX는 뚜렷하게 나쁘다(Table 6: STSb 기준 MEAN 87.44 / MAX 69.92).
    → 여기서도 MEAN 고정.
  - §4 Evaluation: 유사도는 항상 cosine-similarity로 계산.
    Manhattan/Euclidean으로 바꿔도 결과 차이가 거의 없다고 보고.
  - §3 Triplet Objective: max(||s_a - s_p|| - ||s_a - s_n|| + ε, 0), ε=1,
    거리 척도는 Euclidean. ← 사용자가 말한 "margin"이 이 ε.

중요한 구분 (혼동하기 쉬움)
  · triplet margin ε : **학습 시** 양성/음성 쌍을 얼마나 벌릴지의 하이퍼파라미터
  · decision threshold τ : **추론 시** cos-sim을 민감/비민감으로 자르는 값
  요구사항 4의 "임계값을 낮춰 Recall을 높인다"는 τ 이야기다.
  τ는 threshold.py에서 튜닝하고, ε는 아래 scan_margin으로 별도 탐색한다.

모델 (한/영 컬럼을 함께 다루므로 다국어 모델이 기본)
  - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  ← 기본
  - "intfloat/multilingual-e5-base"
  - 한국어 전용이면 "jhgan/ko-sbert-sts", "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

이 파이프라인에서 SBERT의 역할 (요구사항)
  1단계(TF-IDF RF + TF-IDF 코사인)가 놓친 잔여분만 받아서 재탐지하는
  **구제(rescue) 단계**다. 1단계 양성은 건드리지 않는다.
  따라서 SBERT 임계값은 전체 분포가 아니라 **잔여분 위에서** 튜닝해야 한다.
  (pipeline.calibrate가 그렇게 한다)
"""


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class _TfidfEncoder:
    """sentence-transformers 미설치 환경용 폴백 인코더.

    문자 n-gram TF-IDF + LSA. 의미 유사도 품질은 SBERT보다 확실히 낮다.
    파이프라인 배선/테스트 용도로만 쓸 것.
    """

    def __init__(self, dim: int = 256):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                   sublinear_tf=True)
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False
        self._dim = dim

    def fit(self, corpus: Sequence[str]):
        X = self.vec.fit_transform(corpus)
        self.svd.n_components = min(self._dim, max(2, min(X.shape) - 1))
        self.svd.fit(X)
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str], **_) -> np.ndarray:
        if not self._fitted:
            self.fit(list(texts))
        return self.svd.transform(self.vec.transform(texts))


@dataclass
class SbertConfig:
    model_name: str = DEFAULT_MODEL
    pooling: str = "mean"          # SBERT 기본값 (Table 6 근거)
    batch_size: int = 64
    normalize: bool = True
    prototype_agg: str = "centroid"  # "centroid" | "max"
    force_backend: Optional[str] = None   # "tfidf" 로 두면 1단계용 어휘 코사인


class PrototypeSimilarity:
    """카테고리 프로토타입 문장 → 센트로이드 → 컬럼 문서와의 cos-sim."""

    def __init__(self, config: Optional[SbertConfig] = None,
                 fallback: bool = True):
        self.cfg = config or SbertConfig()
        self.backend = "sbert"
        self._model = None
        if self.cfg.force_backend == "tfidf":
            self.backend = "tfidf"
            self._model = _TfidfEncoder()
            self.codes, self.centroids = [], None
            self._proto_texts, self._proto_owner = [], []
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.cfg.model_name)
        except Exception:
            if not fallback:
                raise
            # SBERT가 기본 경로다. 라이브러리 미설치 시에만 약한 TF-IDF 해시로
            # 폴백하는데, 조용히 넘어가면 의미검증이 약화된 줄 모르고 운영하게
            # 되므로 눈에 띄게 경고한다. (설치: pip install sentence-transformers)
            import warnings
            warnings.warn(
                "sentence-transformers 미설치 → SBERT 3단계가 TF-IDF 해시 폴백으로 "
                "동작합니다(의미검증 약화). 실제 SBERT를 쓰려면 "
                "`pip install sentence-transformers` 후 재실행하세요.",
                RuntimeWarning, stacklevel=2)
            self.backend = "tfidf"
            self._model = _TfidfEncoder()
        self.codes: List[str] = []
        self.centroids: Optional[np.ndarray] = None
        self._proto_texts: List[str] = []
        self._proto_owner: List[int] = []

    # -- 인코딩 -----------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "sbert":
            emb = self._model.encode(list(texts),
                                     batch_size=self.cfg.batch_size,
                                     convert_to_numpy=True,
                                     show_progress_bar=False)
        else:
            emb = self._model.encode(list(texts))
        return _l2(np.asarray(emb, dtype=np.float64)) if self.cfg.normalize else emb

    # -- 프로토타입 구축 ---------------------------------------------------
    def build(self, extra_prototypes: Optional[Dict[str, Sequence[str]]] = None
              ) -> "PrototypeSimilarity":
        texts, owners, codes = [], [], []
        for code, cat in CATEGORIES.items():
            protos = list(cat.prototypes)
            # 키워드도 짧은 프로토타입으로 편입 (사전 신호 흡수)
            protos += [f"{cat.name_ko} 관련 항목: {kw}" for kw in cat.keywords[:12]]
            if extra_prototypes and code in extra_prototypes:
                protos += list(extra_prototypes[code])
            ci = len(codes)
            codes.append(code)
            for p in protos:
                texts.append(p)
                owners.append(ci)
        if self.backend == "tfidf":
            self._model.fit(texts)
        emb = self.encode(texts)
        self.codes, self._proto_texts, self._proto_owner = codes, texts, owners
        owners_arr = np.asarray(owners)
        if self.cfg.prototype_agg == "centroid":
            cents = np.vstack([_l2(emb[owners_arr == i].mean(axis=0, keepdims=True))
                               for i in range(len(codes))])
        else:
            cents = emb  # max 집계는 similarity()에서 처리
        self.centroids = cents
        self._proto_emb = emb
        return self

    # -- 값 단위 의미 투표 ------------------------------------------
    def _description_embeddings(self):
        """카테고리 설명문 임베딩 (값 투표 전용 — 1·3단계 프로토타입/보정에는 영향 없음).
        설명문 = 카테고리 이름(영/한) + 키워드 + 값 사전 대표어. 실제 SBERT 에서만 의미가 있다."""
        if getattr(self, "_desc_emb", None) is None:
            from .config import VALUE_LEXICONS
            texts = []
            for code, cat in CATEGORIES.items():
                vals = ", ".join(list(VALUE_LEXICONS.get(code, ()))[:15])
                kws = ", ".join(list(cat.keywords)[:12])
                texts.append(f"{cat.name_en} ({cat.name_ko}): {kws}" + (f"; e.g. {vals}" if vals else ""))
            self._desc_codes = list(CATEGORIES.keys())
            self._desc_emb = self.encode(texts)
        return self._desc_emb, self._desc_codes

    def value_vote(self, values_per_column: Sequence[Sequence[str]], k: int = 20
                   ) -> Tuple[np.ndarray, List[str]]:
        """컬럼마다 대표값(최대 k개 고유값)을 하나씩 임베딩해 카테고리 설명문과의
        평균 유사도를 낸다 → (n_cols, n_cats), 카테고리 순서.
        컬럼명이 처음 보는 이름이어도 값의 '뜻'으로 법정 의심을 찾기 위한 간접 근거."""
        D, codes = self._description_embeddings()
        out = np.zeros((len(values_per_column), len(codes)))
        for i, vals in enumerate(values_per_column):
            uniq = [str(v) for v in dict.fromkeys(str(x) for x in vals if str(x).strip())][:k]
            if not uniq:
                continue
            E = self.encode(uniq)
            out[i] = (E @ D.T).mean(axis=0)
        return out, codes

    # -- 유사도 -----------------------------------------------------------
    def similarity(self, texts: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        """returns (n_texts, n_categories) cos-sim 행렬, 카테고리 코드 순서."""
        if self.centroids is None:
            self.build()
        E = self.encode(texts)
        if self.cfg.prototype_agg == "centroid":
            S = E @ self.centroids.T
        else:
            raw = E @ self._proto_emb.T
            owners = np.asarray(self._proto_owner)
            S = np.vstack([raw[:, owners == i].max(axis=1)
                           for i in range(len(self.codes))]).T
        return S, self.codes

    def sensitivity_score(self, texts: Sequence[str]) -> np.ndarray:
        """민감 카테고리 최대 유사도 − 비민감 카테고리 유사도.

        절대 cos-sim은 모델·도메인마다 스케일이 달라 임계값이 흔들린다.
        비민감 프로토타입과의 상대 마진으로 정규화하면 τ가 안정된다.
        """
        S, codes = self.similarity(texts)
        return sensitivity_from_similarity(S, codes)

    def legal_score(self, texts: Sequence[str]) -> np.ndarray:
        S, codes = self.similarity(texts)
        return legal_from_similarity(S, codes)


# ---------------------------------------------------------------------------
# 유사도 행렬 → 점수 (인코딩 없이 재사용)
#   similarity() 한 번으로 얻은 S 에서 민감/법정 마진을 바로 계산한다.
#   예전에는 pipeline.run 이 같은 텍스트를 similarity/sensitivity_score/
#   legal_score 로 각각 다시 인코딩해 SBERT 인코딩이 3~4배 중복됐다.
# ---------------------------------------------------------------------------
def sensitivity_from_similarity(S: np.ndarray, codes: Sequence[str]) -> np.ndarray:
    """민감 카테고리 최대 유사도 − 비민감(NON_SENSITIVE) 유사도."""
    codes = list(codes)
    neg_idx = codes.index(NEGATIVE_CODE)
    sens_idx = [i for i, c in enumerate(codes) if c != NEGATIVE_CODE]
    return S[:, sens_idx].max(axis=1) - S[:, neg_idx]


def legal_from_similarity(S: np.ndarray, codes: Sequence[str]) -> np.ndarray:
    """법정 카테고리 최대 유사도 − 그 외 카테고리 최대 유사도."""
    li = [i for i, c in enumerate(codes) if c in LEGAL_CODES]
    oi = [i for i, c in enumerate(codes) if c not in LEGAL_CODES]
    return S[:, li].max(axis=1) - S[:, oi].max(axis=1)


# ---------------------------------------------------------------------------
# (선택) 도메인 파인튜닝 — triplet margin ε 탐색
# ---------------------------------------------------------------------------
def finetune_triplet(train_triplets: Sequence[Tuple[str, str, str]],
                     model_name: str = DEFAULT_MODEL,
                     margin: float = 1.0,
                     epochs: int = 1,
                     batch_size: int = 16,
                     out_path: str = "./sbert-pii-ft"):
    """(anchor, positive, negative) 삼중항으로 파인튜닝.

    SBERT §3: 거리 척도 Euclidean, ε=1이 논문 기본값.
    민감정보 도메인에서는 ε를 키울수록 클래스 간 분리는 커지지만
    경계 근처 양성이 음성 쪽으로 밀려 Recall이 떨어질 수 있다.
    반드시 scan_margin으로 검증셋 Recall을 보고 고를 것.
    """
    from sentence_transformers import (InputExample, SentenceTransformer,
                                       losses)
    from torch.utils.data import DataLoader

    model = SentenceTransformer(model_name)
    examples = [InputExample(texts=[a, p, n]) for a, p, n in train_triplets]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.EUCLIDEAN,
        triplet_margin=margin,
    )
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=int(0.1 * len(loader) * epochs),
              output_path=out_path)
    return model


def scan_margin(train_triplets, val_texts, val_labels,
                margins: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
                min_recall: float = 0.95, **kw) -> List[dict]:
    """margin ε별로 파인튜닝 → recall 제약 하 임계값 튜닝 → 비교표 반환."""
    rows = []
    for m in margins:
        model = finetune_triplet(train_triplets, margin=m,
                                 out_path=f"./sbert-ft-m{m}", **kw)
        sim = PrototypeSimilarity(SbertConfig(model_name=f"./sbert-ft-m{m}"))
        sim._model = model
        sim.build()
        scores = sim.sensitivity_score(val_texts)
        res = tune_threshold(np.asarray(val_labels), scores,
                             mode="recall_constrained", min_recall=min_recall)
        rows.append({"margin": m, **res.as_dict()})
    return rows



