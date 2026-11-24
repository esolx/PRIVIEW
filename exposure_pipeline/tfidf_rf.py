# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import (Dict, List, Optional, Sequence, Tuple)

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

"""tfidf_rf.py — 1단계 TF-IDF + RandomForest 분류기"""

from .config import LEGAL_CODES

# ===  tfidf_rf.py

"""
TF-IDF + Random Forest 분류기.

논문 대비 변경점 (요구사항 3)
-----------------------------------------------------------------
논문은 SMOTE로 소수 클래스를 오버샘플링한다. 여기서는 쓰지 않는다.

  · 입력이 이미 가명처리된 데이터다. SMOTE는 소수 클래스 이웃 사이를
    선형보간해 **없던 레코드를 만들어낸다.** 가명정보의 원본 분포를
    변형시키므로, 뒤에 이어지는 EMD(원본↔가명 분포 거리) 계산이
    합성 샘플로 오염된다. 위험도 평가 파이프라인 전체가 무의미해진다.
  · 합성 소수 샘플이 실존 정보주체 근방에 생성되면 그 자체가
    재식별 표면을 넓힌다.

대체 수단: RandomForestClassifier(class_weight=...)
  - "balanced"            : n/(n_classes * n_c) 를 전체 학습셋 기준으로
  - "balanced_subsample"  : 각 트리의 부트스트랩 표본마다 재계산 (기본값)
  둘 다 손실에 가중치를 줄 뿐 표본을 생성하지 않는다.

  ★ 소수 클래스 가중을 더 밀어야 하면 dict로 직접 준다:
     class_weight={"HEALTH": 5.0, ...}  ← 법정 민감정보 FN 비용 반영
"""


@dataclass
class RFConfig:
    # TF-IDF (word + char_wb 병렬 — 한국어 컬럼명 오타/축약 대응)
    word_ngram: Tuple[int, int] = (1, 2)
    char_ngram: Tuple[int, int] = (2, 4)
    max_features_word: int = 30000
    max_features_char: int = 50000
    min_df: int = 1
    sublinear_tf: bool = True
    use_char: bool = True
    use_struct: bool = True

    # Random Forest
    n_estimators: int = 400
    max_depth: Optional[int] = 20
    min_samples_leaf: int = 2
    min_samples_split: int = 4
    max_features: object = "sqrt"
    class_weight: object = "balanced_subsample"   # ← SMOTE 대체
    n_jobs: int = -1
    random_state: int = 42

    # 법정 클래스 추가 학습 손실 가중. 기본 1.0(추가 부스트 없음).
    # class_weight의 불균형 보정 위에 법정 ×3을 다시 얹으면 사람이 NON_SENSITIVE로
    # 교정한 의료/정치 문맥까지 법정 클래스로 끌리는 현상이 확인되어 기본 해제했다.
    legal_class_weight_boost: float = 1.0


class TfidfRandomForest:
    def __init__(self, config: Optional[RFConfig] = None):
        self.cfg = config or RFConfig()
        self.word_vec: Optional[TfidfVectorizer] = None
        self.char_vec: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.clf: Optional[RandomForestClassifier] = None
        self.classes_: np.ndarray = np.array([])

    # -- 피처 -------------------------------------------------------------
    def _fit_features(self, texts: Sequence[str], struct: Optional[np.ndarray]):
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=self.cfg.word_ngram,
            max_features=self.cfg.max_features_word, min_df=self.cfg.min_df,
            sublinear_tf=self.cfg.sublinear_tf, token_pattern=r"(?u)\S+",
        )
        blocks = [self.word_vec.fit_transform(texts)]
        if self.cfg.use_char:
            self.char_vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=self.cfg.char_ngram,
                max_features=self.cfg.max_features_char, min_df=self.cfg.min_df,
                sublinear_tf=self.cfg.sublinear_tf,
            )
            blocks.append(self.char_vec.fit_transform(texts))
        if self.cfg.use_struct and struct is not None:
            self.scaler = StandardScaler()
            blocks.append(sparse.csr_matrix(self.scaler.fit_transform(struct)))
        return sparse.hstack(blocks, format="csr")

    def _transform_features(self, texts: Sequence[str], struct: Optional[np.ndarray]):
        blocks = [self.word_vec.transform(texts)]
        if self.char_vec is not None:
            blocks.append(self.char_vec.transform(texts))
        if self.scaler is not None and struct is not None:
            blocks.append(sparse.csr_matrix(self.scaler.transform(struct)))
        return sparse.hstack(blocks, format="csr")

    def _class_weight(self, y: Sequence[str]):
        cw = self.cfg.class_weight
        if not isinstance(cw, str) or self.cfg.legal_class_weight_boost == 1.0:
            return cw
        # 문자열 모드 + 법정 부스트가 필요하면 dict로 확장
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        n, k = counts.sum(), len(classes)
        base = {c: n / (k * cnt) for c, cnt in zip(classes, counts)}
        for c in base:
            if c in LEGAL_CODES:
                base[c] *= self.cfg.legal_class_weight_boost
        return base

    # -- 학습/추론 ---------------------------------------------------------
    def fit(self, texts: Sequence[str], y: Sequence[str],
            struct: Optional[np.ndarray] = None,
            sample_weight: Optional[Sequence[float]] = None) -> "TfidfRandomForest":
        X = self._fit_features(texts, struct)
        self.clf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            min_samples_split=self.cfg.min_samples_split,
            max_features=self.cfg.max_features,
            class_weight=self._class_weight(y),   # ★ SMOTE 대체 지점
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            bootstrap=True,
        )
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.clf.fit(X, np.asarray(y), sample_weight=sw)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba(self, texts: Sequence[str],
                      struct: Optional[np.ndarray] = None) -> np.ndarray:
        return self.clf.predict_proba(self._transform_features(texts, struct))

    def predict(self, texts, struct=None) -> np.ndarray:
        return self.clf.predict(self._transform_features(texts, struct))

    def proba_frame(self, texts, struct=None) -> Dict[str, np.ndarray]:
        P = self.predict_proba(texts, struct)
        return {c: P[:, i] for i, c in enumerate(self.classes_)}

    # -- 평가 (논문 Table 1 형식: accuracy / CV mean / CV std) ---------------
    def evaluate_cv(self, texts, y, struct=None, n_splits: int = 5) -> Dict[str, float]:
        """Stratified CV. accuracy 하나 대신 macro/weighted F1을 함께 보고한다.

        중요: 벡터라이저를 fold 밖에서 fit하면 데이터 누수가 생기므로,
        각 fold에서 새 모델을 직접 학습해 TF-IDF까지 train fold로만 fit한다.
        """
        y = np.asarray(y)
        min_count = int(np.min(np.unique(y, return_counts=True)[1]))
        k = max(2, min(n_splits, min_count))
        cv = StratifiedKFold(n_splits=k, shuffle=True,
                             random_state=self.cfg.random_state)
        from sklearn.metrics import accuracy_score, f1_score
        acc, macro, weighted = [], [], []
        texts = np.asarray(list(texts), dtype=object)
        struct_arr = None if struct is None else np.asarray(struct)
        for tr, va in cv.split(texts, y):
            fold = TfidfRandomForest(self.cfg)
            tr_struct = None if struct_arr is None else struct_arr[tr]
            va_struct = None if struct_arr is None else struct_arr[va]
            fold.fit(texts[tr].tolist(), y[tr].tolist(), tr_struct)
            pred = fold.predict(texts[va].tolist(), va_struct)
            acc.append(accuracy_score(y[va], pred))
            macro.append(f1_score(y[va], pred, average="macro", zero_division=0))
            weighted.append(f1_score(y[va], pred, average="weighted", zero_division=0))

        # train accuracy는 과적합 gap 확인용으로만 제공
        probe = TfidfRandomForest(self.cfg).fit(texts.tolist(), y.tolist(), struct_arr)
        train_pred = probe.predict(texts.tolist(), struct_arr)
        return {
            "train_accuracy": float(accuracy_score(y, train_pred)),
            "cv_mean": float(np.mean(macro)),  # 기존 키 호환: 이제 macro-F1
            "cv_std": float(np.std(macro)),
            "cv_macro_f1": float(np.mean(macro)),
            "cv_weighted_f1": float(np.mean(weighted)),
            "cv_accuracy": float(np.mean(acc)),
            "n_folds": k,
        }

    def report(self, texts, y_true, struct=None) -> str:
        return classification_report(np.asarray(y_true),
                                     self.predict(texts, struct),
                                     zero_division=0)

    def top_features(self, k: int = 25) -> List[Tuple[str, float]]:
        names = list(self.word_vec.get_feature_names_out())
        if self.char_vec is not None:
            names += [f"char::{f}" for f in self.char_vec.get_feature_names_out()]
        imp = self.clf.feature_importances_[:len(names)]
        order = np.argsort(imp)[::-1][:k]
        return [(names[i], float(imp[i])) for i in order]



