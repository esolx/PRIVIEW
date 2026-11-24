# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import (Dict, Literal, Optional, Sequence, Tuple)

import numpy as np

"""threshold.py — Recall 우선 임계값 튜닝 · 이중 임계값(회색지대)"""

# (리팩토링) 상위 네임스페이스 전파용 star import 제거 — 사용 심볼 없음

# ===  threshold.py

"""
Recall 우선 임계값 튜닝 (요구사항 4).

문제 설정
  민감정보를 놓치면(FN) 위험도 평가가 아예 작동하지 않는다 → 비용 매우 큼
  비민감을 민감이라 하면(FP) 검수 부하만 늘어난다              → 비용 작음
  ⇒ 임계값 τ를 아래로 밀어 Recall을 끌어올린다. 대신 Precision 손실은
    반드시 수치로 확인하고 감수 범위 안인지 판단해야 한다.

제공 모드
  1) recall_constrained : Recall ≥ min_recall 을 만족하는 τ 중 Precision 최대
                          (기본. "최소한 이만큼은 잡는다"를 계약처럼 고정)
  2) fbeta              : F_beta 최대화. beta=2면 Recall을 Precision의 2배로 중시
  3) cost               : 기대비용 c_fn·FN + c_fp·FP 최소화
  4) youden             : TPR − FPR 최대 (클래스 균형이 맞을 때만 권장)

dual_threshold
  τ_low(Recall 확보) 와 τ_high(Precision 확보) 사이를 회색지대로 두고
  사람 검수로 보낸다. 자동 판정과 검수 부하를 동시에 통제하는 실무적 절충.
"""


Mode = Literal["recall_constrained", "fbeta", "cost", "youden"]


@dataclass
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    fbeta: float
    beta: float
    tp: int
    fp: int
    fn: int
    tn: int
    mode: str
    expected_cost: Optional[float] = None
    note: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"[{self.mode}] τ={self.threshold:.4f} | "
                f"R={self.recall:.4f} P={self.precision:.4f} "
                f"F1={self.f1:.4f} F{self.beta:g}={self.fbeta:.4f} | "
                f"TP={self.tp} FP={self.fp} FN={self.fn} TN={self.tn}"
                + (f" | cost={self.expected_cost:.2f}" if self.expected_cost else ""))


def _confusion(y: np.ndarray, pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    return tp, fp, fn, tn


def _metrics_at(y: np.ndarray, scores: np.ndarray, tau: float, beta: float):
    pred = (scores >= tau).astype(int)
    tp, fp, fn, tn = _confusion(y, pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    b2 = beta ** 2
    fb = ((1 + b2) * p * r / (b2 * p + r)) if (b2 * p + r) else 0.0
    return p, r, f1, fb, tp, fp, fn, tn


def candidate_thresholds(scores: np.ndarray, max_grid: int = 2000) -> np.ndarray:
    """관측 점수 사이 중점들을 후보로. 양끝에 여유값 추가."""
    u = np.unique(np.asarray(scores, dtype=float))
    if len(u) > max_grid:
        u = np.quantile(u, np.linspace(0, 1, max_grid))
        u = np.unique(u)
    mids = (u[:-1] + u[1:]) / 2 if len(u) > 1 else u
    return np.unique(np.concatenate([[u.min() - 1e-6], mids, [u.max() + 1e-6]]))


def sweep(y_true: Sequence[int], scores: Sequence[float],
          beta: float = 2.0, max_grid: int = 2000) -> list:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    rows = []
    for tau in candidate_thresholds(s, max_grid):
        p, r, f1, fb, tp, fp, fn, tn = _metrics_at(y, s, tau, beta)
        rows.append({"threshold": float(tau), "precision": p, "recall": r,
                     "f1": f1, "fbeta": fb, "tp": tp, "fp": fp,
                     "fn": fn, "tn": tn})
    return rows


def tune_threshold(y_true: Sequence[int],
                   scores: Sequence[float],
                   mode: Mode = "recall_constrained",
                   min_recall: float = 0.95,
                   beta: float = 2.0,
                   c_fn: float = 10.0,
                   c_fp: float = 1.0,
                   max_grid: int = 2000) -> ThresholdResult:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0:
        raise ValueError("양성 샘플이 없다. 임계값 튜닝 불가.")

    taus = candidate_thresholds(s, max_grid)
    best, best_key, note = None, None, ""

    for tau in taus:
        p, r, f1, fb, tp, fp, fn, tn = _metrics_at(y, s, tau, beta)
        cost = c_fn * fn + c_fp * fp

        if mode == "recall_constrained":
            if r < min_recall:
                continue
            key = (p, r, tau)          # 제약 만족 중 Precision 최대, 동률이면 τ 큰 쪽
        elif mode == "fbeta":
            key = (fb, r, tau)
        elif mode == "cost":
            key = (-cost, r, tau)
        elif mode == "youden":
            tpr = r
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            key = (tpr - fpr, r, tau)
        else:
            raise ValueError(f"unknown mode: {mode}")

        if best_key is None or key > best_key:
            best_key = key
            best = ThresholdResult(float(tau), p, r, f1, fb, beta,
                                   tp, fp, fn, tn, mode, float(cost))

    if best is None:
        # Recall 제약을 어떤 τ로도 못 맞춘 경우 → 최소 τ로 전량 양성 처리
        tau = float(taus[0])
        p, r, f1, fb, tp, fp, fn, tn = _metrics_at(y, s, tau, beta)
        note = (f"Recall {min_recall:.2f}를 만족하는 τ가 없다. "
                f"점수 분포가 두 클래스를 분리하지 못한다는 뜻이므로 "
                f"임계값이 아니라 표현(임베딩/피처)을 먼저 손볼 것.")
        best = ThresholdResult(tau, p, r, f1, fb, beta, tp, fp, fn, tn,
                               mode, float(c_fn * fn + c_fp * fp), note)
    else:
        best.note = note
    return best


def dual_threshold(y_true: Sequence[int], scores: Sequence[float],
                   min_recall: float = 0.98,
                   min_precision: float = 0.90,
                   beta: float = 2.0) -> Dict:
    """τ_low(자동 음성 컷) / τ_high(자동 양성 컷) / 회색지대 비율.

      score <  τ_low            → 비민감으로 자동 확정
      τ_low ≤ score < τ_high    → 사람 검수 (회색지대)
      score ≥ τ_high            → 민감으로 자동 확정
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)

    low = tune_threshold(y, s, mode="recall_constrained",
                         min_recall=min_recall, beta=beta)

    high, best_key = None, None
    for tau in candidate_thresholds(s):
        p, r, f1, fb, tp, fp, fn, tn = _metrics_at(y, s, tau, beta)
        if p < min_precision:
            continue
        key = (r, -tau)
        if best_key is None or key > best_key:
            best_key = key
            high = ThresholdResult(float(tau), p, r, f1, fb, beta,
                                   tp, fp, fn, tn, "precision_constrained")
    if high is None or high.threshold < low.threshold:
        high = low

    gray = float(np.mean((s >= low.threshold) & (s < high.threshold)))
    missed = float(np.mean(s[y == 1] < low.threshold)) if (y == 1).any() else 0.0
    return {
        "tau_low": low.threshold,
        "tau_high": high.threshold,
        "low": low,
        "high": high,
        "gray_zone_rate": gray,
        "auto_negative_rate": float(np.mean(s < low.threshold)),
        "auto_positive_rate": float(np.mean(s >= high.threshold)),
        "missed_positive_rate": missed,   # 회색지대 아래로 샌 실제 양성 비율
    }


def calibrate_by_quantile(scores: Sequence[float], target_flag_rate: float = 0.30
                          ) -> float:
    """정답셋이 아직 없을 때의 임시 방편.

    상위 target_flag_rate 비율을 민감 후보로 올리는 분위수 컷.
    성능 보장이 전혀 없으므로 검수 라벨이 모이면 즉시 tune_threshold로 교체.
    """
    return float(np.quantile(np.asarray(scores, dtype=float),
                             1.0 - target_flag_rate))


def fuse_scores(rf_score: np.ndarray, sbert_score: np.ndarray,
                rule: str = "or", weight: float = 0.5,
                rf_tau: float = 0.5, sbert_tau: float = 0.0) -> np.ndarray:
    """RF 확률과 SBERT 유사도 결합.

      "or"     : 둘 중 하나라도 자기 임계값을 넘으면 양성 (Recall 최대화, 기본)
      "and"    : 둘 다 넘어야 양성 (Precision 우선)
      "linear" : w·RF + (1−w)·SBERT_normalized  → 연속 점수 (τ 재튜닝 필요)
    """
    rf = np.asarray(rf_score, dtype=float)
    sb = np.asarray(sbert_score, dtype=float)
    if rule == "or":
        return np.maximum(rf - rf_tau, sb - sbert_tau)
    if rule == "and":
        return np.minimum(rf - rf_tau, sb - sbert_tau)
    if rule == "linear":
        sb_n = (sb - sb.min()) / max(sb.ptp(), 1e-12)
        return weight * rf + (1 - weight) * sb_n
    raise ValueError(f"unknown rule: {rule}")



