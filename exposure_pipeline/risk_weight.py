# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import (Optional, Sequence, Tuple)

import numpy as np

"""risk_weight.py — 위험도 가중치 — 법정 1.5 고정 / 비법정 SV_adjusted 5단계"""

from .config import CATEGORIES, LEGAL_CODES, NEGATIVE_CODE, category_weight
from .sensitivity_ref import sensitivity_score_for

# ===  risk_weight.py

"""
속성 위험도 가중치 — 비법정(Track B) 산식을 논문1틀 + 논문3 참조표로 재구성.

============================================================================
왜 preference 하드코딩을 버렸나
============================================================================
이전 구현은 카테고리마다 preference(프라이버시 선호 prior)를 손으로 박아두고
PPM-SACG(논문1)의 복합 민감도
        SV_adjusted = w1·SV_entropy + w2·SV_preference
를 계산했다. 그런데 SV_preference는 논문1에서 "사용자 100명 설문 평균"으로
만든 값이라, 설문을 새로 돌리지 않는 우리 프로젝트에서는 근거가 없다.
임의로 채운 preference를 전부 제거하고, 그 자리에 실증 참조표를 넣는다.

============================================================================
채택한 산식 (가산형: 논문1 구조 유지 + 논문3 참조표)
============================================================================
        SV_adjusted = w1·SV_entropy + w2·SV_reference
  · SV_entropy   = 1 − H/H_max      (논문1 Eq.(5) 규약: 작을수록 민감)
  · SV_reference = 1 − score10/10   (논문3 지각 민감도를 논문1 척도로 반전)
                   → 논문3 점수↑(더 민감) 이면 SV_reference↓ 로 맞춘다.

왜 곱셈형(논문4 가중 엔트로피)이 아니라 가산형인가
  논문4의 E/M = (H+log2 w)/(H_max+log2 w) 는 w=1일 때 정확히 1−SV_entropy로
  논문1과 극성이 맞지만(수학적 정합 OK), 실제 데이터에서 **분포가 균등한
  컬럼(H≈H_max)** 은 w가 소거되어 SV가 0(전혀 안 민감)으로 붕괴한다.
  성별(M/F)·소득구간처럼 균등한 범주형이 전부 비민감 처리되는 결함이라,
  민감속성 탐지·등급이 목적인 이 파이프라인에는 부적합하다. 그래서 논문1의
  가산 구조를 유지하되 SV_preference 자리에 논문3 참조표를 넣는다.
  (논문4 곱셈형은 참고용으로 weighted_exposure()에 남겨 둔다.)

방향(극성) 정합
  세 성분(SV_entropy, SV_reference, 최종 SV_adjusted) 모두 논문1 척도
  "작을수록 민감"으로 통일했다. 따라서 grade5/grade4의 등급 경계와 후속
  파이프라인(τ 등)을 손대지 않고 재사용할 수 있다.

Track A (법정 민감정보)는 이 산식을 쓰지 않는다. 종전대로 LEGAL_WEIGHT 고정.
"""





# §4.1.1 5단계 프라이버시 선호 척도 (= 사용자 요구 5단계). 등급 라벨로 재사용.
LEVELS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
# 복합 민감도 → 5단계 경계 (§4.3 등급 경계 0.25/0.5/0.75에 0.125 중점 보정)
DEFAULT_CUTS: Tuple[float, ...] = (0.125, 0.375, 0.625, 0.875)


@dataclass
class RiskWeightConfig:
    """비법정 위험도 산식 설정.

    w1, w2        : SV_adjusted = w1·SV_entropy + w2·SV_reference 의 가중치.
                    논문1 기본값 0.6 / 0.4.
    cuts          : 복합 민감도 → 5단계 양자화 경계.
    drop_penalty  : 원본 대비 엔트로피 감소(가명처리) 반영 강도. 원본이 있을 때만.
    exposure_scale: (참고용) 논문4 곱셈형을 쓸 때의 w 스케일 K. 기본 산식은
                    가산형이라 사용하지 않는다.
    """
    w1: float = 0.6
    w2: float = 0.4
    cuts: Tuple[float, ...] = DEFAULT_CUTS
    drop_penalty: float = 0.5
    exposure_scale: float = 6.0


# ---------------------------------------------------------------------------
# 엔트로피 기본량
# ---------------------------------------------------------------------------
def entropy_bits(values: Sequence) -> Tuple[float, float, int]:
    """(H, H_max, n_unique).  H는 log2 기준 (논문과 동일)."""
    vals = [str(v) for v in values if v is not None and str(v) != "nan"]
    n = len(vals)
    if n <= 1:
        return 0.0, 0.0, len(set(vals))
    counts = Counter(vals)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    h_max = math.log2(len(counts)) if len(counts) > 1 else 0.0
    return float(h), float(h_max), len(counts)


def sv_entropy(values: Sequence) -> float:
    """SV_entropy = 1 − H/H_max ∈ [0,1] (논문1 Eq.(5): 작을수록 민감)."""
    h, h_max, _ = entropy_bits(values)
    if h_max <= 0:
        return 1.0   # 단일값: 구별력 0 → 전혀 안 민감
    return float(np.clip(1.0 - h / h_max, 0.0, 1.0))


def sv_reference(category: str) -> float:
    """SV_reference = 1 − score10/10 (논문3 지각 민감도 → 논문1 척도로 반전).

    논문3 점수가 높을수록(더 민감) SV_reference는 낮아진다(더 민감).
    """
    s10 = sensitivity_score_for(category)
    return float(np.clip(1.0 - s10 / 10.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# (참고용) 논문4 가중 엔트로피 노출량 — 기본 산식엔 미사용
# ---------------------------------------------------------------------------
def score_to_weight(score10: float, scale: float) -> float:
    """논문3 점수(0~10) → 가중치 w ≥ 1 (논문4 곱셈형용). w = 1 + (s/10)·scale."""
    s = float(np.clip(score10, 0.0, 10.0)) / 10.0
    return 1.0 + s * float(scale)


def weighted_exposure(values: Sequence, w: float) -> float:
    """논문4 E(X)/M(X) = (H+log2 w)/(H_max+log2 w). 참고용.

    ⚠️ 분포가 균등(H≈H_max)한 컬럼은 w가 소거되어 1로 붕괴한다. 그래서 기본
    산식(composite_sensitivity)은 이 함수를 쓰지 않는다. 논문4 원식과의
    대조·검증 목적으로만 남겨 둔다.
    """
    h, h_max, _ = entropy_bits(values)
    if h_max <= 0 or w <= 0:
        return 0.0
    log_w = math.log2(w)
    denom = h_max + log_w
    if denom <= 0:
        return 0.0
    return float(np.clip((h + log_w) / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 가명처리 감쇄
# ---------------------------------------------------------------------------
def entropy_drop(original: Sequence, pseudonymized: Sequence) -> float:
    """가명처리로 깎인 정규화 엔트로피(0~1). 원본이 있을 때만 사용."""
    h0, m0, _ = entropy_bits(original)
    h1, m1, _ = entropy_bits(pseudonymized)
    r0 = h0 / m0 if m0 > 0 else 0.0
    r1 = h1 / m1 if m1 > 0 else 0.0
    return float(max(0.0, r0 - r1))


# ---------------------------------------------------------------------------
# 복합 민감도 (가산형)
# ---------------------------------------------------------------------------
def composite_sensitivity(values: Sequence,
                          category: str,
                          cfg: Optional[RiskWeightConfig] = None,
                          original: Optional[Sequence] = None) -> float:
    """SV_adjusted = w1·SV_entropy + w2·SV_reference  (논문1 §4.1.2 Step 3 구조).

    SV_reference는 논문3 참조표 점수를 논문1 척도로 반전한 값.
    원본이 있으면 가명처리로 깎인 엔트로피만큼 민감도를 완화(SV 상향).
    """
    cfg = cfg or RiskWeightConfig()
    e = sv_entropy(values)
    r = sv_reference(category)
    s = cfg.w1 * e + cfg.w2 * r
    if original is not None:
        # 원본 대비 엔트로피가 깎였으면 구별력 손실 → 덜 민감(SV↑).
        s = s + (1.0 - s) * (cfg.drop_penalty * entropy_drop(original, values))
    return float(np.clip(s, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 등급화 (논문1 구간 그대로)
# ---------------------------------------------------------------------------
def grade5(s: float, cuts: Sequence[float] = DEFAULT_CUTS) -> float:
    """복합 민감도 → 5단계 {0, 0.25, 0.5, 0.75, 1}.  (작을수록 민감)"""
    i = int(np.searchsorted(np.asarray(cuts, dtype=float), float(s), side="right"))
    return LEVELS[min(i, len(LEVELS) - 1)]


def grade4(s: float) -> int:
    """§4.3 등급 Level 1~4 (논문1 SV_adjusted 기준, 참고용).

    주의: 논문1 척도에서 SV_adjusted는 '작을수록 민감'이다. 그런데 §4.3
    등급 정의(Level 4 = [0.75,1] = 핵심 민감)는 '클수록 민감'이라 논문 본문
    자체에 방향 불일치가 있다. 여기서는 논문 §4.3 문자 그대로 구현하되,
    실제 위험도 가중치는 sensitivity_level5()(민감도 방향)를 쓴다.
    """
    return int(min(4, max(1, math.floor(float(s) / 0.25) + 1)))


def sensitivity_level5(sv_adjusted: float,
                       cuts: Sequence[float] = DEFAULT_CUTS) -> float:
    """SV_adjusted(작을수록 민감) → 민감도 가중치 5단계(클수록 민감).

    위험도 가중치는 '높을수록 위험'이어야 Track A(LEGAL_WEIGHT=1.5)와 방향이
    맞는다. SV_adjusted는 반대 방향이므로 1−SV_adjusted를 등급화한다.
        SV_adjusted 낮음(=매우 민감)  → 5단계 1.0 (최고)
        SV_adjusted 높음(=덜 민감)     → 5단계 0.0 (최저)
    """
    return grade5(1.0 - float(sv_adjusted), cuts)


def sensitivity_level4(sv_adjusted: float) -> int:
    """SV_adjusted → 민감도 Level 1~4 (클수록 민감). Level 4 = 핵심 민감."""
    return int(min(4, max(1, math.floor((1.0 - float(sv_adjusted)) / 0.25) + 1)))


# ---------------------------------------------------------------------------
# 결과 컨테이너 + 진입점
# ---------------------------------------------------------------------------
@dataclass
class AttributeWeight:
    column: str
    category: str
    track: str                    # "A" | "B" | "-"
    weight: float                 # 최종 위험도 가중치
    basis: str
    h_norm: float = 0.0           # H/H_max
    sv_entropy: float = 0.0       # 1 − H/H_max
    sv_reference: float = 0.0     # 1 − score10/10 (논문3)
    ref_score: float = 0.0        # 논문3 원점수(0~10)
    composite: float = 0.0        # SV_adjusted
    level5: Optional[float] = None
    level4: Optional[int] = None


def attribute_weight(column: str, category: str, values: Sequence,
                     cfg: Optional[RiskWeightConfig] = None,
                     original: Optional[Sequence] = None) -> AttributeWeight:
    cfg = cfg or RiskWeightConfig()
    cat = CATEGORIES.get(category, CATEGORIES[NEGATIVE_CODE])
    h, h_max, _ = entropy_bits(values)
    h_norm = h / h_max if h_max > 0 else 0.0
    e = sv_entropy(values)

    # ---- Track A : 법정 민감정보 → LEGAL_WEIGHT 일괄 고정 -------------------
    if category in LEGAL_CODES:
        w = category_weight(category)
        return AttributeWeight(
            column=column, category=category, track="A", weight=float(w),
            basis=f"법정({cat.basis}) → 법정 가중치 {w} 고정",
            h_norm=h_norm, sv_entropy=e, sv_reference=0.0, ref_score=0.0,
            composite=1.0)

    if category == NEGATIVE_CODE:
        return AttributeWeight(column, category, "-", 0.0,
                               "비민감 판정 → 가중치 0", h_norm, e, 0.0, 0.0,
                               0.0, 0.0, 1)

    # ---- Track B : 비법정 → 논문1틀 가산 + 논문3 참조표 -------------------
    #   SV_adjusted는 '작을수록 민감'(논문1 척도). 위험도 가중치는 '높을수록
    #   위험'이어야 Track A(1.5)와 방향이 맞으므로 sensitivity_level5로 뒤집는다.
    s10 = sensitivity_score_for(category)
    r = sv_reference(category)
    s = composite_sensitivity(values, category, cfg, original)
    lv5 = sensitivity_level5(s, cfg.cuts)     # 민감도 5단계(클수록 민감)
    lv4 = sensitivity_level4(s)               # 민감도 Level 1~4(클수록 민감)
    return AttributeWeight(
        column=column, category=category, track="B", weight=float(lv5),
        basis=(f"비법정 → 논문1틀 가산 · 논문3 참조표(score={s10:.1f}) "
               f"(H/Hmax={h_norm:.3f}, SV_e={e:.3f}, SV_ref={r:.3f}, "
               f"SV_adj={s:.3f} → 민감도 5단계 {lv5}, Level {lv4})"),
        h_norm=h_norm, sv_entropy=e, sv_reference=r, ref_score=s10,
        composite=s, level5=lv5, level4=lv4)


def weight_table(df, decisions, cfg: Optional[RiskWeightConfig] = None,
                 original_df=None):
    rows = []
    for d in decisions:
        vals = df[d.column].dropna().tolist() if d.column in df.columns else []
        orig = (original_df[d.column].dropna().tolist()
                if original_df is not None and d.column in original_df.columns
                else None)
        rows.append(attribute_weight(d.column, d.category, vals, cfg, orig))
    return rows


