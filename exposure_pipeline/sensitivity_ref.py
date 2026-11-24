# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict


"""sensitivity_ref.py — 논문3(Schomakers 2019) 지각 민감도 참조표 — 비법정 전용"""

# (리팩토링) 상위 네임스페이스 전파용 star import 제거 — 사용 심볼 없음

# ===  sensitivity_ref.py

"""
논문3 민감도 참조표 — Schomakers et al. (2019), IJIM 46:142-150.

"Internet users' perceptions of information sensitivity – insights from Germany"
독일 592명이 40개 데이터 유형의 지각된 민감도를 10점 척도로 평가한 실증값.
논문3은 의도적으로 usage context 없이(context-free) 측정했으므로, 상황 무관한
보편 민감도 prior로 쓰기에 적합하다. 국가 간 절대값 차이는 있으나 순위(rank
order)는 매우 유사(독-미 r=.91, 독-브라질 r=.87)하다는 것이 논문의 핵심 주장이라,
순위 기반으로 쓰면 문화 편향이 상당히 완화된다.

주의
  · 이 표는 **비법정(Track B)** 카테고리에만 쓴다. 법정 민감정보는 LEGAL_WEIGHT
    고정이라 점수를 참조하지 않는다.
  · 논문3에 정확히 대응하는 유형이 없는 카테고리는, 가장 가까운 유형의 값 또는
    같은 군(群)의 대표값으로 근사했다. 근사한 항목은 주석에 표시.
  · 논문3에 전혀 근거를 찾을 수 없는 경우 DEFAULT_SCORE로 폴백한다(그리고
    이런 컬럼이 나오면 로그로 남겨 참조표를 보강하도록).
  · 실효 범위는 약 3.0(name of pet) ~ 9.3(passwords). /10 정규화는 문제 없다.

값 출처: 논문3 Fig.2 / Table 3·4의 독일 표본 평균(1~10 스케일).
"""



# 논문3에 없는 카테고리가 나올 때의 폴백. 중앙값 근처(중간 민감도).
DEFAULT_SCORE: float = 5.0


# ---------------------------------------------------------------------------
# 비법정 카테고리 코드 → 논문3 지각 민감도 점수(독일 표본, 0~10)
# ---------------------------------------------------------------------------
#   코드는 config.NON_LEGAL_CATEGORIES 와 일치해야 한다.
#   값 옆 주석은 매핑에 사용한 논문3 원 항목.
SENSITIVITY_SCORE: Dict[str, float] = {
    # 소득, 자산 ← INCOME LEVEL(≈6.0) 과 FINANCIAL ACCOUNT(9.22)의 중간대.
    #   소득 자체는 income level에 가깝게 잡되 자산 포함으로 소폭 상향.
    "INCOME_ASSET":    6.3,
    # 신용, 금융거래 ← CREDIT SCORE(≈7.5), FINANCIAL ACCOUNT NUMBERS(9.22), 
    #   CREDIT CARD(≈8.9) 군. 계좌, 카드 포함하므로 높게.
    "CREDIT_FINANCE":  8.2,
    # 학력, 교육 ← 논문3에 직접 항목 없음. 저민감군(occupation 6.0 미만) 근사.
    "EDUCATION":       4.6,   # 근사
    # 직업, 고용 ← OCCUPATION(≈5.8).
    "OCCUPATION":      5.8,
    # 위치, 이동 ← GPS LOCATION(≈7.4), HOME ADDRESS(≈7.3).
    "LOCATION":        7.4,
    # 통신, 접속기록 ← IP ADDRESS(≈7.6), browsing history(≈6.6), PRIVATE PHONE(7.6).
    #   접속로그, 통화기록 혼합군, IP/전화 기준.
    "COMMUNICATION":   7.2,
    # 소비, 구매이력 ← shopping behavior(≈5.4), INCOME 하위. 저민감.
    "CONSUMPTION":     4.4,
    # 자녀 수 ← NUMBER OF CHILDREN(≈4.3), 논문3의 less-sensitive 군.
    "CHILDREN_COUNT": 4.3,
    # 단말, 온라인 식별자 ← DEVICE/온라인 식별자는 논문3에 직접 없음.
    #   IP ADDRESS(7.6)보다 낮고 광고식별자 성격 → 중상.
    "DEVICE_ONLINE":   6.0,   # 근사
}


def sensitivity_score_for(category: str) -> float:
    """카테고리 코드 → 논문3 민감도 점수(0~10). 없으면 DEFAULT_SCORE.

    로그가 필요하면 호출 측에서 미스 여부를 판단하도록 has_score를 함께 제공.
    """
    return float(SENSITIVITY_SCORE.get(category, DEFAULT_SCORE))


def has_score(category: str) -> bool:
    """논문3 참조표에 직접 매핑이 있는지."""
    return category in SENSITIVITY_SCORE


def missing_categories(categories) -> list:
    """주어진 카테고리들 중 참조표에 없어 폴백을 쓰게 되는 것들."""
    return [c for c in categories if c not in SENSITIVITY_SCORE]


