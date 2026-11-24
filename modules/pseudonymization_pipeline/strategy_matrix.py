# -*- coding: utf-8 -*-
"""위험축 × 속성유형 전략 추천 매트릭스."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .categorical import detect_hierarchical_structure
from .models import AXIS_LABEL, AXIS_PRIORITY, AttrType, AttributeRisk, RiskLevel, StrategyResult, Technique, _LEVEL_ORDER

# 각 셀은 (기법, 설명) 후보 리스트 — search_best_strategy()가 후보끼리
# 유용성(정보손실) + 안전마진을 비교해서 최종 기법을 채택한다. 후보가 1개뿐인
# 셀은 비교할 게 없어 그 1개가 그대로 채택된다(기존 단일-기법 동작과 동일).
STRATEGY_MATRIX = {
    ("reid", AttrType.CATEGORICAL): [
        (Technique.GENERALIZATION, "상위범주(ClustEm4Ano)"),
        (Technique.LOCAL_SUPPRESSION, "희귀값(등가클래스 작은) 셀만 억제"),
        (Technique.PARTIAL_MASK, "값 뒷부분 마스킹 → 앞부분(상위 정보)만 유지"),
    ],
    ("reid", AttrType.NUMERIC): [
        (Technique.BINNING, "분포 특성(왜도/첨도/표준편차)에 따라 5종 중 자동 선택"),
        (Technique.RANK_SWAP, "인접 순위 범위 내 값 교환"),
        (Technique.TOP_BOTTOM_CODING, "분포 꼬리 극단값을 분위수 임계값으로 대체"),
    ],
    ("reid", AttrType.DATE): [
        (Technique.DATE_TRUNCATE, "주/월/분기/연 단위 일반화"),
        (Technique.DATE_JITTER, "레코드별 독립 ±일수 교란"),
        (Technique.TOP_BOTTOM_CODING, "극단적으로 이르거나 늦은 날짜를 임계일로 묶음"),
    ],

    ("aggregation", AttrType.CATEGORICAL): [
        (Technique.DECORRELATION, "속성값 일부 순열 교환 → QI-민감속성 상관관계 완화"),
        (Technique.PRAM, "Invariant PRAM → 분포 유지하며 값 확률적 교란, 결합관계 약화"),
    ],
    ("aggregation", AttrType.NUMERIC): [
        (Technique.DECORRELATION, "속성값 일부 순열 교환 → QI-민감속성 상관관계 완화"),
    ],
    ("aggregation", AttrType.DATE): [
        (Technique.DECORRELATION, "날짜값 일부 순열 교환 → QI-민감속성 상관관계 완화"),
        (Technique.DATE_JITTER, "레코드·컬럼별 독립 지터 → 날짜 간 간격 교란, 조합 패턴 약화"),
    ],

    ("exposure_dist", AttrType.CATEGORICAL): [
        (Technique.GENERALIZATION, "상위범주"),
        (Technique.PARTIAL_MASK, "값 뒷부분 마스킹"),
    ],
    ("exposure_dist", AttrType.NUMERIC): [
        (Technique.MICROAGGREGATION, "k개 그룹 평균"),
        (Technique.BINNING, "분포 특성 기반 5종 자동 선택 구간화"),
        (Technique.TOP_BOTTOM_CODING, "분포 꼬리 극단값을 분위수 임계값으로 대체"),
    ],
    ("exposure_dist", AttrType.DATE): [
        (Technique.DATE_TRUNCATE, "주/월/분기/연 단위 일반화"),
        (Technique.TOP_BOTTOM_CODING, "극단적으로 이르거나 늦은 날짜를 임계일로 묶음"),
    ],

    ("exposure_link", AttrType.CATEGORICAL): [
        (Technique.GENERALIZATION, "상위범주 (EE 위험 높음+불필요시 억제)"),
        (Technique.SWAP, "매칭키 속성 순열 교환"),
        (Technique.LOCAL_SUPPRESSION, "희귀값(등가클래스 작은) 셀만 억제"),
    ],
    ("exposure_link", AttrType.NUMERIC): [
        (Technique.MICROAGGREGATION, "k개 그룹 평균"),
        (Technique.BINNING, "분포 특성 기반 5종 자동 선택 구간화"),
        (Technique.TOP_BOTTOM_CODING, "분포 꼬리 극단값을 분위수 임계값으로 대체"),
    ],
    ("exposure_link", AttrType.DATE): [
        (Technique.GENERALIZATION, "월/연 단위 일반화"),
        (Technique.TOP_BOTTOM_CODING, "극단적으로 이르거나 늦은 날짜를 임계일로 묶음"),
    ],
}


def recommend_strategy_candidates(
    attr: str,
    risk: AttributeRisk,
    attr_type: AttrType,
    series: Optional[pd.Series] = None,
    dictionary_column: bool = False,
    region_hint: bool = False,
    target_level=None,
) -> list[StrategyResult]:
    """최고 위험등급인 축들의 (axis, attr_type) 셀 후보를 합쳐서 StrategyResult로 반환한다.

    - 단일 위험: 그 축 셀의 후보 그대로.
    - 다중 위험(최고 등급 축이 2개 이상): 해당 축 셀들의 후보 합집합(중복 기법은 1개로).
      search_strategy()가 원래 H였던 축 '전부'가 M 이하로 내려가야 달성으로 치므로,
      한 축만 해결하는 기법은 탐색 단계에서 자연스럽게 탈락한다. 각 후보의 detail에
      어느 축 셀에서 왔는지 기록한다.

    오버라이드(후보 1개로 확정, 비교 없음), 위에서부터 먼저 걸리는 것 적용:
      - 목적외사용(purpose_fit=False) → 삭제
      - 최고 등급이 L → 처리없음
      - 목적상 불필요(needed_for_purpose=False) → 전체 억제. 위험축·속성유형과 무관.
        목적상 필요 없는 컬럼을 굳이 복잡하게 처리할 이유가 없고, 억제는 모든 축을 해소한다.
      - 범주형 + 고유값 2개 이하(series가 주어진 경우) → PRAM

    부분마스킹은 조건부 후보다 — 준식별자에서 부분마스킹이 일반화로서 의미를
    가지려면 값의 앞부분이 상위 개념이어야 하므로(주소, 계층 코드), 아래 경우엔
    후보에서 뺀다:
      - series가 주어지지 않음 (구조를 확인할 수 없음 → 안전하게 제외)
      - dictionary_column=True (질병·직업처럼 사전 기반 일반화가 있는 컬럼)
      - detect_hierarchical_structure()가 계층 구조를 찾지 못함
    region_hint=True면 지역 컬럼으로 보고 주소 판정의 행정구역 접미사 조건을 생략한다.

    [v24] target_level을 주면 '최고 등급 축'이 아니라 '목표보다 높은 축 전부'의 셀 후보를 합친다.
    (목표 L에서 재식별 H + 집계 M인 속성은 두 축 셀의 후보를 모두 비교) None이면 v23과 같다."""
    # 4-5. 목적외사용 -> 삭제만 적용 (이미 파이프라인 상위에서 처리되지만 방어적으로 남겨둠)
    if not risk.purpose_fit:
        return [StrategyResult(attr, attr_type, "purpose", RiskLevel.H, Technique.DELETE,
                                "전체 삭제", "목적에 부합하지 않는 속성")]

    axis, level = risk.dominant_axis()
    if level == RiskLevel.L:
        return [StrategyResult(attr, attr_type, axis, level, Technique.NONE, "-", "위험도 낮음, 재처리 불필요")]

    top_axes = [a for a in AXIS_PRIORITY if getattr(risk, a) == level]
    if target_level is not None:
        t = _LEVEL_ORDER[getattr(target_level, "value", str(target_level))]
        above = [a for a in AXIS_PRIORITY if _LEVEL_ORDER[getattr(risk, a).value] > t]
        top_axes = above or top_axes

    # 목적상 불필요 → 위험축, 속성유형 무관 전체 억제 (저카디널리티 PRAM보다 우선)
    if not risk.needed_for_purpose:
        return [StrategyResult(attr, attr_type, axis, level, Technique.SUPPRESSION,
                                "목적상 불필요 → 전체 억제",
                                f"목적상 불필요 컬럼 (H축: {' / '.join(AXIS_LABEL.get(a, a) for a in top_axes)})")]

    # 저카디널리티(고유값≤2) 범주형 → PRAM 고정 (위험축 무관 정책 오버라이드)
    if attr_type == AttrType.CATEGORICAL and series is not None and series.nunique() <= 2:
        return [StrategyResult(attr, attr_type, axis, level, Technique.PRAM, "저카디널리티(고유값≤2) 고정")]

    # 부분마스킹 활성화 여부는 컬럼 단위로 한 번만 판정
    mask_detail = None
    if series is not None and not dictionary_column:
        kind, info = detect_hierarchical_structure(series, region_hint=region_hint)
        if kind is not None:
            mask_detail = (f"부분마스킹 후보 활성화 — {'주소형' if kind == 'address' else '계층코드형'} "
                           f"구조 감지({info['reason']})")

    merged: dict = {}   # technique -> {"note", "sources"} (삽입 순서 = 축 우선순위 순)
    for a in top_axes:
        for tech, note in STRATEGY_MATRIX[(a, attr_type)]:
            if tech == Technique.PARTIAL_MASK and mask_detail is None:
                continue
            entry = merged.setdefault(tech, {"note": note, "sources": []})
            entry["sources"].append(AXIS_LABEL.get(a, a))

    candidates = []
    for tech, entry in merged.items():
        parts = []
        if len(top_axes) > 1:
            parts.append(f"다중위험 후보 출처: {' / '.join(entry['sources'])} 셀")
        if tech == Technique.PARTIAL_MASK:
            parts.append(mask_detail)
        # intensity(note)는 파이프라인에서 실제 강도 설명으로 덮어써지므로 근거는 detail에 둔다
        candidates.append(StrategyResult(attr, attr_type, axis, level, tech, entry["note"], "; ".join(parts)))
    return candidates


def recommend_strategy(attr: str, risk: AttributeRisk, attr_type: AttrType) -> StrategyResult:
    """단일-후보 호환용. 셀에 후보가 여러 개면 첫 번째(기존 기본 기법)를 반환한다.
    실제 후보 비교가 필요하면 recommend_strategy_candidates()를 쓸 것."""
    return recommend_strategy_candidates(attr, risk, attr_type)[0]
