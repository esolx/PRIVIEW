# -*- coding: utf-8 -*-
"""위험도/속성유형/전략 관련 공통 타입 정의."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# =============================================================================
# 0-B. 공통 타입 정의
# =============================================================================

class RiskLevel(str, Enum):
    H = "H"
    M = "M"
    L = "L"


_LEVEL_ORDER = {"H": 2, "M": 1, "L": 0}


class AttrType(str, Enum):
    NUMERIC = "수치형"
    CATEGORICAL = "범주형"   # 텍스트형 포함, 문자열 컬럼은 전부 범주형으로 취급
    DATE = "날짜형"


class Technique(str, Enum):
    GENERALIZATION = "일반화"
    BINNING = "구간화"
    SUPPRESSION = "억제"
    MICROAGGREGATION = "미세집계"
    DATE_TRUNCATE = "날짜일반화(절단)"
    PRAM = "사후확률화(PRAM)"
    RANK_SWAP = "랭크 스와핑"
    LOCAL_SUPPRESSION = "로컬 억제"
    SWAP = "스와핑"
    DECORRELATION = "비상관화"
    PARTIAL_MASK = "부분마스킹"
    TOP_BOTTOM_CODING = "상하단코딩"
    DATE_JITTER = "날짜지터링"
    DELETE = "삭제"
    NONE = "처리없음"


AXIS_LABEL = {
    "reid": "재식별",
    "aggregation": "집계",
    "exposure_dist": "노출(분포)",
    "exposure_link": "노출(외부연결)",
}
# 여러 축이 동시에 H면 재식별 > 노출(외부연결) > 노출(분포) > 집계 순으로 지배축 결정
AXIS_PRIORITY = ["reid", "exposure_link", "exposure_dist", "aggregation"]


@dataclass
class AttributeRisk:
    """속성별 위험도 평가 결과. 외부 위험평가 모듈의 산출물을 그대로 담아 입력."""
    reid: RiskLevel = RiskLevel.L
    aggregation: RiskLevel = RiskLevel.L
    exposure_dist: RiskLevel = RiskLevel.L
    exposure_link: RiskLevel = RiskLevel.L
    purpose_fit: bool = True          # False = 목적외사용 -> 무조건 삭제
    needed_for_purpose: bool = True   # 목적상 필요한 컬럼인지. False + H 위험 → 위험축, 속성유형 무관 전체 억제
                                      # (run_pipeline(purpose_columns=...)을 주면 그 목록 기준으로 덮어씀)
    ee_high: bool = False             # 현재 추천 로직에서 사용 안 함 — 목적불필요 억제가 위험축 무관으로
                                      # 바뀌면서 외부연결 전용 억제 조건이 흡수됨. 기존 호출 코드 호환용으로만 유지
    risk_scores: dict = field(default_factory=dict)
    # 축별 연속값(H/M/L 라벨의 근거가 된 원값 그대로 — 정규화하지 않는다).
    # search.py는 원래 H였던 축이 정확히 1개일 때만 이 값을 강도/후보 비교에
    # 쓴다. 채워지지 않은 축이 있으면 그 축은 비교 대상에서 빠지고, 축이 2개
    # 이상 겹치거나(다중위험) 아예 비어있으면 기존 동작(마지막 강도/유용성
    # 최소 선택)으로 자동 폴백한다. 키: "reid", "aggregation", "exposure_dist",
    # "exposure_link".
    errors: dict = field(default_factory=dict)
    # 재평가 중 실패한 축과 오류 내용 {"reid": "RuntimeError: ...", ...}.
    # 실패한 축은 안전한 쪽(H)으로 채우되, 실패했다는 사실은 여기 남겨서
    # '강도를 올려도 H 유지'와 '평가 자체가 실패'를 구분할 수 있게 한다.

    def dominant_axis(self) -> tuple[str, RiskLevel]:
        axes = {a: getattr(self, a) for a in AXIS_PRIORITY}
        max_level = max(axes.values(), key=lambda lv: _LEVEL_ORDER[lv.value])
        for a in AXIS_PRIORITY:  # 우선순위 순으로 최고 등급과 일치하는 첫 축 선택
            if axes[a] == max_level:
                return a, max_level
        return AXIS_PRIORITY[0], max_level

    def h_axis_count(self) -> int:
        """식별/집계/노출 네 축 중 H인 축 개수. 2개 이상이면 다중위험으로 간주."""
        return sum(1 for a in AXIS_PRIORITY if getattr(self, a) == RiskLevel.H)


@dataclass
class PlanDoc:
    """가명처리 결과서(계획서). 매칭키 예외처리 판단에만 사용."""
    matching_keys: set = field(default_factory=set)
    technique_applied: dict = field(default_factory=dict)  # attr -> "record_suppression" | "sampling" | ...


@dataclass
class StrategyResult:
    attribute: str
    attr_type: AttrType
    dominant_axis: str
    dominant_level: RiskLevel
    technique: Technique
    intensity: str = ""
    detail: str = ""


@dataclass
class ProcessingReport:
    deleted_purpose: list = field(default_factory=list)
    matching_key_skipped: list = field(default_factory=list)
    skipped: list = field(default_factory=list)                 # 위험도가 H가 아니라 재처리 안 한 속성
    no_effect: list = field(default_factory=list)               # H지만 모든 후보가 전 단계 무변경 → 기존 가명값 유지
    reprocessed: list = field(default_factory=list)             # list[StrategyResult]
    protection_degree: dict = field(default_factory=dict)       # attr -> "H → M [성공(안전)]" 형태 문자열
    utility_loss: dict = field(default_factory=dict)            # attr -> float
    search_log: dict = field(default_factory=dict)              # attr -> [{"label","utility","new_risk","achieved"}, ...]
    candidate_comparison: dict = field(default_factory=dict)    # attr -> 후보 기법별 [{"technique","ul","achieved","score"}, ...]


