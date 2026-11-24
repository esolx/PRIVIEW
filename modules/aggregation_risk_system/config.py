# -*- coding: utf-8 -*-
"""
집계 위험도 분석 시스템 — 설정(Config).

분석 파이프라인 전체(데이터 로딩, 모델, 위험도 구간화, 최빈 통합)에서 쓰이는
파라미터를 하나의 데이터클래스로 모아둔 모듈입니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AggregationConfig:
    data_file: str = "data/features.csv"
    target_col: str = "target"
    out_dir: str = "results/aggregation/"
    feature_cols: List[str] = field(default_factory=list)  # 비어있으면 타깃 제외 전체
    categorical_cols: List[str] = field(default_factory=list)  # 비어있으면 문자열/카테고리 dtype 자동 감지
    high_cardinality_threshold: int = 10  # 이 값보다 카테고리 종류가 많으면 One-Hot 대신 빈도 인코딩
    min_combo_size: int = 2
    max_combo_size: int = 4
    test_size: float = 0.2
    seed: int = 42
    n_jobs: int = -1
    save_every: int = 100  # 조합 N개마다 중간 저장
    # 위험 임계값 (accuracy 기준 고정)
    random_baseline: float = 0.5  # 무작위 수준 임계값 (random_baseline_mode="fixed"일 때만 그대로 사용)
    # "majority": 평가 데이터에서 다수 클래스만 찍었을 때의 정확도를 우연 수준으로 쓴다
    #       (타깃이 80:20이면 0.8). "fixed": random_baseline 값을 그대로 쓴다 (v24까지)
    random_baseline_mode: str = "majority"
    # 최빈 통합 시 동률 우선순위
    risk_priority: Dict[str, int] = field(default_factory=lambda: {
        "고위험": 3, "위험": 2, "중간": 1, "안심": 0
    })

    # ── 속성별 위험 등급 파라미터 (attribute_risk) ───────────
    # correctmatch_toolkit의 classify_attribute_risk와 동일 기준값 사용
    attribute_top_n: int = 10                # "상위 조합"으로 볼 개수
    attribute_combo_high_cut: float = 0.7    # 상위 조합 중 "거의 항상 등장"으로 볼 비율 (attribute_grade_method="top_n")
    # ── v25: 처리 전 경계 고정 ─────────────────────────────
    # 처리 전후를 같은 자로 재기 위한 설정. 처리 전(1단계) 평가에서 뽑은 모델별 경계를
    # fixed_thresholds로 넘기면 경계를 다시 뽑지 않고 그 값으로 라벨을 붙인다.
    # 형식: {모델명: {"inflection": [high, mid], "gradient": [high, mid]}}
    fixed_thresholds: Optional[Dict] = None
    # True면 경계를 뽑은 뒤에도 순위가 아니라 정확도 값과 경계를 비교해 라벨을 붙인다.
    # (처리 전과 처리 후에 같은 규칙을 쓰기 위함. False면 v24처럼 처리 전은 순위 기준)
    label_by_value: bool = True
    # True면 재가명처리 확정(commit) 때 경계를 갱신하지 않는다 (v24는 갱신)
    freeze_thresholds: bool = True
    # 속성 등급 산출 방식
    #   "label_share": 그 속성이 들어간 조합들의 H/M/L 비율 중 최댓값 (데이터셋 규칙과 같은 방식, 절대 기준)
    #   "top_n":       정확도 상위 N개 조합 등장 비율 (v24까지, 순위 기준)
    attribute_grade_method: str = "label_share"

    # ── 위험도 구간화 파라미터 (RiskClassifier) ─────────────
    spline_smoothing: float = 1.0          # UnivariateSpline 평활화 계수(s)
    inflection_zone_ratio: float = 1 / 3   # 두 전이점 탐색 위치(1/3, 2/3 지점)
    inflection_zone_window: int = 50       # 전이점 탐색 구간 반경(±window, 고정폭) — 아래 비율이 None일 때만 사용
    # 탐색 구간 반경을 조합 수에 비례하게 (반경 = 조합 수 × 비율, 최소 1).
    #       고정 50은 조합이 100개 안팎이면 구간이 전체를 덮어 경계가 최솟값으로 무너진다. None이면 고정폭
    inflection_zone_window_ratio: Optional[float] = 1 / 6
    inflection_zone_fallback_ratio: float = 0.33  # 탐색구간이 빌 때 폴백 위치 비율
    gradient_first_zone_frac: float = 0.25   # 변화율 1차 전이점 탐색 구간(상단 비율)
    gradient_second_zone_frac: float = 0.5   # 변화율 2차 전이점 탐색 구간 폭(비율)
    gradient_rolling_window: int = 3         # 이동평균 차분 윈도우 크기
    fallback_quantile_high: float = 0.75     # 스플라인 실패 시 고위험 컷 분위수
    fallback_quantile_mid: float = 0.50      # 스플라인 실패 시 위험 컷 분위수

    # ── 최빈 통합 파라미터 (ModeFusion) ─────────────────────
    fusion_critical_threshold: float = 0.5   # 고위험 단독 확정 비율 기준
    fusion_majority_threshold: float = 0.6   # 위험/중간/안심 다수결 확정 비율 기준

    # ── 모델 하이퍼파라미터 (ModelFactory) ───────────────────
    n_estimators: int = 100                  # 트리 계열(RF/ExtraTrees/XGB/LGB/CatBoost) 공통
    knn_neighbors: int = 5
    lr_max_iter: int = 1000
    mlp_hidden_layers: Tuple[int, ...] = (64, 32)
    mlp_max_iter: int = 500
    pca_variance_ratio: float = 0.95         # PCA_RandomForest 파이프라인의 설명분산 비율
