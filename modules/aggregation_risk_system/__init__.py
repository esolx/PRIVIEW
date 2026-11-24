# -*- coding: utf-8 -*-
"""
aggregation_risk_system
=========================

집계 위험도 분석 시스템 (다모델·이중기준 최빈 통합).

비식별 특성 조합(2개 이상 부분집합)으로부터 민감정보(이진분류)를 얼마나 잘
예측해내는지를 10개 ML 모델로 측정하고, 변곡점 기반 + 변화율 기반의 이중 기준을
20개 라벨(모델 10개 × 기준 2개)로 산출한 뒤 최빈 통합하여 조합별 위험 등급과
데이터셋 전체의 H/M/L 등급을 확정합니다.

모듈 구성:
    config          전체 파이프라인 파라미터 (AggregationConfig)
    utils           공통 유틸 (타임스탬프, CSV 저장)
    preprocessing   범주형(문자열) 컬럼 자동 감지 및 조합 단위 One-Hot 인코딩
    models          ML 모델 팩토리 (ModelFactory)
    risk_classifier 이중 기준 위험도 구간화 (RiskClassifier)
    mode_fusion     최빈 통합 (ModeFusion)
    visualization   결과 시각화 (분포 산점도/파이차트/Top20/박스플롯)
    analyzer        핵심 분석 엔진 (AggregationRiskAnalyzer)
    cli             명령줄 실행 진입점
"""

from .analyzer import AggregationRiskAnalyzer
from .attribute_risk import classify_attribute_risk
from .config import AggregationConfig
from .mode_fusion import ModeFusion
from .models import ModelFactory
from .preprocessing import convert_date_columns, detect_categorical_columns, detect_date_columns, encode_combo
from .risk_classifier import RiskClassifier
from .utils import now, write_csv

__all__ = [
    "AggregationConfig",
    "classify_attribute_risk",
    "AggregationRiskAnalyzer",
    "ModelFactory",
    "RiskClassifier",
    "ModeFusion",
    "detect_categorical_columns",
    "encode_combo",
    "detect_date_columns",
    "convert_date_columns",
    "now",
    "write_csv",
]
