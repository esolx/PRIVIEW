"""
correctmatch_toolkit
=====================

CorrectMatch(Rocher et al., 2019)를 파이썬에서 사용하기 위한 래퍼와,
연속형 속성 이산화(binning) 및 H/M/L 위험 등급 산출 기능을 제공하는 패키지.

모듈 구성:
    julia_bridge        Julia(CorrectMatch.jl) 런타임 초기화 및 데이터 변환
    core                원본 CorrectMatch 로직 (uniqueness, correctness, 모델 피팅/샘플링)
    discretization      연속형 → 이산형 변환 (binning)
    pipeline            이산화 + CorrectMatch 평가 파이프라인 (전체/개별/조합 분석)
    risk_classification 속성별 / 데이터셋 전체 H/M/L 위험 등급 산출
    data_io             데이터셋 파일 로딩
    cli                 명령줄 실행 진입점

원본 저작권 고지:
# CorrectMatch
# Copyright © 2019 Université catholique de Louvain, UCLouvain
# Copyright © 2019 Imperial College London
# by Luc Rocher, Julien Hendrickx, Yves-Alexandre de Montjoye
# GNU General Public License v3 (또는 그 이후 버전)
"""

from .core import (
    correctness,
    fit_model,
    individual_correctness,
    individual_uniqueness,
    sample_model,
    uniqueness,
)
from .data_io import load_dataset
from .discretization import (
    auto_bins,
    detect_continuous_columns,
    discretize_column,
    discretize_dataframe,
    freedman_diaconis_bins,
    sturges_bins,
)
from .pipeline import analyze_all_combinations, assess_individual_risk, assess_reidentification_risk
from .risk_classification import classify_attribute_risk, dataset_level_risk

__all__ = [
    # core
    "uniqueness",
    "correctness",
    "fit_model",
    "sample_model",
    "individual_uniqueness",
    "individual_correctness",
    # discretization
    "sturges_bins",
    "freedman_diaconis_bins",
    "auto_bins",
    "discretize_column",
    "detect_continuous_columns",
    "discretize_dataframe",
    # pipeline
    "assess_reidentification_risk",
    "assess_individual_risk",
    "analyze_all_combinations",
    # risk_classification
    "classify_attribute_risk",
    "dataset_level_risk",
    # data_io
    "load_dataset",
]
