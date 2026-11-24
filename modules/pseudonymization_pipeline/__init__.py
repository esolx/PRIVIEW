# -*- coding: utf-8 -*-
"""
가명처리 전략 추천 및 재적용 파이프라인

흐름:
  1. 원본 데이터셋 + 가명처리 데이터셋 + 속성별 위험도(재식별/집계/노출-분포/노출-외부연결, H/M/L) 입력
  2. H로 판정된 속성만 대상으로 선정 (매칭키, 목적외사용 컬럼은 제외)
  3. 원본 데이터셋 기준으로 속성 유형(수치형/범주형/날짜형) 판별
  4. 위험축 × 속성유형 매트릭스로 처리 기법 추천
  5. 강도 사다리(약→강)를 순서대로 시도하면서 재평가 함수로 실제 위험도가 M 이하로
     내려갔는지 확인, 달성한 첫 단계(=정보손실 최소)를 채택
  6. 처리 결과를 가명처리 데이터셋에 반영, 결과 리포트 생성

이 모듈은 위험도 산출 자체는 하지 않는다. 재식별/집계/노출 위험평가는 외부 모듈의
결과값(RiskLevel)을 입력으로 받고, 재평가(5단계)도 reassess_fn으로 외부 함수를
주입받는 구조다. 기본값은 테스트용 스텁(_demo_reassess).

빠른 사용법:
    from pseudonymization_pipeline import run_pipeline, AttributeRisk, RiskLevel, PlanDoc, print_report

    risk_report = {"age": AttributeRisk(reid=RiskLevel.H)}
    result_df, report = run_pipeline(original_df, pseudo_df, risk_report, domain_mapping={})
    print_report(report)
"""
from .models import (
    RiskLevel, AttrType, Technique, AXIS_LABEL, AXIS_PRIORITY,
    AttributeRisk, PlanDoc, StrategyResult, ProcessingReport,
)
from .dictionaries import (
    ALIASES, EXTRA_DISEASES,
    load_kcd8_mapping,
    load_region_mapping_ko, load_region_mapping_en, load_combined_region_mapping,
    load_occupation_mapping_ko, load_occupation_mapping_en, load_combined_occupation_mapping,
)
from .attribute_type import detect_attr_type
from .strategy_matrix import STRATEGY_MATRIX, recommend_strategy, recommend_strategy_candidates
from .categorical import (
    apply_invariant_pram, cluster_categorical_clustem4ano, generalize_categorical_series,
    local_suppress_categorical, permute_values, partial_mask, detect_hierarchical_structure,
    label_cluster, default_embedding_fn,
    DISEASE_COLUMN_KEYWORDS, OCCUPATION_COLUMN_KEYWORDS, REGION_COLUMN_KEYWORDS,
    is_disease_column, is_occupation_column, is_region_column,
)
from .numeric import (
    select_binning_strategy_by_shape, compare_binning_strategies, BINNING_FUNCS,
    binning_log_max, binning_equal_frequency, binning_kmeans, binning_decision_tree, binning_density_based,
    microaggregate, rank_swap, top_bottom_code,
)
from .date_processing import date_truncate, date_jitter, date_top_bottom_code
from .utility import utility_il1, utility_ncp_taxonomy, utility_ncp_range, change_rate
from .apply_strategy import apply_strategy, compute_utility
from .reassessment import _demo_reassess, format_protection
from .search import (
    STRENGTH_LADDER, search_strategy, search_best_strategy, composite_score,
    set_base_seed, get_base_seed, trial_seed, get_current_trial,
)
from .pipeline import run_pipeline
from .report import print_report

__all__ = [
    "RiskLevel", "AttrType", "Technique", "AXIS_LABEL", "AXIS_PRIORITY",
    "AttributeRisk", "PlanDoc", "StrategyResult", "ProcessingReport",
    "ALIASES", "EXTRA_DISEASES", "load_kcd8_mapping",
    "load_region_mapping_ko", "load_region_mapping_en", "load_combined_region_mapping",
    "load_occupation_mapping_ko", "load_occupation_mapping_en", "load_combined_occupation_mapping",
    "detect_attr_type",
    "STRATEGY_MATRIX", "recommend_strategy", "recommend_strategy_candidates",
    "apply_invariant_pram", "cluster_categorical_clustem4ano", "generalize_categorical_series",
    "local_suppress_categorical", "permute_values", "partial_mask", "detect_hierarchical_structure",
    "label_cluster", "default_embedding_fn",
    "DISEASE_COLUMN_KEYWORDS", "OCCUPATION_COLUMN_KEYWORDS", "REGION_COLUMN_KEYWORDS",
    "is_disease_column", "is_occupation_column", "is_region_column",
    "select_binning_strategy_by_shape", "compare_binning_strategies", "BINNING_FUNCS",
    "binning_log_max", "binning_equal_frequency", "binning_kmeans", "binning_decision_tree", "binning_density_based",
    "microaggregate", "rank_swap", "top_bottom_code",
    "date_truncate", "date_jitter", "date_top_bottom_code",
    "utility_il1", "utility_ncp_taxonomy", "utility_ncp_range", "change_rate",
    "apply_strategy", "compute_utility",
    "format_protection", "STRENGTH_LADDER", "search_strategy", "search_best_strategy", "composite_score",
    "set_base_seed", "get_base_seed", "trial_seed", "get_current_trial",
    "run_pipeline", "print_report",
]
