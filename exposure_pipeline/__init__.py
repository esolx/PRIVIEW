"""노출 위험도 평가 파이프라인 — 기능별 모듈 패키지."""
from .pipeline import (ColumnDecision, ExposurePipeline, PipelineConfig,
                       RiskInput, compare_baselines)
# 주의: 같은 이름의 composite_sensitivity 가 두 모듈에 있다.
#   ,  risk_weight.composite_sensitivity : 단일 속성 값 → 5단계 가중치용 민감도
#   ,  metrics.composite_sensitivity     : 민감속성 '집합'의 Combo(쌍 AMI/nMI 결합)
#   패키지 최상위에서는 후자를 combo_sensitivity 로 내보낸다.
from .risk_weight import (attribute_weight, composite_sensitivity,
                          sv_entropy, sv_reference, sensitivity_level5,
                          RiskWeightConfig)  # noqa: F401
from .cassed_lite import (value_rescue, detect_track_b_by_values,
                          gazetteer_score, FinetunedCassedRescuer)  # noqa: F401
from .metrics import (compute_metric_matrix, emd_risk, coverage_risk,
                      composite_sensitivity as combo_sensitivity,
                      external_nmi, granularity_risk, external_exposure,
                      nmi, ami, METRIC_SPECS, METRIC_ORDER, MetricSpec,
                      MetricsConfig, metrics_config,
                      internal_metrics, external_metrics,
                      identifier_like_columns,
                      sensitive_presence_mask, presence_rule,
                      presence_info, PresenceInfo,
                      emd_result, composite_sensitivity_detail, MetricResult,
                      target_na_reason, nmi_defined, external_nmi_detail,
                      pair_valid_count, PAIR_MIN_VALID,
                      STATUS_NUMERIC, STATUS_DETERMINED, STATUS_NA,
                      STATUS_KO)  # noqa: F401
from .exposure_stage import (run_exposure_metrics, run_from_detection,
                             assign_roles, matrix_to_view_json,
                             build_all_columns_matrix,
                             RoleAssignment,
                             describe_metrics, select_metrics,
                             flag_by_reference, render_heatmap,
                             render_metric_views,
                             integrated_risk, risk_components, render_risk_curve,
                             assess_dataset, classify_dataset, exceed_ratio,
                             topk_actions, topk_actions_split, ACTION_ORDERS,
                             format_assessment, DatasetRisk,
                             metric_status, status_counts, determined_attributes,
                             evaluation_omissions, OMITTED_LABEL,
                             value_band, DEFAULT_ATTR_THRESHOLD, RHO_CUTS,
                             DEFAULT_MIN_TIER, COVERAGE_FLOOR,
                             RISK_VALUE_CUTS, DATASET_GRADE_LABELS,
                             BAND_COLORS, BAND_LEGEND_KO)  # noqa: F401
from .runtime import (read_csv_any, load_calibration, build_detector,
                      CalibrationSet)  # noqa: F401
