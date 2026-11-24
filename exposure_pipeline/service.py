# -*- coding: utf-8 -*-
"""
service.py — 대시보드/서버용 진입점 (import 해서 호출)
======================================================
학습된 rf_model.joblib 을 주입한 ExposurePipeline 로
  탐지(0~5단계) → 내부/외부 위험 행렬 → 데이터셋 H/M/L 판정 → (선택)PNG
까지 한 번에 돌리고, **출력 대신 구조화된 dict/DataFrame** 을 돌려준다.
CLI(run_integrated.py)와 달리 print 하지 않으므로 대시보드가 그대로 렌더링하면 된다.

사용 (대시보드 백엔드)
----------------------
    from exposure_pipeline.service import RiskService
    import pandas as pd

    svc = RiskService(model_path="rf_model.joblib")   # 앱 시작 시 1회 구성
    result = svc.analyze(pd.read_csv("업로드.csv"))    # 요청마다 호출(파이프라인 재사용)

    result["assessment"]["grade"]      # "High" | "Middle" | "Low"
    result["assessment"]["topk"]       # [{"rank","attribute","internal_risk","external_risk"}, ...]
    result["detection"]                # 컬럼별 탐지 결과 (list[dict])
    result["exposure_matrix"]          # 속성×지표 위험 (list[dict])

    pngs = svc.render(result, out_prefix="run1", df=업로드_df)  # 필요하면 그림 파일 생성

[리팩토링] 수정 사항
  · rf_text_mode 기본값 'raw' → 'rf'. 저장 RF 는 profile_to_rf_text 로 학습되는데
    서비스만 'raw'(힌트 토큰 포함 원문)를 넣어 RF 입력 분포가 어긋났다.
    (compas_800 에서 CLI 는 민감 7개, 서비스는 1개로 결과가 달랐음)
  · 노출지표 tier 기본값을 CLI 와 같은 ES.DEFAULT_MIN_TIER 로 통일.
  · 모델 로드·보정은 runtime.build_detector 한 곳을 공유(CLI 와 동일 경로).
  · coverage_floor · 지표 설정(MetricsConfig) 을 인자로 받는다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from . import exposure_stage as ES
from . import metrics as M
from .runtime import DEFAULT_RF_TEXT_MODE, build_detector


def _r4(v) -> Optional[float]:
    """[v9.6] 수치는 소수 4자리, 확정·비적용(NaN)은 None(JSON null)."""
    return None if pd.isna(v) else round(float(v), 4)


def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame → records. NaN 은 None 으로 (NaN 은 JSON 표준이 아님)."""
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


class RiskService:
    """탐지→위험→판정을 한 객체로 감싼 서비스.

    · model_path 의 rf_model.joblib 을 주입해 쓴다(재학습 없음).
    · 파일이 없으면 부트스트랩 학습으로 폴백(경고 문자열을 model_source 에 남김).
    · analyze() 는 여러 번 호출해도 같은 파이프라인을 재사용한다(대시보드에 유리).
    ⚠ 지표 설정은 metrics 모듈 전역을 블록 단위로 바꿔 계산한다. 설정이 다른
      요청을 스레드로 동시에 처리하지 말 것(프로세스 단위로 분리 권장).
    """

    def __init__(self, model_path: Optional[str] = "rf_model.joblib",
                 lang: str = "both", rf_text_mode: str = DEFAULT_RF_TEXT_MODE,
                 calibration_path: Optional[str] = "calibration_holdout.csv",
                 min_tier: Sequence[str] = ES.DEFAULT_MIN_TIER,
                 coverage_floor: float = ES.COVERAGE_FLOOR,
                 metrics_cfg: Optional[M.MetricsConfig] = None,
                 dataset_specific_rules: bool = True,
                 learned_dir: Optional[str] = "learned",
                 lexicon_dirs: Sequence[str] = ()):
        self.pipe, self.calibration, meta = build_detector(
            model_path=model_path, calibration_path=calibration_path,
            lang=lang, rf_text_mode=rf_text_mode,
            dataset_specific_rules=dataset_specific_rules,
            allow_bootstrap_model=True,
            learned_dir=learned_dir, lexicon_dirs=lexicon_dirs)
        self.model_source = meta["model_source"]
        self.min_tier = tuple(min_tier)
        self.coverage_floor = float(coverage_floor)
        self.metrics_cfg = (metrics_cfg or M.MetricsConfig()).validate()

    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame,
                threshold: float = ES.DEFAULT_ATTR_THRESHOLD,
                expected_sensitive: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """한 데이터셋을 분석해 구조화된 결과 dict 를 반환.

        threshold : 속성 통합위험 초과 임계 τ (기본 0.5).
        반환 dict 의 밑줄(_) 키는 시각화용 원본 객체(대시보드에서 무시해도 됨).
        """
        result = self.pipe.run(df)
        det = self.pipe.to_frame(result)

        with M.metrics_config(self.metrics_cfg):
            out = ES.run_from_detection(df, result, min_tier=self.min_tier,
                                        expected_sensitive=expected_sensitive)
        omissions = out["omissions"]
        roles = out["roles"]
        matrix = out["matrix"]

        assessment: Optional[Dict[str, Any]] = None
        assess_obj = None
        if not matrix.empty:
            assess_obj = ES.assess_dataset(matrix, threshold=threshold,
                                           omitted=omissions["attribute"].tolist(),
                                           coverage_floor=self.coverage_floor)
            assessment = {
                "grade": assess_obj.grade,             # High/Middle/Low
                "rho": round(float(assess_obj.rho), 4),
                "threshold": float(assess_obj.threshold),
                "n_exceed": int(assess_obj.n_exceed),
                "n_total": int(assess_obj.n_total),
                "exceed": list(assess_obj.exceed),
                # 내부/외부 분포 위험을 하나로 합치지 않고 각각 보고
                "topk": [{"rank": r, "attribute": a,
                          "internal_risk": _r4(iv),
                          "external_risk": _r4(ev)}
                         for (r, a, iv, ev) in ES.topk_actions_split(assess_obj)],
                # 내부 기준 / 외부 기준 조치 목록을 따로
                "topk_internal": [{"rank": r, "attribute": a,
                                   "internal_risk": _r4(iv),
                                   "external_risk": _r4(ev)}
                                  for (r, a, iv, ev) in ES.topk_actions_split(
                                      assess_obj, order_by="internal")],
                "topk_external": [{"rank": r, "attribute": a,
                                   "internal_risk": _r4(iv),
                                   "external_risk": _r4(ev)}
                                  for (r, a, iv, ev) in ES.topk_actions_split(
                                      assess_obj, order_by="external")],
                "risk_components": _records(
                    assess_obj.components.reset_index()
                    .rename(columns={"index": "attribute"}).round(4)),
                # 산출 상태 보고 (확정, 비적용은 0 으로 대체하지 않음)
                "priority_review_determined": list(assess_obj.determined),
                "excluded_from_ranking": list(assess_obj.unscored),
                "evaluation_omitted": list(assess_obj.omitted),
                "status_counts": {k: {c: int(v) for c, v in row.items()}
                                  for k, row in assess_obj.status_counts.iterrows()},
                "report_text": ES.format_assessment(assess_obj, action_order="split"),
            }

        matrix_records: List[Dict[str, Any]] = (
            [] if matrix.empty
            else _records(matrix.reset_index().rename(columns={"index": "attribute"})))

        return {
            "model_source": self.model_source,
            "settings": {"min_tier": list(self.min_tier),
                         "coverage_floor": self.coverage_floor,
                         "metrics": {k: v for k, v in self.metrics_cfg.__dict__.items()}},
            "detection": det.to_dict(orient="records"),
            "sensitive_columns": det[det.is_sensitive].column.tolist(),
            "exposure_matrix": matrix_records,
            # 평가 누락 (점수 0 아님). expected_sensitive 를 주면 미탐지 속성 포함
            "evaluation_omissions": _records(omissions),
            "roles": {"sensitive": roles.sensitive,
                      "ee": roles.ee, "qi": roles.qi,
                      "qi_dropped": out.get("qi_dropped", [])},
            "assessment": assessment,
            # ---- 시각화/후처리용 원본 (대시보드는 안 써도 됨) ----
            "_result": result, "_matrix": matrix, "_assess": assess_obj,
            "_category_map": roles.category_map,
            "_result_roles": roles,
        }

    # ------------------------------------------------------------------
    def render(self, analysis: Dict[str, Any],
               out_prefix: str = "dashboard",
               df: Optional[pd.DataFrame] = None) -> Dict[str, Optional[str]]:
        """analyze() 결과로 히트맵·위험곡선 PNG 를 만들어 경로를 반환.

        matplotlib 미설치면 해당 값은 None. df 를 넘기면 민감+비민감 맥락
        히트맵(heatmap_allcols)도 만든다.
        """
        matrix = analysis.get("_matrix")
        assess_obj = analysis.get("_assess")
        heat = heat_all = curve = None
        if matrix is not None and not matrix.empty:
            heat = ES.render_heatmap(
                matrix, out_path=f"{out_prefix}_heatmap.png",
                category_map=analysis.get("_category_map"),
                coverage_floor=self.coverage_floor)
            if df is not None:
                m_all = ES.build_all_columns_matrix(
                    df, analysis["_result_roles"], sensitive_matrix=matrix,
                    omitted=[r["attribute"] for r in analysis.get("evaluation_omissions", [])])
                heat_all = ES.render_heatmap(
                    m_all, out_path=f"{out_prefix}_heatmap_allcols.png",
                    category_map=analysis.get("_category_map"),
                    coverage_floor=self.coverage_floor)
            if assess_obj is not None:
                curve = ES.render_risk_curve(
                    assess_obj, out_path=f"{out_prefix}_risk_curve.png")
        return {"heatmap": heat, "heatmap_allcols": heat_all,
                "risk_curve": curve}
