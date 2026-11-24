# -*- coding: utf-8 -*-
"""파이프라인 오케스트레이터 — run_pipeline()."""
from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Callable, Optional

import pandas as pd

from .models import AttributeRisk, PlanDoc, ProcessingReport, RiskLevel, Technique, _LEVEL_ORDER
from .attribute_type import detect_attr_type
from .categorical import is_disease_column, is_occupation_column, is_region_column
from .strategy_matrix import recommend_strategy_candidates
from .search import search_best_strategy, as_level
from .reassessment import _demo_reassess, format_protection

def run_pipeline(
    original_df: pd.DataFrame,
    pseudo_df: pd.DataFrame,
    risk_report: dict,                       # {attr: AttributeRisk}  (가명처리 데이터셋 기준 위험평가 결과)
    domain_mapping: dict,                     # 범주형 일반화용 사전 (예: cluster_labeling.load_kcd8_mapping())
    plan_doc: Optional[PlanDoc] = None,       # 매칭키 예외처리 판단용 계획서
    reassess_fn: Optional[Callable[[pd.DataFrame, str], AttributeRisk]] = None,
    commit_fn: Optional[Callable[[pd.DataFrame, str], None]] = None,
    # 속성 하나의 최종 기법/강도가 확정된 직후(다음 속성으로 넘어가기 전) 호출된다.
    # reassess_fn이 캐시(예: 집계위험 threshold 캐시)를 쓰는 경우, 강도 사다리
    # 탐색 중의 "시험 호출"과 달리 이 시점의 결과만 캐시에 영구 반영해야 다음
    # 속성 재평가부터 이번 확정 결과가 정확히 반영된다. reassess_fn이 자체적으로
    # 시험/확정을 구분하지 않는다면 None으로 둬도 기존 동작과 동일하다.
    disease_columns: Optional[set] = None,    # 질병/진단명 컬럼 명시 지정 (None이면 이름으로 자동 감지)
    ladder: Optional[list] = None,            # 강도 사다리 커스텀 지정 (None이면 STRENGTH_LADDER 사용)
    occupation_mapping: Optional[dict] = None,  # 직업분류 사전 (load_combined_occupation_mapping() 등)
    occupation_columns: Optional[set] = None,   # 직업 컬럼 명시 지정 (None이면 이름으로 자동 감지)
    region_mapping: Optional[dict] = None,      # 지역분류 사전 (load_combined_region_mapping() 등)
    region_columns: Optional[set] = None,       # 지역 컬럼 명시 지정 (None이면 이름으로 자동 감지)
    purpose_columns: Optional[set] = None,
    # 이용 목적상 필요한 컬럼 목록. 지정하면 여기 없는 컬럼은 전부 '목적상 불필요'로 보고,
    # H 위험이 있으면 위험축, 속성유형과 무관하게 전체 억제한다(각 AttributeRisk의
    # needed_for_purpose 값을 덮어씀). None이면 속성별 needed_for_purpose 값을 그대로 쓴다.
    # 목적외사용(purpose_fit=False → 삭제)과 매칭키 예외는 이 목록보다 먼저 적용된다.
    target_level=RiskLevel.M,
    # 목표 등급. M(기본)은 v23과 같다 — H 속성만 처리하고 H→M 첫 달성에서 멈춤.
    # L이면 M 속성도 처리 대상이 되고, 목표보다 높은 축 전부가 L이 될 때까지 강도를 올린다.
) -> tuple[pd.DataFrame, ProcessingReport]:
    plan_doc = plan_doc or PlanDoc()
    target_level = as_level(target_level)
    reassess_fn = reassess_fn or _demo_reassess
    report = ProcessingReport()
    result_df = pseudo_df.copy()

    if purpose_columns is not None:
        purpose_columns = set(purpose_columns)
        unknown = purpose_columns - set(original_df.columns)
        if unknown:
            # 오타 등으로 필요한 컬럼이 목록에서 빠지면 그 컬럼이 통째로 억제되므로 경고
            warnings.warn(f"purpose_columns에 원본 데이터셋에 없는 컬럼이 있습니다: {sorted(unknown)}")

    for attr, risk in risk_report.items():
        if attr not in original_df.columns:
            continue

        if purpose_columns is not None:
            risk = replace(risk, needed_for_purpose=(attr in purpose_columns))

        # 9. 매칭키 예외 처리 — record_suppression / sampling 아니면 절대 건드리지 않음
        if attr in plan_doc.matching_keys:
            applied = plan_doc.technique_applied.get(attr, "")
            if applied not in ("record_suppression", "sampling"):
                report.matching_key_skipped.append(attr)
                continue

        # 2-1 / 4-5. 목적외사용 -> 무조건 삭제
        if not risk.purpose_fit:
            if attr in result_df.columns:
                result_df = result_df.drop(columns=[attr])
            report.deleted_purpose.append(attr)
            continue

        axis, level = risk.dominant_axis()
        if _LEVEL_ORDER[level.value] <= _LEVEL_ORDER[target_level.value]:
            # 이미 목표 등급 이하 → 재처리 대상 아님 (목표 M이면 v23처럼 H만 타겟팅)
            report.skipped.append(attr)
            continue

        attr_type = detect_attr_type(original_df[attr])

        # 3. 로컬 삭제 (가명처리 데이터셋 기준)
        if attr in result_df.columns:
            result_df = result_df.drop(columns=[attr])

        # 5. 전략 후보 추천 — 오버라이드(목적불필요 억제, 저카디널리티 PRAM)와 매트릭스 후보
        # 선정은 전부 recommend_strategy_candidates에서 처리. H인 축이 여러 개면 그 축들 셀의
        # 후보 합집합을 비교하고, 원래 H였던 축 전부가 M 이하가 돼야 달성으로 친다.
        # 부분마스킹 조건부 활성화용 컬럼 판별 — apply_strategy()의 도메인 컬럼 판별과 같은 규칙
        is_disease = (attr in disease_columns) if disease_columns is not None else is_disease_column(attr)
        is_occupation = (attr in occupation_columns) if occupation_columns is not None else is_occupation_column(attr)
        is_region = (attr in region_columns) if region_columns is not None else is_region_column(attr)
        candidates = recommend_strategy_candidates(
            attr, risk, attr_type, series=original_df[attr],
            dictionary_column=is_disease or is_occupation, region_hint=is_region,
            target_level=target_level,
        )
        if len(candidates) == 1 and candidates[0].technique == Technique.NONE:
            report.skipped.append(attr)
            continue

        # 6~8. 후보별 강도 사다리 반복탐색(약→강, reassess_fn으로 첫 M달성 지점 확인) +
        # 유용성손실, 안전마진 비교로 최종 기법 채택. 후보가 1개뿐이면 비교 없이 그대로 채택.
        processed_series, extra, ul, new_risk, attempt_log, h_axes, chosen_strategy, all_candidates = search_best_strategy(
            attr, original_df[attr], candidates, domain_mapping, risk,
            reassess_fn, result_df, target_level=target_level,
            disease_columns=disease_columns, ladder=ladder,
            occupation_mapping=occupation_mapping, occupation_columns=occupation_columns,
            region_mapping=region_mapping, region_columns=region_columns,
        )
        if processed_series is None:
            # 모든 후보가 사다리 전 단계에서 무변경 — 원본값을 붙이면 기존 가명처리가
            # 되돌려지므로, 가명처리 데이터셋의 기존 값을 그대로 복원하고 별도 기록한다.
            if attr in pseudo_df.columns:
                result_df[attr] = pseudo_df[attr].values
            report.no_effect.append(attr)
            report.search_log[attr] = attempt_log
            report.candidate_comparison[attr] = all_candidates
            continue

        chosen_strategy.intensity = extra.get("intensity_note", chosen_strategy.intensity)

        # 7. 재부착
        result_df[attr] = processed_series.values

        # attr의 기법/강도가 이번에 확정됨 — reassess_fn이 시험/확정을 구분해
        # 캐시하는 구조라면(예: 집계위험 threshold 캐시), 여기서 딱 한 번만
        # 확정 반영해야 강도 사다리 탐색 중 탈락한 시행값이 캐시에 안 남는다.
        if commit_fn is not None:
            commit_fn(result_df, attr)

        # 유용성(정보손실) — 억제처럼 별도 계산식이 없는 기법은 완전손실(1.0)로 기록
        report.utility_loss[attr] = ul

        # 8. 재평가 결과 -> 보호정도(원래 H였던 축 전부의 H→M 여부) 확인
        report.protection_degree[attr] = format_protection(risk, new_risk, h_axes, target_level)
        report.search_log[attr] = attempt_log
        report.candidate_comparison[attr] = all_candidates

        report.reprocessed.append(chosen_strategy)

    return result_df, report

