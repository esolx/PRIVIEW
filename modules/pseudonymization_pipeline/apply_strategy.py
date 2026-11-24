# -*- coding: utf-8 -*-
"""속성 하나에 대해 실제로 기법을 적용하는 디스패처."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .models import AttrType, StrategyResult, Technique
from .categorical import (
    apply_invariant_pram, generalize_categorical_series,
    local_suppress_categorical, permute_values, partial_mask,
    is_disease_column, is_occupation_column, is_region_column,
)
from .numeric import BINNING_FUNCS, select_binning_strategy_by_shape, microaggregate, rank_swap, top_bottom_code
from .date_processing import date_truncate, date_jitter, date_top_bottom_code
from .utility import utility_il1, utility_ncp_range, utility_ncp_taxonomy

def apply_strategy(
    series: pd.Series,
    strategy: StrategyResult,
    domain_mapping: dict,
    k_anonymity: int = 3,
    local_suppress_k: int = 3,
    microagg_k: int = 3,
    date_unit: str = "M",
    pram_p_keep: float = 0.90,
    disease_columns: Optional[set] = None,
    n_bins: int = 5,
    occupation_mapping: Optional[dict] = None,
    occupation_columns: Optional[set] = None,
    region_mapping: Optional[dict] = None,
    region_columns: Optional[set] = None,
    mask_ratio: float = 0.5,
    tbc_quantile: float = 0.05,
    jitter_days: int = 15,
) -> tuple[pd.Series, dict]:
    extra: dict = {}
    t, at = strategy.technique, strategy.attr_type

    if t == Technique.SUPPRESSION:
        extra["intensity_note"] = "전체 억제 (대표기호 '*')"
        extra["suppressed"] = True
        return pd.Series(["*"] * len(series), index=series.index), extra

    # 질병/직업/지역 컬럼(이름으로 자동 감지, 또는 각 *_columns로 직접 지정)은
    # '일반화' 후보를 실행할 때만 도메인 사전을 사용한다 (사전매칭 우선 + 미매칭값만
    # 클러스터링). 로컬억제/스와핑/비상관화 후보는 이 분기를 그냥 지나쳐서 원래
    # 병명, 직업명, 지역명 문자열 그대로에 대해 자기 방식대로 처리한다 — 세 기법이
    # 사전 개입 없이 공정하게 비교되도록.
    is_disease = (strategy.attribute in disease_columns) if disease_columns is not None else is_disease_column(strategy.attribute)
    if at == AttrType.CATEGORICAL and is_disease and t == Technique.GENERALIZATION:
        gen, cinfo = generalize_categorical_series(
            series, domain_mapping, k_anonymity=k_anonymity, dict_first=True,
        )
        extra["cluster_info"] = cinfo
        extra["intensity_note"] = f"[질병 컬럼 예외] 사전매칭 우선 + 미매칭값만 K={k_anonymity}-익명성 클러스터링"
        return gen, extra

    is_occupation = (strategy.attribute in occupation_columns) if occupation_columns is not None else is_occupation_column(strategy.attribute)
    if at == AttrType.CATEGORICAL and is_occupation and occupation_mapping and t == Technique.GENERALIZATION:
        gen, cinfo = generalize_categorical_series(
            series, occupation_mapping, k_anonymity=k_anonymity, dict_first=True,
        )
        extra["cluster_info"] = cinfo
        extra["intensity_note"] = f"[직업 컬럼 예외] 사전매칭 우선 + 미매칭값만 K={k_anonymity}-익명성 클러스터링"
        return gen, extra

    is_region = (strategy.attribute in region_columns) if region_columns is not None else is_region_column(strategy.attribute)
    if at == AttrType.CATEGORICAL and is_region and region_mapping and t == Technique.GENERALIZATION:
        gen, cinfo = generalize_categorical_series(
            series, region_mapping, k_anonymity=k_anonymity, dict_first=True,
        )
        extra["cluster_info"] = cinfo
        extra["intensity_note"] = f"[지역 컬럼 예외] 사전매칭 우선 + 미매칭값만 K={k_anonymity}-익명성 클러스터링"
        return gen, extra

    # 고유값 2개 이하(성별 등)인 범주형은 위험축, 추천기법과 무관하게 PRAM으로 대체.
    # pipeline.py가 이제 이 경우 애초에 PRAM 후보 1개로 candidates를 고정해서 넘기지만,
    # apply_strategy()가 다른 경로(직접 호출 등)로 GENERALIZATION/SUPPRESSION 외의
    # 기법을 받아도 방어적으로 동일하게 처리한다.
    if at == AttrType.CATEGORICAL and t != Technique.DELETE and t != Technique.PRAM and series.nunique() <= 2:
        strategy.technique = Technique.PRAM
        pram_series, note, p_info = apply_invariant_pram(series, p_keep=pram_p_keep)
        extra["pram_info"] = p_info
        extra["intensity_note"] = note
        return pram_series, extra

    if t == Technique.PRAM and at == AttrType.CATEGORICAL:
        pram_series, note, p_info = apply_invariant_pram(series, p_keep=pram_p_keep)
        extra["pram_info"] = p_info
        extra["intensity_note"] = note
        return pram_series, extra

    if t == Technique.GENERALIZATION and at == AttrType.CATEGORICAL:
        gen, cinfo = generalize_categorical_series(series, domain_mapping, k_anonymity=k_anonymity)
        extra["cluster_info"] = cinfo
        extra["intensity_note"] = f"K={k_anonymity}-익명성(레코드 수 기준) 클러스터, purity≥0.6 다수결라벨"
        return gen, extra

    if t == Technique.BINNING and at == AttrType.NUMERIC:
        chosen, stats = select_binning_strategy_by_shape(series.values)
        binned, note = BINNING_FUNCS[chosen](series, n_bins=n_bins)
        binned = binned.round(2)
        extra["intensity_note"] = f"{note} [분포특성: skew={stats['skewness']}, kurt={stats['kurtosis']}, std={stats['std']} → {chosen} 채택]"
        return binned, extra

    if t == Technique.DATE_TRUNCATE:
        trunc, intervals, note = date_truncate(series, unit=date_unit)
        extra["intervals"] = intervals
        extra["intensity_note"] = note
        return trunc, extra

    if t == Technique.GENERALIZATION and at == AttrType.DATE:
        # 노출-외부연결 날짜형: "월/연 단위" 일반화. 사다리의 date_unit(W/M/Q/Y)을
        # 월, 연 두 단계로만 올려서 쓴다 (주 단위는 외부 매칭 차단에 너무 약하고,
        # 분기는 표의 '월/연' 정의에 없으므로 연으로 올림).
        unit = {"W": "M", "M": "M", "Q": "Y", "Y": "Y"}.get(date_unit, "Y")
        trunc, intervals, note = date_truncate(series, unit=unit)
        extra["intervals"] = intervals
        extra["intensity_note"] = note + " [월/연 단위 일반화]"
        return trunc, extra

    if t == Technique.DATE_JITTER and at == AttrType.DATE:
        jittered, note, info = date_jitter(series, max_days=jitter_days)
        extra["jitter_info"] = info
        extra["intensity_note"] = note
        return jittered, extra

    if t == Technique.TOP_BOTTOM_CODING and at == AttrType.DATE:
        coded, note, intervals, info = date_top_bottom_code(series, tail_q=tbc_quantile)
        extra["intervals"] = intervals
        extra["tbc_info"] = info
        extra["intensity_note"] = note
        return coded, extra

    if t == Technique.TOP_BOTTOM_CODING and at == AttrType.NUMERIC:
        coded, note, info = top_bottom_code(series, tail_q=tbc_quantile)
        extra["tbc_info"] = info
        extra["intensity_note"] = note
        return coded.round(2), extra

    if t == Technique.PARTIAL_MASK and at == AttrType.CATEGORICAL:
        masked, note, info = partial_mask(series, mask_ratio=mask_ratio)
        extra["cluster_info"] = info["cluster_info"]   # NCP(taxonomy)로 일반화와 같은 기준 비교
        extra["intensity_note"] = note
        return masked, extra

    if t == Technique.MICROAGGREGATION and at == AttrType.NUMERIC:
        agg, note = microaggregate(series, k=microagg_k)
        agg = agg.round(2)
        extra["intensity_note"] = note
        return agg, extra

    if t == Technique.RANK_SWAP and at == AttrType.NUMERIC:
        swapped, note = rank_swap(series, k_anonymity=k_anonymity)
        extra["intensity_note"] = note
        return swapped, extra

    if t == Technique.LOCAL_SUPPRESSION and at == AttrType.CATEGORICAL:
        sup, note, info = local_suppress_categorical(series, k_anonymity=local_suppress_k)
        extra["local_suppression_info"] = info
        extra["intensity_note"] = note
        return sup, extra

    if t in (Technique.SWAP, Technique.DECORRELATION):
        swap_fraction = 1.0 - pram_p_keep  # 기존 사다리의 pram_p_keep(유지율) 재사용: 강도↑ → 스와핑 비율↑
        swapped, note, frac = permute_values(series, swap_fraction=swap_fraction)
        extra["intensity_note"] = note
        extra["swap_fraction"] = frac
        return swapped, extra

    extra["intensity_note"] = "-"
    return series, extra  # Technique.NONE 등 폴백


def compute_utility(attr_type: AttrType, orig_series: pd.Series, processed_series: pd.Series, extra: dict) -> Optional[float]:
    if extra.get("suppressed"):
        # 전체 억제는 속성유형과 무관하게 정보 전부 손실 (수치형에 IL1을 계산하면 '*' 때문에 실패)
        return 1.0
    if "pram_info" in extra:
        # PRAM은 교란확률 theta 자체가 "레코드가 원값에서 바뀔 확률"이므로 그대로 정보손실 지표로 사용
        p_diag = np.mean([extra["pram_info"]["transition_matrix"][i][i] for i in range(len(extra["pram_info"]["categories"]))])
        return round(1.0 - p_diag, 4)
    if "local_suppression_info" in extra:
        # 로컬 억제는 일부 셀만 '*'로 대체 -> 억제된 비율을 그대로 정보손실로 사용
        # (전체 억제처럼 무조건 1.0 폴백을 쓰면 일부만 억제한 이점이 비교에서 사라짐)
        total = len(orig_series)
        return round(extra["local_suppression_info"]["n_suppressed"] / total, 4) if total else 1.0
    if attr_type == AttrType.NUMERIC:
        # 랭크 스와핑, 비상관화도 값 자체는 원본 분포에서 나온 실제값이라 IL1이 그대로 의미 있는 지표.
        # 상하단코딩도 꼬리값만 임계값으로 바뀌므로 IL1로 그 손실만큼 반영된다.
        return utility_il1(orig_series.values, processed_series.values)
    if attr_type == AttrType.DATE and "jitter_info" in extra:
        # 날짜지터링은 값이 구간이 아니라 '다른 날짜'가 되므로 일수 기준 IL1을 쓴다
        # (수치형 교란 기법과 같은 척도).
        o = pd.to_datetime(orig_series, errors="coerce")
        p = pd.to_datetime(processed_series, errors="coerce")
        base = pd.Timestamp("1970-01-01")
        return utility_il1((o - base).dt.days.values, (p - base).dt.days.values)
    if attr_type == AttrType.DATE and "intervals" in extra:
        return utility_ncp_range(orig_series, extra["intervals"])
    if attr_type == AttrType.CATEGORICAL and "cluster_info" in extra:
        return utility_ncp_taxonomy(extra["cluster_info"], n_total_leaves=orig_series.astype(str).nunique())
    if "swap_fraction" in extra:
        # 범주형 스와핑/비상관화 폴백: NCP를 계산할 클러스터 정보가 없으므로
        # 교환된 레코드 비율 자체를 정보손실 근사치로 사용
        return extra["swap_fraction"]
    return None

