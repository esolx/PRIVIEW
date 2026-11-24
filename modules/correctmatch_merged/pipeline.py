"""
이산화 + CorrectMatch 평가 파이프라인.

discretization 모듈로 연속형 속성을 이산화한 뒤, core 모듈의 CorrectMatch
로직으로 재식별 위험(uniqueness/correctness)을 계산하는 상위 레벨 함수들입니다.
"""

from __future__ import annotations

import itertools

import pandas as pd

from .core import correctness, fit_model, individual_correctness, individual_uniqueness, uniqueness
from .discretization import discretize_dataframe


def assess_reidentification_risk(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    method: str = "quantile",
    n_bins: int | None = None,
    custom_edges: dict[str, list[float]] | None = None,
    bin_rule: str = "fd",
) -> dict:
    """연속형 속성을 이산화한 뒤 uniqueness/correctness를 계산.

    continuous_cols를 지정하지 않으면 어떤 컬럼이 연속형인지 자동으로 판단합니다.
    """
    discretized_df = discretize_dataframe(
        df,
        continuous_cols=continuous_cols,
        method=method,
        n_bins=n_bins,
        custom_edges=custom_edges,
        bin_rule=bin_rule,
    )

    return {
        "discretized_df": discretized_df,
        "uniqueness": uniqueness(discretized_df),
        "correctness": correctness(discretized_df),
    }


def assess_individual_risk(
    df: pd.DataFrame,
    individual_index: int,
    n: int,
    continuous_cols: list[str] | None = None,
    method: str = "quantile",
    n_bins: int | None = None,
    custom_edges: dict[str, list[float]] | None = None,
    bin_rule: str = "fd",
    exact_marginal: bool = True,
) -> dict:
    """특정 레코드 하나에 대해 individual_uniqueness / individual_correctness 계산.

    continuous_cols를 지정하지 않으면 어떤 컬럼이 연속형인지 자동으로 판단합니다.
    """
    discretized_df = discretize_dataframe(
        df,
        continuous_cols=continuous_cols,
        method=method,
        n_bins=n_bins,
        custom_edges=custom_edges,
        bin_rule=bin_rule,
    )

    model = fit_model(discretized_df, exact_marginal=exact_marginal)
    individual = discretized_df.iloc[individual_index]

    return {
        "individual_uniqueness": individual_uniqueness(model, individual, n),
        "individual_correctness": individual_correctness(model, individual, n),
    }


def analyze_all_combinations(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    method: str = "quantile",
    n_bins: int | None = None,
    custom_edges: dict[str, list[float]] | None = None,
    min_size: int = 1,
    max_size: int | None = None,
    columns: list[str] | None = None,
    bin_rule: str = "fd",
    k_anon_k: int = 3,
) -> pd.DataFrame:
    """연속형 속성을 이산화한 뒤, 가능한 모든 속성 조합(부분집합)에 대해
    uniqueness/correctness를 계산해서 표로 반환합니다.

    Args:
        min_size: 분석할 조합의 최소 속성 개수 (기본 1개짜리 단일 속성부터)
        max_size: 분석할 조합의 최대 속성 개수. None이면 전체 속성 개수까지
                  (전체 컬럼 수가 많으면 조합 수가 2^N으로 폭발적으로 늘어나므로 주의)
        columns: 조합 분석에 포함할 컬럼만 지정하고 싶을 때 사용. None이면 전체 컬럼 사용
        k_anon_k: k-익명성 기준 인원 (기본 3명). min_group_size/k_violation_ratio 계산에 사용.

    Returns:
        각 행이 하나의 속성 조합인 DataFrame.
        컬럼: combination(속성 조합 튜플), size(속성 개수), uniqueness, correctness,
        min_group_size(이 조합의 동질집합 중 가장 작은 크기),
        k_violation_ratio(k_anon_k명 미만 동질집합에 속한 레코드 비율, 0~1).
        uniqueness 내림차순으로 정렬되어 반환됨 (위험도가 높은 조합이 위로).
    """
    discretized_df = discretize_dataframe(
        df,
        continuous_cols=continuous_cols,
        method=method,
        n_bins=n_bins,
        custom_edges=custom_edges,
        bin_rule=bin_rule,
    )

    target_cols = columns if columns is not None else list(discretized_df.columns)
    n = len(target_cols)
    max_size = n if max_size is None else min(max_size, n)

    total_combos = sum(
        len(list(itertools.combinations(target_cols, r))) for r in range(min_size, max_size + 1)
    )
    print(f"[조합 분석] 속성 {n}개 중 크기 {min_size}~{max_size} 조합 총 {total_combos}개 계산 시작...")

    rows = []
    for r in range(min_size, max_size + 1):
        for combo in itertools.combinations(target_cols, r):
            subset = discretized_df[list(combo)]
            group_sizes = subset.groupby(list(combo)).size()
            rows.append(
                {
                    "combination": combo,
                    "size": r,
                    "uniqueness": uniqueness(subset),
                    "correctness": correctness(subset),
                    "min_group_size": int(group_sizes.min()),
                    "k_violation_ratio": float(group_sizes[group_sizes < k_anon_k].sum() / len(subset)),
                }
            )

    result_df = (
        pd.DataFrame(rows)
        .sort_values(["uniqueness", "correctness"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return result_df
