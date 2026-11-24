"""
원본 CorrectMatch(__init__.py) 로직.

이산형 다변량 데이터셋에 대한 uniqueness/correctness 계산, Gaussian Copula
모델 피팅 및 샘플링, 개별 레코드 단위 위험도 추정을 제공합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .julia_bridge import cm, jl, series_to_vector, to_integer_array, to_julia_data


def uniqueness(arr: np.ndarray | pd.DataFrame) -> float:
    """이산형 다변량 데이터셋에서 고유 개체 비율(uniqueness)을 계산."""
    arr = to_integer_array(arr)
    return cm.uniqueness(arr)


def correctness(arr: np.ndarray | pd.DataFrame) -> float:
    """이산형 다변량 데이터셋에서 정확히 식별되는 개체 비율(correctness)을 계산."""
    arr = to_integer_array(arr)
    return cm.correctness(arr)


def fit_model(
    arr: np.ndarray | pd.DataFrame,
    *,
    exact_marginal: bool = True,
    adaptative_threshold: int = 100,
    mi_abs_tol: float = 1e-5,
):
    """이산형 다변량 데이터셋에 Gaussian Copula 모델을 피팅하고 Julia 객체를 반환."""
    data = to_julia_data(arr)
    return cm.fit_mle(
        cm.GaussianCopula,
        data,
        exact_marginal=exact_marginal,
        adaptative_threshold=adaptative_threshold,
        mi_abs_tol=mi_abs_tol,
    )


def sample_model(m, size: int) -> np.ndarray | pd.DataFrame:
    """주어진 Gaussian Copula 모델에서 `size`개의 합성 개체 데이터셋을 샘플링."""
    res = jl.rand(m, size)
    if cm.returns_dataframe(m):
        return jl.PythonCall.pytable(res)
    return np.array(res)


def individual_uniqueness(m, indiv: np.ndarray | pd.Series, n: int) -> float:
    """특정 레코드 하나에 대한 개별 uniqueness를 추정."""
    indiv = series_to_vector(indiv)
    return cm.individual_uniqueness(m, indiv, n)


def individual_correctness(m, indiv: np.ndarray | pd.Series, n: int) -> float:
    """특정 레코드 하나에 대한 개별 correctness를 추정."""
    indiv = series_to_vector(indiv)
    return cm.individual_correctness(m, indiv, n)
