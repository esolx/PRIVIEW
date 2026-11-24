"""
Julia 런타임 초기화 및 CorrectMatch.jl 바인딩.

이 모듈을 임포트하는 순간 Julia 세션이 시작되고, CorrectMatch.jl / DataFrames.jl이
설치되어 있지 않으면 자동으로 설치합니다 (최초 1회, 몇 분 소요될 수 있음).

원본 저작권 고지:
# CorrectMatch
# Copyright © 2019 Université catholique de Louvain, UCLouvain
# Copyright © 2019 Imperial College London
# by Luc Rocher, Julien Hendrickx, Yves-Alexandre de Montjoye
# GNU General Public License v3 (또는 그 이후 버전)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from juliacall import Main as jl  # noqa: N813

try:
    jl.seval("using CorrectMatch")
except Exception:
    print("[안내] CorrectMatch.jl이 설치되어 있지 않아 자동으로 설치합니다 (처음 한 번만 실행됨, 몇 분 걸릴 수 있음)...")
    jl.seval('import Pkg; Pkg.add("CorrectMatch")')
    jl.seval("using CorrectMatch")

try:
    jl.seval("using DataFrames")
except Exception:
    jl.seval('import Pkg; Pkg.add("DataFrames")')
    jl.seval("using DataFrames")

jl.seval("using PythonCall")

# CorrectMatch.jl 네임스페이스 (다른 모듈에서 `from .julia_bridge import cm, jl` 형태로 사용)
cm = jl.CorrectMatch


def to_julia_data(arr: np.ndarray | pd.DataFrame):
    """입력 데이터를 적절한 Julia 형식(numpy array 또는 DataFrame)으로 변환."""
    if isinstance(arr, np.ndarray):
        return arr

    if isinstance(arr, pd.DataFrame):
        return jl.DataFrame(jl.PythonCall.pytable(arr))

    raise TypeError("Input data must be a numpy array or a pandas DataFrame.")


def to_integer_array(arr: np.ndarray | pd.DataFrame) -> np.ndarray:
    """pandas DataFrame을 정수 numpy array로 변환 (단순 함수용)."""
    if isinstance(arr, pd.DataFrame):
        result = np.empty((len(arr), len(arr.columns)), dtype=np.int64)
        for i, col in enumerate(arr.columns):
            if arr[col].dtype == "category":
                result[:, i] = arr[col].cat.codes
            elif not pd.api.types.is_numeric_dtype(arr[col]):
                # object dtype뿐 아니라 pandas 3.0+의 새 문자열 dtype(StringDtype/str)까지 포괄
                result[:, i] = pd.Categorical(arr[col]).codes
            else:
                result[:, i] = arr[col].to_numpy()
        return result
    return arr


def series_to_vector(indiv: np.ndarray | pd.Series) -> list | np.ndarray:
    """pandas Series를 Julia에 적합한 형식으로 변환."""
    if isinstance(indiv, pd.Series):
        if pd.api.types.is_numeric_dtype(indiv.dtype):
            return indiv.to_numpy()
        return list(indiv.values)
    return indiv
