"""
연속형 → 이산형 변환 (binning).

원본 CorrectMatch는 이산형 속성만 다룰 수 있으므로, 연속형 속성을 구간화해서
이산형으로 변환한 뒤 core 모듈의 로직을 그대로 태우기 위한 전처리 유틸입니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sturges_bins(n: int) -> int:
    """Sturges' rule로 적정 구간 수를 계산: k = 1 + log2(n) (반올림, 최소 1).

    표본 수(n)만으로 결정되는 일반적인 히스토그램 구간 수 공식.
    사람이 임의로 n_bins를 정하는 대신, 데이터 크기에 근거한 값을 자동으로 씀.
    표본 수가 클수록 구간을 상대적으로 적게 잡는 경향이 있어(로그 증가),
    개인 단위 위험도가 잘 안 드러나고 뭉뚱그려질 수 있음 (n=1000 → 11구간).
    """
    if n <= 1:
        return 1
    return max(1, round(1 + np.log2(n)))


def freedman_diaconis_bins(series: pd.Series) -> int:
    """Freedman-Diaconis rule로 적정 구간 수를 계산.

    구간폭 = 2 × IQR(x) / n^(1/3),  구간 수 = 범위(range) / 구간폭

    Sturges와 달리 데이터의 분산·이상치(IQR)까지 반영해서, 표본이 클수록
    Sturges보다 훨씬 세밀한 구간 수를 주는 경향이 있음 (n=1000에서 보통 Sturges의
    1.5~2배 수준). IQR이 0이면(값이 대부분 동일) 계산이 불가능하므로 Sturges로 대체.
    """
    x = series.dropna().values
    n = len(x)
    if n <= 1:
        return 1

    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return sturges_bins(n)  # IQR=0이면 계산 불가 → Sturges로 대체

    bin_width = 2 * iqr / (n ** (1 / 3))
    if bin_width <= 0:
        return sturges_bins(n)

    data_range = x.max() - x.min()
    return max(1, round(data_range / bin_width))


def auto_bins(series: pd.Series, bin_rule: str = "fd") -> int:
    """bin_rule에 따라 적정 구간 수를 자동 계산.

    bin_rule:
        - "sturges": 표본 수만 반영 (k = 1 + log2(n))
        - "fd" (Freedman-Diaconis): 분산·이상치(IQR)까지 반영, 대체로 더 세밀한 구간 제공
    """
    if bin_rule == "sturges":
        return sturges_bins(series.dropna().shape[0])
    if bin_rule == "fd":
        return freedman_diaconis_bins(series)
    raise ValueError(f"지원하지 않는 bin_rule입니다: {bin_rule} ('sturges' 또는 'fd'만 지원)")


def discretize_column(
    series: pd.Series,
    method: str = "quantile",
    n_bins: int | None = None,
    custom_edges: list[float] | None = None,
    bin_rule: str = "fd",
) -> pd.Series:
    """단일 연속형 컬럼을 지정된 방식으로 구간화하여 이산형(카테고리) 시리즈로 반환.

    method:
        - "quantile"    : 등빈도(equal-frequency) 구간화.
        - "equal_width" : 등간격(equal-width) 구간화.
        - "custom"      : custom_edges로 직접 지정한 경계값 사용.

    n_bins: 구간 개수. None이면 bin_rule에 따라 자동 계산합니다 (method="custom"에서는 사용되지 않음).
    bin_rule: n_bins가 None일 때 사용할 자동 계산 규칙. "sturges"(기본) 또는 "fd"(Freedman-Diaconis).
    """
    if method == "custom":
        if custom_edges is None:
            raise ValueError("method='custom'이면 custom_edges를 반드시 지정해야 합니다.")
        binned = pd.cut(series, bins=custom_edges)
    else:
        effective_n_bins = n_bins if n_bins is not None else auto_bins(series, bin_rule)
        if method == "quantile":
            binned = pd.qcut(series, q=effective_n_bins, duplicates="drop")
        elif method == "equal_width":
            binned = pd.cut(series, bins=effective_n_bins)
        else:
            raise ValueError(f"지원하지 않는 method입니다: {method}")

    # 구간(Interval)을 문자열 레이블의 category dtype으로 변환
    # (to_integer_array가 category/object dtype을 코드화해서 처리할 수 있도록)
    return binned.astype(str).astype("category")


def detect_continuous_columns(
    df: pd.DataFrame,
    unique_ratio_threshold: float = 0.05,
    min_unique: int = 20,
) -> list[str]:
    """DataFrame에서 연속형으로 추정되는 컬럼을 자동으로 탐지.

    판단 기준:
        1. dtype이 숫자형(int/float)이어야 함 — 문자열/카테고리형/불리언은 애초에 이산형으로 간주.
        2. float dtype이면 그 자체로 연속형으로 판단 (소수점 값은 정확한 재현이 거의 불가능하므로).
        3. int dtype이면, 고유값 개수가 min_unique 이상이면서
           전체 대비 고유값 비율이 unique_ratio_threshold를 넘을 때만 연속형으로 판단
           (예: 0/1 플래그나 등급처럼 정수형이지만 사실상 이산형인 컬럼을 걸러내기 위함).

    Args:
        unique_ratio_threshold: 고유값 개수 / 전체 행 수 비율의 최소 기준값 (기본 5%).
        min_unique: 연속형으로 판단하기 위한 최소 고유값 개수 (기본 20개).

    Returns:
        연속형으로 판단된 컬럼명 리스트.
    """
    continuous_cols = []
    n_rows = len(df)

    for col in df.columns:
        series = df[col]

        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue  # 문자열/카테고리/불리언 → 이산형으로 간주하고 건너뜀

        n_unique = series.nunique(dropna=True)

        if pd.api.types.is_float_dtype(series):
            continuous_cols.append(col)
        elif n_unique >= min_unique and (n_unique / n_rows) >= unique_ratio_threshold:
            continuous_cols.append(col)
        # 그 외 정수형(예: 자녀 수, 등급)은 이산형으로 간주하고 그대로 둠

    return continuous_cols


def discretize_dataframe(
    df: pd.DataFrame,
    continuous_cols: list[str] | None = None,
    method: str = "quantile",
    n_bins: int | None = None,
    custom_edges: dict[str, list[float]] | None = None,
    unique_ratio_threshold: float = 0.05,
    min_unique: int = 20,
    bin_rule: str = "fd",
) -> pd.DataFrame:
    """DataFrame 내 연속형 컬럼들을 일괄 이산화하고, 나머지 컬럼은 그대로 반환.

    continuous_cols를 지정하지 않으면(None) detect_continuous_columns로 자동 탐지합니다.
    n_bins를 지정하지 않으면(None) bin_rule에 따라 컬럼별로 자동 계산합니다
    ("sturges": 표본 수만 반영 / "fd": Freedman-Diaconis, 분산·이상치까지 반영해 더 세밀함).
    """
    result = df.copy()
    custom_edges = custom_edges or {}

    if continuous_cols is None:
        continuous_cols = detect_continuous_columns(
            result,
            unique_ratio_threshold=unique_ratio_threshold,
            min_unique=min_unique,
        )
        if continuous_cols:
            print(f"[자동 탐지] 연속형으로 판단된 컬럼: {continuous_cols}")
        else:
            print("[자동 탐지] 연속형으로 판단된 컬럼 없음 (모든 컬럼을 이산형으로 처리)")

    for col in continuous_cols:
        if col not in result.columns:
            raise KeyError(f"'{col}' 컬럼이 DataFrame에 없습니다.")

        edges = custom_edges.get(col) if method == "custom" else None
        result[col] = discretize_column(
            result[col],
            method=method,
            n_bins=n_bins,
            custom_edges=edges,
            bin_rule=bin_rule,
        )
        if n_bins is None and method != "custom":
            applied = auto_bins(df[col], bin_rule)
            rule_label = "Sturges' rule" if bin_rule == "sturges" else "Freedman-Diaconis rule"
            print(
                f"[{rule_label} 자동 적용] '{col}' → 구간 수 {applied}개 "
                f"(n={df[col].dropna().shape[0]})"
            )

    return result
