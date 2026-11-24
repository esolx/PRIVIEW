# -*- coding: utf-8 -*-
"""
범주형(문자열) 특성 인코딩.

문자열(object)/카테고리(category)/불리언 dtype 컬럼을 범주형으로 자동 감지하고,
조합(combination) 단위로 인코딩. 저카디널리티는 One-Hot, 고카디널리티는
빈도 인코딩으로 자동 분기 (encode_combo 참고). 전체 데이터셋을 미리 통째로
인코딩하지 않는 이유는, 조합 탐색 로직이 원본 컬럼명(예: "city")을 기준으로
동작하기 때문 — 미리 인코딩해버리면 "city"가 "city_Seoul", "city_Busan"
등 여러 컬럼으로 쪼개져서 조합 매칭이 꺠진다.

날짜형(YYYY-MM-DD, YYYY-MM 등) 문자열 컬럼은 그대로 두면 범주형으로 오인되어
시간적 순서·간격 정보가 사라지므로(빈도 인코딩은 "몇 번 등장했는지"만 보고
"언제인지"는 반영하지 못함), detect_date_columns()로 먼저 걸러내 연속형
(해당 컬럼의 최솟값 대비 경과일수)으로 변환한 뒤 나머지만 범주형 감지 대상으로
넘긴다.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd

_DATE_PATTERN = re.compile(r"^\d{4}-\d{1,2}(-\d{1,2})?$")  # YYYY-MM 또는 YYYY-MM-DD


def detect_date_columns(df: pd.DataFrame, feature_cols: List[str], min_match_ratio: float = 0.95) -> List[str]:
    """문자열 컬럼 중 날짜(YYYY-MM-DD, YYYY-MM) 형식인 컬럼을 자동 감지.

    두 단계로 판별한다:
    1) 정규식으로 "YYYY-MM" 또는 "YYYY-MM-DD" 패턴에 맞는 값의 비율을 먼저 거른다
       (임의의 숫자·문자열이 우연히 datetime으로 파싱되는 오탐을 막기 위함).
    2) 패턴 매칭 비율이 min_match_ratio(기본 95%) 이상이면 pd.to_datetime으로
       실제 파싱까지 재확인한다.

    이미 숫자 코드로 인코딩된 이산형이나 일반 범주형 문자열(예: "Seoul")은
    이 패턴에 맞지 않으므로 감지되지 않는다.
    """
    date_cols = []
    for col in feature_cols:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            continue
        pattern_match_ratio = non_null.str.match(_DATE_PATTERN).mean()
        if pattern_match_ratio < min_match_ratio:
            continue
        parsed = pd.to_datetime(non_null, errors="coerce")
        parse_success_ratio = parsed.notna().mean()
        if parse_success_ratio >= min_match_ratio:
            date_cols.append(col)
    return date_cols


def convert_date_columns(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """날짜형 컬럼을 "해당 컬럼 내 최솟값 대비 경과일수"인 연속형(float)으로 변환.

    컬럼명과 조합 매칭 대상은 그대로 유지하고 값만 바뀐다 — 그래야 이미 만들어진
    조합 탐색 로직(원본 컬럼명 기준)이 그대로 동작한다. 파싱에 실패한 값은
    NaN이 되며, 이후 파이프라인의 기존 dropna() 단계에서 해당 행이 자연스럽게
    제외된다(날짜 변환 단계에서 별도로 행을 지우지 않음).
    """
    result = df.copy()
    for col in date_cols:
        parsed = pd.to_datetime(result[col], errors="coerce")
        min_date = parsed.min()
        result[col] = (parsed - min_date).dt.days.astype(float)
    return result


def detect_categorical_columns(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """문자열(object)/카테고리(category)/불리언 dtype 컬럼을 범주형으로 자동 판단.
.
    """
    cat_cols = []
    for col in feature_cols:
        series = df[col]
        # object dtype(구버전 pandas 문자열)뿐 아니라 pandas 2.x/3.x의
        # StringDtype, arrow-backed str dtype까지 포괄해야 문자열 컬럼을 놓치지 않음
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        ):
            cat_cols.append(col)
    return cat_cols


def encode_combo(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: List[str],
    high_cardinality_threshold: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """특성 조합 하나(X_train/X_test 서브셋)에 대해 범주형 컬럼을 인코딩.

    카테고리 종류가 high_cardinality_threshold 이하면 One-Hot 인코딩(정보 손실 없음),
    그보다 많으면(고카디널리티) 빈도 인코딩(각 카테고리를 train 내 등장 비율 숫자로
    치환)으로 처리

    train 데이터 기준으로 만든 인코딩을 test에 그대로 적용합니다:
    - One-Hot: test에만 있는 새 카테고리 → 버려짐 / train에만 있던 카테고리 → 0으로 채워짐
    - 빈도 인코딩: test에만 있는 새 카테고리(train에서 못 본 값) → 0으로 채워짐

    조합 안에 범주형 컬럼이 하나도 없으면 원본을 그대로 반환합니다(연속형/이미
    숫자 코드인 이산형만 있는 조합은 기존과 동일하게 동작).
    """
    cat_in_combo = [c for c in categorical_cols if c in X_train.columns]
    if not cat_in_combo:
        return X_train, X_test

    onehot_cols = [c for c in cat_in_combo if X_train[c].nunique() <= high_cardinality_threshold]
    freq_cols = [c for c in cat_in_combo if c not in onehot_cols]

    X_tr = X_train.copy()
    X_te = X_test.copy()

    # ── 고카디널리티 컬럼: 빈도 인코딩 ──
    for c in freq_cols:
        freq_map = X_tr[c].value_counts(normalize=True)
        X_tr[c] = X_tr[c].map(freq_map).astype(float)
        X_te[c] = X_te[c].map(freq_map).fillna(0.0).astype(float)

    # ── 저카디널리티 컬럼: One-Hot 인코딩 ──
    if onehot_cols:
        X_tr = pd.get_dummies(X_tr, columns=onehot_cols)
        X_te = pd.get_dummies(X_te, columns=onehot_cols)
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    return X_tr, X_te
