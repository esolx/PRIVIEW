"""데이터 유형에 맞춰 선택한 컬럼을 재가명처리한다."""
from __future__ import annotations

from typing import Any

import pandas as pd


METHOD_LABELS = {
    "mask": "부분 마스킹",
    "rare_group": "희소 범주 묶기",
    "numeric_bin": "수치 구간화",
    "date_month": "연월 단위로 축소",
}

DIRECT_ID_HINTS = (
    "name", "이름", "성명", "phone", "전화", "휴대", "email", "메일",
    "address", "주소", "resident", "주민", "passport", "여권", "account", "계좌",
    "id", "번호",
)
DATE_HINTS = ("date", "일자", "날짜", "생년", "birth", "가입일", "등록일")


def suggest_method(series: pd.Series, column_name: str) -> str:
    """컬럼명과 자료형을 기준으로 기본 처리 방법을 제안한다."""
    lowered = str(column_name).lower()
    if any(hint in lowered for hint in DATE_HINTS) or pd.api.types.is_datetime64_any_dtype(series):
        return "date_month"
    if any(hint in lowered for hint in DIRECT_ID_HINTS):
        return "mask"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric_bin"
    return "rare_group" if series.nunique(dropna=True) <= max(30, len(series) * 0.2) else "mask"


def _mask_value(value: Any) -> Any:
    if pd.isna(value):
        return value
    text = str(value)
    if "@" in text:
        local, domain = text.split("@", 1)
        return (local[:1] + "***" if local else "***") + "@" + domain
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 7:
        return "*" * max(4, len(text) - 4) + text[-4:]
    if len(text) <= 2:
        return text[:1] + "*"
    return text[:1] + "*" * (len(text) - 2) + text[-1:]


def _numeric_bins(series: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    unique_count = int(numeric.nunique(dropna=True))
    if unique_count < 2:
        return series.astype("string")
    q = max(2, min(int(bins), unique_count))
    binned = pd.qcut(numeric, q=q, duplicates="drop")
    result = binned.astype("string")
    return result.where(numeric.notna(), pd.NA)


def _date_to_month(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    result = parsed.dt.strftime("%Y-%m").astype("string")
    original = series.astype("string")
    return result.where(parsed.notna(), original)


def process_dataframe(df: pd.DataFrame, strategies: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """선택한 전략을 적용하고 컬럼별 처리 요약을 반환한다."""
    output = df.copy()
    summary: list[dict[str, Any]] = []
    for strategy in strategies:
        column = str(strategy.get("column", ""))
        method = str(strategy.get("method", ""))
        parameter = strategy.get("parameter")
        if column not in output.columns:
            raise ValueError(f"컬럼 '{column}'을 데이터셋에서 찾을 수 없습니다.")
        if method not in METHOD_LABELS:
            raise ValueError(f"'{column}'의 처리 방법을 확인해 주세요.")

        original = output[column].copy()
        if method == "mask":
            output[column] = original.map(_mask_value)
        elif method == "rare_group":
            min_count = max(2, int(parameter or 5))
            counts = original.value_counts(dropna=True)
            rare_values = set(counts[counts < min_count].index.tolist())
            output[column] = original.map(lambda value: "기타" if pd.notna(value) and value in rare_values else value)
        elif method == "numeric_bin":
            output[column] = _numeric_bins(original, int(parameter or 5))
        elif method == "date_month":
            output[column] = _date_to_month(original)

        before = original.astype("string").fillna("<NA>")
        after = output[column].astype("string").fillna("<NA>")
        changed = int((before != after).sum())
        summary.append({
            "column": column,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "changed_rows": changed,
            "before_unique": int(original.nunique(dropna=True)),
            "after_unique": int(output[column].nunique(dropna=True)),
        })
    return output, summary
