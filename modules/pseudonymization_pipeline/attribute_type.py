# -*- coding: utf-8 -*-
"""속성 유형(수치형/범주형/날짜형) 판별."""
from __future__ import annotations

import pandas as pd

from .models import AttrType

def detect_attr_type(series: pd.Series, date_parse_threshold: float = 0.9) -> AttrType:
    if pd.api.types.is_numeric_dtype(series):
        return AttrType.NUMERIC
    if pd.api.types.is_datetime64_any_dtype(series):
        return AttrType.DATE
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() >= date_parse_threshold:
        return AttrType.DATE
    return AttrType.CATEGORICAL  # 문자열은 텍스트형 별도 판별 없이 전부 범주형


