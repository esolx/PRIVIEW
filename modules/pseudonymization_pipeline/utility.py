# -*- coding: utf-8 -*-
"""유용성(정보손실) 지표 — IL1(수치형), NCP taxonomy(범주형), NCP range(날짜형)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

def utility_il1(x_orig, x_pseudo) -> float:
    """연속형 IL1 = mean(|x - x'|) / (sqrt(2) * S)  (단일 변수 기준)"""
    x_orig = np.asarray(x_orig, dtype=float)
    x_pseudo = np.asarray(x_pseudo, dtype=float)
    s = np.nanstd(x_orig)
    if s == 0:
        return 0.0
    diffs = np.abs(x_orig - x_pseudo)
    return float(np.nanmean(diffs) / (math.sqrt(2) * s))


def utility_ncp_taxonomy(cluster_info: dict, n_total_leaves: int) -> float:
    """범주형/텍스트형: NCP_A(t) = size(u) / |A|, 레코드(멤버) 가중 평균"""
    if n_total_leaves == 0 or not cluster_info:
        return 0.0
    total_members = 0
    weighted = 0.0
    for info in cluster_info.values():
        size_u = info["n_members"]
        weighted += size_u * (size_u / n_total_leaves)
        total_members += size_u
    return weighted / total_members if total_members else 0.0


def utility_ncp_range(series: pd.Series, intervals: pd.Series) -> float:
    """날짜형(구간 일반화): NCP_Ai(t) = (z_i - y_i) / |A_i|"""
    dt = pd.to_datetime(series, errors="coerce")
    domain_days = (dt.max() - dt.min()).days or 1
    widths = [(iv.right - iv.left).days for iv in intervals if isinstance(iv, pd.Interval)]
    if not widths:
        return 0.0
    return float(np.mean(widths) / domain_days)


def change_rate(orig: pd.Series, processed: pd.Series, attr_type) -> float:
    """원본 대비 값이 실제로 바뀐 레코드 비율 (0.0이면 무변경).

    단순 `!=` 비교는 세 가지 경우에 틀리므로 보정한다:
      - 결측: NaN != NaN 이 True라서 무변경인데 '바뀜'으로 셈 → 둘 다 결측이면 같음으로 처리
      - 날짜: 원본은 datetime, 처리값은 'YYYY-MM-DD' 문자열 → 같은 문자열 형식으로 맞춰 비교
      - 수치: 64 vs 64.0, 부동소수 오차 → np.isclose로 비교 ('*' 같은 비수치값은 바뀜으로 처리)
    """
    from .models import AttrType  # 순환 import 방지용 지연 import

    o = pd.Series(orig).reset_index(drop=True)
    p = pd.Series(processed).reset_index(drop=True)
    if len(o) == 0:
        return 0.0
    if len(o) != len(p):
        return 1.0  # 길이가 다르면 비교 불가 — 무변경으로 오판하지 않도록 '전부 바뀜' 처리

    both_na = o.isna().to_numpy() & p.isna().to_numpy()

    if attr_type == AttrType.NUMERIC:
        o_num = pd.to_numeric(o, errors="coerce").to_numpy(dtype=float)
        p_num = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float)
        same = np.isclose(o_num, p_num, equal_nan=False)
    elif attr_type == AttrType.DATE:
        o_str = pd.to_datetime(o, errors="coerce").dt.strftime("%Y-%m-%d")
        p_str = p.astype(object).map(lambda v: v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
        same = (o_str.astype(object).to_numpy() == p_str.astype(object).to_numpy())
    else:
        same = (o.astype(object).map(str).to_numpy() == p.astype(object).map(str).to_numpy())

    changed = ~(same | both_na)
    return float(changed.mean())
