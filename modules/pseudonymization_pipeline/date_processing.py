# -*- coding: utf-8 -*-
"""날짜형 처리 — 절단(일반화), 지터링, 상하단코딩."""
from __future__ import annotations

import numpy as np
import pandas as pd

def date_truncate(series: pd.Series, unit: str = "M") -> tuple[pd.Series, pd.Series, str]:
    dt = pd.to_datetime(series, errors="coerce")
    period = dt.dt.to_period(unit)
    intervals = period.apply(lambda p: pd.Interval(p.start_time, p.end_time) if pd.notna(p) else np.nan)
    truncated_repr = period.astype(str)
    unit_label = {"Y": "연", "Q": "분기", "M": "월"}.get(unit, unit)
    note = f"날짜 일반화 - {unit_label} 단위 절단(truncate)"
    return truncated_repr, intervals, note


_DATE_FMT = "%Y-%m-%d"


def date_jitter(series: pd.Series, max_days: int = 15) -> tuple[pd.Series, str, dict]:
    """날짜지터링 — 레코드마다 독립적으로 [-max_days, +max_days] 범위의 정수 일수를 더한다.

    속성(컬럼) 단위로 호출되고 레코드마다 난수를 새로 뽑으므로, 같은 레코드의 서로
    다른 날짜 컬럼에는 서로 다른 오프셋이 들어간다 → 날짜 간 간격이 흔들려 날짜
    조합 패턴(집계 위험)도 함께 약해진다. 개인별 공통 오프셋(date shifting)과는
    다르다는 점에 유의 — 그 방식은 간격이 보존돼 집계 위험에는 효과가 없다.

    주의: 컬럼을 하나씩 처리하는 구조라 날짜 컬럼 간 순서 제약(예: 입원일 ≤ 퇴원일)은
    여기서 보장하지 않는다. 필요하면 파이프라인 후처리에서 검사/보정할 것.
    """
    dt = pd.to_datetime(series, errors="coerce")
    offsets = np.random.randint(-max_days, max_days + 1, size=len(dt))
    jittered = dt + pd.to_timedelta(offsets, unit="D")
    out = jittered.dt.strftime(_DATE_FMT)
    out = out.where(dt.notna(), np.nan)
    valid = dt.notna().to_numpy()
    mean_abs = float(np.abs(offsets[valid]).mean()) if valid.any() else 0.0
    note = f"날짜지터링 (레코드별 독립 ±{max_days}일, 평균 이동 {mean_abs:.1f}일)"
    return pd.Series(out.values, index=series.index), note, {"max_days": max_days, "mean_abs_shift": round(mean_abs, 2)}


def date_top_bottom_code(series: pd.Series, tail_q: float = 0.05) -> tuple[pd.Series, str, pd.Series, dict]:
    """날짜 상하단코딩 — 하위/상위 tail_q 분위수보다 이르거나 늦은 날짜를 임계일로 묶는다.
    코딩된 값은 "≤YYYY-MM-DD" / "≥YYYY-MM-DD"로 표기하고, 나머지는 원래 날짜(일 단위) 유지.

    NCP(range) 유용성 계산용 intervals도 함께 반환한다: 코딩된 레코드는
    [최솟값, 하단임계] 또는 [상단임계, 최댓값] 구간, 원값 유지 레코드는 폭 0 구간.
    """
    dt = pd.to_datetime(series, errors="coerce")
    valid = dt.dropna()
    lo = valid.quantile(tail_q).floor("D")
    hi = valid.quantile(1.0 - tail_q).ceil("D")
    dmin, dmax = valid.min(), valid.max()

    out, intervals = [], []
    n_low = n_high = 0
    for d in dt:
        if pd.isna(d):
            out.append(np.nan)
            intervals.append(np.nan)
        elif d < lo:
            out.append(f"≤{lo.strftime(_DATE_FMT)}")
            intervals.append(pd.Interval(dmin, lo, closed="both"))
            n_low += 1
        elif d > hi:
            out.append(f"≥{hi.strftime(_DATE_FMT)}")
            intervals.append(pd.Interval(hi, dmax, closed="both"))
            n_high += 1
        else:
            out.append(d.strftime(_DATE_FMT))
            intervals.append(pd.Interval(d, d, closed="both"))

    note = (f"날짜 상하단코딩 (하위/상위 {tail_q*100:g}% 절단: "
            f"≤{lo.strftime(_DATE_FMT)} {n_low}건, ≥{hi.strftime(_DATE_FMT)} {n_high}건)")
    info = {"low": lo, "high": hi, "n_low": n_low, "n_high": n_high}
    return (pd.Series(out, index=series.index), note,
            pd.Series(intervals, index=series.index, dtype=object), info)
