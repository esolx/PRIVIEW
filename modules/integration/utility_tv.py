# -*- coding: utf-8 -*-
"""재가명처리 결과의 TV 유용성 — 표시 전용.

실험 1(추론 기준 실험)에서 정한 주 조건을 대시보드 재가명처리에 옮겨 쓴다.

  허용 폭(질의별) = 같은 크기 재표본(기본 200회)의 TV 95% 지점 × √2
  판정           = 처리본 TV ≤ 허용 폭

이 결과는 후보를 탈락시키거나 강도를 고르는 데 쓰지 않는다. 재가명처리가 끝난 뒤
처리 전후 비교 화면과 결과 파일에 함께 보여주기만 한다.

대시보드용으로 바꾼 부분
  - 질의: 처리한 컬럼마다 (1) 그 컬럼의 1차원 분포, (2) 그 컬럼 × 집계 타깃의 결합 분포.
    실험 1의 민감속성 S 자리에 집계 타깃을 쓴다. 고정 Q 쌍(속성 두 개 × S) 질의는 없다.
  - 일반화값 균등 배분: 처리값이 원래 값 영역에 없는 값(구간, 상위 범주, '*')이면,
    그 처리값을 받은 레코드들의 원래 값 집합에 질량을 균등하게 나눈다.
  - 수치형은 원본 기준 분위 구간(기본 10개)으로 바꾼 뒤 비교한다. 처리값이 수치면
    같은 구간에 넣고, 수치가 아니면(구간 문자열 등) 위의 균등 배분을 쓴다.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

N_BOOT = 200
QUANTILE = 0.95
FACTOR = math.sqrt(2)
NUMERIC_BINS = 10
NA_TOKEN = "<NA>"


class _Coder:
    """원본 컬럼의 값 영역(토큰)과, 처리값 → 토큰 분포 변환을 맡는다."""

    def __init__(self, original: pd.Series, numeric_bins: int = NUMERIC_BINS):
        self.original = original.reset_index(drop=True)
        numeric = pd.to_numeric(self.original, errors="coerce")
        n_valid = int(self.original.notna().sum())
        self.is_numeric = (n_valid > 0 and int(numeric.notna().sum()) == n_valid
                           and int(numeric.nunique(dropna=True)) > numeric_bins)
        if self.is_numeric:
            _, edges = pd.qcut(numeric.dropna(), q=numeric_bins, retbins=True, duplicates="drop")
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges = edges
        self.tokens = self.encode_original(self.original)
        self.domain = set(self.tokens)

    def _bin(self, values: pd.Series) -> pd.Series:
        codes = pd.cut(values, bins=self.edges, labels=False, include_lowest=True)
        return codes.map(lambda c: NA_TOKEN if pd.isna(c) else f"bin{int(c)}")

    def encode_original(self, series: pd.Series) -> pd.Series:
        series = series.reset_index(drop=True)
        if self.is_numeric:
            out = self._bin(pd.to_numeric(series, errors="coerce"))
            return out.where(series.notna(), NA_TOKEN).astype(str)
        return series.map(lambda v: NA_TOKEN if pd.isna(v) else str(v)).astype(str)

    def weights(self, processed: pd.Series) -> list[dict]:
        """레코드별 {원본 토큰: 질량}. 원래 영역에 없는 처리값은 균등 배분."""
        processed = processed.reset_index(drop=True)
        point = [None] * len(processed)
        if self.is_numeric:
            num = pd.to_numeric(processed, errors="coerce")
            binned = self._bin(num)
            for i, v in enumerate(processed):
                if pd.isna(v):
                    point[i] = NA_TOKEN
                elif not pd.isna(num.iloc[i]):
                    point[i] = binned.iloc[i]
        else:
            for i, v in enumerate(processed):
                key = NA_TOKEN if pd.isna(v) else str(v)
                if key in self.domain:
                    point[i] = key
        # 균등 배분: 처리값별로 그 값을 받은 레코드들의 원래 토큰 집합
        covered: dict = defaultdict(set)
        keys = processed.map(lambda v: NA_TOKEN if pd.isna(v) else str(v))
        for i, p in enumerate(point):
            if p is None:
                covered[keys.iloc[i]].add(self.tokens.iloc[i])
        uniform = {k: {t: 1.0 / len(ts) for t in ts} for k, ts in covered.items()}
        return [{p: 1.0} if p is not None else uniform[keys.iloc[i]] for i, p in enumerate(point)]


def _dist_from_tokens(*token_cols: pd.Series) -> dict:
    keys = list(zip(*[c.tolist() for c in token_cols]))
    n = len(keys)
    out: dict = defaultdict(float)
    for k in keys:
        out[k] += 1.0 / n
    return out


def _dist_from_weights(*weight_cols: list) -> dict:
    n = len(weight_cols[0])
    out: dict = defaultdict(float)
    for i in range(n):
        combos = [((), 1.0)]
        for col in weight_cols:
            combos = [(k + (t,), w * m) for k, w in combos for t, m in col[i].items()]
        for k, w in combos:
            out[k] += w / n
    return out


def _tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def tv_report(original_df: pd.DataFrame, processed_df: pd.DataFrame, columns: list,
              target_col: Optional[str] = None, n_boot: int = N_BOOT, quantile: float = QUANTILE,
              factor: float = FACTOR, seed: int = 0, numeric_bins: int = NUMERIC_BINS) -> dict:
    """처리 컬럼별 TV와 재표본 허용 폭을 계산한다. 판정에는 쓰지 않는 표시용 결과."""
    original_df = original_df.reset_index(drop=True)
    processed_df = processed_df.reset_index(drop=True)
    if len(original_df) != len(processed_df):
        return {"status": "skipped", "message": "처리 전후 행 수가 달라 TV를 계산하지 않았습니다.",
                "queries": [], "columns": []}
    cols = [c for c in columns if c in original_df.columns and c in processed_df.columns]
    use_target = bool(target_col) and target_col in original_df.columns and target_col in processed_df.columns
    coders = {c: _Coder(original_df[c], numeric_bins) for c in set(cols) | ({target_col} if use_target else set())}

    queries = []
    for c in cols:
        queries.append((c, [c]))
        if use_target and c != target_col:
            queries.append((c, [c, target_col]))

    rng = np.random.default_rng(seed)
    n = len(original_df)
    boot_idx = [rng.integers(0, n, n) for _ in range(n_boot)]

    rows = []
    for column, query_cols in queries:
        orig_tokens = [coders[q].tokens for q in query_cols]
        p_orig = _dist_from_tokens(*orig_tokens)
        p_proc = _dist_from_weights(*[coders[q].weights(processed_df[q]) for q in query_cols])
        tv = _tv(p_orig, p_proc)
        # 재표본 TV: 원본 결합 토큰을 정수 코드로 바꿔 bincount로 빠르게 계산
        codes, uniques = pd.factorize(pd.Series(list(zip(*[t.tolist() for t in orig_tokens]))))
        base = np.bincount(codes, minlength=len(uniques)) / n
        boot = [0.5 * float(np.abs(base - np.bincount(codes[idx], minlength=len(uniques)) / n).sum())
                for idx in boot_idx]
        q95 = float(np.quantile(boot, quantile)) if boot else 0.0
        allowed = q95 * factor
        rows.append({"column": column, "query": " × ".join(query_cols), "tv": round(tv, 4),
                     "bootstrap_q95": round(q95, 4), "allowed": round(allowed, 4),
                     "pass": bool(tv <= allowed + 1e-12)})

    by_column = []
    for c in cols:
        qs = [r for r in rows if r["column"] == c]
        by_column.append({"column": c, "max_tv": max(r["tv"] for r in qs),
                          "n_queries": len(qs), "n_pass": sum(r["pass"] for r in qs),
                          "pass": all(r["pass"] for r in qs)})
    return {
        "status": "success", "display_only": True,
        "criterion": f"질의별 TV ≤ 같은 크기 재표본 {n_boot}회의 TV {int(quantile * 100)}% 지점 × √2",
        "note": "표시용 결과입니다. 후보 기법·강도 선택에는 쓰지 않았습니다.",
        "target_col": target_col if use_target else None,
        "queries": rows, "columns": by_column,
        "n_columns": len(by_column), "n_columns_pass": sum(c["pass"] for c in by_column),
    }
