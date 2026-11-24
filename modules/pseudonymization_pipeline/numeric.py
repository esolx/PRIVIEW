# -*- coding: utf-8 -*-
"""수치형 처리 — 분포특성 기반 구간화 전략 선택(5종) + 미세집계 + 랭크스와핑 + 상하단코딩."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def select_binning_strategy_by_shape(x: np.ndarray) -> tuple[str, dict]:
    """
    Copula 논문 Table 1(임계값 기준표) 그대로 — 위험등급이 아니라 그 속성 값 분포의
    왜도/첨도/표준편차로 5종 구간화 전략 중 하나를 고른다. 조건이 겹칠 수 있어서
    표에 나온 순서(로그빈닝→등빈도→K평균→결정트리→밀도기반)대로 먼저 만족하는
    조건을 채택한다.

    kurtosis는 정규분포=3 기준(첨도, excess 아님)으로 계산 — 등빈도/K평균 조건이
    "Kurtosis < 3(or excess≈0)"으로 정규분포 기준값을 쓰고 있어서 통일함.
    """
    s = pd.Series(np.asarray(x, dtype=float)).dropna()
    skew = float(s.skew())
    kurt = float(s.kurt()) + 3.0  # pandas.kurt()는 excess kurtosis라 +3으로 환산
    std = float(s.std())

    stats = {"skewness": round(skew, 3), "kurtosis": round(kurt, 3), "std": round(std, 3)}

    if abs(skew) > 2 or kurt >= 7:
        return "log_binning", stats
    if std < 0.01:
        return "equal_frequency", stats
    if abs(skew) < 0.5 and kurt < 3:
        return "kmeans", stats
    if std > 0.5 or abs(skew) > 1:
        return "decision_tree", stats
    return "density_based", stats


def compare_binning_strategies(x: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """
    5종 구간화 전략 비교 (참고/정보용). 품질지표 = 구간내분산/전체분산 (낮을수록 좋음).
    실제 채택은 select_binning_strategy_by_shape()의 분포 특성 기준으로 결정된다.
    """
    x = np.asarray(x, dtype=float)
    total_var = np.var(x) if np.var(x) > 0 else 1e-9
    rows = []

    def within_ratio(bin_ids):
        s = pd.Series(x).groupby(bin_ids, observed=True)
        within = s.apply(lambda g: np.var(g) * len(g)).sum() / len(x)
        return within / total_var

    # equal-frequency
    try:
        eq = pd.qcut(x, q=n_bins, duplicates="drop")
        rows.append(("equal_frequency", within_ratio(eq)))
    except Exception:
        pass

    # log-binning
    x_pos = x - x.min() + 1.0
    edges = np.unique(np.logspace(0, np.log10(x_pos.max()), n_bins + 1))
    if len(edges) > 2:
        logb = pd.cut(x_pos, bins=edges, include_lowest=True)
        rows.append(("log_binning", within_ratio(logb)))

    # k-means 기반
    km = KMeans(n_clusters=min(n_bins, len(np.unique(x))), n_init=10, random_state=0)
    km_labels = km.fit_predict(x.reshape(-1, 1))
    rows.append(("kmeans", within_ratio(km_labels)))

    # decision-tree 기반 (분산감소 기준 CART로 근사: x -> x 자기예측, leaf=구간)
    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor(max_leaf_nodes=n_bins, random_state=0)
    tree.fit(x.reshape(-1, 1), x)
    tree_labels = tree.apply(x.reshape(-1, 1))
    rows.append(("decision_tree", within_ratio(tree_labels)))

    # density 기반 (히스토그램 국소최소점 근사)
    hist, edges_h = np.histogram(x, bins=max(20, n_bins * 4))
    # 국소 최소를 경계로 사용, 목표 구간 수에 맞춰 상위 n_bins-1개 최소점 선택
    minima_idx = [i for i in range(1, len(hist) - 1) if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]]
    minima_idx = sorted(minima_idx, key=lambda i: hist[i])[: max(0, n_bins - 1)]
    cut_points = sorted(edges_h[i] for i in minima_idx)
    if cut_points:
        density_edges = [x.min() - 1e-9] + cut_points + [x.max() + 1e-9]
        dens = pd.cut(x, bins=sorted(set(density_edges)), include_lowest=True)
        rows.append(("density_based", within_ratio(dens)))

    df = pd.DataFrame(rows, columns=["strategy", "within_var_ratio"]).sort_values("within_var_ratio")
    return df.reset_index(drop=True)


def binning_log_max(series: pd.Series, n_bins: int = 5) -> tuple[pd.Series, str]:
    """대표값=구간 상단(log-scale) — 극단적 왜도/첨도(꼬리가 아주 긴 분포)에 적합."""
    x = series.astype(float)
    x_shift = x - x.min() + 1.0
    edges = np.unique(np.logspace(0, np.log10(x_shift.max()), n_bins + 1))
    binned = pd.cut(x_shift, bins=edges, include_lowest=True)
    upper = pd.Series([iv.right if pd.notna(iv) else np.nan for iv in binned], index=series.index, dtype=float)
    represented = upper + x.min() - 1.0
    note = f"Log-binning, 대표값=구간상단(Log_max), 구간수={len(edges) - 1}"
    return pd.Series(represented.values, index=series.index), note


def binning_equal_frequency(series: pd.Series, n_bins: int = 5) -> tuple[pd.Series, str]:
    """대표값=구간 상단 — 표준편차가 매우 작은(거의 평평한) 분포에 적합."""
    x = series.astype(float)
    binned = pd.qcut(x, q=n_bins, duplicates="drop")
    upper = pd.Series([iv.right if pd.notna(iv) else np.nan for iv in binned], index=series.index, dtype=float)
    note = f"Equal-frequency, 대표값=구간상단, 구간수={binned.cat.categories.size if hasattr(binned, 'cat') else n_bins}"
    return upper, note


def binning_kmeans(series: pd.Series, n_bins: int = 5) -> tuple[pd.Series, str]:
    """대표값=클러스터 중심(centroid) — 거의 대칭이고 정규분포에 가까운 분포에 적합."""
    x = series.astype(float)
    valid = x.notna()
    km = KMeans(n_clusters=min(n_bins, valid.sum()), n_init=10, random_state=0)
    labels = km.fit_predict(x[valid].values.reshape(-1, 1))
    centers = km.cluster_centers_.ravel()
    represented = pd.Series(np.nan, index=series.index, dtype=float)
    represented[valid] = centers[labels]
    note = f"K-means, 대표값=클러스터중심, 구간수={min(n_bins, valid.sum())}"
    return represented, note


def binning_decision_tree(series: pd.Series, n_bins: int = 5) -> tuple[pd.Series, str]:
    """대표값=리프(leaf) 평균 — 분산이 크고 흩어져 있는 분포에 적합."""
    from sklearn.tree import DecisionTreeRegressor
    x = series.astype(float)
    valid = x.notna()
    tree = DecisionTreeRegressor(max_leaf_nodes=n_bins, random_state=0)
    tree.fit(x[valid].values.reshape(-1, 1), x[valid].values)
    leaf_ids = tree.apply(x[valid].values.reshape(-1, 1))
    leaf_means = pd.Series(x[valid].values, index=leaf_ids).groupby(level=0).mean()
    represented = pd.Series(np.nan, index=series.index, dtype=float)
    represented[valid] = pd.Series(leaf_ids, index=x[valid].index).map(leaf_means).values
    note = f"Decision Tree, 대표값=리프평균, 리프수={len(leaf_means)}"
    return represented, note


def binning_density_based(series: pd.Series, n_bins: int = 5) -> tuple[pd.Series, str]:
    """대표값=구간 평균 — 그 외 나머지(밀집/희소가 섞인 분포)에 대한 기본값.
    히스토그램 국소최소점을 경계로 잡아 밀집 구간을 나눈다."""
    x = series.astype(float)
    valid = x[x.notna()].values
    hist, edges_h = np.histogram(valid, bins=max(20, n_bins * 4))
    minima_idx = [i for i in range(1, len(hist) - 1) if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]]
    minima_idx = sorted(minima_idx, key=lambda i: hist[i])[: max(0, n_bins - 1)]
    cut_points = sorted(edges_h[i] for i in minima_idx)
    if not cut_points:
        cut_points = list(np.quantile(valid, np.linspace(0, 1, n_bins + 1)[1:-1]))
    density_edges = sorted(set([valid.min() - 1e-9] + cut_points + [valid.max() + 1e-9]))
    binned = pd.cut(x, bins=density_edges, include_lowest=True)
    means = x.groupby(binned, observed=True).transform("mean")
    note = f"Density-based, 대표값=구간평균, 구간수={len(density_edges) - 1}"
    return means, note


BINNING_FUNCS = {
    "log_binning": binning_log_max,
    "equal_frequency": binning_equal_frequency,
    "kmeans": binning_kmeans,
    "decision_tree": binning_decision_tree,
    "density_based": binning_density_based,
}


def microaggregate(series: pd.Series, k: int = 3) -> tuple[pd.Series, str]:
    x = series.astype(float).values
    n = len(x)
    order = np.argsort(x)
    result = np.empty(n)
    i = 0
    while i < n:
        group = order[i:i + k]
        if n - (i + k) < k and (i + k) < n:  # 마지막 그룹이 k 미만이면 직전 그룹과 합쳐서 처리
            group = order[i:]
            result[group] = x[group].mean()
            i = n
            continue
        result[group] = x[group].mean()
        i += k
    note = f"미세집계 K={k} (그룹 내 평균값 대체, k-익명성 준수)"
    return pd.Series(result, index=series.index), note


def rank_swap(series: pd.Series, k_anonymity: int = 3) -> tuple[pd.Series, str]:
    """랭크 스와핑 — 정렬 순위 기준 ±k_anonymity 범위 내 무작위 레코드와 값을 교환.
    분포·순서는 거의 보존하면서 개별 레코드-값의 1:1 매칭만 깨뜨린다.
    강도 사다리의 k_anonymity를 그대로 '교환 윈도우 폭'으로 재사용 — 크게 잡을수록
    더 먼 순위끼리도 교환돼 보호수준이 올라간다."""
    vals = series.astype(float).values
    n = len(vals)
    order = np.argsort(vals)          # order[i] = i번째로 작은 값의 원래 위치
    window = max(1, k_anonymity)
    perm = np.arange(n)
    for i in range(n):
        j = min(n - 1, max(0, i + np.random.randint(-window, window + 1)))
        perm[i], perm[j] = perm[j], perm[i]
    new_vals = np.empty(n)
    new_vals[order] = vals[order[perm]]
    note = f"랭크 스와핑 (인접 순위 ±{window} 범위 내 값 교환)"
    return pd.Series(new_vals, index=series.index), note


def top_bottom_code(series: pd.Series, tail_q: float = 0.05) -> tuple[pd.Series, str, dict]:
    """상하단코딩 — 하위 tail_q, 상위 tail_q 분위수 바깥의 값을 각 분위수 값으로 눌러 담는다.
    분포 꼬리의 극단값(소수라서 그 자체로 식별되기 쉬운 값)만 처리하고 가운데는 원값 유지.
    수치형 컬럼이 그대로 수치형으로 남도록 임계값(숫자)으로 대체한다 — 어떤 값이
    코딩됐는지는 반환 info의 low/high 임계값과 건수로 확인할 수 있다."""
    x = series.astype(float)
    lo = float(x.quantile(tail_q))
    hi = float(x.quantile(1.0 - tail_q))
    n_low = int((x < lo).sum())
    n_high = int((x > hi).sum())
    coded = x.clip(lower=lo, upper=hi)
    note = (f"상하단코딩 (하위/상위 {tail_q*100:g}% 절단: ≤{lo:,.2f} → {lo:,.2f} {n_low}건, "
            f"≥{hi:,.2f} → {hi:,.2f} {n_high}건)")
    return coded, note, {"low": lo, "high": hi, "n_low": n_low, "n_high": n_high}
