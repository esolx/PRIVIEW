# -*- coding: utf-8 -*-
"""
위험도 구간화 (이중 기준).

- 변곡점 기반: 스플라인 평활화 후 2차 미분 최댓값 지점을 전이점으로
- 변화율 기반: 1차 차분·기울기·이동평균·곡률 4개 지표의 전이점 중앙값
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from .config import AggregationConfig


class RiskClassifier:
    """
    변곡점 기반 + 변화율 기반 두 가지 기준으로 위험도 구간화
    """
    LABELS = ["고위험", "위험", "중간", "안심"]

    def __init__(self, config: AggregationConfig):
        self.random_baseline = config.random_baseline
        self.spline_smoothing = config.spline_smoothing
        self.zone_ratio = config.inflection_zone_ratio
        self.zone_window = config.inflection_zone_window
        self.zone_window_ratio = getattr(config, "inflection_zone_window_ratio", None)
        self.zone_fallback_ratio = config.inflection_zone_fallback_ratio
        self.grad_first_zone_frac = config.gradient_first_zone_frac
        self.grad_second_zone_frac = config.gradient_second_zone_frac
        self.grad_rolling_window = config.gradient_rolling_window
        self.fallback_q_high = config.fallback_quantile_high
        self.fallback_q_mid = config.fallback_quantile_mid

    # ── 변곡점 기반 ─────────────────────────────────────
    def inflection_based(self, accuracies: np.ndarray, return_thresholds: bool = False):
        """
        정확도 내림차순 정렬 후 스플라인 2차미분 최댓값을 변곡점으로 구간 분류
        Critical(고위험) / High(위험) / Middle(중간) / Low(안심)

        return_thresholds=True면 (labels, threshold_high, threshold_mid)를 반환한다.
        threshold_high/mid는 전이점 인덱스(t1, t2)가 아니라 그 지점의 실제 정확도
        값(sorted_acc[t1], sorted_acc[t2])이다 — 인덱스는 배열 구성(콤보 집합)이
        바뀌면 의미가 달라지지만, 정확도 값 자체는 캐시해서 재사용할 수 있다
        (classify_with_cached_thresholds 참고).
        """
        n = len(accuracies)
        sorted_idx = np.argsort(accuracies)[::-1]
        sorted_acc = accuracies[sorted_idx]
        indices = np.arange(n)

        # 스플라인 평활화
        try:
            spline = UnivariateSpline(indices, sorted_acc, s=self.spline_smoothing)
        except Exception:
            # fallback: quantile 기반
            labels = self._quantile_fallback(accuracies)
            if return_thresholds:
                th_high, th_mid = self._quantile_fallback_thresholds(accuracies)
                return labels, th_high, th_mid
            return labels

        # 두 전이점 탐색 (각 구간에서 2차미분 최대)
        # 반경을 조합 수에 비례하게 (zone_window_ratio). None이면 v24처럼 고정 폭
        window = (max(1, int(round(n * self.zone_window_ratio)))
                  if self.zone_window_ratio else self.zone_window)
        third = int(n * self.zone_ratio)
        # [v25 버그 수정] 탐색 구간은 '정렬 순위'여야 한다. v24는 sorted_idx[...](원래 조합 번호)를
        # 넣어서, 스플라인을 조합 번호 위치에서 미분하고 그 번호를 순위처럼 써서 경계가 엉뚱한 곳에 잡혔다.
        zone1 = np.arange(max(0, third - window), min(n, third + window))
        zone2 = np.arange(max(0, 2*third - window), min(n, 2*third + window))

        def strongest_inflection(zone_idx):
            if len(zone_idx) == 0:
                return int(n * self.zone_fallback_ratio)
            d2 = np.abs(spline.derivative(n=2)(zone_idx.astype(float)))
            return int(zone_idx[np.argmax(d2)])

        inf1 = strongest_inflection(zone1)
        inf2 = strongest_inflection(zone2)
        t1, t2 = sorted([inf1, inf2])

        # 라벨 부여 (원본 순서 기준)
        labels = ["안심"] * n
        for rank, orig_i in enumerate(sorted_idx):
            acc = accuracies[orig_i]
            if acc <= self.random_baseline:
                labels[orig_i] = "안심"
            elif rank <= t1:
                labels[orig_i] = "고위험"
            elif rank <= t2:
                labels[orig_i] = "위험"
            else:
                labels[orig_i] = "중간"

        if return_thresholds:
            # t1/t2는 정렬배열 인덱스이므로, 그 위치의 실제 정확도 값을 threshold로 반환
            th_high = float(sorted_acc[t1])
            th_mid = float(sorted_acc[t2])
            return labels, th_high, th_mid
        return labels

    # ── 변화율 기반 ─────────────────────────────────────
    def gradient_based(self, accuracies: np.ndarray, return_thresholds: bool = False):
        """
        4가지 변화율 지표(1차차분/기울기/이동평균/곡률)의
        전이점 중앙값을 경계로 구간 분류

        return_thresholds=True면 (labels, threshold_high, threshold_mid)를 반환한다.
        이 함수의 t1/t2도 inflection_based와 마찬가지로 정렬배열 상의 인덱스라서,
        diff1/grad/roll_diff/curvature 자체(이웃 콤보에 의존하는 값)가 아니라
        그 인덱스에 해당하는 정확도 값(sorted_acc[t1], sorted_acc[t2])만 캐시하면
        재계산 없이 재사용할 수 있다.
        """
        n = len(accuracies)
        sorted_idx = np.argsort(accuracies)[::-1]
        sorted_acc = accuracies[sorted_idx]

        # 4가지 변화율 지표
        def find_transition(metric_arr: np.ndarray, zone_frac: float) -> int:
            zone_end = max(2, int(n * zone_frac))
            if zone_end >= len(metric_arr):
                zone_end = len(metric_arr) - 1
            return int(np.argmax(np.abs(metric_arr[:zone_end])))

        diff1 = np.abs(np.diff(sorted_acc, prepend=sorted_acc[0]))           # 1차 차분
        grad = np.abs(np.gradient(sorted_acc))                                 # 기울기
        roll_diff = (pd.Series(sorted_acc)
                     .rolling(self.grad_rolling_window, center=True).mean()
                     .diff().abs().fillna(0).to_numpy())                       # 이동평균 차분
        # 곡률 근사: |y''| / (1+y'^2)^(3/2)
        d1 = np.gradient(sorted_acc)
        d2 = np.gradient(d1)
        curvature = np.abs(d2) / (1 + d1**2) ** 1.5

        metrics = [diff1, grad, roll_diff, curvature]
        t1_candidates, t2_candidates = [], []

        for m in metrics:
            # 상단 1차 구간 내에서 첫 번째 전이
            t1_candidates.append(find_transition(m, self.grad_first_zone_frac))
            # 1차 구간 이후 ~ 2차 구간 폭 만큼에서 두 번째 전이 (인덱스 오프셋)
            offset = max(1, int(n * self.grad_first_zone_frac))
            sub = m[offset:]
            if len(sub) > 0:
                window = max(1, int(n * self.grad_second_zone_frac))
                t2_candidates.append(offset + int(np.argmax(np.abs(sub[:window]))))
            else:
                t2_candidates.append(n // 2)

        t1 = int(np.median(t1_candidates))
        t2 = int(np.median(t2_candidates))
        if t1 >= t2:
            t2 = min(t1 + 1, n - 1)

        labels = ["안심"] * n
        for rank, orig_i in enumerate(sorted_idx):
            acc = accuracies[orig_i]
            if acc <= self.random_baseline:
                labels[orig_i] = "안심"
            elif rank <= t1:
                labels[orig_i] = "고위험"
            elif rank <= t2:
                labels[orig_i] = "위험"
            else:
                labels[orig_i] = "중간"

        if return_thresholds:
            th_high = float(sorted_acc[t1])
            th_mid = float(sorted_acc[t2])
            return labels, th_high, th_mid
        return labels

    # ── 캐시된 threshold로 재계산 없이 즉시 분류 ────────────
    def classify_with_cached_thresholds(
        self, accuracies: np.ndarray, threshold_high: float, threshold_mid: float
    ) -> List[str]:
        """
        inflection_based/gradient_based를 return_thresholds=True로 한 번 돌려서
        얻은 threshold_high/threshold_mid를 그대로 받아, 스플라인/차분 재계산 없이
        accuracy 값을 두 threshold와 비교만 해서 라벨을 매긴다.

        주의: threshold_high >= threshold_mid 여야 하며(원래 t1 <= t2였으므로
        sorted_acc[t1] >= sorted_acc[t2]), 이 함수는 그 순서를 강제하지 않고
        그대로 신뢰한다 — 호출 측(재가명처리 루프)이 baseline에서 뽑은 값을
        그대로 넘기는 걸 전제로 한다.
        """
        labels = []
        for acc in accuracies:
            if acc <= self.random_baseline:
                labels.append("안심")
            elif acc >= threshold_high:
                labels.append("고위험")
            elif acc >= threshold_mid:
                labels.append("위험")
            else:
                labels.append("중간")
        return labels

    def _quantile_fallback_thresholds(self, accuracies: np.ndarray) -> tuple:
        """_quantile_fallback과 동일한 분위수 기준을 threshold 스칼라 쌍으로 반환."""
        high_acc = accuracies[accuracies >= self.random_baseline]
        if len(high_acc) == 0:
            return self.random_baseline, self.random_baseline
        q1 = float(np.quantile(high_acc, self.fallback_q_high))
        q2 = float(np.quantile(high_acc, self.fallback_q_mid))
        return q1, q2

    def _quantile_fallback(self, accuracies: np.ndarray) -> List[str]:
        """스플라인 실패 시 분위수 기반 폴백"""
        labels = []
        high_acc = accuracies[accuracies >= self.random_baseline]
        if len(high_acc) == 0:
            return ["안심"] * len(accuracies)
        q1 = float(np.quantile(high_acc, self.fallback_q_high))
        q2 = float(np.quantile(high_acc, self.fallback_q_mid))
        for acc in accuracies:
            if acc <= self.random_baseline:
                labels.append("안심")
            elif acc >= q1:
                labels.append("고위험")
            elif acc >= q2:
                labels.append("위험")
            else:
                labels.append("중간")
        return labels
