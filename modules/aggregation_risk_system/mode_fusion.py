# -*- coding: utf-8 -*-
"""최빈 통합 — 이중기준 × 다모델(최대 20개) 라벨을 하나의 최종 등급으로 합침."""

from __future__ import annotations

from collections import Counter
from typing import List

from .config import AggregationConfig


class ModeFusion:
    def __init__(self, config: AggregationConfig):
        self.risk_priority = config.risk_priority
        self.critical_threshold = config.fusion_critical_threshold
        self.majority_threshold = config.fusion_majority_threshold

    def fuse(self, labels: List[str]) -> str:
        """20개 라벨 최빈 통합 (동률 시 상위 위험 우선)"""
        count = Counter(labels)
        total = len(labels)
        high = count.get("고위험", 0)
        med = count.get("위험", 0)
        mid = count.get("중간", 0)
        low = count.get("안심", 0)

        # 유동 기준 통합 (논문 규칙)
        if high >= total * self.critical_threshold:
            return "고위험"
        if (high + med) >= total * self.majority_threshold:
            return "위험"
        if mid >= total * self.majority_threshold:
            return "중간"
        if low >= total * self.majority_threshold:
            return "안심"

        # 최빈값, 동률 시 상위 위험 우선
        max_count = max(count.values())
        candidates = [l for l, c in count.items() if c == max_count]
        return sorted(candidates, key=lambda x: -self.risk_priority.get(x, 0))[0]
