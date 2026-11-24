# -*- coding: utf-8 -*-
"""v25 집계 모듈: 변곡점 탐색 구간(순위·비례 폭), 우연 수준(다수 클래스), 처리 전 경계 고정, 라벨 비율 속성 등급."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from modules.aggregation_risk_system.attribute_risk import classify_attribute_risk_by_label_share
from modules.aggregation_risk_system.config import AggregationConfig
from modules.aggregation_risk_system.risk_classifier import RiskClassifier


class Inflection(unittest.TestCase):
    def test_thresholds_are_taken_from_rank_zones(self):
        # 정렬했을 때 1/3, 2/3 지점에서 꺾이는 곡선 — 경계가 그 근처 정확도로 잡혀야 한다
        acc = np.r_[np.linspace(0.90, 0.85, 20), np.linspace(0.75, 0.70, 20), np.linspace(0.60, 0.55, 20)]
        shuffled = np.random.default_rng(0).permutation(acc)
        clf = RiskClassifier(AggregationConfig(random_baseline=0.5))
        _, high, mid = clf.inflection_based(shuffled, return_thresholds=True)
        self.assertGreater(high, 0.69)   # v24 버그(조합 번호를 순위로 씀)에서는 입력 순서에 따라 아무 값
        self.assertGreater(mid, 0.54)
        self.assertGreaterEqual(high, mid)
        # 입력 순서가 바뀌어도 같은 경계 (순위 기준이므로)
        _, high2, mid2 = clf.inflection_based(acc[::-1].copy(), return_thresholds=True)
        self.assertEqual((high, mid), (high2, mid2))

    def test_window_scales_with_combo_count(self):
        clf = RiskClassifier(AggregationConfig())
        self.assertAlmostEqual(clf.zone_window_ratio, 1 / 6)
        clf_fixed = RiskClassifier(AggregationConfig(inflection_zone_window_ratio=None))
        self.assertIsNone(clf_fixed.zone_window_ratio)


class Baseline(unittest.TestCase):
    def _analyzer(self, y, fixed=None, mode="majority"):
        from modules.aggregation_risk_system.analyzer import AggregationRiskAnalyzer
        cfg = AggregationConfig(fixed_thresholds=fixed, random_baseline_mode=mode)
        a = AggregationRiskAnalyzer.__new__(AggregationRiskAnalyzer)
        a.config = cfg
        a.risk_classifier = RiskClassifier(cfg)
        a.y_train = pd.Series(y[:80]); a.y_test = pd.Series(y[80:])
        a._set_random_baseline()
        return a

    def test_majority_class_accuracy(self):
        y = [0] * 64 + [1] * 16 + [0] * 16 + [1] * 4      # 80:20
        self.assertAlmostEqual(self._analyzer(y).risk_classifier.random_baseline, 0.8)

    def test_fixed_mode_keeps_config_value(self):
        y = [0] * 64 + [1] * 16 + [0] * 16 + [1] * 4
        self.assertAlmostEqual(self._analyzer(y, mode="fixed").risk_classifier.random_baseline, 0.5)

    def test_reassessment_uses_before_baseline(self):
        y = [0] * 64 + [1] * 16 + [0] * 16 + [1] * 4
        a = self._analyzer(y, fixed={"__random_baseline__": 0.61})
        self.assertAlmostEqual(a.risk_classifier.random_baseline, 0.61)


class LabelShare(unittest.TestCase):
    def test_grade_is_plurality_of_its_combos(self):
        risk_df = pd.DataFrame({
            "Feature_Combination": ["a+b", "a+c", "b+c", "a+d"],
            "Final_Risk": ["고위험", "위험", "안심", "안심"],
        })
        g = classify_attribute_risk_by_label_share(risk_df).set_index("attribute")
        self.assertEqual(g.loc["a", "grade"], "H")      # 3개 중 H 2
        self.assertEqual(g.loc["b", "grade"], "H")      # H 1, L 1 동률 → 위험 쪽
        self.assertEqual(g.loc["c", "grade"], "H")      # H 1, L 1
        self.assertEqual(g.loc["d", "grade"], "L")      # 위험 조합 없음 → L (v24 상위 N 방식에선 순위로 M 가능)
        self.assertAlmostEqual(g.loc["a", "high_share"], 2 / 3, places=3)


class Evidence(unittest.TestCase):
    def test_aggregation_rows_are_attribute_h_share(self):
        from modules.integration.final_risk import compute_integrated_risk_from_modules
        modules = {"aggregation": {"status": "success", "dataset_grade": "H",
                                   "attribute_grade_method": "label_share",
                                   "attribute_scores": {"a": 0.4, "b": 0.9},
                                   "attribute_grades": {"a": "L", "b": "H"}}}
        comp = compute_integrated_risk_from_modules(modules, pd.DataFrame())["components"][0]
        self.assertEqual(comp["top_attribute"], "b")
        self.assertEqual([r["attribute"] for r in comp["attributes"]], ["b", "a"])
        self.assertIn("H 비율", comp["attributes"][0]["metric_label"])


if __name__ == "__main__":
    unittest.main()
