# -*- coding: utf-8 -*-
"""재평가 전용 집계 안심 기준 완화: 처음 평가는 그대로, 재평가에서만 우연 수준 + 완화폭."""
import unittest

import pandas as pd

import app
from modules.aggregation_risk_system.config import AggregationConfig
from modules.aggregation_risk_system.risk_classifier import RiskClassifier


def _analyzer(fixed):
    from modules.aggregation_risk_system.analyzer import AggregationRiskAnalyzer
    cfg = AggregationConfig(fixed_thresholds=fixed)
    a = AggregationRiskAnalyzer.__new__(AggregationRiskAnalyzer)
    a.config = cfg
    a.risk_classifier = RiskClassifier(cfg)
    y = [0] * 60 + [1] * 40
    a.y_train, a.y_test = pd.Series(y[:80]), pd.Series(y[80:] [:10] + [1] * 10)
    a._set_random_baseline()
    return a


class SafeMargin(unittest.TestCase):
    def test_initial_assessment_has_no_margin(self):
        a = _analyzer(None)
        self.assertEqual(a.reassess_safe_margin, 0.0)

    def test_reassessment_raises_safe_cutoff(self):
        base = {"__random_baseline__": 0.54}
        self.assertAlmostEqual(_analyzer(base).risk_classifier.random_baseline, 0.54)
        relaxed = _analyzer(app._reassess_thresholds(base))
        self.assertAlmostEqual(relaxed.risk_classifier.random_baseline, 0.54 + app.REASSESS_SAFE_MARGIN)

    def test_reassess_thresholds_keeps_original(self):
        base = {"__random_baseline__": 0.54}
        app._reassess_thresholds(base)
        self.assertNotIn("__reassess_safe_margin__", base)
        self.assertIsNone(app._reassess_thresholds(None))

    def test_label_uses_safe_cutoff(self):
        clf = RiskClassifier(AggregationConfig())
        clf.random_baseline = 0.59
        self.assertEqual(clf.classify_with_cached_thresholds([0.58, 0.60, 0.70], 0.68, 0.62),
                         ["안심", "중간", "고위험"])


if __name__ == "__main__":
    unittest.main()
