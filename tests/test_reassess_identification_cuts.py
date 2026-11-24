# -*- coding: utf-8 -*-
"""재평가 전용 식별 조합 등급 기준: 처음 평가는 기본값, 재평가만 REASSESS_ID_COMBO_CUTS."""
import unittest

import importlib.util
from pathlib import Path

import pandas as pd

import app

# 식별 패키지는 불러올 때 Julia가 필요하므로, 등급 계산 파일만 직접 불러온다
_spec = importlib.util.spec_from_file_location(
    "risk_classification", Path(app.__file__).parent / "modules/correctmatch_merged/risk_classification.py")
_rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rc)
classify_attribute_risk = _rc.classify_attribute_risk


def _combo_df():
    # 상위 10개 위험 조합 중 a는 1번(10%), b는 8번(80%) 등장, c는 7번(70%)
    rows = [{"combination": ("a",), "size": 1, "k_violation_ratio": 0.0, "uniqueness": 0.0, "correctness": 0.0},
            {"combination": ("b",), "size": 1, "k_violation_ratio": 0.0, "uniqueness": 0.0, "correctness": 0.0},
            {"combination": ("c",), "size": 1, "k_violation_ratio": 0.0, "uniqueness": 0.0, "correctness": 0.0}]
    combos = [("a", "b")] + [("b", "c")] * 6 + [("b", "d"), ("c", "d"), ("d", "e")]
    for i, combo in enumerate(combos):
        rows.append({"combination": combo, "size": 2, "k_violation_ratio": 0.9 - i * 0.01,
                     "uniqueness": 0.9 - i * 0.01, "correctness": 0.005})
    return pd.DataFrame(rows)


class ReassessIdentificationCuts(unittest.TestCase):
    def grades(self, **cuts):
        df = classify_attribute_risk(_combo_df(), top_n=10, **cuts).set_index("attribute")
        return {a: df.loc[a, "final_grade"] for a in ("a", "b", "c")}

    def test_first_assessment_default_cuts(self):
        self.assertEqual(self.grades(), {"a": "M", "b": "H", "c": "H"})

    def test_reassessment_cuts(self):
        g = self.grades(**app.REASSESS_ID_COMBO_CUTS)
        self.assertEqual(g["a"], "L")   # 10% 등장 → 저위험 (30% 이하)
        self.assertEqual(g["b"], "H")   # 80% → 고위험 유지
        self.assertEqual(g["c"], "M")   # 70% → 중위험

    def test_reassessment_low_cut_value(self):
        self.assertEqual(app.REASSESS_ID_COMBO_CUTS, {"combo_high_cut": 0.8, "combo_low_cut": 0.3})
        g = classify_attribute_risk(_combo_df(), top_n=10, combo_high_cut=0.8, combo_low_cut=0.3)
        self.assertIn("final_grade", g.columns)


if __name__ == "__main__":
    unittest.main()
