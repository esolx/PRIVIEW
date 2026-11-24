import unittest

import pandas as pd

from modules.integration.final_risk import compute_integrated_risk_from_modules


class IntegratedRiskRegressionTest(unittest.TestCase):
    def test_two_high_axes_cannot_be_reduced_by_column_impact(self):
        df = pd.DataFrame({"나이": [20, 30], "직업": ["A", "B"]})
        modules = {
            "identification": {"status": "success", "dataset_grade": "H",
                               "attributes": [{"attribute": "나이", "individual_risk_score": 80,
                                               "final_grade": "H"}]},
            "aggregation": {"status": "success", "dataset_grade": "H",
                            "target_col": "직업", "top_combinations": []},
            "exposure": {"status": "success", "dataset_grade": "L",
                         "attribute_scores": {"나이": 0.1}},
        }
        result = compute_integrated_risk_from_modules(modules, df)
        self.assertEqual(result["grade"], "H")
        self.assertEqual(result["highest_modules"], ["식별 위험", "집계 위험"])
        self.assertEqual(result["score_kind"], "ordinal_grade")

    def test_missing_axis_requires_review(self):
        result = compute_integrated_risk_from_modules({
            "identification": {"status": "success", "dataset_grade": "L"},
            "aggregation": {"status": "error"},
            "exposure": {"status": "success", "dataset_grade": "L"},
        }, pd.DataFrame({"x": [1]}))
        self.assertEqual(result["grade"], "U")
        self.assertEqual(result["missing_modules"], ["aggregation"])


if __name__ == "__main__":
    unittest.main()
