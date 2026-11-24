# -*- coding: utf-8 -*-
"""재가명처리 자동 추천: 집계 고위험 속성 중 H 비율 상위 절반만 (상위 절반 버전), 라운드는 1회."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import app


class RecommendAggregationHigh(unittest.TestCase):
    def test_all_aggregation_high_recommended(self):
        tmp = Path(tempfile.mkdtemp())
        old = app.RESULT_DIR
        app.RESULT_DIR = tmp
        try:
            job = "a" * 32
            (tmp / job).mkdir()
            scores = {"a": 1.0, "b": 0.95, "c": 0.8, "d": 0.7, "e": 0.6, "f": 0.2}
            grades = {"a": "H", "b": "H", "c": "H", "d": "H", "e": "H", "f": "L"}
            result = {"analysis_config": {}, "modules": {"aggregation": {
                "status": "success", "dataset_grade": "H", "attribute_grade_method": "label_share",
                "attribute_scores": scores, "attribute_grades": grades}}}
            (tmp / job / "integrated_result.json").write_text(json.dumps(result), encoding="utf-8")
            df = pd.DataFrame({c: [1, 2] for c in scores})
            cols, reasons = app._recommended_columns(job, df)
            agg = [c for c in cols if "H 비율 상위 절반" in reasons[c]]
            self.assertEqual(sorted(agg), ["a", "b", "c"])     # 고위험 5개 중 H 비율 상위 3개 (올림)
            self.assertIn("H 비율 100%", reasons["a"])
        finally:
            app.RESULT_DIR = old

    def test_single_round_by_default(self):
        self.assertEqual(app.DEFAULT_MAX_ROUNDS, 1)


if __name__ == "__main__":
    unittest.main()
