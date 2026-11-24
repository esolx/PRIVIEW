"""시도별 시드 재현성 · 오류 기록 · 강도별 저장을 확인한다 (위험평가 모듈 없이 스텁 재평가)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modules.pseudonymization_pipeline import (
    AttributeRisk, RiskLevel, run_pipeline, set_base_seed, trial_seed,
)
from modules.integration.strength_log import save_strength_search


def _data():
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "income": rng.integers(1000, 9000, 200),
        "age": rng.integers(20, 80, 200),
        "target": rng.integers(0, 2, 200),
    })


def _never_pass(df, attr):
    # 항상 H → 사다리 전 단계를 다 시도하게 만든다 (랜덤 기법 결과를 모두 얻기 위해)
    return AttributeRisk(reid=RiskLevel.H, risk_scores={"reid": 9.0})


def _failing(df, attr):
    return AttributeRisk(reid=RiskLevel.H, errors={"reid": "RuntimeError: julia 없음"})


def _run(columns, reassess=_never_pass):
    df = _data()
    risk = {c: AttributeRisk(reid=RiskLevel.H, risk_scores={"reid": 9.0}) for c in columns}
    _, report = run_pipeline(df, df, risk, domain_mapping={}, reassess_fn=reassess)
    return df, risk, report


def _trials(report, attr):
    out = {}
    for cand in report.candidate_comparison[attr]:
        for e in cand["log"]:
            if e.get("processed") is not None:
                out[(cand["technique"].value, e["label"])] = pd.Series(e["processed"]).astype(str).tolist()
    return out


class TrialSeedTest(unittest.TestCase):
    def tearDown(self):
        set_base_seed(None)

    def test_same_seed_same_result_even_with_other_random_use(self):
        set_base_seed(42)
        a = _trials(_run(["age"])[2], "age")
        np.random.rand(1000)                      # 중간에 전역 난수를 써도
        set_base_seed(42)
        b = _trials(_run(["income", "age"])[2], "age")   # 앞에 다른 컬럼을 추가해도
        self.assertEqual(a, b)

    def test_different_seed_changes_random_technique(self):
        set_base_seed(42)
        a = _trials(_run(["age"])[2], "age")
        set_base_seed(0)
        b = _trials(_run(["age"])[2], "age")
        swap = [k for k in a if "스와핑" in k[0]]
        self.assertTrue(swap, "랜덤 기법 후보가 있어야 비교할 수 있다")
        self.assertTrue(any(a[k] != b[k] for k in swap))

    def test_seed_recorded_and_stable(self):
        set_base_seed(42)
        report = _run(["age"])[2]
        entry = report.candidate_comparison["age"][0]["log"][0]
        self.assertEqual(entry["seed"], trial_seed("age", entry["technique"], entry["label"], 42))
        self.assertEqual(trial_seed("age", "x", "1단계", 42), trial_seed("age", "x", "1단계", 42))

    def test_no_seed_keeps_old_behaviour(self):
        set_base_seed(None)
        report = _run(["age"])[2]
        self.assertIsNone(report.candidate_comparison["age"][0]["log"][0]["seed"])

    def test_errors_logged_and_saved(self):
        set_base_seed(42)
        df, risk, report = _run(["age"], reassess=_failing)
        with tempfile.TemporaryDirectory() as tmp:
            info = save_strength_search(report, df, risk, tmp, base_seed=42)
            trials = pd.read_csv(info["trial_csv"])
            self.assertTrue(trials["has_error"].all())
            self.assertIn("julia", trials["식별_error"].iloc[0])
            self.assertEqual(trials["final_adopted"].sum(), 1)
            first = trials.dropna(subset=["trial_data"]).iloc[0]["trial_data"]
            saved = pd.read_csv(Path(tmp) / first)
            self.assertEqual(list(saved.columns), ["original", "processed"])
            self.assertEqual(len(saved), len(df))
            self.assertTrue((Path(tmp) / "candidate_comparison.csv").exists())
            self.assertTrue((Path(tmp) / "strength_search_log.json").exists())


if __name__ == "__main__":
    unittest.main()
