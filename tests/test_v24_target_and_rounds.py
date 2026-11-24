# -*- coding: utf-8 -*-
"""v24: 목표 등급(L) 탐색, M 속성 처리, 라운드 루프 멈춤 규칙, TV 표시용 계산을 확인한다.
위험평가 모듈 없이 스텁 재평가로 돌린다."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modules.pseudonymization_pipeline import (
    AttrType, AttributeRisk, RiskLevel, StrategyResult, Technique,
    format_protection, get_current_trial, run_pipeline, search_strategy,
)
from modules.integration.utility_tv import tv_report

H, M, L = RiskLevel.H, RiskLevel.M, RiskLevel.L
RUNG = {"1단계(최약)": 1, "2단계": 2, "3단계": 3, "4단계": 4, "5단계(최강)": 5}


def _series():
    return pd.Series(np.random.default_rng(0).integers(0, 1000, 300), name="x")


def _by_rung(levels):
    """강도 단계 → 재식별 등급 스텁. levels[i] = (i+1)단계의 결과"""
    calls = []

    def fn(df, attr):
        rung = RUNG[get_current_trial()["rung"]]
        calls.append(rung)
        return AttributeRisk(reid=levels[rung - 1])
    return fn, calls


def _search(levels, target, start=H):
    s = _series()
    fn, calls = _by_rung(levels)
    strat = StrategyResult("x", AttrType.NUMERIC, "reid", start, Technique.BINNING)
    out = search_strategy("x", s, strat, {}, AttributeRisk(reid=start), fn,
                          pd.DataFrame({"x": s}), target_level=target)
    return out, calls


class SearchTarget(unittest.TestCase):
    def test_target_m_stops_at_first_m(self):
        (_, extra, _, new_risk, _, _), calls = _search([H, M, M, L, L], "M")
        self.assertEqual(calls, [1, 2])
        self.assertEqual(extra["chosen_rung"], "2단계")
        self.assertEqual(new_risk.reid, M)

    def test_target_l_continues_to_l(self):
        (_, extra, _, new_risk, log, _), calls = _search([H, M, M, L, L], "L")
        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual(extra["chosen_rung"], "4단계")
        self.assertEqual(extra["m_rung"], "2단계")
        self.assertEqual(new_risk.reid, L)
        self.assertTrue(log[-1]["achieved"])

    def test_target_l_unreached_picks_m_rung_with_least_loss(self):
        (_, extra, ul, new_risk, log, _), calls = _search([H, M, M, M, M], "L")
        self.assertEqual(calls, [1, 2, 3, 4, 5])
        # 전부 M이면 정보손실이 가장 작은(=가장 약한) M 단계
        m_logs = [e for e in log if e["achieved_m"]]
        best = min(m_logs, key=lambda e: e["utility"])
        self.assertEqual(extra["chosen_rung"], best["label"])
        self.assertEqual(new_risk.reid, M)
        self.assertIn("L 미달성", extra["intensity_note"])

    def test_m_attribute_is_target_under_l(self):
        (_, extra, _, new_risk, _, axes), calls = _search([M, L, L, L, L], "L", start=M)
        self.assertEqual(axes, ["reid"])
        self.assertEqual(extra["chosen_rung"], "2단계")


class PipelineTarget(unittest.TestCase):
    def _run(self, target):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"a": rng.integers(0, 1000, 200), "b": rng.integers(0, 1000, 200)})
        risk = {"a": AttributeRisk(reid=H), "b": AttributeRisk(aggregation=M)}
        return run_pipeline(df, df, risk, {}, reassess_fn=lambda d, attr: AttributeRisk(),
                            target_level=target)[1]

    def test_m_target_skips_m_attribute(self):
        rep = self._run("M")
        self.assertEqual([s.attribute for s in rep.reprocessed], ["a"])
        self.assertIn("b", rep.skipped)

    def test_l_target_processes_m_attribute(self):
        rep = self._run("L")
        self.assertEqual(sorted(s.attribute for s in rep.reprocessed), ["a", "b"])

    def test_protection_labels(self):
        before = AttributeRisk(reid=H)
        self.assertIn("성공", format_protection(before, AttributeRisk(reid=M), ["reid"], "M"))
        self.assertIn("개선", format_protection(before, AttributeRisk(reid=M), ["reid"], "L"))
        self.assertIn("성공", format_protection(before, AttributeRisk(reid=L), ["reid"], "L"))


class Rounds(unittest.TestCase):
    """라운드 루프의 멈춤 규칙 — 재가명처리·재평가 함수를 스텁으로 바꿔 확인"""

    def _run(self, grades, max_rounds=3, summaries=None, counts=None):
        import app
        df = pd.DataFrame({"a": range(50), "y": [0, 1] * 25})
        source = {"analysis_config": {"target_col": "y"}, "modules": {},
                  "integrated": {"grade": "H"}}
        state = {"i": 0}
        summaries = summaries or [[{"column": "a"}]] * len(grades)

        def fake_repseudo(result, selected, current, out_dir, **kw):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return current.copy(), summaries[state["i"]], None, None

        def fake_reassess(processed, config, target_col, out_dir, progress=None):
            g = grades[state["i"]]
            state["i"] += 1
            return {"identification": {"status": "success", "dataset_grade": g}}, []

        def fake_integrated(modules, df):
            return {"grade": modules["identification"]["dataset_grade"]}

        orig = (app.run_repseudonymization_from_result, app.reassess_dataframe,
                app.compute_integrated_risk_from_modules, app._attrs_above_target)
        app.run_repseudonymization_from_result = fake_repseudo
        app.reassess_dataframe = fake_reassess
        app.compute_integrated_risk_from_modules = fake_integrated
        # 목표 초과 속성 수: counts가 있으면 [처리 전, 1라운드 후, 2라운드 후, ...] 순서로 돌려준다
        seq = iter(counts) if counts else None
        app._attrs_above_target = lambda result, d, t, sel=None: ["a"] * (
            next(seq) if seq else {"H": 3, "M": 2, "L": 0}[(result.get("integrated") or {}).get("grade")])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = app.run_repseudonymization_rounds(source, None, df, Path(tmp), "L", max_rounds)
                self.assertTrue((Path(tmp) / "rounds_summary.json").exists())
                return out
        finally:
            (app.run_repseudonymization_from_result, app.reassess_dataframe,
             app.compute_integrated_risk_from_modules, app._attrs_above_target) = orig

    def test_stops_when_target_reached(self):
        out = self._run(["M", "L", "L"])
        self.assertEqual(len(out["rounds"]), 2)
        self.assertIn("도달", out["stop_reason"])

    def test_max_three_rounds(self):
        # 매 라운드 목표 초과 속성 수가 줄어 계속 개선 중이어도 3라운드에서 멈춤
        out = self._run(["M", "M", "M", "M"], counts=[6, 5, 4, 3, 2])
        self.assertEqual(len(out["rounds"]), 3)
        self.assertEqual(out["stop_reason"], "최대 라운드 도달")

    def test_stops_without_improvement(self):
        out = self._run(["H", "H", "H"])
        self.assertEqual(len(out["rounds"]), 1)
        self.assertIn("개선 없음", out["stop_reason"])

    def test_tv_is_reported_not_used(self):
        out = self._run(["L"])
        self.assertEqual(out["utility_tv"]["status"], "success")
        self.assertTrue(out["utility_tv"]["display_only"])


class Tv(unittest.TestCase):
    def test_identical_is_zero(self):
        df = pd.DataFrame({"c": list("aabbbc") * 50, "y": [0, 1, 0] * 100})
        r = tv_report(df, df, ["c"], "y", n_boot=50)
        self.assertEqual([q["tv"] for q in r["queries"]], [0.0, 0.0])
        self.assertTrue(all(q["pass"] for q in r["queries"]))

    def test_generalized_value_spread_uniformly(self):
        df = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 30 + ["c"] * 20})
        p = df.copy()
        p.loc[p.c.isin(["b", "c"]), "c"] = "기타"
        r = tv_report(df, p, ["c"], n_boot=50)
        # '기타' 50%를 b,c에 25%씩 → |0.3-0.25|+|0.2-0.25| = 0.1 → TV 0.05
        self.assertAlmostEqual(r["queries"][0]["tv"], 0.05, places=4)


if __name__ == "__main__":
    unittest.main()
