# -*- coding: utf-8 -*-
"""저장된 집계 재평가 결과에 '안심 기준 완화폭'을 바꿔 적용해 보는 도구 (모델 재학습 없음, 몇 초).

사용법 (app.py가 있는 폴더에서):
    python tools/relabel_aggregation.py <1단계 결과 폴더> <재평가 집계 폴더>

예:
    python tools/relabel_aggregation.py ^
        results\\<1단계 job_id> ^
        results\\repseudo-<task_id>\\round_assessment\\aggregation

- 1단계 결과 폴더: integrated_result.json이 있는 곳 (처리 전 경계를 여기서 읽음)
- 재평가 집계 폴더: combination_accuracies.csv가 있는 곳
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.aggregation_risk_system.analyzer import AggregationRiskAnalyzer  # noqa: E402
from modules.aggregation_risk_system.config import AggregationConfig  # noqa: E402
from modules.aggregation_risk_system.mode_fusion import ModeFusion  # noqa: E402
from modules.aggregation_risk_system.risk_classifier import RiskClassifier  # noqa: E402

_HML = {"고위험": "H", "위험": "H", "중간": "M", "안심": "L"}


def relabel(acc_df: pd.DataFrame, thresholds: dict, margin: float):
    fixed = {**thresholds, "__reassess_safe_margin__": margin}
    cfg = AggregationConfig(fixed_thresholds=fixed)
    a = AggregationRiskAnalyzer.__new__(AggregationRiskAnalyzer)
    a.config = cfg
    a.risk_classifier = RiskClassifier(cfg)
    a.mode_fusion = ModeFusion(cfg)
    a.risk_classifier.random_baseline = float(thresholds["__random_baseline__"]) + margin
    a.config.random_baseline = a.risk_classifier.random_baseline
    a.models = {c[: -len("_Accuracy")]: None for c in acc_df.columns
                if c.endswith("_Accuracy") and c != "Mean_Accuracy"}
    a._cached_thresholds = {}
    a.threshold_source = "fixed"
    with contextlib.redirect_stdout(io.StringIO()):
        risk_df = a.classify_risks(acc_df.copy())
    hml = risk_df["Final_Risk"].map(_HML).value_counts(normalize=True)
    shares = {g: float(hml.get(g, 0.0)) for g in ("H", "M", "L")}
    best = max(shares.values())
    grade = max((g for g, v in shares.items() if v == best), key={"H": 2, "M": 1, "L": 0}.get)
    return grade, shares, a.risk_classifier.random_baseline


def main():
    stage1, agg_dir = Path(sys.argv[1]), Path(sys.argv[2])
    result = json.loads((stage1 / "integrated_result.json").read_text(encoding="utf-8"))
    thresholds = ((result.get("modules") or {}).get("aggregation") or {}).get("thresholds")
    if not thresholds or "__random_baseline__" not in thresholds:
        sys.exit("1단계 결과에 처리 전 집계 경계가 없습니다. v25로 1단계 평가를 다시 실행한 결과 폴더를 지정하세요.")
    acc_df = pd.read_csv(agg_dir / "combination_accuracies.csv")
    print(f"조합 {len(acc_df)}개, 처리 전 우연 수준 {thresholds['__random_baseline__']:.4f}\n")
    print("완화폭  안심 기준   등급   고위험   중위험   저위험(안심)")
    for margin in (0.0, 0.05, 0.10, 0.15, 0.20):
        grade, s, cut = relabel(acc_df, thresholds, margin)
        print(f"{margin:5.2f}   {cut:7.4f}     {grade}    {s['H']:6.1%}   {s['M']:6.1%}   {s['L']:6.1%}")


if __name__ == "__main__":
    main()
