# -*- coding: utf-8 -*-
"""무변경 단계 스킵 동작 점검.   실행: python test_noop_skip.py

(1) change_rate가 결측·날짜·수치 타입 차이를 '변경'으로 오판하지 않는지
(2) 사다리에서 무변경 단계는 재평가 함수가 호출되지 않는지
(3) 재평가가 무조건 통과(스텁)여도 무변경 후보가 채택되지 않는지
(4) 모든 후보가 무변경이면 원본값이 아니라 기존 가명값이 유지되는지
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from modules.pseudonymization_pipeline import (
    AttrType, AttributeRisk, RiskLevel, StrategyResult, Technique,
    change_rate, run_pipeline, search_strategy,
)

fails = []
def check(name, cond):
    print(f"{'✓' if cond else '✗'} {name}")
    if not cond:
        fails.append(name)

# (1) change_rate
cat = pd.Series(["a", "b", np.nan, "c"])
check("범주형: 결측 포함 동일값 → 0", change_rate(cat, cat.copy(), AttrType.CATEGORICAL) == 0.0)
check("범주형: 1건 '*' → 0.25", change_rate(cat, pd.Series(["a", "*", np.nan, "c"]), AttrType.CATEGORICAL) == 0.25)
num = pd.Series([64, 70, np.nan, 81])
check("수치형: 64 vs 64.0 → 0", change_rate(num, num.astype(float), AttrType.NUMERIC) == 0.0)
check("수치형: '*' 전체억제 → 1", change_rate(num, pd.Series(["*"] * 4), AttrType.NUMERIC) == 1.0)
dt = pd.Series(pd.to_datetime(["2021-05-11", "2022-01-01", None, "2023-03-03"]))
same_str = pd.Series(["2021-05-11", "2022-01-01", np.nan, "2023-03-03"])
check("날짜형: datetime vs 같은 날짜 문자열 → 0", change_rate(dt, same_str, AttrType.DATE) == 0.0)
check("날짜형: 월 절단 '2021-05' → 바뀜",
      change_rate(dt, pd.Series(["2021-05", "2022-01", np.nan, "2023-03"]), AttrType.DATE) == 0.75)

# (2)(3) 로컬 억제: 희귀값이 빈도 4라서 k=1,3(1~2단계)은 무변경, k=5(3단계)부터 억제됨
s = pd.Series(["x"] * 30 + ["y"] * 30 + ["rare"] * 4)
calls = []
def always_pass(df, attr):
    calls.append(attr)
    return AttributeRisk(reid=RiskLevel.M)
risk = AttributeRisk(reid=RiskLevel.H)
strat = StrategyResult("col", AttrType.CATEGORICAL, "reid", RiskLevel.H, Technique.LOCAL_SUPPRESSION)
processed, extra, ul, new_risk, log, _ = search_strategy(
    "col", s, strat, {}, risk, always_pass, pd.DataFrame({"col": s}))
check("무변경 1~2단계는 재평가 미호출 (호출 1회)", len(calls) == 1)
check("무변경 단계는 skipped로 기록", [e.get("skipped") is not None for e in log] == [True, True, False])
check("채택 결과는 실제로 값이 바뀜", processed is not None and (processed == "*").sum() == 4)

# (4) 모든 후보가 무변경 → 기존 가명값 유지
# 고유값 3개가 각각 20건: 일반화는 모든 값이 k=12 이상이라 클러스터가 안 합쳐지고(사전도 없음),
# 로컬억제는 억제할 희귀값이 없고, 부분마스킹은 계층구조가 없어 후보에서 빠짐 → 전부 무변경
orig = pd.DataFrame({"job": ["교사"] * 20 + ["간호사"] * 20 + ["요리사"] * 20})
pseudo = pd.DataFrame({"job": ["PSEUDO"] * 60})
res, rep = run_pipeline(orig, pseudo, {"job": AttributeRisk(reid=RiskLevel.H)}, {}, reassess_fn=always_pass)
check("전 후보 무변경 → no_effect에 기록", rep.no_effect == ["job"])
check("전 후보 무변경 → 원본이 아니라 기존 가명값 유지", (res["job"] == "PSEUDO").all())
check("전 후보 무변경 → reprocessed에 안 들어감", not rep.reprocessed)

print("\n실패:", fails if fails else "없음")
