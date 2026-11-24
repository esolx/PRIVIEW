# -*- coding: utf-8 -*-
"""STRATEGY_MATRIX의 모든 (위험축 × 속성유형) 셀, 모든 후보 기법을 강도 사다리 5단계
전부로 실행해서 (1) 에러 없이 동작하는지 (2) 실제로 값이 바뀌는지 (3) 유용성손실이
계산되는지 확인한다. 결측값이 섞인 데이터로 돌린다.   실행: python test_strategy_matrix.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from modules.pseudonymization_pipeline import (
    STRATEGY_MATRIX, STRENGTH_LADDER, AttrType, RiskLevel, StrategyResult,
    apply_strategy, compute_utility, change_rate,
)

np.random.seed(42)
n = 200
cat = pd.Series(np.random.choice(
    ["서울시 강남구 역삼동", "서울시 강남구 삼성동", "서울시 종로구 청운동",
     "부산시 해운대구 우동", "홍길동", "김철수", "이영희", "A"], size=n))
num = pd.Series(np.random.lognormal(8, 1, size=n).round(0))
date = pd.Series(pd.to_datetime("2020-01-01") + pd.to_timedelta(np.random.randint(0, 1500, size=n), unit="D"))
cat.iloc[[3, 7]] = np.nan
num.iloc[[5]] = np.nan
date.iloc[[9]] = pd.NaT
data = {AttrType.CATEGORICAL: cat, AttrType.NUMERIC: num, AttrType.DATE: date}

fails = []
for (axis, at), cands in STRATEGY_MATRIX.items():
    for tech, _ in cands:
        row = []
        for rung in STRENGTH_LADDER:
            s = data[at]
            st = StrategyResult("col", at, axis, RiskLevel.H, tech)
            try:
                out, extra = apply_strategy(
                    s, st, {}, k_anonymity=rung["k_anonymity"], local_suppress_k=rung["local_suppress_k"],
                    microagg_k=rung["microagg_k"], date_unit=rung["date_unit"], pram_p_keep=rung["pram_p_keep"],
                    n_bins=rung["n_bins"], mask_ratio=rung["mask_ratio"],
                    tbc_quantile=rung["tbc_quantile"], jitter_days=rung["jitter_days"],
                )
                ul = compute_utility(at, s, out, extra)
                # NaN≠NaN, 타입 차이를 보정한 변경률 (단순 != 비교는 결측을 '바뀜'으로 셈)
                changed = change_rate(s, out, at)
                if ul is None or changed == 0 or extra.get("intensity_note") == "-":
                    fails.append((axis, at.value, tech.value, rung["label"], ul, changed))
                row.append(f"{ul:.3f}/{changed:.0%}" if ul is not None else f"None/{changed:.0%}")
            except Exception as e:
                fails.append((axis, at.value, tech.value, rung["label"], repr(e)))
                row.append("ERR")
        print(f"{axis:14s} {at.value:4s} {tech.value:14s} | " + "  ".join(row))

print("\n[유용성손실/변경된 레코드 비율] 1단계 → 5단계")
print("실패:", fails if fails else "없음")

# 샘플 출력 확인
st = lambda at, t: StrategyResult("col", at, "reid", RiskLevel.H, t)
from modules.pseudonymization_pipeline import Technique as T
print("\n[샘플]")
print(pd.DataFrame({
    "원본": cat.head(6),
    "부분마스킹(0.5)": apply_strategy(cat, st(AttrType.CATEGORICAL, T.PARTIAL_MASK), {}, mask_ratio=0.5)[0].head(6),
}).to_string())
print(pd.DataFrame({
    "원본": date.head(4),
    "지터(±15)": apply_strategy(date, st(AttrType.DATE, T.DATE_JITTER), {}, jitter_days=15)[0].head(4),
    "상하단(20%)": apply_strategy(date, st(AttrType.DATE, T.TOP_BOTTOM_CODING), {}, tbc_quantile=0.2)[0].head(4),
}).to_string())
out, ex = apply_strategy(num, st(AttrType.NUMERIC, T.TOP_BOTTOM_CODING), {}, tbc_quantile=0.05)
print(ex["intensity_note"])

# ── 부분마스킹 조건부 활성화: 계층 구조 감지 ──
from modules.pseudonymization_pipeline import detect_hierarchical_structure
expect = {
    "address": ["서울시 강남구 역삼동", "서울시 강남구 삼성동", "서울시 종로구 청운동", "부산시 해운대구 우동"],
    "code": ["I10.9", "I11.0", "E11.9", "E78.5", "J00.0"],          # KCD
    None: ["고혈압", "고지혈증", "당뇨", "폐렴"],                      # 자유 텍스트 범주
}
extra_none = [["세무 및 관세 사무원", "세무 및 회계 사무원", "국회의원"], ["A", "B", "AB", "O"]]
print("\n[계층 구조 감지]")
ok = True
for want, vals in list(expect.items()) + [(None, v) for v in extra_none]:
    got = detect_hierarchical_structure(pd.Series(vals * 3))[0]
    ok &= (got == want)
    print(f"  {vals[0]:14s} … → {got}  {'✓' if got == want else '✗ (기대: ' + str(want) + ')'}")
print("감지 테스트:", "통과" if ok else "실패")
