# -*- coding: utf-8 -*-
"""
속성별 위험 등급(H/M/L) — 조합 위험도만으로 산출.

correctmatch_toolkit(가명처리 적정성 검토 시스템)의 classify_attribute_risk()와
동일한 기준을 그대로 재사용한다: 상위 N개 조합(정확도 높은 순) 중 이 속성이
등장한 비율로 H/M/L을 매긴다. 그쪽과 달리 여기서는 "개인 단독 위험도"(size=1
조합)를 따로 보지 않는다 — 이 프로젝트의 min_combo_size 기본값이 2라서 size=1
결과가 없는 게 보통이고, 요청도 "조합 위험도만" 보라는 것이었으므로 상위 조합
등장 빈도 하나만으로 등급을 정한다.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def _combo_grade(appearance_ratio: float, high_cut: float = 0.7, low_cut: float = 0.0) -> str:
    """상위 조합 중 이 속성이 등장한 비율을 기준으로 H/M/L 분류.

    appearance_ratio: 상위 N개 조합 중 이 속성이 포함된 조합의 비율 (0~1).
        - high_cut 이상: 상위 조합에 거의 항상 등장 → H
        - low_cut 초과 ~ high_cut 미만: 상위 조합에 일부 등장 → M
        - low_cut 이하(보통 0): 상위 조합에 전혀 등장하지 않음 → L
    """
    if appearance_ratio >= high_cut:
        return "H"
    if appearance_ratio > low_cut:
        return "M"
    return "L"


def classify_attribute_risk(
    risk_df: pd.DataFrame,
    top_n: int = 10,
    combo_high_cut: float = 0.7,
) -> pd.DataFrame:
    """조합별 정확도(risk_df)만으로 개별 속성의 위험 등급(H/M/L)을 산출.

    Args:
        risk_df: compute_combination_accuracies()/classify_risks()의 결과.
            'Feature_Combination'(예: "age+priors_count"), 'Mean_Accuracy' 컬럼이
            반드시 있어야 한다.
        top_n: "상위 조합"으로 볼 개수 (기본 10, correctmatch_toolkit과 동일).
        combo_high_cut: "거의 항상 등장"으로 볼 비율 임계값 (기본 0.7 = 70%,
            correctmatch_toolkit과 동일).

    Returns:
        속성별 등장 횟수/비율/등급을 담은 DataFrame. grade 기준 내림차순 정렬
        (H가 위로, 동률이면 appearance_ratio 높은 순).
    """
    all_attrs: List[str] = sorted(
        {attr for combo in risk_df["Feature_Combination"] for attr in combo.split("+")}
    )

    top_combos = risk_df.sort_values("Mean_Accuracy", ascending=False).head(top_n)
    top_combo_sets = [set(c.split("+")) for c in top_combos["Feature_Combination"]]
    n_top = len(top_combo_sets)

    records = []
    for attr in all_attrs:
        appearance_count = sum(1 for combo_set in top_combo_sets if attr in combo_set)
        appearance_ratio = appearance_count / n_top if n_top > 0 else 0.0
        grade = _combo_grade(appearance_ratio, combo_high_cut)
        records.append(
            {
                "attribute": attr,
                f"top{top_n}_appearance_count": appearance_count,
                f"top{top_n}_appearance_ratio": round(appearance_ratio, 2),
                "grade": grade,
            }
        )

    result_df = pd.DataFrame(records)
    order = {"H": 2, "M": 1, "L": 0}
    result_df["_order"] = result_df["grade"].map(order)
    result_df = (
        result_df.sort_values(["_order", f"top{top_n}_appearance_ratio"], ascending=[False, False])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return result_df


# ── v25: 조합 라벨 비율 기준 (절대 기준) ───────────────────────
_HML = {"고위험": "H", "위험": "H", "중간": "M", "안심": "L"}
_ORDER = {"H": 2, "M": 1, "L": 0}


def classify_attribute_risk_by_label_share(risk_df: pd.DataFrame) -> pd.DataFrame:
    """그 속성이 들어간 조합들의 라벨 비율로 속성 등급을 매긴다 (v25 기본).

    데이터셋 등급과 같은 규칙을 속성 단위로 적용한다:
      H 비율 = 고위험+위험, M 비율 = 중간, L 비율 = 안심 → 세 비율 중 최댓값 (동률이면 더 위험한 쪽)

    조합 라벨이 처리 전 경계로 매겨져 있으면 이 등급도 처리 전 기준의 절대값이다.
    v24의 상위 N개 등장 방식은 순위 기준이라 상위 N개 안의 속성은 처리와 무관하게
    M 이상이었지만, 이 방식에서는 그 속성이 들어간 위험 조합이 줄면 등급이 내려간다.
    점수(high_share)는 H 비율 — 강도 탐색에서 실패 단계끼리 비교할 때 쓴다(낮을수록 좋음).
    """
    combos = risk_df["Feature_Combination"].astype(str).str.split("+")
    levels = risk_df["Final_Risk"].map(_HML).fillna("L")
    all_attrs: List[str] = sorted({a for combo in combos for a in combo})
    records = []
    for attr in all_attrs:
        mask = combos.apply(lambda c, a=attr: a in c)
        n = int(mask.sum())
        counts = levels[mask].value_counts()
        shares = {g: (float(counts.get(g, 0)) / n if n else 0.0) for g in ("H", "M", "L")}
        best = max(shares.values())
        grade = max((g for g, v in shares.items() if v == best), key=_ORDER.get)
        records.append({"attribute": attr, "n_combos": n,
                        "high_share": round(shares["H"], 4), "mid_share": round(shares["M"], 4),
                        "low_share": round(shares["L"], 4), "grade": grade})
    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df
    result_df["_order"] = result_df["grade"].map(_ORDER)
    return (result_df.sort_values(["_order", "high_share"], ascending=[False, False])
            .drop(columns="_order").reset_index(drop=True))
