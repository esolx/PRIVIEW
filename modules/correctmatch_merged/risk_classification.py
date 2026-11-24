"""
속성별 / 데이터셋 전체 위험 등급(H/M/L) 산출.

pipeline.analyze_all_combinations()의 결과를 입력받아, 개별 속성 위험도와
상위 조합 등장 빈도를 함께 고려해 최종 등급을 매깁니다.
"""

from __future__ import annotations

import pandas as pd

_RISK_ORDER = {"L": 0, "M": 1, "H": 2}


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


def _individual_grade(risk_score: float, high_cut: float = 8, low_cut: float = 1) -> str:
    """개별 속성의 Risk_Score(correctness×100, 0~100)를 고정 임계값으로 H/M/L 분류.

    기본 임계값(H≥8, M 1~8, L<1)은 실제 6개 데이터셋 + 설문형 합성 데이터셋 18개
    (총 24개, 305개 준식별자 속성)의 risk_score 분포를 실증 분석해서 도출한 값입니다
    (90th percentile≈8.0, 중앙값≈1.0, 이 지점부터 값 간 간격도 눈에 띄게 벌어짐).
    """
    if risk_score >= high_cut:
        return "H"
    if risk_score >= low_cut:
        return "M"
    return "L"


def classify_attribute_risk(
    combo_df: pd.DataFrame,
    top_n: int = 10,
    individual_high_cut: float = 8,
    individual_low_cut: float = 1,
    combo_high_cut: float = 0.7,
    combo_low_cut: float = 0.0,
) -> pd.DataFrame:
    """개별 속성 위험도(고정 임계값) + 상위 조합 등장 빈도(H/M/L)를 max()로 합쳐 속성별 최종 등급을 산출.

    로직:
        1. 개인 등급: 속성 하나만의 Risk_Score(= correctness κ_x × 100, 논문에서 정의하는
           "매칭됐을 때 정확히 재식별될 확률" 자체)를 고정 임계값으로 분류.
        2. 조합 등급: k-익명성을 이미 충족하는(=3명 미만 동질집합이 하나도 없는) 조합은
           후보에서 제외한 뒤, 남은 조합 중 uniqueness·correctness 기준 상위 top_n개를
           추려 이 속성이 등장한 비율로 분류 (기본: 70% 이상 등장 시 H, 일부 등장 시 M, 전무하면 L).
           이 게이트 덕분에, 가명처리 후 재평가에서 "예전엔 안전했지만 다른 조합들이 더
           낮아지면서 상대적으로 1위가 된" 조합은 — 실제로 3-익명성을 만족한다면 — 절대
          후보에 못 들어와서 H로 튀지 않음.
        3. 최종 등급 = max(개인 등급, 조합 등급)  — 둘 중 더 위험한 쪽으로 판정.

    Args:
        combo_df: analyze_all_combinations()의 결과 (size=1 조합이 반드시 포함되어 있어야 함,
            k_violation_ratio 컬럼이 있어야 k-익명성 게이트가 적용됨 — 없으면 게이트 없이 진행.
            게이트 기준 k값은 analyze_all_combinations(k_anon_k=...) 호출 시 이미 정해짐, 기본 3명).
        top_n: "상위 조합"으로 볼 개수 (기본 10, Top-10 기준).
        individual_high_cut, individual_low_cut: 개인 등급 고정 임계값 (기본 8, 1 —
            24개 데이터셋 실증 분석으로 도출, 아래 _individual_grade 참고).
        combo_high_cut: 조합 등급에서 "거의 항상 등장"으로 볼 비율 임계값 (기본 0.7 = 70%).

    Returns:
        속성별 개인/조합/최종 등급과 근거 수치를 담은 DataFrame. final_grade 기준 내림차순 정렬.
    """
    all_attrs = sorted({attr for combo in combo_df["combination"] for attr in combo})

    # k-익명성 게이트: 이미 k-익명성을 충족하는(=3명 미만 동질집합이 없는) 조합은 후보에서 제외
    if "k_violation_ratio" in combo_df.columns:
        candidate_combos = combo_df[combo_df["k_violation_ratio"] > 0]
    else:
        candidate_combos = combo_df

    # 상위 조합(Top-N) 추출: uniqueness 내림차순, 동률이면 correctness 내림차순
    top_combos = candidate_combos.sort_values(
        ["uniqueness", "correctness"], ascending=[False, False]
    ).head(top_n)

    records = []
    for attr in all_attrs:
        # ── 1. 개인 등급 (고정 임계값) ──
        # Risk_Score = correctness(κ_x) × 100.
        # 논문(Rocher et al., 2019)에서 κ_x 자체가 이미 "매칭됐을 때 정확히 재식별될
        # 확률(최종 재식별률)"로 완결된 정의이므로, uniqueness와 별도로 평균 낼 필요가 없음
        # (κ_x=1은 정확히 uniqueness=1인 경우와 같아 uniqueness 정보가 이미 반영되어 있음).
        individual_row = combo_df[
            (combo_df["size"] == 1) & (combo_df["combination"] == (attr,))
        ]
        individual_correctness_val = (
            individual_row["correctness"].iloc[0] if not individual_row.empty else 0.0
        )
        individual_risk_score = individual_correctness_val * 100
        individual_grade = _individual_grade(individual_risk_score, individual_high_cut, individual_low_cut)

        # ── 2. 조합 등급 (상위 조합 중 등장 빈도) ──
        appearance_count = top_combos["combination"].apply(lambda c, a=attr: a in c).sum()
        appearance_ratio = appearance_count / len(top_combos) if len(top_combos) > 0 else 0.0
        combo_grade_val = _combo_grade(appearance_ratio, combo_high_cut, combo_low_cut)

        # ── 3. 최종 등급 = max(개인, 조합) ──
        final_grade = max(individual_grade, combo_grade_val, key=lambda g: _RISK_ORDER[g])

        # ── 4. 위험점수 (강도 사다리 실패 시 "M에 얼마나 가까운지" 비교용) ──
        # final_grade를 결정지은 쪽(개인/조합 중 더 위험한 쪽)의 원값을 그대로 쓴다.
        # 같은 축(reid) 안에서만 비교할 때 쓰는 값이라 다른 축과 스케일을 맞출
        # 필요가 없다 — 서로 다른 축끼리는 아예 비교하지 않기로 했기 때문.
        if _RISK_ORDER[individual_grade] >= _RISK_ORDER[combo_grade_val]:
            risk_score = individual_risk_score
        else:
            risk_score = appearance_ratio

        records.append(
            {
                "attribute": attr,
                "individual_risk_score": round(individual_risk_score, 2),
                "individual_grade": individual_grade,
                f"top{top_n}_appearance_count": int(appearance_count),
                f"top{top_n}_appearance_ratio": round(appearance_ratio, 2),
                "combo_grade": combo_grade_val,
                "final_grade": final_grade,
                "risk_score": round(risk_score, 4),
            }
        )

    result_df = pd.DataFrame(records)
    result_df["_order"] = result_df["final_grade"].map(_RISK_ORDER)
    result_df = result_df.sort_values(
        ["_order", "individual_risk_score"], ascending=[False, False]
    ).drop(columns="_order").reset_index(drop=True)
    return result_df


def dataset_level_risk(attribute_risk_df: pd.DataFrame) -> str:
    """속성별 final_grade의 최빈값으로 데이터셋 전체 식별 위험 등급을 산출.

    최빈값이 여러 개로 동률일 경우, 더 위험한 등급(H > M > L)을 채택합니다 (보수적 판정).
    """
    counts = attribute_risk_df["final_grade"].value_counts()
    max_count = counts.max()
    tied_grades = counts[counts == max_count].index.tolist()
    return max(tied_grades, key=lambda g: _RISK_ORDER[g])
