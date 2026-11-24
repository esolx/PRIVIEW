# -*- coding: utf-8 -*-
"""세 위험 축의 자체 판정을 보존해 통합한다.

각 모듈의 수치는 서로 다른 의미이므로 공통 확률이나 피해 점수로 재해석하지 않는다.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

_LABEL = {"L": "저위험", "M": "중위험", "H": "고위험"}
_RANK = {"L": 0, "M": 1, "H": 2}
_MODULE_LABEL = {
    "identification": "식별 위험", "aggregation": "집계 위험", "exposure": "노출 위험",
}


def _grade(value) -> Optional[str]:
    grade = str(value or "").upper()
    return grade if grade in _RANK else None


def _finite_score(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 6) if 0 <= number <= 1 else None


def _evidence(name: str, module: dict) -> tuple[str, str, list[dict]]:
    """원 모듈의 설명용 근거. 통합 판정에는 이 숫자를 사용하지 않는다."""
    if name == "identification":
        rows = []
        for row in module.get("attributes") or []:
            column = str(row.get("attribute") or "")
            if not column:
                continue
            try:
                score = float(row.get("individual_risk_score"))
                score = round(score / 100, 6)
                if not 0 <= score <= 1:
                    score = None
            except (TypeError, ValueError):
                score = None
            rows.append({"attribute": column, "risk_score": score,
                         "grade": _grade(row.get("final_grade")),
                         "metric_label": "CorrectMatch 속성별 정확 매칭률"})
        rows.sort(key=lambda r: (_RANK.get(r["grade"], -1), r["risk_score"] or 0), reverse=True)
        return (rows[0]["attribute"] if rows else "-", "CorrectMatch 속성별 결과", rows)
    if name == "aggregation" and module.get("attribute_grade_method") == "label_share" \
            and module.get("attribute_scores"):
        # 속성별 H 비율 — 그 속성이 들어간 조합 중 위험 이상 판정을 받은 조합의 비율.
        # 처리 전 경계로 재므로 처리 전후 같은 속성끼리 비교할 수 있다.
        grades = module.get("attribute_grades") or {}
        rows = [{"attribute": str(column), "risk_score": _finite_score(score),
                 "grade": _grade(grades.get(column)),
                 "metric_label": "속성이 들어간 조합 중 위험 이상 비율(H 비율)"}
                for column, score in module["attribute_scores"].items()]
        rows.sort(key=lambda r: (_RANK.get(r["grade"], -1), r["risk_score"] or 0), reverse=True)
        return (rows[0]["attribute"] if rows else "-", "집계 모듈의 속성별 H 비율", rows)
    if name == "aggregation":
        rows = [{"attribute": str(row.get("Feature_Combination") or "-"),
                 "risk_score": None, "grade": _grade(row.get("Final_Risk")),
                 "metric_label": "집계 모듈의 조합별 판정"}
                for row in module.get("top_combinations") or []]
        return (rows[0]["attribute"] if rows else str(module.get("target_col") or "-"),
                "집계 모듈의 데이터셋 판정", rows)
    rows = [{"attribute": str(column), "risk_score": _finite_score(score),
             "grade": None, "metric_label": "노출 모듈의 속성별 점수"}
            for column, score in (module.get("attribute_scores") or {}).items()]
    rows.sort(key=lambda r: r["risk_score"] or 0, reverse=True)
    return (rows[0]["attribute"] if rows else "-", "노출 모듈의 데이터셋 판정", rows)


def compute_integrated_risk_from_modules(modules: dict, df: pd.DataFrame) -> Optional[dict]:
    components, missing = [], []
    for name in ("identification", "aggregation", "exposure"):
        module = modules.get(name) or {}
        grade = _grade(module.get("dataset_grade")) if module.get("status") == "success" else None
        if grade is None:
            missing.append(name)
            continue
        attribute, basis, attributes = _evidence(name, module)
        components.append({"module": name, "label": _MODULE_LABEL[name],
                           "grade": grade, "grade_label": _LABEL[grade],
                           "score": _RANK[grade], "top_attribute": attribute,
                           "basis": basis, "attributes": attributes})
    if not components:
        return None
    driver = max(components, key=lambda row: row["score"])
    complete = not missing
    high = [row["label"] for row in components if row["grade"] == driver["grade"]]
    return {
        "grade": driver["grade"] if complete else "U",
        "grade_label": driver["grade_label"] if complete else "확인 필요",
        "score": driver["score"] if complete else None,
        "numeric": driver["score"] if complete else None,
        "score_kind": "ordinal_grade", "method": "세 위험 축의 데이터셋 등급 중 최고 등급",
        "formula": "통합 등급 = max(식별 등급, 집계 등급, 노출 등급), L < M < H",
        "components": components, "driver_module": driver["module"],
        "driver_label": driver["label"], "driver_attribute": driver["top_attribute"],
        "highest_modules": high,
        "rationale": f"{', '.join(high)}에서 가장 높은 등급인 {driver['grade_label']}이 산출되었습니다." if complete
                     else f"{', '.join(missing)} 분석이 완료되지 않아 통합 등급을 보류합니다.",
        "completeness": "complete" if complete else "partial", "missing_modules": missing,
    }


def compute_integrated_risk(identification_grade: Optional[str],
                            aggregation_grade: Optional[str],
                            exposure_grade: Optional[str] = None) -> Optional[dict]:
    values = [(name, _grade(value)) for name, value in (
        ("identification", identification_grade), ("aggregation", aggregation_grade),
        ("exposure", exposure_grade))]
    valid = [(name, grade) for name, grade in values if grade]
    if not valid:
        return None
    name, grade = max(valid, key=lambda item: _RANK[item[1]])
    return {"grade": grade, "grade_label": _LABEL[grade], "numeric": _RANK[grade],
            "method": "세 위험 축의 데이터셋 등급 중 최고 등급",
            "driver_module": name, "driver_label": _MODULE_LABEL[name], "components": []}
