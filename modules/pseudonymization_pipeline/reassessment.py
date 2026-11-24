# -*- coding: utf-8 -*-
"""재평가 결과 요약 및 테스트용 스텁."""
from __future__ import annotations

import pandas as pd

from .models import AXIS_LABEL, AttributeRisk, RiskLevel, _LEVEL_ORDER

def _demo_reassess(df: pd.DataFrame, attr: str) -> AttributeRisk:
    """테스트용 재평가 스텁. 실제 운영에서는 위험평가 모듈을 호출하는 함수로
    교체해서 reassess_fn으로 주입한다."""
    return AttributeRisk(
        reid=RiskLevel.M, aggregation=RiskLevel.M,
        exposure_dist=RiskLevel.M, exposure_link=RiskLevel.M,
    )


def format_protection(risk: AttributeRisk, new_risk: AttributeRisk, axes: list,
                      target_level=RiskLevel.M) -> str:
    """재평가 대상이 된 축들(목표보다 높았던 축 전부) 각각의 before→after를 요약한 문자열.
    예: '재식별 H→M[성공] / 노출(외부연결) H→H[실패]'
    [v24] 목표 등급 이하로 내려가면 성공. 목표 L에서 H→M은 '개선'."""
    t = _LEVEL_ORDER[getattr(target_level, "value", str(target_level))]
    parts = []
    for a in axes:
        b = getattr(risk, a).value
        af = getattr(new_risk, a).value
        if _LEVEL_ORDER[af] <= t and _LEVEL_ORDER[af] < _LEVEL_ORDER[b]:
            mark = "성공"
        elif af == "H":
            mark = "실패"
        elif _LEVEL_ORDER[af] < _LEVEL_ORDER[b]:
            mark = "개선"
        else:
            mark = "동일"
        parts.append(f"{AXIS_LABEL.get(a, a)} {b}→{af}[{mark}]")
    return " / ".join(parts)

