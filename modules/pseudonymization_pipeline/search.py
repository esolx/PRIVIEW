# -*- coding: utf-8 -*-
"""강도 사다리 반복탐색 — 약한 강도부터 시도해 실제 재평가로 목표 등급 달성 확인.

v24: 목표 등급(target_level)을 받는다. 기본값 M은 v23과 같은 동작(H→M 첫 달성에서 멈춤),
L이면 목표보다 높은 축 전부가 L이 될 때까지 사다리를 계속 올라간다."""
from __future__ import annotations

import hashlib
import random
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .models import AXIS_PRIORITY, AttributeRisk, RiskLevel, StrategyResult, _LEVEL_ORDER
from .apply_strategy import apply_strategy, compute_utility
from .utility import change_rate


def _ord(level) -> int:
    return _LEVEL_ORDER[getattr(level, "value", level)]


def as_level(level) -> RiskLevel:
    """'L' / RiskLevel.L 둘 다 받는다."""
    return level if isinstance(level, RiskLevel) else RiskLevel(str(level).upper())


def axes_above(risk: AttributeRisk, target_level=RiskLevel.M) -> list:
    """목표 등급보다 높은 축 (AXIS_PRIORITY 순). 목표 M이면 v23의 'H였던 축'과 같다."""
    t = _ord(as_level(target_level))
    return [a for a in AXIS_PRIORITY if _ord(getattr(risk, a)) > t]


def _level_key(new_risk: AttributeRisk, axes: list) -> tuple:
    """도달 등급 비교용 (낮을수록 좋음): (대상 축 중 최고 등급, 대상 축 등급 합)."""
    ords = [_ord(getattr(new_risk, a)) for a in axes] or [0]
    return (max(ords), sum(ords))


def _reached_label(new_risk: AttributeRisk, axes: list) -> str:
    return {2: "H", 1: "M", 0: "L"}[_level_key(new_risk, axes)[0]]


# ── 실험 재현용 시드 ──────────────────────────────────────────────
# PRAM, 스와핑, 날짜 지터링 등은 전역 np.random을 쓴다. 기본 시드를 정하면
# 시도(컬럼, 기법, 강도 단계)마다 고유한 시드를 만들어 그 시도 직전에 다시 넣는다.
# 그래서 컬럼을 추가하거나 순서를 바꾸거나 일부 단계가 건너뛰어져도,
# 같은 (컬럼, 기법, 단계)는 항상 같은 처리본이 나온다.
# 기본 시드를 정하지 않으면(None) 기존처럼 매번 무작위로 동작한다.
_BASE_SEED: Optional[int] = None
_CURRENT_TRIAL: dict = {}


def set_base_seed(seed: Optional[int]) -> None:
    """실험용 기본 시드를 정한다. None이면 시드 고정을 끈다."""
    global _BASE_SEED
    _BASE_SEED = None if seed is None else int(seed)


def get_base_seed() -> Optional[int]:
    return _BASE_SEED


def trial_seed(attr: str, technique: str, rung_label: str,
               base_seed: Optional[int] = None) -> Optional[int]:
    """(기본 시드, 컬럼, 기법, 단계)로 시도별 시드를 만든다.
    파이썬 내장 hash()는 실행할 때마다 값이 바뀌므로 sha256을 쓴다."""
    base = _BASE_SEED if base_seed is None else base_seed
    if base is None:
        return None
    key = f"{base}|{attr}|{technique}|{rung_label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def get_current_trial() -> dict:
    """reassess_fn 안에서 지금 몇 번째 시도(컬럼·기법·단계·시드)를 평가 중인지 확인한다."""
    return dict(_CURRENT_TRIAL)

STRENGTH_LADDER = [
    {"label": "1단계(최약)", "k_anonymity": 2,  "local_suppress_k": 1,  "pram_p_keep": 0.95, "microagg_k": 2,  "n_bins": 10, "date_unit": "W",
     "mask_ratio": 0.3,  "tbc_quantile": 0.01,  "jitter_days": 3},
    {"label": "2단계",       "k_anonymity": 3,  "local_suppress_k": 3,  "pram_p_keep": 0.90, "microagg_k": 3,  "n_bins": 7,  "date_unit": "M",
     "mask_ratio": 0.5,  "tbc_quantile": 0.025, "jitter_days": 7},
    {"label": "3단계",       "k_anonymity": 5,  "local_suppress_k": 5,  "pram_p_keep": 0.80, "microagg_k": 5,  "n_bins": 5,  "date_unit": "Q",
     "mask_ratio": 0.6,  "tbc_quantile": 0.05,  "jitter_days": 15},
    {"label": "4단계",       "k_anonymity": 8,  "local_suppress_k": 8,  "pram_p_keep": 0.70, "microagg_k": 8,  "n_bins": 3,  "date_unit": "Y",
     "mask_ratio": 0.75, "tbc_quantile": 0.10,  "jitter_days": 30},
    {"label": "5단계(최강)", "k_anonymity": 12, "local_suppress_k": 12, "pram_p_keep": 0.50, "microagg_k": 12, "n_bins": 2,  "date_unit": "Y",
     "mask_ratio": 0.9,  "tbc_quantile": 0.20,  "jitter_days": 90},
]


def search_strategy(
    attr: str,
    orig_series: pd.Series,
    strategy: StrategyResult,
    domain_mapping: dict,
    risk: AttributeRisk,
    reassess_fn: Callable[[pd.DataFrame, str], AttributeRisk],
    base_result_df: pd.DataFrame,
    disease_columns: Optional[set] = None,
    ladder: Optional[list] = None,
    occupation_mapping: Optional[dict] = None,
    occupation_columns: Optional[set] = None,
    region_mapping: Optional[dict] = None,
    region_columns: Optional[set] = None,
    target_level=RiskLevel.M,
) -> tuple[pd.Series, dict, float, AttributeRisk, list, list]:
    """
    가장 약한 강도부터 순서대로 시도하면서 reassess_fn으로 M(또는 그 이하) 달성
    여부를 확인하고, 달성한 첫 단계에서 멈춘다. 약한 것부터 보므로 처음 M을
    달성한 단계가 곧 정보손실이 가장 낮은 후보다.

    끝까지 M이 안 나오면(achieved=False), 재평가한 단계들 중 "그나마 M에 가장
    가까웠던" 단계를 채택한다 — 강도가 세다고 항상 더 안전해진다는 보장이
    없기 때문에, 무조건 마지막(가장 강한) 단계를 채택하지 않는다.
      · 원래 H였던 축이 1개: 그 축의 risk_scores(원값)가 가장 낮은 단계.
      · 2개 이상(다중위험): 원값은 스케일이 달라 못 더하므로, 라벨만 보고 아직
        H로 남은 축이 가장 적은 단계.
    잔여위험이 같으면 더 강한(뒤쪽) 단계를 채택한다. 축이 1개인데 risk_scores가
    없으면 비교할 값이 없어 기존처럼 마지막 단계를 채택한다(achieved=False로 표시).

    처리 전후 값이 같은(변경률 0) 단계는 재평가 없이 건너뛴다 — 아무것도 바꾸지 않은
    결과를 재평가에 넘기면, 재평가가 느슨할 때(스텁·미평가 축 등) 손실 0인 무변경
    결과가 '달성'으로 채택되고, 파이프라인은 원본값을 그대로 결과에 붙이게 된다.
    전 단계가 무변경이면 processed=None, extra["all_noop"]=True로 반환한다
    (호출 측에서 이 후보를 탈락시킨다).

    지배축 하나만 보지 않고 원래 H였던 축 전부가 M 이하로 내려갔는지 확인한다.
    reassess_fn은 재식별/집계/노출을 함께 재평가하는 합성 함수를 넣는 걸 전제로 한다.

    [v24] target_level
      - M(기본): 위 설명 그대로 (v23과 동일).
      - L: '목표보다 높은 축'(H·M 축 전부)이 L이 되는 첫 단계에서 멈춘다. 끝까지 L이 안 되면
        M을 달성한 단계들 중 (도달 등급, 정보손실)이 가장 작은 단계를 채택한다. M도 못 찍으면
        v23의 잔여위험 규칙으로 고른다.

    반환: (선택된 처리결과, extra, 정보손실, 재평가결과, 단계별 시도기록, 목표보다 높았던 축 목록)
    """
    ladder = ladder or STRENGTH_LADDER
    attr_type = strategy.attr_type
    target_level = as_level(target_level)
    t_ord = _ord(target_level)
    h_axes = axes_above(risk, target_level)   # 목표 M이면 'H였던 축'과 같다
    attempt_log: list = []
    attempt_records: list = []  # 무변경(건너뜀)을 제외한, 실제 재평가한 단계들의 (label, processed, extra, ul, new_risk, achieved)
    chosen = None

    technique_value = getattr(strategy.technique, "value", str(strategy.technique))
    for rung in ladder:
        seed = trial_seed(attr, technique_value, rung["label"])
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # local_suppress_k는 커스텀 사다리(ladder=...)에 없을 수 있어 k_anonymity로 방어적 폴백
        processed, extra = apply_strategy(
            orig_series, strategy, domain_mapping,
            k_anonymity=rung["k_anonymity"], local_suppress_k=rung.get("local_suppress_k", rung["k_anonymity"]),
            microagg_k=rung["microagg_k"],
            date_unit=rung["date_unit"], pram_p_keep=rung["pram_p_keep"],
            disease_columns=disease_columns, n_bins=rung["n_bins"],
            occupation_mapping=occupation_mapping, occupation_columns=occupation_columns,
            region_mapping=region_mapping, region_columns=region_columns,
            # 신규 기법 파라미터 — 커스텀 사다리에 없으면 apply_strategy 기본값 사용
            mask_ratio=rung.get("mask_ratio", 0.5),
            tbc_quantile=rung.get("tbc_quantile", 0.05),
            jitter_days=rung.get("jitter_days", 15),
        )
        rate = change_rate(orig_series, processed, attr_type)
        if rate == 0.0:
            # 무변경 → 재평가 호출 없이 다음 단계로
            attempt_log.append({"label": rung["label"], "technique": technique_value, "seed": seed,
                                "utility": None, "new_risk": None,
                                "achieved": False, "achieved_m": False, "reached": None,
                                "change_rate": 0.0, "skipped": "무변경 — 재평가 생략",
                                "errors": {}, "processed": None})
            continue

        ul = compute_utility(attr_type, orig_series, processed, extra)
        ul = ul if ul is not None else 1.0

        temp_df = base_result_df.copy()
        temp_df[attr] = processed.values
        _CURRENT_TRIAL.clear()
        _CURRENT_TRIAL.update({"attribute": attr, "technique": technique_value,
                               "rung": rung["label"], "seed": seed})
        try:
            new_risk = reassess_fn(temp_df, attr)
        finally:
            _CURRENT_TRIAL.clear()
        achieved = all(_ord(getattr(new_risk, a)) <= t_ord for a in h_axes)
        achieved_m = all(getattr(new_risk, a) != RiskLevel.H for a in h_axes)

        # processed: 이 단계의 처리본(저장용). 저장 코드가 파일로 쓰고 JSON에서는 뺀다.
        attempt_log.append({"label": rung["label"], "technique": technique_value, "seed": seed,
                            "utility": ul, "new_risk": new_risk,
                            "achieved": achieved, "achieved_m": achieved_m,
                            "reached": _reached_label(new_risk, h_axes),
                            "change_rate": round(rate, 4),
                            "errors": dict(getattr(new_risk, "errors", {}) or {}),
                            "processed": processed})
        record = (rung["label"], processed, extra, ul, new_risk, achieved, achieved_m)
        attempt_records.append(record)
        chosen = record
        if achieved:
            break

    if chosen is None:
        # 사다리 전 단계가 무변경 — 이 기법으로는 이 컬럼을 처리할 수 없음
        note = f" [강도탐색: 전 {len(attempt_log)}단계 무변경 — 이 기법으로 처리 불가]"
        return None, {"intensity_note": note, "all_noop": True}, None, risk, attempt_log, h_axes

    fallback_used = False
    m_records = [(i, rec) for i, rec in enumerate(attempt_records) if rec[6]]
    if not chosen[5] and m_records:
        # 목표(L)는 못 찍었지만 M은 달성한 단계가 있음 → (도달 등급, 정보손실)이 가장 작은 단계
        chosen = min(m_records, key=lambda ir: (_level_key(ir[1][4], h_axes), ir[1][3], ir[0]))[1]
    elif not chosen[5]:  # 재평가한 단계 중 M을 달성한 게 하나도 없음 (v23 규칙)
        scored = [(rec, _residual_risk(rec[4], h_axes)) for rec in attempt_records]
        if all(s is None for _, s in scored):
            # 비교할 값이 없음(축 1개인데 risk_scores 미제공) — 기존 동작 유지
            chosen = attempt_records[-1]
            fallback_used = True
        else:
            # 잔여위험이 같으면 더 강한(뒤쪽) 단계를 채택 — 기존 폴백과 같은 방향
            best_i = min(range(len(scored)),
                         key=lambda i: (scored[i][1] if scored[i][1] is not None else float("inf"), -i))
            chosen = scored[best_i][0]

    label, processed, extra, ul, new_risk, achieved, achieved_m = chosen
    n_skipped = sum(1 for e in attempt_log if e.get("skipped"))
    skip_note = f", 무변경 {n_skipped}단계 건너뜀" if n_skipped else ""
    if achieved:
        status = f"전체축 {target_level.value}달성"
    elif achieved_m:
        status = (f"{target_level.value} 미달성 — M 달성 단계 중 "
                  f"도달 등급·정보손실 최소({label}) 채택")
    elif fallback_used:
        status = "H 유지(risk_scores 없어 최강 단계 채택)"
    elif len(h_axes) > 1:
        status = f"H 유지({label} 채택 — 시도한 강도 중 H로 남은 축 최소)"
    else:
        status = f"H 유지({label} 채택 — 시도한 강도 중 잔여위험 최소)"
    suffix = f" [강도탐색: {label}에서 {status}, 재평가 {len(attempt_log) - n_skipped}회{skip_note}]"
    extra["intensity_note"] = extra.get("intensity_note", "") + suffix
    extra["chosen_rung"] = label   # 이 기법에서 채택된 강도 단계 (저장용)
    extra["target_level"] = target_level.value
    extra["reached_level"] = _reached_label(new_risk, h_axes)
    extra["m_rung"] = next((r[0] for r in attempt_records if r[6]), None)   # M을 처음 달성한 단계
    return processed, extra, ul, new_risk, attempt_log, h_axes


def _residual_risk(new_risk: AttributeRisk, h_axes: list) -> Optional[float]:
    """실패한 시도끼리 비교할 '잔여위험' 값. 낮을수록 M에 가깝다.

    - h_axes(원래 H였던 축)가 1개: 그 축의 risk_scores 원값. 값이 없으면 None.
    - h_axes가 2개 이상(다중위험): 축마다 원값의 스케일이 달라 더할 수 없으므로
      원값은 쓰지 않고, 라벨만 보고 '아직 H로 남은 축의 개수'를 센다. 한 축만
      내려간 시도가 둘 다 H인 시도보다 낫다고 보는 것. risk_scores가 없어도 계산된다.

    같은 호출 안에서 비교되는 시도들은 h_axes가 모두 같으므로 서로 다른 종류의
    값(원값 vs 개수)이 한 비교에 섞이지 않는다."""
    if not h_axes:
        return None
    if len(h_axes) == 1:
        return new_risk.risk_scores.get(h_axes[0])
    return float(sum(1 for a in h_axes if getattr(new_risk, a) == RiskLevel.H))


def _safety_margin(new_risk: AttributeRisk, h_axes: list) -> float:
    """원래 H였던 축들이 평균적으로 H에서 몇 단계 내려왔는지 (0~1 정규화: M만 달성=0.5, L까지=1.0)."""
    if not h_axes:
        return 0.0
    drops = [(2 - _LEVEL_ORDER[getattr(new_risk, a).value]) for a in h_axes]
    return (sum(drops) / len(drops)) / 2


def composite_score(ul: float, new_risk: AttributeRisk, h_axes: list, lam: float = 0.15) -> float:
    """낮을수록 좋음. 유용성손실이 비슷한 후보끼리는 더 확실히 안전해진(L까지 간) 쪽이 이긴다.
    lam은 안전마진을 얼마나 중시할지 가중치 — 기본값 0.15는 마진이 유용성손실 차이가
    작을 때만 타이브레이커로 작용하도록 잡은 값이다."""
    return ul - lam * _safety_margin(new_risk, h_axes)


def search_best_strategy(
    attr: str,
    orig_series: pd.Series,
    candidates: list,          # [StrategyResult, ...] — 최고등급 축(들)의 셀 후보 (다중위험이면 합집합)
    domain_mapping: dict,
    risk: AttributeRisk,
    reassess_fn: Callable[[pd.DataFrame, str], AttributeRisk],
    base_result_df: pd.DataFrame,
    lam: float = 0.15,
    target_level=RiskLevel.M,
    **kwargs,
) -> tuple:
    """
    후보 기법마다 search_strategy()로 첫 M달성 지점까지 탐색한 뒤,
    achieved=True인 후보들 중 composite_score 최소인 걸 채택한다.

    achieved인 후보가 하나도 없으면(무변경 후보 제외), 각 후보의 최종 시도 결과로
    잔여위험이 가장 낮은 후보를 채택한다(기준은 search_strategy와 같다: 축 1개면
    risk_scores 원값, 다중위험이면 H로 남은 축 개수). 잔여위험이 같으면 유용성
    손실(ul)이 작은 후보. 축이 1개인데 risk_scores가 없으면 기존처럼 ul 최소로 폴백한다.

    후보가 1개뿐이면 비교 없이 그 결과를 그대로 반환한다(기존 단일-기법 동작과 동일).

    [v24] target_level=L이면 achieved는 'L 달성'. L을 찍은 후보가 없으면 M을 달성한 후보 중
    (도달 등급, 유용성 손실)이 가장 작은 후보를 채택하고, 그것도 없으면 위의 잔여위험 규칙을 쓴다.

    반환: (선택된 처리결과, extra, 정보손실, 재평가결과, 단계별 시도기록,
           H였던 축 목록, 채택된 StrategyResult, 후보 전체 비교결과 리스트)
    """
    results = []
    for cand in candidates:
        processed, extra, ul, new_risk, log, h_axes = search_strategy(
            attr, orig_series, cand, domain_mapping, risk,
            reassess_fn, base_result_df, target_level=target_level, **kwargs,
        )
        all_noop = extra.get("all_noop", False)
        achieved = (not all_noop) and any(e["achieved"] for e in log)
        achieved_m = (not all_noop) and new_risk is not None and bool(h_axes) and \
            all(getattr(new_risk, a) != RiskLevel.H for a in h_axes)
        score = composite_score(ul, new_risk, h_axes, lam) if achieved else float("inf")
        results.append({
            "strategy": cand, "technique": cand.technique,
            "processed": processed, "extra": extra,
            "ul": ul, "new_risk": new_risk, "log": log, "h_axes": h_axes,
            "achieved": achieved, "achieved_m": achieved_m,
            "reached": None if all_noop else _reached_label(new_risk, h_axes),
            "score": score, "all_noop": all_noop,
        })

    succeeded = [r for r in results if r["achieved"]]
    applicable = [r for r in results if not r["all_noop"]]   # 무변경 후보는 폴백 대상에서도 제외
    m_reached = [r for r in applicable if r["achieved_m"]]
    if succeeded:
        chosen = min(succeeded, key=lambda r: r["score"])
    elif m_reached:
        # 목표(L)를 찍은 후보는 없지만 M을 달성한 후보가 있음 → (도달 등급, 정보손실) 최소
        chosen = min(m_reached, key=lambda r: (_level_key(r["new_risk"], r["h_axes"]), r["ul"]))
    elif applicable:
        scored = [(r, _residual_risk(r["new_risk"], r["h_axes"])) for r in applicable]
        if all(s is None for _, s in scored):
            chosen = min(applicable, key=lambda r: r["ul"])  # 비교할 값 없음 — 기존 동작 폴백
        else:
            # 잔여위험이 같으면 유용성손실(ul)이 작은 후보
            chosen = min(scored, key=lambda rs: (rs[1] if rs[1] is not None else float("inf"),
                                                 rs[0]["ul"]))[0]
    else:
        # 모든 후보가 전 단계 무변경 — 채택할 처리결과가 없음 (pipeline에서 기존 가명값 유지)
        r0 = results[0]
        return (None, r0["extra"], None, r0["new_risk"], r0["log"], r0["h_axes"], r0["strategy"], results)

    if len(results) > 1:
        comparison_note = " / ".join(
            (f"{r['technique'].value}: 무변경✗" if r["all_noop"]
             else f"{r['technique'].value}: ul={r['ul']:.4f}{'✓' if r['achieved'] else '✗'}")
            for r in results
        )
        chosen["extra"]["intensity_note"] = chosen["extra"].get("intensity_note", "") + \
            f" [기법비교: {comparison_note} → {chosen['technique'].value} 채택]"

    return (chosen["processed"], chosen["extra"], chosen["ul"], chosen["new_risk"],
            chosen["log"], chosen["h_axes"], chosen["strategy"], results)

