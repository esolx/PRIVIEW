"""
exposure_stage.py — 탐지 → 지표 계산 통합 어댑터
================================================

README의 파이프라인 흐름에서 "[이후 예정] EMD · Coverage · Combo · PMI" 자리를
채우는 모듈. 탐지 결과(ColumnDecision들)와 원본 데이터프레임, EE 사전을 받아
5개 노출 지표를 계산하고 (속성 × 지표) 위험 행렬을 반환한다.

역할 분류
--------
탐지 결과로부터 각 컬럼을 세 역할로 나눈다:
  · 민감속성  : is_sensitive=True 인 컬럼 (지표의 행)
  · EE 속성   : EE 사전에 매칭되고 ee_score ≥ 0.7 (외부 노출 계산용)
  · QI 후보   : 비민감이면서 준식별 가능한 컬럼 (동치류·희귀도 계산용)
                기본은 "비민감 전체". EE 속성도 QI로 함께 쓸 수 있다.

산출은 두 층이다:
  (A) 분석용 — (속성 × 지표) 위험 행렬. 지표를 하나로 합치지 않고 그대로 둔다.
      metric_framework(describe_metrics / select_metrics / flag_by_reference)로
      이용자가 지표를 선택·해석한다. (각 지표는 '무엇을 재는가'만 설명한다.)
  (B) 판정용 — 데이터셋 H/M/L 등급 + 조치 순서(top-k).
      속성별 통합위험 = max(내부 최댓값, 외부 최댓값) → 임계 τ 초과 판정 →
      초과 비율 ρ → ρ로 데이터셋을 High/Middle/Low로 분류 → 초과 속성을 위험
      내림차순으로 세워 조치 순서를 준다. (assess_dataset / topk_actions)
  ⚠️ (B)의 등급·순서는 요약일 뿐, (A)의 지표별 해석을 대체하지 않는다.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

from . import metrics as M

try:
    from . import ee_dictionary as eedict
    _HAS_EE = True
except Exception:
    _HAS_EE = False


# ===========================================================================
# 역할 분류
# ===========================================================================

@dataclass
class RoleAssignment:
    sensitive: List[str] = field(default_factory=list)
    ee: List[str] = field(default_factory=list)         # ee_score ≥ threshold
    qi: List[str] = field(default_factory=list)
    ee_score_map: Dict[str, float] = field(default_factory=dict)
    category_map: Dict[str, str] = field(default_factory=dict)


def assign_roles(columns: Sequence[str],
                 sensitive_cols: Sequence[str],
                 category_map: Optional[Dict[str, str]] = None,
                 ee_threshold: float = 0.7,
                 qi_include_ee: bool = True) -> RoleAssignment:
    """
    컬럼을 민감/EE/QI 역할로 분류.

    · 민감속성 : sensitive_cols 그대로
    · EE 속성  : EE 사전 매칭 & ee_score ≥ threshold
    · QI 후보  : 비민감 컬럼 전체 (qi_include_ee=False면 EE 제외)
    """
    sensitive = list(sensitive_cols)
    sens_set = set(sensitive)
    ee_cols: List[str] = []
    ee_score_map: Dict[str, float] = {}

    if _HAS_EE:
        matched = eedict.match_columns(list(columns))
        for col, attr in matched.items():
            if attr is not None and attr.ee_score >= ee_threshold:
                ee_cols.append(col)
                ee_score_map[col] = attr.ee_score

    non_sensitive = [c for c in columns if c not in sens_set]
    if qi_include_ee:
        qi = non_sensitive
    else:
        qi = [c for c in non_sensitive if c not in set(ee_cols)]

    return RoleAssignment(
        sensitive=sensitive,
        ee=[c for c in ee_cols if c not in sens_set],
        qi=qi,
        ee_score_map=ee_score_map,
        category_map=category_map or {},
    )


# ===========================================================================
# 메인 진입점
# ===========================================================================

def filter_qi_by_cardinality(df: pd.DataFrame,
                             qi_cols: Sequence[str],
                             max_unique_ratio: float = 0.9,
                             k: int = 5,
                             target_kanon_ratio: float = 0.5) -> Tuple[List[str], List[str]]:
    """QI 후보에서 고유식별자/과세분 컬럼을 제거해 동치류가 실제로 형성되게 한다.

    준식별자(QI)의 취지는 '여러 사람이 공유하는' 속성으로 동치류(익명 그룹)를
    만드는 것이다. 값이 거의 고유한 컬럼(member_no·row_id)이나, 개별로는
    안 높아도 '조합'하면 모두를 유일하게 만드는 컬럼(signup_date, 연속 점수)이
    QI에 섞이면 모든 행이 크기 1 그룹이 되어 EMD·Coverage·세분성이 포화(=1.0)된다.
    위험이 실제로 높아서가 아니라 '식별자를 QI로 잘못 넣어서'다.

    2단계로 제거한다:
      1) 단일 컬럼 필터: 고유값 비율(nunique/N) > max_unique_ratio → 즉시 제외.
      2) 조합 필터(그리디): 남은 QI 조합의 k-익명 위반 비율이 목표
         (target_kanon_ratio) 이하로 내려갈 때까지, 고유율이 높은 컬럼부터
         하나씩 제거. 조합이 지나치게 세분화돼 그룹이 안 생기는 것을 막는다.
         (단, 최소 1개 컬럼은 남긴다.)

    반환: (사용할 QI, 제외된 QI)
    """
    n = len(df)
    if n == 0:
        return list(qi_cols), []
    cols = [c for c in qi_cols if c in df.columns]
    dropped: List[str] = []

    # --- 1) 단일 컬럼 고유율 필터 ---
    kept = []
    for c in cols:
        ratio = df[c].nunique(dropna=False) / n
        if ratio > max_unique_ratio:
            dropped.append(c)
        else:
            kept.append(c)

    # --- 2) 조합 k-익명 위반 비율 그리디 완화 ---
    def kanon_violation(cols_):
        if not cols_:
            return 0.0
        key = df[cols_].astype(str).fillna("nan").agg("\x1f".join, axis=1)  # pandas 3: NaN 유지 → join TypeError 방지
        freq = key.map(key.value_counts()).values
        return float((freq < k).sum() / n)

    # 고유율 높은 순으로 후보 정렬(먼저 뺄 대상)
    kept.sort(key=lambda c: df[c].nunique(dropna=False) / n, reverse=True)
    while len(kept) > 1 and kanon_violation(kept) > target_kanon_ratio:
        dropped.append(kept.pop(0))   # 가장 고유율 높은 것부터 제거

    # 원래 순서 보존해서 반환
    kept_set = set(kept)
    kept_ordered = [c for c in cols if c in kept_set]
    return kept_ordered, dropped


def run_exposure_metrics(df: pd.DataFrame,
                         sensitive_cols: Sequence[str],
                         category_map: Optional[Dict[str, str]] = None,
                         ee_threshold: float = 0.7,
                         qi_include_ee: bool = True,
                         qi_max_unique_ratio: float = 0.9,
                         bins: int = 10) -> Dict:
    """
    탐지 결과를 받아 (속성 × 지표) 위험 행렬을 계산한다.

    Parameters
    ----------
    df             : 원본 데이터프레임
    sensitive_cols : 탐지된 민감속성 컬럼명 리스트
    category_map   : {컬럼: 카테고리코드} (표시용, 선택)
    qi_max_unique_ratio : QI에서 제외할 고유값 비율 임계(기본 0.9).
        member_no·row_id 같은 (준)식별자가 QI에 섞여 지표가 포화되는 것을 막는다.
        None 이면 필터 비활성(구버전 동작).

    Returns
    -------
    dict with keys:
      · roles       : RoleAssignment (qi는 필터 후 최종 QI)
      · qi_dropped  : 카디널리티로 제외된 QI 컬럼 리스트
      · matrix      : DataFrame (index=민감속성, cols=지표) — 메인 산출물
      · summary     : DataFrame (참고용 요약 스칼라, 보조)
    """
    roles = assign_roles(df.columns, sensitive_cols,
                         category_map=category_map,
                         ee_threshold=ee_threshold,
                         qi_include_ee=qi_include_ee)

    # (보정) QI 후보에서 고유식별자/과세분 컬럼 제거 → 동치류가 실제로 형성됨
    qi_dropped: List[str] = []
    if qi_max_unique_ratio is not None:
        kept, qi_dropped = filter_qi_by_cardinality(
            df, roles.qi, max_unique_ratio=qi_max_unique_ratio)
        roles.qi = kept

    matrix = M.compute_metric_matrix(
        df,
        sensitive_cols=roles.sensitive,
        qi_cols=roles.qi,
        ee_cols=roles.ee,
        ee_score_map=roles.ee_score_map,
        ee_threshold=ee_threshold,
        bins=bins,
        category_map=roles.category_map,      # 부재(0 의 의미) 판정에 카테고리 사용
    )

    # 카테고리 채우기
    if not matrix.empty and roles.category_map:
        matrix["category"] = [roles.category_map.get(a, "") for a in matrix.index]

    # 분석 산출물은 (속성 × 지표) 행렬(지표를 합치지 않음). 이 행렬을 assess_dataset
    # 에 넘기면 데이터셋 H/M/L 등급 + 조치 순서(top-k)를 별도로 얻을 수 있다.
    return {"roles": roles, "qi_dropped": qi_dropped, "matrix": matrix}


# 노출 지표에 포함할 탐지 tier (진입점 공통 기본값).
#   [리팩토링] 예전에는 run_integrated.py(CLI) 기본이 WEAK 포함, 이 함수와
#   service.py 기본이 CONFIRMED, PROBABLE 만이라 같은 데이터가 진입점마다 다른
#   민감속성 집합, 등급을 냈다. 한 상수로 통일한다(CLI 기존 동작 = WEAK 포함).
#   WEAK 이면서 is_sensitive=True 인 것은 '법정 + 독립근거 부족(검수 대상)'이다.
DEFAULT_MIN_TIER: Tuple[str, ...] = ("CONFIRMED", "PROBABLE", "WEAK")


# ---------------------------------------------------------------------------
# 평가 누락 — 논문 3.2절: "탐지되지 않은 민감속성은 점수 0이 아닌 평가 누락으로 보고"
#   평가 누락 속성은 지표를 계산하지 않으며(0 으로 채우지 않음) 요약값, 순위, ρ 에서 빠진다.
# ---------------------------------------------------------------------------
OMITTED_LABEL = "(평가 누락)"
OMISSION_COLUMNS = ["attribute", "expected_category", "detected_category",
                    "detected_tier", "reason"]


def evaluation_omissions(df_columns: Sequence[str],
                         evaluated: Sequence[str],
                         expected_sensitive: Optional[Dict[str, str]] = None,
                         detection_result=None,
                         min_tier: Sequence[str] = DEFAULT_MIN_TIER) -> pd.DataFrame:
    """민감속성이지만 지표가 계산되지 않은 속성 목록 (평가 누락).

    expected_sensitive : {컬럼: 기대 범주} — 정답·검토 라벨·코드북 등 기준 민감속성.
                         주어지면 이 중 평가 집합에 없는 속성을 모두 보고한다.
    detection_result   : 탐지 결과. 기준이 없어도, 민감으로 판정됐지만 tier 가
                         min_tier 밖이라 평가에서 빠진 속성은 보고한다.
    """
    evaluated = set(map(str, evaluated))
    cols = set(map(str, df_columns))
    dec = {str(d.column): d for d in getattr(detection_result, "decisions", []) or []}
    rows: Dict[str, Dict] = {}

    def _add(attr, exp_cat, reason):
        d = dec.get(attr)
        rows.setdefault(attr, {
            "attribute": attr, "expected_category": exp_cat or "",
            "detected_category": getattr(d, "category", "") if d else "",
            "detected_tier": getattr(d, "tier", "") if d else "",
            "reason": reason})

    for attr, exp_cat in (expected_sensitive or {}).items():
        attr = str(attr)
        if attr in evaluated:
            continue
        d = dec.get(attr)
        if attr not in cols:
            why = "데이터에 컬럼 없음"
        elif d is None:
            why = "탐지 결과 없음"
        elif getattr(d, "is_sensitive", False):
            why = f"민감 판정됐으나 tier {d.tier} 가 평가 범위 밖"
        else:
            why = "탐지되지 않음 (비민감 판정)"
        _add(attr, exp_cat, why)
    for attr, d in dec.items():
        if (attr not in evaluated and attr in cols and getattr(d, "is_sensitive", False)
                and getattr(d, "tier", "NONE") not in min_tier):
            _add(attr, (expected_sensitive or {}).get(attr, ""),
                 f"민감 판정됐으나 tier {d.tier} 가 평가 범위 밖")
    return pd.DataFrame(list(rows.values()), columns=OMISSION_COLUMNS)


def run_from_detection(df: pd.DataFrame,
                       detection_result,
                       ee_threshold: float = 0.7,
                       qi_include_ee: bool = True,
                       bins: int = 10,
                       min_tier: Sequence[str] = DEFAULT_MIN_TIER,
                       expected_sensitive: Optional[Dict[str, str]] = None) -> Dict:
    """
    탐지 파이프라인의 RiskInput(result)을 직접 받아 지표 단계를 실행한다.

    detection_result.decisions 에서 is_sensitive=True 이고 tier가 min_tier에
    속하는 컬럼을 민감속성으로 뽑는다. 확정·유력만 보려면
    min_tier=("CONFIRMED", "PROBABLE").

    [v9.6] expected_sensitive({컬럼: 기대 범주})를 주면 그중 평가되지 않은 속성을
    결과의 "omissions"(평가 누락)로 보고한다. 이 속성들은 점수 0 이 아니다.
    """
    sens, cat_map = [], {}
    if any(not isinstance(c, str) for c in df.columns):
        df = df.copy()                              # 탐지 단계와 같은 문자열 컬럼명
        df.columns = [str(c) for c in df.columns]
    for d in detection_result.decisions:
        if getattr(d, "is_sensitive", False) and getattr(d, "tier", "NONE") in min_tier:
            if d.column in df.columns:
                sens.append(d.column)
                cat_map[d.column] = getattr(d, "category", "")

    out = run_exposure_metrics(
        df, sensitive_cols=sens, category_map=cat_map,
        ee_threshold=ee_threshold, qi_include_ee=qi_include_ee, bins=bins)
    out["omissions"] = evaluation_omissions(
        df.columns, sens, expected_sensitive=expected_sensitive,
        detection_result=detection_result, min_tier=min_tier)
    return out


NON_SENSITIVE_LABEL = "(비민감)"


def build_all_columns_matrix(df: pd.DataFrame,
                             roles: "RoleAssignment",
                             sensitive_matrix: Optional[pd.DataFrame] = None,
                             ee_threshold: float = 0.7,
                             bins: int = 10,
                             sensitive_first: bool = True,
                             omitted: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """민감속성(track A+B) + 비민감 컬럼을 모두 행으로 두는 '맥락용' 행렬 (히트맵 2).

    비민감 컬럼은 탐지 단계에서 이미 non-sensitive 로 확정됐으므로 위험 지표를
    계산하지 않는다. 행은 두되 지표 값은 NaN, category 는 "(비민감)" 으로 둔다.
    목적은 '민감속성의 위험이 데이터셋 전체에서 어디에 몰려 있는지'와
    '비민감 컬럼이 얼마나 많은지'를 한 화면에서 보여주는 것이다.

    민감속성 행은 sensitive_matrix(히트맵 1 행렬)의 값을 그대로 복사한다.
    → 두 히트맵에서 같은 셀은 항상 같은 값(재계산 없음).
    sensitive_matrix 가 없으면 roles 로 한 번 계산한다(하위호환).

    반환: index=컬럼명, columns=[category, is_sensitive, emd, coverage, combo,
          ee_match, granularity, ...]. 비민감 행의 지표 열은 NaN.
    """
    cols = list(df.columns)
    if not cols:
        return pd.DataFrame()

    if sensitive_matrix is None:
        sensitive_matrix = M.compute_metric_matrix(
            df, sensitive_cols=roles.sensitive, qi_cols=roles.qi,
            ee_cols=roles.ee, ee_score_map=roles.ee_score_map,
            ee_threshold=ee_threshold, bins=bins,
            category_map=roles.category_map)
        if not sensitive_matrix.empty and roles.category_map:
            sensitive_matrix["category"] = [
                roles.category_map.get(a, "") for a in sensitive_matrix.index]

    sens_rows = [c for c in cols if c in set(sensitive_matrix.index)]
    omit_set = set(map(str, omitted or [])) - set(sens_rows)
    omit_rows = [c for c in cols if c in omit_set]                 # 평가 누락
    nons_rows = [c for c in cols if c not in set(sens_rows) and c not in omit_set]
    order = (sens_rows + omit_rows + nons_rows) if sensitive_first else cols

    out = sensitive_matrix.reindex(order).copy()   # 비민감, 평가 누락 행은 전부 NaN
    if "category" not in out.columns:
        out["category"] = ""
    out["category"] = out["category"].astype(object)
    out.loc[nons_rows, "category"] = NON_SENSITIVE_LABEL
    out.loc[omit_rows, "category"] = OMITTED_LABEL
    out["is_sensitive"] = [c in set(sens_rows) for c in out.index]
    out["evaluation"] = ["evaluated" if c in set(sens_rows)
                         else "omitted" if c in omit_set else "non_sensitive"
                         for c in out.index]
    out.index.name = "attribute"
    return out


def matrix_to_view_json(matrix: pd.DataFrame,
                        category_map: Optional[Dict[str, str]] = None) -> List[Dict]:
    """
    (속성×지표) 행렬을 뷰(히트맵 위젯)용 JSON 레코드로 변환.
    각 레코드: {attribute, category, emd, coverage, combo, ee_match, granularity,
               {지표}_status ...}. [v9.6] 확정·비적용 값은 null.
    """
    if matrix.empty:
        return []
    cols = ["emd", "coverage", "combo", "ee_match", "granularity"]
    st = {c: metric_status(matrix, c) for c in cols}
    recs = []
    for attr, row in matrix.iterrows():
        rec = {"attribute": attr,
               "category": (category_map or {}).get(attr, row.get("category", ""))}
        for c in cols:
            v = row.get(c, np.nan)
            rec[c] = None if pd.isna(v) else round(float(v), 3)
            rec[f"{c}_status"] = st[c].get(attr, M.STATUS_NA)
        recs.append(rec)
    return recs


def _cell_label(v: float, status: str) -> str:
    """히트맵 셀 글자: 수치면 0.00, 아니면 상태(확정/N/A)."""
    if not pd.isna(v):
        return f"{v:.2f}"
    return "확정" if status == M.STATUS_DETERMINED else "N/A"


# ===========================================================================
# 지표 선택 프레임워크 (분석용 ,  (A) 층)
# ---------------------------------------------------------------------------
# 지표들은 단위, 의미가 서로 달라서 하나의 최종값으로 합치면(덧셈, 곱셈 모두)
# 근거가 약해진다. 그래서 '분석' 단계에서는 합치지 않고 (속성×지표) 행렬 그대로
# 두되, 이용자가
#   (1) 목적에 맞는 지표를 고르고(select_metrics),
#   (2) 그 지표가 무엇을 재는지 설명을 보고(describe_metrics),
#   (3) 참고 기준선으로 '주목할 셀'을 짚는다(flag_by_reference).
# 데이터셋 단위의 H/M/L 등급, 조치 순서가 필요하면 '판정' 단계(assess_dataset)를
# 쓴다 — 둘은 목적이 다른 별도 층이다.
# ===========================================================================

from .metrics import (METRIC_SPECS, METRIC_ORDER,           # noqa: E402
                      internal_metrics, external_metrics)


def describe_metrics(keys: Optional[Sequence[str]] = None) -> str:
    """지표 설명을 사람이 읽는 텍스트로. 각 지표가 '무엇을 재는가' + 참고 기준선.

    이용자는 이 설명을 보고 자신의 목적(예: 재식별 방지 vs 동질성 노출 방지)에
    맞는 지표를 select_metrics로 골라 해석한다. (장단점 서술은 두지 않는다.)
    """
    keys = list(keys) if keys else METRIC_ORDER
    axis_ko = {"internal": "내부 분포 위험", "external": "외부 분포 위험"}
    lines = ["지표 설명  (전부 0~1, 클수록 위험)", "=" * 68]
    for k in keys:
        s = METRIC_SPECS[k]
        lines.append(f"[{s.name_ko}] {s.name_en}  ·  축: {axis_ko[s.axis]}")
        lines.append(f"  측정      : {s.measures}")
        lines.append(f"  참고기준선: {s.ref_watch:.2f} 이상 주목  ({s.ref_note})")
        lines.append("")
    lines.append("※ 참고기준선은 고정 등급이 아니라 데이터·맥락에 따라 조정할 "
                 "실무 '주목선'입니다.")
    return "\n".join(lines)


def select_metrics(matrix: pd.DataFrame,
                   keys: Optional[Sequence[str]] = None,
                   axis: Optional[str] = None) -> pd.DataFrame:
    """행렬에서 이용자가 고른 지표 열만 남긴 부분 행렬.

    keys : 볼 지표 키 리스트(예: ["emd","coverage"]). None이면 전체.
    axis : "internal" 또는 "external" 로 한 축만 선택(keys와 함께 쓰면 교집합).
    """
    if matrix is None or matrix.empty:
        return matrix
    chosen = list(keys) if keys else list(METRIC_ORDER)
    if axis == "internal":
        chosen = [k for k in chosen if k in internal_metrics()]
    elif axis == "external":
        chosen = [k for k in chosen if k in external_metrics()]
    cols = [c for c in chosen if c in matrix.columns]
    if "category" in matrix.columns:          # 있으면 맨 앞에 붙여 맥락 유지
        cols = ["category"] + cols
    return matrix[cols]


def flag_by_reference(matrix: pd.DataFrame,
                      keys: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """각 지표의 '참고기준선'을 넘는 (속성) 목록을 지표별로 반환.

    등급을 매기는 게 아니라 '한 번 더 볼' 후보를 가리키는 포인터일 뿐이다.
    이용자는 기준선 자체를 자신의 데이터에 맞게 조정할 수 있다.
    반환: {지표키: [기준선 초과 속성, ...]}
    """
    if matrix is None or matrix.empty:
        return {}
    keys = list(keys) if keys else METRIC_ORDER
    out: Dict[str, List[str]] = {}
    for k in keys:
        if k not in matrix.columns:
            continue
        thr = METRIC_SPECS[k].ref_watch
        out[k] = [str(a) for a, v in matrix[k].items() if float(v) >= thr]
    return out


_CJK_FP_CACHE = "__unset__"


def _cjk_fontprop():
    """한글 라벨용 CJK 폰트(FontProperties).

    우선순위: (1) 패키지에 동봉한 NanumGothic → (2) 시스템 설치 CJK 폰트 →
    (3) None. 동봉 폰트가 있으므로 한글 폰트가 없는 환경(Colab/VSCode 등)에서도
    라벨이 □□□(tofu)로 깨지지 않는다. 찾으면 rcParams 전역 폰트로도 등록해
    fontproperties 를 명시하지 않은 title/legend 까지 한글이 렌더된다.
    """
    global _CJK_FP_CACHE
    if _CJK_FP_CACHE != "__unset__":
        return _CJK_FP_CACHE
    try:
        import os as _os
        import matplotlib.font_manager as fm
        import matplotlib as _mpl
    except Exception:
        _CJK_FP_CACHE = None
        return None

    fp = None
    # (1) 동봉 폰트
    bundled = _os.path.join(_os.path.dirname(__file__), "assets", "NanumGothic.ttf")
    if _os.path.exists(bundled):
        try:
            fm.fontManager.addfont(bundled)
            fp = fm.FontProperties(fname=bundled)
        except Exception:
            fp = None
    # (2) 시스템 폰트
    if fp is None:
        for cand in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic",
                     "Malgun Gothic", "AppleGothic", "Noto Sans CJK SC"):
            try:
                path = fm.findfont(fm.FontProperties(family=cand),
                                   fallback_to_default=False)
                fp = fm.FontProperties(fname=path)
                break
            except Exception:
                continue
    # 전역 등록(명시 안 한 텍스트까지 한글 렌더) + 마이너스 기호 깨짐 방지
    if fp is not None:
        try:
            _mpl.rcParams["font.family"] = fp.get_name()
            _mpl.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass
    _CJK_FP_CACHE = fp
    return fp


def render_heatmap(matrix: pd.DataFrame,
                   out_path: str = "exposure_heatmap.png",
                   category_map: Optional[Dict[str, str]] = None,
                   merge_coverage_emd: bool = False,
                   coverage_floor: Optional[float] = None,
                   title: Optional[str] = None,
                   max_nonsensitive_rows: Optional[int] = None,
                   ns_style: str = "tint",
                   coverage_bars: bool = True) -> Optional[str]:
    """(속성×지표) 행렬을 히트맵 PNG로 저장 — 내부=파랑, 외부=빨강 계열.

    [v9.5] coverage_bars=True(기본): Coverage 를 히트맵 칸에서 빼고, 오른쪽에 컬럼별
    회색 가로 막대(있음 비율 0~1)로 따로 그린다. 점선 = coverage_floor(기본 0.30).
    히트맵의 EMD 칸은 Coverage 를 곱하지 않은 순수 EMD 다.
    merge_coverage_emd=True 는 예전 표시('EMD·Coverage = Coverage × EMD' 한 칸, 막대 없음).
    두 옵션 모두 표시용이며 원본 행렬·CSV·위험계산(risk_components)에는 영향이 없다.

    [비민감 행] build_all_columns_matrix() 결과처럼 지표 값이 전부 NaN 인 행
    (또는 category == "(비민감)")은 '확정 비민감' 으로 보고 수치를 그리지 않는다.
      · ns_style="tint"(기본): 내부(파랑) 열 묶음·외부(빨강) 열 묶음을 각각
        하나의 박스로 옅게 깔고 사선 해칭, 행 가운데에 "non-sensitive" 표시.
        속성명은 민감 행과 같은 검정. 해칭으로 "0.00(위험 없음)" 과 구분한다.
      · ns_style="gray": 예전 방식(행 전체 연회색 + 가운데 "non-sensitive").
      · 비민감 행 표시 개수(max_nonsensitive_rows):
          None(기본) → 전부 표시 / 0 → "비민감 컬럼 N개" 요약 한 줄만
          k>0  → 앞 k개만 표시하고 나머지는 "… 외 M개" 한 줄로 요약.
    비민감 행이 없으면 기존과 동일하게 그린다.
    matplotlib 없으면 None.
    """
    try:
        import warnings
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except Exception:
        return None
    if matrix is None or matrix.empty:
        return None

    fp = _cjk_fontprop()

    # --- 민감 / 비민감 행 분리 ------------------------------------------------
    metric_cols = [k for k in METRIC_ORDER if k in matrix.columns]
    all_nan = matrix[metric_cols].isna().all(axis=1) if metric_cols else \
        pd.Series(True, index=matrix.index)
    if "category" in matrix.columns:
        cat_s = matrix["category"].astype(str)
        is_ns = all_nan | cat_s.isin([NON_SENSITIVE_LABEL, OMITTED_LABEL])
        omitted_names = set(map(str, matrix.index[(cat_s == OMITTED_LABEL).values]))
    else:
        is_ns = all_nan
        omitted_names = set()
    sens_m = matrix[~is_ns.values]
    ns_names = [str(a) for a in matrix.index[is_ns.values]]
    n_ns = len(ns_names)
    n_omit = len(omitted_names & set(ns_names))
    n_ns -= n_omit                                  # 제목의 '비민감 N개'에는 누락 제외

    def _ns_text(name):                             # 평가 누락 행은 따로 표시
        return "평가 누락 (not evaluated)" if name in omitted_names else "non-sensitive"
    # ── 비민감 행 표시 개수 조절 ──────────────────────────────────────
    omit_rows = [a for a in ns_names if a in omitted_names]    # 평가 누락은 항상 전부 표시
    pure_ns = [a for a in ns_names if a not in omitted_names]
    if max_nonsensitive_rows is None or n_ns <= max_nonsensitive_rows:
        ns_rows = pure_ns                                   # 전부 표시
    elif max_nonsensitive_rows <= 0:
        ns_rows = [f"비민감 컬럼 {n_ns}개 (목록 생략)"]        # 요약 한 줄만
    else:
        k = int(max_nonsensitive_rows)                      # 앞 k개 + 요약
        ns_rows = pure_ns[:k] + [f"… 외 비민감 {n_ns - k}개"]
    ns_rows = omit_rows + ns_rows

    # --- 표시할 (라벨, 축, 값벡터, 상태벡터) — 민감 행 기준 ---------------------
    disp: List[Tuple[str, str, "object", "object"]] = []
    if coverage_bars:
        merge_coverage_emd = False
    has_merge = (merge_coverage_emd and "emd" in sens_m.columns
                 and "coverage" in sens_m.columns)
    if has_merge:
        emd_cov = (sens_m["coverage"].astype(float)
                   * sens_m["emd"].astype(float)).values
        disp.append(("EMD·Coverage", "internal", emd_cov,
                     metric_status(sens_m, "emd").values))
        for k in METRIC_ORDER:
            if k in ("emd", "coverage") or k not in sens_m.columns:
                continue
            disp.append((METRIC_SPECS[k].name_ko, METRIC_SPECS[k].axis,
                         sens_m[k].astype(float).values,
                         metric_status(sens_m, k).values))
    else:
        for k in metric_cols:
            if coverage_bars and k == "coverage":
                continue                                    # 오른쪽 막대 패널로
            disp.append((METRIC_SPECS[k].name_ko, METRIC_SPECS[k].axis,
                         sens_m[k].astype(float).values,
                         metric_status(sens_m, k).values))
    show_bars = coverage_bars and "coverage" in sens_m.columns

    s_attrs = [str(a) for a in sens_m.index]
    n_s = len(s_attrs)
    attrs = s_attrs + ns_rows
    nrow = len(attrs)
    ncol = len(disp)
    blue = LinearSegmentedColormap.from_list("blue", ["#eaf2fb", "#0d3b66"])
    red = LinearSegmentedColormap.from_list("red", ["#fbeaea", "#7a1f1f"])
    NS_FILL, NS_TEXT = "#f3f3f3", "#a0a0a0"

    if show_bars:
        fig = plt.figure(figsize=(1.25 * ncol + 4.6, 0.5 * nrow + 2.6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25 * ncol, 1.9], wspace=0.04)
        ax = fig.add_subplot(gs[0])
        bx = fig.add_subplot(gs[1], sharey=ax)
    else:
        fig, ax = plt.subplots(figsize=(1.25 * ncol + 2.8, 0.5 * nrow + 2.2))
        bx = None

    # 민감 블록: 지표별 색 + 수치
    if n_s:
        for j, (label, axis_kind, vals, stats) in enumerate(disp):
            cmap = blue if axis_kind == "internal" else red
            ax.imshow(np.asarray(vals, dtype=float).reshape(-1, 1), cmap=cmap,
                      vmin=0, vmax=1, aspect="auto",
                      extent=[j - 0.5, j + 0.5, n_s - 0.5, -0.5])
            for i in range(n_s):
                v = float(vals[i])
                if np.isnan(v):                     # 확정, 비적용: 수치 대신 상태
                    det = stats[i] == M.STATUS_DETERMINED
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1.0, 1.0,
                        facecolor="#fff4d6" if det else "#f3f3f3",
                        edgecolor="white", linewidth=1.0, zorder=1))
                    ax.text(j, i, _cell_label(v, stats[i]), ha="center", va="center",
                            color="#8a5a00" if det else "#a0a0a0", fontsize=8.5,
                            fontproperties=fp, zorder=2)
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v >= 0.5 else "#222", fontsize=9)

    # 비민감 블록 (수치 없음, 위험 미산출)
    if ns_style == "gray":
        # 예전 방식: 연회색 행 + "non-sensitive"
        for r in range(n_s, nrow):
            ax.add_patch(plt.Rectangle((-0.5, r - 0.5), ncol, 1.0,
                                       facecolor=NS_FILL, edgecolor="white",
                                       linewidth=1.0, zorder=1))
            ax.text((ncol - 1) / 2, r, _ns_text(attrs[r]), ha="center",
                    va="center", color=NS_TEXT, fontsize=9, style="italic",
                    fontproperties=fp, zorder=2)
    else:
        # tint 방식: 내부(파랑) 열 묶음 / 외부(빨강) 열 묶음을 각각 하나의 박스로
        #  — 옅은 톤 + 사선 해칭. 행 경계는 흰 선, 각 행 가운데에 "non-sensitive".
        NS_TINT = {"internal": ("#f4f8fd", "#d3e1f1"),   # (바탕, 해칭선)
                   "external": ("#fdf5f5", "#f0d4d4")}
        n_ns_rows = nrow - n_s
        if n_ns_rows > 0:
            # 같은 축(internal/external)이 연속된 열 구간 → 박스 1개씩
            spans, start = [], 0
            for j in range(1, ncol + 1):
                if j == ncol or disp[j][1] != disp[start][1]:
                    spans.append((start, j, disp[start][1]))
                    start = j
            for (a, b, axis_kind) in spans:
                face, hatch_c = NS_TINT.get(axis_kind, NS_TINT["internal"])
                ax.add_patch(plt.Rectangle(
                    (a - 0.5, n_s - 0.5), b - a, n_ns_rows,
                    facecolor=face, edgecolor=hatch_c, hatch="////",
                    linewidth=0, zorder=1))
            # 행 경계만 흰 선으로 (열 사이 세로선 없음 → 행마다 파랑 박스 1개 + 빨강 박스 1개)
            for r in range(n_s + 1, nrow):
                ax.axhline(r - 0.5, color="white", linewidth=1.2, zorder=2)
            # 행마다 전체 폭 가운데에 "non-sensitive"
            for r in range(n_s, nrow):
                omit = attrs[r] in omitted_names
                ax.text((ncol - 1) / 2, r, _ns_text(attrs[r]), ha="center",
                        va="center", color="#8a5a00" if omit else NS_TEXT,
                        fontsize=9, fontproperties=fp, zorder=3)

    # 구분선 (민감 ↔ 비민감)
    if n_s and len(ns_rows):
        if ns_style == "gray":
            ax.axhline(n_s - 0.5, color="#555", linewidth=1.4, zorder=3)
        else:
            ax.axhline(n_s - 0.5, color="#6b7280", linewidth=1.0,
                       linestyle=(0, (4, 3)), zorder=3)

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([d[0] for d in disp], rotation=0 if show_bars else 30,
                       ha="center" if show_bars else "left",
                       fontsize=9, fontproperties=fp)
    ax.xaxis.tick_top()
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(attrs, fontsize=9, fontproperties=fp)
    # 비민감 속성명도 민감 컬럼과 동일한 색(검정) — 별도 스타일 없음
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)

    if show_bars:
        # 내부/외부 열 묶음 머리글 (색 선 + 라벨) — 막대 패널 머리글과 같은 높이
        spans, start = [], 0
        for j in range(1, ncol + 1):
            if j == ncol or disp[j][1] != disp[start][1]:
                spans.append((start, j, disp[start][1]))
                start = j
        for (a, b, axis_kind) in spans:
            col = "#4a6f93" if axis_kind == "internal" else "#a8605e"
            lab = "내부 (분포)" if axis_kind == "internal" else "외부 (연결)"
            ax.plot([a - 0.45, b - 0.55], [-1.05, -1.05], color=col, lw=2,
                    clip_on=False, solid_capstyle="butt")
            ax.text((a + b - 1) / 2, -1.15, lab, ha="center", va="bottom",
                    color=col, fontsize=9, fontproperties=fp, clip_on=False)
        floor = COVERAGE_FLOOR if coverage_floor is None else float(coverage_floor)
        cov = sens_m["coverage"].astype(float).values
        bx.barh(np.arange(n_s), cov, height=0.55, color="#a6a6a6", zorder=2)
        for i, v in enumerate(cov):
            if np.isnan(v):
                continue
            if v > 0.75:
                bx.text(v - 0.03, i, f"{v:.2f}", va="center", ha="right",
                        fontsize=8.5, color="white", zorder=3)
            else:
                bx.text(v + 0.03, i, f"{v:.2f}", va="center", ha="left",
                        fontsize=8.5, color="#444", zorder=3)
        for r in range(n_s, nrow):                       # 비민감: 미산출
            bx.text(0.5, r, "–", va="center", ha="center", color=NS_TEXT, fontsize=9)
        bx.plot([floor, floor], [-0.5, n_s - 0.5], ls="--", lw=1, color="#777", zorder=1)
        if n_s and len(ns_rows):
            bx.axhline(n_s - 0.5, color="#6b7280", linewidth=1.0,
                       linestyle=(0, (4, 3)), zorder=3)
        bx.set_xlim(0, 1.0)
        bx.set_xticks([0, floor, 1])
        bx.set_xticklabels(["0", f"{floor:g}\n(floor)", "1"], fontsize=8)
        bx.xaxis.tick_top()
        bx.tick_params(axis="both", length=0, labelleft=False)
        bx.text(0.5, -1.15, "Coverage (있음 비율)", ha="center", va="bottom",
                color="#666", fontsize=9, fontproperties=fp, clip_on=False)
        for sp in bx.spines.values():
            sp.set_visible(False)

    _title = title or "Exposure risk matrix (blue=internal / red=external; 0~1)"
    if n_ns or n_omit:
        _title += (f"\n민감 {n_s}개 / 전체 {n_s + n_ns + n_omit}개 컬럼"
                   + (f"  (회색 = 확정 비민감 {n_ns}개, 위험 미산출)"
                      if ns_style == "gray" else
                      f"  (빗금 = 확정 비민감 {n_ns}개, 위험 미산출)")
                   + (f"  · 평가 누락 {n_omit}개" if n_omit else ""))
    if show_bars:
        # 표 머리(내부/외부 묶음 라벨) 바로 위에 — 그림 높이와 무관하게 간격 고정
        ax.set_title(_title, fontsize=10, fontproperties=fp, pad=44)
    else:
        ax.set_title(_title, fontsize=10, fontproperties=fp, pad=12)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not show_bars:
            fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight" if show_bars else None)
    plt.close(fig)
    return out_path


def render_metric_views(matrix: pd.DataFrame,
                        out_prefix: str = "metric_view",
                        keys: Optional[Sequence[str]] = None,
                        combined: bool = True) -> Dict[str, Optional[str]]:
    """지표별 '개별 뷰' — 각 지표를 따로 히트맵으로 그린다 (문서: 개별 뷰 + 통합).

    통합 히트맵(render_heatmap)이 5개 지표를 한 판에 모아 보여준다면, 이 함수는
    지표를 '하나씩' 떼어 (속성 × 그 지표 하나) 세로 히트맵으로 그린다. 지표를
    합치지 않고 각각 해석하라는 프레임워크에 맞춘 시각화다.

    Parameters
    ----------
    out_prefix : 파일 접두사. 개별 파일은 f"{out_prefix}_{key}.png".
    keys       : 그릴 지표(기본: 전체 5개, METRIC_ORDER 순).
    combined   : True면 소형 다중 패널(small-multiples) 한 장도 함께 생성
                 (f"{out_prefix}_all.png"). 논문/화면에 '개별 뷰'를 한 눈에.

    Returns
    -------
    dict : {지표key: png경로 or None, ..., ("all": 통합패널경로 if combined)}.
           matplotlib 미설치/빈 행렬이면 값은 None.
    """
    out: Dict[str, Optional[str]] = {}
    try:
        import warnings
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except Exception:
        return out
    if matrix is None or matrix.empty:
        return out

    fp = _cjk_fontprop()
    keys = [k for k in (keys or METRIC_ORDER) if k in matrix.columns]
    attrs = [str(a) for a in matrix.index]
    blue = LinearSegmentedColormap.from_list("blue", ["#eaf2fb", "#0d3b66"])
    red = LinearSegmentedColormap.from_list("red", ["#fbeaea", "#7a1f1f"])

    def _draw_single(ax, k):
        spec = METRIC_SPECS[k]
        cmap = blue if spec.axis == "internal" else red
        col = matrix[k].astype(float).values.reshape(-1, 1)
        stats = metric_status(matrix, k).values
        ax.imshow(col, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, v in enumerate(col[:, 0]):
            ax.text(0, i, _cell_label(v, stats[i]), ha="center", va="center",
                    color=("white" if v >= 0.5 else "#222") if not np.isnan(v) else "#8a8a8a",
                    fontsize=9, fontproperties=fp)
        ax.set_xticks([0])
        ax.set_xticklabels([spec.name_ko], fontsize=9, fontproperties=fp)
        ax.set_yticks(range(len(attrs)))
        ax.set_yticklabels(attrs, fontsize=9, fontproperties=fp)
        axis_ko = "내부" if spec.axis == "internal" else "외부"
        ax.set_title(f"{spec.name_ko} ({axis_ko})", fontsize=10, fontproperties=fp)
        ax.tick_params(length=0)

    # 1) 지표별 개별 파일
    for k in keys:
        fig, ax = plt.subplots(figsize=(2.4, 0.55 * len(attrs) + 1.6))
        _draw_single(ax, k)
        path = f"{out_prefix}_{k}.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
            fig.savefig(path, dpi=130)
        plt.close(fig)
        out[k] = path

    # 2) 소형 다중 패널(개별 뷰를 한 장에)
    if combined and keys:
        fig, axes = plt.subplots(
            1, len(keys), figsize=(2.1 * len(keys) + 1.0, 0.55 * len(attrs) + 2))
        if len(keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, keys):
            _draw_single(ax, k)
            if k != keys[0]:
                ax.set_yticks([])            # 왼쪽 패널만 속성 라벨
        path = f"{out_prefix}_all.png"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.suptitle("지표별 개별 뷰 (내부=파랑 / 외부=빨강)",
                         fontsize=11, fontproperties=fp)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(path, dpi=130)
        plt.close(fig)
        out["all"] = path

    return out


# ===========================================================================
# 데이터셋 위험 등급 (H / M / L) + 조치 우선순위 (top-k)
# ---------------------------------------------------------------------------
# 지표는 '분석'용으로는 합치지 않는다(위 프레임워크). 여기서는 데이터셋 하나를
# 한 등급으로 요약하고, 어디부터 손댈지 순서를 주기 위한 '판정' 단계를 둔다.
# 흐름:
#   1) 속성별 통합 위험  = max(내부 지표 최댓값, 외부 지표 최댓값)   ← integrated_risk
#   2) 임계 τ 를 넘는 속성 판정  (통합위험 ≥ τ 이면 '초과')
#   3) 임계 초과 비율     ρ = (초과 속성 수) / (전체 민감속성 수)     ← exceed_ratio
#   4) ρ 로 데이터셋을 H / M / L 로 분류                             ← classify_dataset
#   5) 초과 속성을 위험 내림차순으로 정렬 → 조치 순서(top-k)         ← topk_actions
#   ⚠️ 등급, 순서는 '무엇을 먼저 볼지' 요약일 뿐, 지표별 해석을 대체하지 않는다.
#      실제 원인, 성격은 반드시 (속성×지표) 행렬로 되돌아가 확인한다.
# ===========================================================================

# --- 임계값 파라미터 (데이터, 정책에 맞게 조정) -------------------------------
DEFAULT_ATTR_THRESHOLD = 0.5        # τ: 속성 통합위험이 이 값 이상이면 '초과'
RHO_CUTS = (0.50, 0.20)             # ρ≥0.50 → High, ρ≥0.20 → Middle, else Low
DATASET_GRADE_LABELS = ("High", "Middle", "Low")

# --- 3색 등급 팔레트 (첨부 예시 스타일: High=빨강 / Middle=노랑 / Low=연두) ----
#   점, 범례 색(진하게) 과 배경 밴드 색(연하게) 을 분리해 둔다.
BAND_COLORS = {"High": "#e2564d", "Middle": "#e8a33c", "Low": "#7fb04f"}
BAND_BG_COLORS = {"High": "#fbe9e7", "Middle": "#fcf3e0", "Low": "#eef5e6"}
BAND_LEGEND_KO = {"High": "High 위험", "Middle": "Middle 중간", "Low": "Low 안심"}
DATASET_GRADE_COLORS = BAND_COLORS   # 데이터셋 등급 배지도 같은 팔레트 사용

# 속성 1개(점)를 통합위험 '값'으로 3등급 분류할 때의 컷 (배경 밴드/점 색용).
#   High 컷은 τ 와 같게 둬서 '빨강 점 = ρ 초과 속성' 이 되도록 정렬.
RISK_VALUE_CUTS = (0.50, 0.25)      # 값≥0.50 → High, ≥0.25 → Middle, else Low


def value_band(value: float,
               cuts: Tuple[float, float] = RISK_VALUE_CUTS) -> str:
    """속성 통합위험 '값' → 3등급 High/Middle/Low (곡선 점·배경 밴드용).

    cuts = (high 하한, middle 하한). 예: (0.50, 0.25)이면
      값 ≥ 0.50 → High,  값 ≥ 0.25 → Middle,  그 외 → Low.
    """
    hi, mid = cuts
    if value >= hi:
        return "High"
    if value >= mid:
        return "Middle"
    return "Low"


# Coverage 게이트 하한: 이 채움율 이상이면 게이트=1(감점 없음), 미만이면
# 선형으로 0까지 낮춘다. Coverage 는 도크스트링대로 '게이트 성격' 지표라
# 위험을 끌어올리는 성분이 아니라, 값이 비면(=노출될 게 없으면) 내부위험을
# 눌러주는 역할만 하도록 둔다. 데이터, 정책에 맞게 조정.
COVERAGE_FLOOR = 0.30


# ---------------------------------------------------------------------------
# 산출 상태 (수치 / 확정 / 비적용)
#   행렬의 '{지표}_status' 열을 읽는다. 상태 열이 없는 예전 행렬은 값이 NaN 이면
#   비적용, 아니면 수치로 본다(하위 호환).
# ---------------------------------------------------------------------------
STATUS_REPORT_METRICS: Tuple[str, ...] = ("emd", "coverage", "combo",
                                          "external_nmi", "granularity", "ee_match")


def metric_status(matrix: pd.DataFrame, key: str) -> pd.Series:
    """속성별 산출 상태 Series (numeric | determined | not_applicable)."""
    col = f"{key}_status"
    if col in matrix.columns:
        return matrix[col].fillna(M.STATUS_NA).astype(str)
    if key not in matrix.columns:
        return pd.Series(M.STATUS_NA, index=matrix.index)
    return matrix[key].isna().map({True: M.STATUS_NA, False: M.STATUS_NUMERIC})


def status_counts(matrix: pd.DataFrame,
                  keys: Sequence[str] = STATUS_REPORT_METRICS) -> pd.DataFrame:
    """지표별 상태별 속성 수 (index=지표, columns=수치/확정/비적용)."""
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=[M.STATUS_KO[s] for s in M.STATUSES])
    rows = {}
    for k in keys:
        if k not in matrix.columns and f"{k}_status" not in matrix.columns:
            continue
        vc = metric_status(matrix, k).value_counts()
        rows[k] = {M.STATUS_KO[s]: int(vc.get(s, 0)) for s in M.STATUSES}
    return pd.DataFrame.from_dict(rows, orient="index")


def determined_attributes(matrix: pd.DataFrame) -> List[str]:
    """확정 상태 성분을 하나라도 가진 속성 — 우선 검토 대상."""
    if matrix is None or matrix.empty:
        return []
    hit = pd.Series(False, index=matrix.index)
    for k in STATUS_REPORT_METRICS:
        if k in matrix.columns or f"{k}_status" in matrix.columns:
            hit |= metric_status(matrix, k).eq(M.STATUS_DETERMINED)
    return [str(a) for a in matrix.index[hit.values]]


def risk_components(matrix: pd.DataFrame, sort: bool = False,
                    coverage_floor: float = COVERAGE_FLOOR) -> pd.DataFrame:
    """각 민감속성의 두 축 위험을 '합치지 않고' 분리해서 반환.

      · internal_risk (내부 분포 위험 max) = coverage_gate × max(EMD, Combo)
      · external_risk (외부 분포 위험 max) = EE_match
        └ coverage_gate = clip(coverage / coverage_floor, 0, 1)
          채움율이 floor 이상이면 1(감점 없음), 값이 비어갈수록 0으로 낮춤.
      · 세분성(granularity)은 외부축에 직접 더하지 않는다. 세분성은 이미
        EE_match = gate × √(external_nmi × granularity) 공식 안에 들어가 있어,
        외부축을 max(EE_match, 세분성)으로 두면 세분성이 이중 계산된다.
        따라서 외부축 = EE_match 하나만 쓰고, 세분성은 EE_match 안에서만 반영한다.
        (granularity 열 자체는 행렬·히트맵·select_metrics 에 진단용으로 남는다.)

    ⚠️ 예전에는 내부 = max(EMD, Coverage, Combo) 였다. 실데이터는 결측이 거의
    없어 Coverage≈1.0 이 되고, max 성분이라 EMD·Combo 가 아무리 낮아도 내부
    위험이 1.0 으로 포화됐다. Coverage 를 '끌어올리는 성분'이 아니라 '게이트'로
    바꿔 이 포화를 없앤다(자세한 설명은 CHANGES 참고).

    반환: (attribute × {internal_risk, external_risk}) DataFrame.
    integrated_risk 와 달리 두 축을 하나의 스칼라로 뭉개지 '않는다'. 이것이
    보고용 '결과값'이다. sort=True 면 내부 위험 내림차순(동률 시 외부)으로 정렬.

    [v9.6] 확정·비적용 성분(NaN)은 max 에서 빠진다. 예) EMD 확정 + Combo 수치
    → 내부 = gate × Combo. 축의 모든 성분이 수치가 아니면 그 축은 NaN.
    """
    if matrix is None or matrix.empty:
        return pd.DataFrame(columns=["internal_risk", "external_risk"],
                            dtype=float)
    # 내부축에서 coverage 는 게이트로 분리, 나머지(EMD, Combo)만 max
    icols = [c for c in internal_metrics()
             if c in matrix.columns and c != "coverage"]
    # 외부축: 세분성(granularity)은 EE_match 공식 √(external_nmi×granularity)
    # 안에서만 반영하고, 여기서 다시 max 로 더하지 않는다(이중 계산 방지).
    # 따라서 external_metrics() 중 granularity 는 제외하고 ee_match 만 쓴다.
    ecols = [c for c in external_metrics()
             if c in matrix.columns and c != "granularity"]
    base = (matrix[icols].astype(float).max(axis=1, skipna=True) if icols
            else pd.Series(np.nan, index=matrix.index))
    if "coverage" in matrix.columns:
        floor = max(1e-9, float(coverage_floor))
        gate = (matrix["coverage"].astype(float) / floor).clip(0.0, 1.0)
    else:
        gate = pd.Series(1.0, index=matrix.index)   # coverage 열 없으면 게이트 없음
    imax = gate * base                                    # 내부 분포 위험 max
    emax = (matrix[ecols].astype(float).max(axis=1, skipna=True) if ecols  # 외부 분포 위험 max
            else pd.Series(np.nan, index=matrix.index))
    comp = pd.DataFrame({"internal_risk": imax, "external_risk": emax})
    if sort:
        comp = comp.sort_values(["internal_risk", "external_risk"],
                                ascending=False)
    return comp


def integrated_risk(matrix: pd.DataFrame, sort: bool = True,
                    coverage_floor: float = COVERAGE_FLOOR) -> pd.Series:
    """[등급 판정·정렬 전용] 내부·외부 위험을 max 로 합친 단일 스칼라.

    ⚠️ 결과 '값'은 더 이상 이 함수로 보고하지 않는다. 내부/외부 분포 위험은
    risk_components() 로 각각 분리해서 낸다(요청: 하나의 값으로 뭉개지 않음).
    이 함수는 τ 초과 판정과 조치 순서 정렬에 필요한 '단일 순위 스칼라'
    (= max(내부, 외부)) 를 만들 때만 내부적으로 쓴다. 따라서 등급 판정은
    "internal ≥ τ 또는 external ≥ τ" 와 수학적으로 동일하다.

    sort=True 면 위험 내림차순 정렬해서 반환.
    [v9.6] 내부·외부 모두 수치가 아닌 속성은 순위 비교에서 빠진다(반환에 없음).
    """
    comp = risk_components(matrix, sort=False, coverage_floor=coverage_floor)
    if comp.empty:
        return pd.Series(dtype=float)
    s = comp.max(axis=1, skipna=True).dropna()   # 내부 max 와 외부 max 중 큰 값(정렬, 판정용)
    s.name = "integrated_risk"
    return s.sort_values(ascending=False) if sort else s


@dataclass
class DatasetRisk:
    """데이터셋 1개의 위험 판정 결과 (H/M/L + 조치 순서)."""
    scores: pd.Series                       # 속성별 순위 스칼라 = max(내부,외부), 내림차순 (정렬, 판정용)
    threshold: float                        # τ: 초과 판정 임계
    exceed: List[str]                       # 임계 초과 속성 (위험 내림차순 = 조치 순서)
    rho: float                              # 초과 비율 = 초과 수 / 전체 수
    n_exceed: int
    n_total: int
    grade: str                              # "High" | "Middle" | "Low"
    rho_cuts: Tuple[float, float]           # (high 하한, middle 하한)
    components: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["internal_risk", "external_risk"], dtype=float))
    # ↑ 속성별 내부/외부 분포 위험 max (합치지 않은 결과값). scores 순서 + 미산출 속성.
    # 산출 상태 보고
    determined: List[str] = field(default_factory=list)   # 확정 성분 보유 → 우선 검토
    unscored: List[str] = field(default_factory=list)     # 내부, 외부 모두 수치 없음 → 순위, ρ 제외
    status_counts: pd.DataFrame = field(default_factory=pd.DataFrame)  # 지표 × 상태별 속성 수
    omitted: List[str] = field(default_factory=list)      # 평가 누락 (점수 0 아님, 요약, 순위 제외)


def classify_dataset(rho: float,
                     cuts: Tuple[float, float] = RHO_CUTS) -> str:
    """임계 초과 비율 ρ → 데이터셋 등급 High/Middle/Low.

    cuts = (high 하한, middle 하한). 예: (0.5, 0.2)이면
      ρ ≥ 0.5 → High,  ρ ≥ 0.2 → Middle,  그 외 → Low.
    """
    hi, mid = cuts
    if rho >= hi:
        return "High"
    if rho >= mid:
        return "Middle"
    return "Low"


def exceed_ratio(scores: pd.Series,
                 threshold: float = DEFAULT_ATTR_THRESHOLD
                 ) -> Tuple[float, List[str], int, int]:
    """임계 τ 를 넘는 속성 비율 ρ 를 계산.

    반환: (rho, exceed_attrs, n_exceed, n_total)
      · exceed_attrs : 통합위험 ≥ τ 인 속성 (위험 내림차순 정렬)
      · rho          : n_exceed / n_total  (n_total=0이면 0.0)
    """
    if scores is None or len(scores) == 0:
        return 0.0, [], 0, 0
    s = scores.dropna().sort_values(ascending=False)   # 미산출(NaN)은 분모에서 제외
    mask = s >= threshold
    exceed = [str(a) for a in s.index[mask.values]]
    n_total = int(len(s))
    n_exceed = int(mask.sum())
    rho = n_exceed / n_total if n_total else 0.0
    return rho, exceed, n_exceed, n_total


def assess_dataset(matrix: pd.DataFrame,
                   threshold: float = DEFAULT_ATTR_THRESHOLD,
                   rho_cuts: Tuple[float, float] = RHO_CUTS,
                   coverage_floor: float = COVERAGE_FLOOR,
                   omitted: Optional[Sequence[str]] = None) -> DatasetRisk:
    """(속성×지표) 행렬 → 데이터셋 H/M/L 판정 + 조치 순서.

      1) integrated_risk 로 속성별 통합위험(max 내부/외부) 산출
         (coverage_floor: 내부축 coverage 게이트 하한, integrated_risk 참조)
      2) τ 초과 속성 판정 → ρ = 초과비율
      3) ρ → H/M/L 등급
      4) 초과 속성 = 위험 내림차순 = 조치 순서(top-k의 모집단)

    [v9.6] 확정·비적용 성분은 0 으로 세지 않는다. 내부·외부 모두 수치가 없는
    속성은 ρ 의 분모와 순위에서 빠지고(unscored), 확정 성분이 있는 속성은
    우선 검토 목록(determined)에 따로 올린다. 상태별 속성 수는 status_counts.
    omitted(평가 누락, evaluation_omissions 결과)는 보고만 하고 ρ·순위에 넣지 않는다.
    """
    scores = integrated_risk(matrix, sort=True, coverage_floor=coverage_floor)
    # 내부/외부 분포 위험을 '합치지 않은' 결과값. scores 정렬 + 미산출 속성은 뒤에.
    comp = risk_components(matrix, sort=False, coverage_floor=coverage_floor)
    unscored = [str(a) for a in comp.index if a not in scores.index]
    comp = comp.reindex(list(scores.index) + unscored)
    rho, exceed, n_exceed, n_total = exceed_ratio(scores, threshold)
    grade = classify_dataset(rho, rho_cuts)
    return DatasetRisk(scores=scores, threshold=threshold, exceed=exceed,
                       rho=rho, n_exceed=n_exceed, n_total=n_total,
                       grade=grade, rho_cuts=rho_cuts, components=comp,
                       determined=determined_attributes(matrix),
                       unscored=unscored,
                       status_counts=status_counts(matrix),
                       omitted=[str(a) for a in (omitted or [])])


def topk_actions(assessment: DatasetRisk,
                 k: Optional[int] = None) -> List[Tuple[int, str, float]]:
    """조치 순서 top-k — 임계 초과 속성을 위험 내림차순으로 (순위, 속성, 위험).

    k=None이면 초과 속성 전체. 등급이 낮아 초과 속성이 없으면 빈 리스트.
    이 순서가 곧 '어느 속성부터 비식별/삭제 조치할지'의 우선순위다.
    """
    if assessment is None or not assessment.exceed:
        return []
    ordered = assessment.exceed if k is None else assessment.exceed[:max(0, k)]
    out: List[Tuple[int, str, float]] = []
    for rank, attr in enumerate(ordered, start=1):
        out.append((rank, attr, float(assessment.scores.get(attr, 0.0))))
    return out


ACTION_ORDERS = ("max", "internal", "external")


def topk_actions_split(assessment: DatasetRisk,
                       k: Optional[int] = None,
                       order_by: str = "max",
                       ) -> List[Tuple[int, str, float, float]]:
    """조치 순서 top-k — 내부/외부 분포 위험을 '분리'해서 반환.

    (순위, 속성, 내부 분포 위험 max, 외부 분포 위험 max).

    order_by — 어떤 위험으로 목록을 만들고 정렬할지 [v9.5]:
      · "max"      : max(내부, 외부) ≥ τ 인 속성, max 내림차순 (기존 = 등급 판정 기준)
      · "internal" : 내부 위험 ≥ τ 인 속성, 내부 위험 내림차순 → 분포·추론 위험 조치 목록
                     (예: 범주화·잡음 추가로 쏠림/얽힘 완화)
      · "external" : 외부 위험 ≥ τ 인 속성, 외부 위험 내림차순 → 연결·재식별 위험 조치 목록
                     (예: QI 일반화·EE 속성 삭제)
    데이터셋 등급(H/M/L)은 항상 max 기준이며 order_by 와 무관하다.
    """
    if assessment is None:
        return []
    if order_by not in ACTION_ORDERS:
        raise ValueError(f"order_by 는 {ACTION_ORDERS} 중 하나: {order_by}")
    comp = assessment.components
    if comp is None or comp.empty:
        return []

    def _vals(attr):
        if attr in comp.index:
            return (float(comp.loc[attr, "internal_risk"]),
                    float(comp.loc[attr, "external_risk"]))
        return 0.0, 0.0

    if order_by == "max":
        ordered = list(assessment.exceed)
    else:
        col = "internal_risk" if order_by == "internal" else "external_risk"
        s = comp[col].astype(float)
        s = s[s >= assessment.threshold].sort_values(ascending=False, kind="stable")
        ordered = [str(a) for a in s.index]
    if k is not None:
        ordered = ordered[:max(0, k)]
    return [(rank, attr) + _vals(attr) for rank, attr in enumerate(ordered, start=1)]


def _fmt_val(v: float, width: int = 12) -> str:
    """수치는 소수 3자리, 미산출(NaN)은 '—'."""
    return f"{'—':>{width}}" if pd.isna(v) else f"{v:>{width}.3f}"


def _format_action_list(acts, header: str) -> List[str]:
    lines = ["", header]
    if not acts:
        lines.append("    (임계 초과 속성 없음)")
        return lines
    lines.append(f"    {'순위':>2}  {'속성':<20}  {'내부 분포 max':>12}  {'외부 분포 max':>12}")
    for rank, attr, iv, ev in acts:
        lines.append(f"    {rank:>2}. {attr:<20}  {_fmt_val(iv)}  {_fmt_val(ev)}")
    return lines


def _format_status_report(a: DatasetRisk) -> List[str]:
    """[v9.6] 상태별 속성 수 + 확정(우선 검토) + 순위 제외 목록."""
    lines = ["", "  산출 상태별 속성 수  (확정·비적용은 0 으로 대체하지 않고 요약·순위에서 제외)"]
    sc = a.status_counts
    if sc is not None and not sc.empty:
        lines.append(f"    {'지표':<14}" + "".join(f"{c:>8}" for c in sc.columns))
        for k, row in sc.iterrows():
            lines.append(f"    {k:<14}" + "".join(f"{int(v):>8}" for v in row.values))
    lines.append("")
    if a.determined:
        lines.append(f"  우선 검토 [확정 — 유효값 1종, 값 확정 가능]: {', '.join(a.determined)}")
    else:
        lines.append("  우선 검토 [확정]: 없음")
    if a.unscored:
        lines.append(f"  순위·ρ 제외 [내부·외부 모두 수치 없음]: {', '.join(a.unscored)}")
    if a.omitted:
        lines.append(f"  평가 누락 [민감속성이나 지표 미계산 — 점수 0 아님]: {', '.join(a.omitted)}")
    return lines


def format_assessment(assessment: DatasetRisk, k: Optional[int] = None,
                      action_order: str = "max") -> str:
    """데이터셋 판정 결과를 사람이 읽는 텍스트로.

    action_order : "max"(기존 한 목록) | "internal" | "external" |
                   "split"(내부 기준 목록과 외부 기준 목록을 따로 둘 다)
    """
    a = assessment
    lines = [
        "데이터셋 위험 등급  (등급 판정 정렬 = max(내부,외부) · 결과값은 내부/외부 분리)",
        "=" * 60,
        f"  임계 τ            : {a.threshold:.2f}  (내부 max 또는 외부 max ≥ τ 이면 초과)",
        f"  초과 속성          : {a.n_exceed} / {a.n_total}"
        + (f"  (수치 없는 {len(a.unscored)}개 제외)" if a.unscored else ""),
        f"  초과 비율 ρ        : {a.rho:.3f}",
        f"  등급 컷 (H/M 하한) : {a.rho_cuts[0]:.2f} / {a.rho_cuts[1]:.2f}",
        f"  → 데이터셋 등급    : {a.grade}",
    ]
    lines += _format_status_report(a)
    orders = ["internal", "external"] if action_order == "split" else [action_order]
    titles = {
        "max": "조치 순서 — max(내부, 외부) ≥ τ, 내림차순 (등급 판정 기준)",
        "internal": "조치 순서 [내부 기준] — 내부 분포 위험 ≥ τ, 내부 위험 내림차순 "
                    "(쏠림·얽힘 → 범주화·잡음 등)",
        "external": "조치 순서 [외부 기준] — 외부 분포 위험 ≥ τ, 외부 위험 내림차순 "
                    "(연결·재식별 → QI 일반화·EE 삭제 등)",
    }
    for o in orders:
        acts = topk_actions_split(a, k=k, order_by=o)
        head = f"  {titles[o]}" + (f"  (top-{len(acts)})" if acts else "")
        lines += _format_action_list(acts, head)
    return "\n".join(lines)


def render_risk_curve(assessment,
                      out_path: str = "risk_curve.png",
                      title: str = "데이터셋 위험 분포 · High / Middle / Low",
                      ylabel: str = "통합 위험 = max(내부, 외부)",
                      xlabel: str = "데이터셋 (위험 내림차순)",
                      value_cuts: Optional[Tuple[float, float]] = None,
                      annotate_topk: Optional[int] = None,
                      show_grade_badge: bool = True) -> Optional[str]:
    """통합위험 분포 곡선을 PNG로 저장 — 세로 3색 밴드(첨부 예시 스타일).

    각 점(민감속성/데이터 단위)을 통합위험 '값'으로 High/Middle/Low 로 나눠
    점을 색칠(High=빨강 / Middle=노랑 / Low=연두)하고, 위험 내림차순으로 정렬돼
    같은 등급이 뭉치는 x-구간에 같은 색의 옅은 세로 배경 밴드를 깐다.
    범례: High 위험 / Middle 중간 / Low 안심.

    assessment : DatasetRisk (assess_dataset의 결과) 또는 점수 Series/dict.
                 Series/dict를 주면 기본 파라미터로 즉석 판정한다.
    value_cuts : 점을 3등급으로 나눌 값 컷 (high 하한, middle 하한).
                 None 이면 (τ, RISK_VALUE_CUTS[1]) — High 컷을 판정 임계 τ 와
                 맞춰 '빨강 점 = τ 초과 속성'이 항상 성립하게 한다.
                 (예전엔 0.50 고정이라 τ 를 바꾸면 점 색과 초과 판정이 어긋났다.)
    matplotlib 없으면 None.
    """
    try:
        import warnings
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except Exception:
        return None

    # Series/dict 로 들어오면 즉석 판정 (하위호환)
    if not isinstance(assessment, DatasetRisk):
        if isinstance(assessment, pd.Series):
            s = assessment.dropna()
        elif isinstance(assessment, dict):
            s = pd.Series(assessment, dtype=float).dropna()
        else:
            s = pd.Series(list(assessment), dtype=float).dropna()
        if s.empty:
            return None
        tmp = s.sort_values(ascending=False)
        rho, exceed, n_ex, n_tot = exceed_ratio(tmp, DEFAULT_ATTR_THRESHOLD)
        assessment = DatasetRisk(scores=tmp, threshold=DEFAULT_ATTR_THRESHOLD,
                                 exceed=exceed, rho=rho, n_exceed=n_ex,
                                 n_total=n_tot,
                                 grade=classify_dataset(rho, RHO_CUTS),
                                 rho_cuts=RHO_CUTS)

    a = assessment
    if a.scores is None or a.scores.empty:
        return None
    if value_cuts is None:
        value_cuts = (float(a.threshold),
                      min(RISK_VALUE_CUTS[1], float(a.threshold)))
    s = a.scores.sort_values(ascending=False)
    y = s.values
    x = list(range(1, len(y) + 1))
    bands = [value_band(float(v), value_cuts) for v in y]   # 점별 등급
    fp = _cjk_fontprop()

    fig, ax = plt.subplots(figsize=(max(8.0, 0.30 * len(y) + 3), 4.6))

    # --- 세로 배경 밴드: 같은 등급이 연속된 x-구간마다 옅은 색 띠 -------------
    #   (위험 내림차순이라 High→Middle→Low 순으로 뭉친다)
    if bands:
        start = 0
        for i in range(1, len(bands) + 1):
            if i == len(bands) or bands[i] != bands[start]:
                b = bands[start]
                # 경계는 점과 점 사이 중간(정수 위치 ±0.5)
                x0 = (start + 1) - 0.5
                x1 = i + 0.5
                ax.axvspan(x0, x1, color=BAND_BG_COLORS[b], zorder=0)
                start = i

    # --- 곡선 + 점(등급 색) --------------------------------------------------
    ax.plot(x, y, "-", color="#9aa0a6", lw=1.1, zorder=2)
    for xi, yi, b in zip(x, y, bands):
        ax.plot(xi, yi, "o", ms=7, color=BAND_COLORS[b],
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    # --- 조치 순서 번호 주석 (top-k, 선택) ----------------------------------
    if annotate_topk:
        acts = topk_actions(a, k=annotate_topk)
        pos = {attr: i for i, attr in enumerate(s.index)}
        for rank, attr, _ in acts:
            if attr in pos:
                xi = pos[attr] + 1
                ax.annotate(f"{rank}", (xi, s.iloc[pos[attr]]),
                            textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=9, fontweight="bold",
                            color="#7a1f1f")

    # --- 축 ------------------------------------------------------------------
    ax.set_ylim(0, max(0.9, float(y.max()) + 0.08))
    ax.set_xlim(0.5, len(y) + 0.5)
    # 속성명이 많지 않으면 라벨을, 많으면 순위 눈금만
    if len(y) <= 20:
        ax.set_xticks(x)
        ax.set_xticklabels([str(a_) for a_ in s.index], rotation=30, ha="right",
                           fontsize=8, fontproperties=fp)
    else:
        ax.set_xticks([t for t in range(1, len(y) + 1) if t % 4 == 1])
    ax.set_ylabel(ylabel, fontproperties=fp)
    ax.set_xlabel(xlabel, fontproperties=fp)
    ax.grid(axis="y", color="#eee", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # --- 범례: High 위험 / Middle 중간 / Low 안심 (플롯 위쪽 바깥에 배치) ----
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                      markerfacecolor=BAND_COLORS[b], markeredgecolor="white",
                      label=BAND_LEGEND_KO[b]) for b in ("High", "Middle", "Low")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.01),
              frameon=False, ncol=3, prop=fp, handletextpad=0.4,
              columnspacing=1.2)

    # --- 제목(데이터셋 등급 + ρ) + 등급 배지 --------------------------------
    sub = f"데이터셋 등급: {a.grade}   ·   초과비율(rho) = {a.n_exceed}/{a.n_total} = {a.rho:.2f}"
    ax.set_title(title + "\n" + sub, fontsize=11, fontproperties=fp, loc="left",
                 color="#222")
    if show_grade_badge:
        ax.text(0.99, 0.86, f"{a.grade}", transform=ax.transAxes,
                ha="right", va="top", fontsize=12, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.3",
                                         fc=BAND_COLORS.get(a.grade, "#555"),
                                         ec="none"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
