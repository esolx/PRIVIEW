"""
metrics.py — 노출 위험 지표 계산 (탐지 이후 단계)
==================================================

민감정보 탐지(pipeline.py)가 끝난 뒤, 탐지된 민감속성/QI/EE 속성을 받아
5개 지표를 **각각 독립적으로** 계산한다. 지표를 하나로 합치지 않는다.
최종 산출은 (속성 × 지표) 위험 행렬이며, 이용자가 지표를 선택해 해석한다.

지표 정의(대화에서 확정)
------------------------
내부 노출 축 — 데이터셋 단독으로 민감속성이 드러나는가:
  · EMD       : 민감속성 '자체' 분포가 균등분포에서 벗어난 정도(분포 편향).
                연속/순서형=등폭 구간화 후 EMD, 범주형=TVD. (개별 속성, QI 무관)
  · Coverage  : 민감속성이 실제로 존재하는 레코드 비율 = 존재수/N (무정보·음성값 제외). (개별 속성, QI 무관)
  · Combo     : 복합 민감도 = 민감속성 i가 다른 민감속성과 얽힌 정도
                noisy-OR 결합 (연속형 등빈도 구간화). 쌍 값은 AMI(우연 보정, 기본)
  ※ QI(준식별자) 조합은 '세분성(granularity)'에서만 쓴다. EMD·Coverage·Combo는
    QI를 쓰지 않는다(문서 정의).

외부 노출 축 — EE 결합으로 노출이 증폭되는가:
  · EE 매칭   : ee_score(≥0.7 EE 판정, 게이트) + 외부 nMI = I*(EE속성; 민감)
  · 세분성    : 1 − 결합고유값/N (조합별 여러 값은 max 집계)

집계 원칙(확정)
--------------
  · 외부 nMI       : max_e I*(EE_e; 민감_s)      — "하나만 연결돼도" OR → max
  · 복합 민감도    : 1 − Π(1 − AMI(민감_i; 민감_j)) — 연쇄 노출 누적 → noisy-OR
                    (쌍 값: AMI 기본 — nMI 의 우연 기준선 누적으로 인한 포화 방지.
                     COMBO_PAIR_METRIC="nmi" 로 기존 방식 사용 가능)
  · 세분성         : max (극단 희귀 조합 = 신호)  — 파라미터 없음
  · 외부 결합      : 𝟙[ee≥0.7] × √(외부nMI × 세분성) — 게이트 × 기하평균

방향·정규화
----------
  · 전부 [0,1], 클수록 위험
  · nMI = I(X;Y)/min(H(X),H(Y))  (0~1 정규화 상호정보량)
  · 구간화: EMD=등폭(equal-width), Combo/nMI=등빈도(equal-frequency)
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Sequence, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


# ===========================================================================
# 공통 유틸 — 구간화 ,  엔트로피 ,  nMI
# ===========================================================================

def _as_series(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(list(x))


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 12


def bin_equal_width(s: pd.Series, bins: int = 10) -> pd.Series:
    """등폭 구간화 — 값 범위를 균등 폭으로 나눔 (EMD용)."""
    s = _as_series(s)
    if not _is_numeric(s):
        return s.astype("object")
    try:
        return pd.cut(s, bins=bins, duplicates="drop").astype("object")
    except ValueError:
        return s.astype("object")


def bin_equal_freq(s: pd.Series, bins: int = 10) -> pd.Series:
    """등빈도 구간화 — 각 구간에 같은 개수가 들어가게 나눔 (MI 편향 완화)."""
    s = _as_series(s)
    if not _is_numeric(s):
        return s.astype("object")
    try:
        return pd.qcut(s, q=bins, duplicates="drop").astype("object")
    except (ValueError, IndexError):
        return s.astype("object")

_DATE_PATTERN = r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"


def _date_to_days(s: pd.Series, min_ratio: float = 0.9) -> Optional[pd.Series]:
    """날짜형 문자열이면 기준일 대비 일수(float)로 바꿔 반환, 아니면 None.

    날짜 문자열은 _is_numeric()에 안 걸려 구간화 없이 원값이 그대로 남는다.
    (COMPAS c_jail_in/out: 800행 중 고유값 424/450 → 아무 속성과도 nMI 0.56~0.79)
    값이 아니라 '시점'이 의미이므로 순서를 살려 수치로 바꾼 뒤 등빈도 구간화한다.
    """
    if pd.api.types.is_numeric_dtype(s):
        return None
    nn = s.dropna().astype(str)
    if len(nn) == 0 or nn.str.match(_DATE_PATTERN).mean() < min_ratio:
        return None
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().sum() < min_ratio * len(nn):
        return None
    return (dt - dt.min()).dt.days.astype(float)

def bin_for_nmi(s: pd.Series, bins: int = 10) -> pd.Series:
    """nMI 계산용 카디널리티 축소 — 숫자·문자열 모두 처리.

    · 숫자(고카디널리티) : 등빈도 구간화(기존 bin_equal_freq).
    · 문자열/범주(고카디널리티) : 빈도 상위 (bins−1)개 값만 그대로 두고 나머지를
      '__OTHER__' 로 묶어 카디널리티를 bins 이하로 축소.

    이렇게 하는 이유: near-unique 문자열(IP·지문·adid 등)은 구간화가 안 먹어
    고유값이 그대로 남고, nMI = I/min(H) 의 min(H) 정규화 때문에 아무 속성과도
    nMI=1.0 으로 포화된다. 카디널리티를 낮춰 이 포화를 없앤다(민감속성 자격은
    그대로 유지 — 준식별자로 옮기는 게 아니다).
    """
    s = _as_series(s)
    if _is_numeric(s):
        return bin_equal_freq(s, bins=bins)
    days = _date_to_days(s)
    if days is not None:                               # 날짜형 → 일수 → 등빈도 구간
        return bin_equal_freq(days, bins=bins).astype(str)
    ss = s.astype(str)
    if ss.nunique(dropna=False) <= bins:
        return ss.astype("object")
    top = list(ss.value_counts().index[: max(1, bins - 1)])
    # 결측은 __OTHER__ 에 섞지 않고 결측으로 둔다 → nmi() 가 pairwise 제외
    # (pandas 3 는 astype(str) 후에도 NaN 유지 → 기존엔 결측이 __OTHER__ 로 묶였음)
    return ss.where(ss.isin(top) | s.isna(), other="__OTHER__").astype("object")


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def _codes(labels: Sequence) -> Tuple[np.ndarray, int]:
    """라벨 → 정수 코드 (pd.factorize). 문자열 비교 대신 정수 연산으로 빠르게."""
    codes, uniq = pd.factorize(np.asarray(labels, dtype=object))
    return codes, len(uniq)


def _entropy(labels: Sequence) -> float:
    codes, k = _codes(labels)
    if len(codes) == 0:
        return 0.0
    return _entropy_from_counts(np.bincount(codes, minlength=k).astype(float))


def _mutual_info(x: Sequence, y: Sequence) -> float:
    """이산 MI (bits). x,y는 이미 구간화/범주화된 라벨.

    [리팩토링] 결합 라벨을 f"{a}\\x1f{b}" 문자열로 만들던 것을 정수 코드 결합
    (cx·ky + cy)으로 바꿔 O(n) 문자열 생성을 없앴다. 값은 동일하다.
    """
    cx, kx = _codes(x)
    cy, ky = _codes(y)
    if len(cx) == 0:
        return 0.0
    hx = _entropy_from_counts(np.bincount(cx, minlength=kx).astype(float))
    hy = _entropy_from_counts(np.bincount(cy, minlength=ky).astype(float))
    joint = cx.astype(np.int64) * ky + cy
    hxy = _entropy_from_counts(np.bincount(joint).astype(float))
    return float(max(0.0, hx + hy - hxy))


PAIR_MIN_VALID = 2       # 쌍별 유효 표본이 이보다 적으면 그 쌍은 정의되지 않음


def _pairwise_complete(x: Sequence, y: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """둘 중 하나라도 결측인 행을 뺀 (x, y) 문자열 배열 (pairwise complete)."""
    xs = pd.Series(list(x), dtype=object)
    ys = pd.Series(list(y), dtype=object)
    miss = {"nan", "none", "<na>", "nat", ""}
    keep = ~(xs.isna() | ys.isna()
             | xs.astype(str).str.strip().str.lower().isin(miss)
             | ys.astype(str).str.strip().str.lower().isin(miss))
    return (xs[keep].astype(str).to_numpy(dtype=object),
            ys[keep].astype(str).to_numpy(dtype=object))


def pair_valid_count(x: Sequence, y: Sequence) -> int:
    """쌍별 유효 표본 수 (두 값이 모두 관측된 행 수)."""
    return int(len(_pairwise_complete(x, y)[0]))


def nmi_defined(x: Sequence, y: Sequence) -> Optional[float]:
    """[v9.6] 논문 식 (6) ν = I(X;Y)/min(H(X),H(Y)). 정의되지 않으면 None.

    정의되지 않는 경우: 쌍별 유효 표본 < 2, 또는 분모 min(H) = 0 (한쪽 값이 상수).
    """
    x, y = _pairwise_complete(x, y)
    if len(x) < PAIR_MIN_VALID:
        return None
    denom = min(_entropy(x), _entropy(y))
    if denom <= 1e-12:
        return None
    return float(np.clip(_mutual_info(x, y) / denom, 0.0, 1.0))


def nmi(x: Sequence, y: Sequence) -> float:
    """정규화 상호정보량  I*(X;Y) = I(X;Y) / min(H(X), H(Y)),  [0,1].

    [수정] 둘 중 하나라도 결측인 행은 제외하고 계산(pairwise complete).
      · 결측이 있는 숫자 컬럼은 구간(Interval)과 NaN이 섞여 정렬 시 TypeError가 났다.
      · 결측을 하나의 값으로 두면 '같은 행이 같이 비어 있다'는 것만으로 MI가 올라간다.
    정의되지 않으면 0 (하위 호환). 상태가 필요하면 nmi_defined().
    """
    v = nmi_defined(x, y)
    return 0.0 if v is None else v


def ami(x: Sequence, y: Sequence) -> float:
    """우연 보정 상호정보량 AMI(X;Y), [0,1] (음수는 0으로 자름).

        AMI = (I − E[I]) / (avg(H(X),H(Y)) − E[I])
        E[I] = 두 속성의 값 분포는 그대로 두고 행만 무작위로 섞었을 때의 기대 MI

    nMI 는 관계가 없어도 분포 모양(고카디널리티·한쪽 쏠림)에 따라 우연 값이
    0.1~0.2 까지 나와, Combo 누적 시 포화를 만든다. AMI 는 이 우연 기준선을
    빼서 무관한 쌍을 0 근처로 만든다 (Vinh, Epps & Bailey, JMLR 2010).
    결측 처리는 nmi() 와 동일(둘 중 하나라도 결측인 행 제외).
    """
    from sklearn.metrics import adjusted_mutual_info_score
    x, y = _pairwise_complete(x, y)
    if len(x) < PAIR_MIN_VALID:
        return 0.0
    x, y = x.astype(str), y.astype(str)
    if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:      # 상수 → 해당 쌍의 연관 0 (논문)
        return 0.0
    return float(np.clip(adjusted_mutual_info_score(x, y), 0.0, 1.0))


# ===========================================================================
# 산출 상태 — 각 성분은 수치값과 함께 상태를 가진다.
#   ,  numeric        (수치)   : 정상 산출. value ∈ [0,1]
#   ,  determined     (확정)   : 유효값이 한 종류 → 대상자가 자료에 있으면 값이 확정됨.
#                               정규화가 정의되지 않으므로 수치 없음(NaN). 우선 검토 대상.
#   ,  not_applicable (비적용) : 계산할 대상, 상대가 없음(유효값 없음, 식별자형 목표,
#                               상대 민감속성 없음, 보조속성(EE), 준식별자(QI) 없음).
# 확정, 비적용은 0 으로 대체하지 않는다(value=NaN). 요약값, 순위 비교에서 빠지고,
# 상태별 속성 수는 따로 보고한다(exposure_stage.status_counts).
# ===========================================================================

STATUS_NUMERIC = "numeric"
STATUS_DETERMINED = "determined"
STATUS_NA = "not_applicable"
STATUS_KO = {STATUS_NUMERIC: "수치", STATUS_DETERMINED: "확정", STATUS_NA: "비적용"}
STATUSES = (STATUS_NUMERIC, STATUS_DETERMINED, STATUS_NA)


@dataclass
class MetricResult:
    """한 성분의 산출 결과. status 가 numeric 이 아니면 value 는 NaN."""
    value: float
    status: str = STATUS_NUMERIC
    reason: str = ""

    @classmethod
    def numeric(cls, v: float, reason: str = "") -> "MetricResult":
        return cls(float(v), STATUS_NUMERIC, reason)

    @classmethod
    def determined(cls, reason: str) -> "MetricResult":
        return cls(float("nan"), STATUS_DETERMINED, reason)

    @classmethod
    def not_applicable(cls, reason: str) -> "MetricResult":
        return cls(float("nan"), STATUS_NA, reason)


# ===========================================================================
# 내부 노출 지표
# ===========================================================================

# EMD 를 계산할 수 없거나 해석하면 안 되는 경우의 표시 (emd_note 열)
EMD_MIN_PRESENT = 30          # 유효 레코드가 이보다 적으면 표본 부족 표시
EMD_NEAR_UNIQUE_RATIO = 0.90  # 고유값 비율이 이 이상이면 식별자형 → 분포 쏠림 정의 불가
EMD_NEAR_UNIQUE_MIN_K = 20


EMD_OUTLIER_IQR_K = 10.0      # Q1−k, IQR ~ Q3+k, IQR 밖이면 이상치 (매우 보수적: 긴 꼬리는 유지)


def emd_valid_values(sensitive: pd.Series,
                     category: Optional[str] = None,
                     _info: Optional["PresenceInfo"] = None
                     ) -> Tuple[pd.Series, Dict[str, int]]:
    """[v9.3] EMD 계산에 쓸 '유효 값'과 제외 사유별 개수.

    유효 = 실제로 관측된 값. **'없음'(0·N·아니오·미가입 등 음성 값)은 포함**한다.
    제외하는 것은 값이 아닌 것뿐이다:
      · missing      : NaN·Null·빈칸·'unknown'·'모름'·'무응답' 등 결측 토큰
      · missing-code : 결측코드 — 코드북(MISSING_OVERRIDES), 동떨어진 0(zero-gap, 나이 0 등),
                       9/99/999/-1 처럼 다른 값과 동떨어진 숫자 코드
      · outlier      : 수치형에서 Q1−10·IQR ~ Q3+10·IQR 밖 (IQR>0 일 때만, 입력 오류 수준)
    """
    s = _as_series(sensitive)
    info = _info or presence_info(s, category)
    vals = info.values if info.values is not None else s
    name = str(getattr(s, "name", "") or "")
    n0 = len(vals)
    excl = {"missing": 0, "missing-code": 0, "outlier": 0}

    keep = vals.notna()
    if not pd.api.types.is_numeric_dtype(vals):
        norm = vals.astype(str).str.strip().str.lower()
        tok = norm.isin(MISSING_TOKENS)
        keep &= ~tok
    excl["missing"] = int(n0 - keep.sum())

    if name in MISSING_OVERRIDES:
        miss = {_norm_token(v) for v in MISSING_OVERRIDES[name]}
        hit = keep & vals.map(lambda v: _norm_token(v) if pd.notna(v) else None).isin(miss)
        excl["missing-code"] += int(hit.sum())
        keep &= ~hit

    if pd.api.types.is_numeric_dtype(vals):
        num = pd.to_numeric(vals, errors="coerce")
        if info.rule.endswith("zero-gap"):                 # 0 = 결측코드로 판정된 컬럼
            hit = keep & num.eq(0)
            excl["missing-code"] += int(hit.sum())
            keep &= ~hit
        codes = _sentinel_codes(num[keep])
        if codes:
            hit = keep & num.isin(codes)
            excl["missing-code"] += int(hit.sum())
            keep &= ~hit
        kv = num[keep]
        if len(kv) >= 20 and kv.nunique() > 12:            # 연속형에만 이상치 규칙
            q1, q3 = kv.quantile(0.25), kv.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - EMD_OUTLIER_IQR_K * iqr, q3 + EMD_OUTLIER_IQR_K * iqr
                hit = keep & ((num < lo) | (num > hi))
                excl["outlier"] = int(hit.sum())
                keep &= ~hit
        return num[keep], excl
    return vals[keep], excl


def emd_detail(sensitive: pd.Series,
               bins: int = 10,
               category: Optional[str] = None,
               _info: Optional["PresenceInfo"] = None) -> Tuple[float, str]:
    """(값, 사유) — 하위 호환용. 확정·비적용이면 값은 NaN (상태는 emd_result 참고)."""
    r = emd_result(sensitive, bins=bins, category=category, _info=_info)
    return r.value, r.reason


def emd_result(sensitive: pd.Series,
               bins: int = 10,
               category: Optional[str] = None,
               _info: Optional["PresenceInfo"] = None) -> MetricResult:
    """EMD 값 + 산출 상태 + 사유(note).

    D_i = EMD(P_i, U_i) / EMD_max   — 민감속성 i 의 **유효 값** 분포가 균등에서 얼마나 쏠렸나.
      · 연속형/순서형 : 등폭 구간화 후 1-Wasserstein, EMD_max = (K−1)/2
      · 날짜형 문자열 : 기준일 대비 일수로 바꿔 연속형과 같게 계산
      · 범주형        : TVD(P_i, U_i) / TVD_max,  TVD_max = 1 − 1/K

    [v9.3] 분포를 '있음' 레코드가 아니라 **유효 레코드**(결측·결측코드·이상치 제외,
    '없음' 포함)로 잰다 — emd_valid_values() 참고.
      · 0/1 여부 컬럼 : 0 과 1 의 비율 쏠림 → 정규화 TVD = |2p − 1|
                        (희귀질환 5% → 0.90, 재범 50:50 → 0)
      · 횟수 컬럼     : 0(전과 없음)이 많으면 그 쏠림이 EMD 에 반영
    Coverage·세분성은 그대로 '있음' 기준이다 (노출 물량·식별 위험).

    [v9.6] 수치를 낼 수 없는 경우 0 이 아니라 상태로 보고한다 (value=NaN):
      · single-value → 확정(determined) : 유효 값이 한 종류. 대상자가 자료에 있으면
                       추가 정보 없이 값이 확정됨 = 기본 추측 가능성 최고. 정규화
                       (EMD_max=(K−1)/2, TVD_max=1−1/K)가 K=1 에서 정의되지 않음.
      · no-valid     → 비적용 : 유효 레코드 없음
      · near-unique  → 비적용 : 식별자형(ID·IP·지문템플릿) — 분포 쏠림 정의 불가
    """
    info = _info or presence_info(_as_series(sensitive), category)
    s, excl = emd_valid_values(sensitive, category, _info=info)
    n = len(s)
    dropped = ", ".join(f"{k} {v}" for k, v in excl.items() if v)
    drop_note = f"제외: {dropped} (유효 n={n})" if dropped else ""
    if n == 0:
        return MetricResult.not_applicable(
            "no-valid: 유효 레코드 없음" + (f" | {drop_note}" if drop_note else ""))
    notes = []
    if n < EMD_MIN_PRESENT:
        notes.append(f"small-sample: 유효 레코드 {n}개(<{EMD_MIN_PRESENT}) — 표본 부족")
    if drop_note:
        notes.append(drop_note)

    days = _date_to_days(s)                       # 날짜 문자열 → 일수 (순서 보존)
    if days is not None:
        s = days

    if _is_numeric(s):
        # 연속형/순서형 — 등폭 구간화 → 균등분포 대비 1-Wasserstein
        binned = bin_equal_width(s, bins=bins)
        cats = pd.Categorical(binned)
        K = len(cats.categories)
        if K <= 1:
            return MetricResult.determined(" | ".join(["single-value: 유효 값 1종"] + notes))
        codes = cats.codes
        p = np.bincount(codes[codes >= 0], minlength=K).astype(float)
        p /= p.sum()
        u = np.full(K, 1.0 / K)                       # 균등분포 U_i
        emd = np.abs(np.cumsum(p) - np.cumsum(u)).sum()
        emd_max = (K - 1) / 2.0                        # 균등 대비 최대 EMD (K≥2 → >0)
        return MetricResult.numeric(float(np.clip(emd / emd_max, 0.0, 1.0)),
                                    " | ".join(notes))

    # 범주형 — TVD(P_i, U_i) / TVD_max (순서가 없으므로 CDF 대신 총변동거리)
    vc = s.astype(str).value_counts(normalize=True)
    K = len(vc)
    if K <= 1:
        return MetricResult.determined(" | ".join(["single-value: 유효 값 1종"] + notes))
    if K >= EMD_NEAR_UNIQUE_MIN_K and K / n >= EMD_NEAR_UNIQUE_RATIO:
        return MetricResult.not_applicable(
            " | ".join([f"near-unique: 고유값 {K}/{n} — 식별자형이라 분포 쏠림 정의 불가 "
                        "(위험은 식별성·EE 쪽에서 볼 것)"] + notes))
    p = vc.values
    u = 1.0 / K                                        # 균등분포
    tvd = 0.5 * np.abs(p - u).sum()                    # ½ Σ|P−U|
    tvd_max = 1.0 - 1.0 / K                            # 한 값에 전부 몰릴 때
    return MetricResult.numeric(float(np.clip(tvd / tvd_max, 0.0, 1.0)), " | ".join(notes))


def emd_risk(sensitive: pd.Series,
             bins: int = 10,
             category: Optional[str] = None,
             _info: Optional["PresenceInfo"] = None) -> float:
    """EMD 값만 반환 (확정·비적용이면 NaN. 상태·사유는 emd_result)."""
    return emd_result(sensitive, bins=bins, category=category, _info=_info).value


# ===========================================================================
# '존재(presence)' 판정 — Coverage ,  EMD ,  세분성이 모두 이 하나를 공유한다.
#   ,  Coverage = 존재 레코드 / 전체 N           (semi-continuous 1부: 있나/없나)
#   ,  EMD      = 존재 레코드 안에서의 분포 쏠림   (2부: 있다면 어떻게 분포하나)
#   ,  세분성   = 존재 레코드 중 k-익명 위반 비율
#
# '부재'는 두 종류를 모두 포함한다. Coverage 관점에서는 둘 다 "이 레코드에는
# 민감값이 없다"이므로 같은 쪽으로 센다.
#   (가) 결측   : 값이 없음 (NaN, 빈칸, 'unknown', 결측코드로 쓰인 0 ...)
#   (나) 음성   : 값은 있는데 '해당 없음'을 뜻함 (질병여부 0, 'N', '아니오' ...)
# 중요한 것은 (가), (나)를 구별하는 게 아니라, 0 이 **'부재'인지 '실제 값'인지**
# 를 가르는 것이다. 예) 질병여부 0 = 부재 / 인종코드 0(=백인) = 실제 값.
#
# 판정에 '카테고리'를 함께 쓴다 (이전엔 값 모양만 봄 → 아래 오판이 있었다)
#   ,  성별 0/1, 인종 정수코드 0~4 → 0 을 부재로 잘못 셈 (coverage 0.52 / 0.57)
#   ,  설문식 1=예/2=아니오, 'Y'/'N', '예'/'아니오', '0.0'/'1.0' → 전부 존재로 셈 (coverage 1.0)
#   ,  나이 0(결측코드)이 30% 미만이면 0세를 실제 값으로 셈
#
# 판정 우선순위 (PresenceInfo.rule 에 어떤 규칙이 적용됐는지 남긴다):
#   1) codebook     : ABSENCE_OVERRIDES[컬럼] 에 지정된 값(+결측)만 부재. 자동규칙 끔.
#   2) bool         : False 부재
#   0) 숫자형 문자열 : '0','1.0',' 2 ' 처럼 전부 숫자면 수치로 바꿔 3) 적용 (라벨 앞에 'str→num:')
#   3) 수치형
#      [범주값 카테고리 — 인종, 정치, 종교, 성적지향, 학력, 직업, 위치 등]
#        이름에 여부/유무/yn/횟수 같은 사건 토큰이 없으면 → category-code
#        (0 도 하나의 범주 값. 결측 토큰만 부재)
#      [사건형 카테고리 — 건강, 범죄, 유전, 노조가입, 소득, 신용 등, 또는 카테고리 모름]
#        a) binary     : 값이 {0,1} 뿐 → 0 부재
#        b) binary-12  : 값이 {1,2} 뿐 + 이름에 여부/유무/yn → 2(=아니오) 부재 ⚠
#        c) zero-gap   : 0 이 나머지 값과 동떨어짐(나이 0, 혈당 0) → 0 = 결측코드로 부재 ⚠
#        d) zero≥r     : 음수 없음 & 0 비율 ≥ ZERO_ABSENCE_RATIO → 0 부재 (semi-continuous)
#        e) name-token : 이름에 사건/횟수 토큰 → 0 부재
#        f) none       : 그 외 → 0 도 실제 값
#   4) 문자열       : 결측 토큰 + (사건형이면) 음성 토큰 부재
#   ⚠ 가 붙은 규칙과, 자동 처리하지 않는 의심 신호(9/99/999 결측코드, 역코딩 의심 등)는
#     PresenceInfo.warning 에 사람이 볼 문구로 남긴다 → 코드북으로 확정할 것.
# ===========================================================================

# (가) 결측, 무정보 토큰 — 모든 컬럼에서 부재
MISSING_TOKENS = frozenset({
    "", "nan", "none", "null", "na", "n/a", "<na>", "nat", "-", "--", ".",
    "unknown", "unk", "미상", "모름", "무응답", "not available",
})
# (나) 음성, 해당없음 토큰 — 사건형 카테고리에서만 부재
#   'n', 'f', '아니오', '아니요', '비해당', '미해당' 추가 (Y/N, 예/아니오 컬럼이
#   전부 '존재'로 세지던 문제). docstring 에는 있었으나 토큰 목록에 빠져 있었다.
NEGATIVE_TOKENS = frozenset({
    "no", "n", "false", "f", "negative", "음성", "normal", "정상", "0",
    "없음", "무", "아니오", "아니요", "아님", "해당없음", "해당 없음",
    "해당사항없음", "해당사항 없음", "비해당", "미해당", "not applicable",
    # '미/비 + 동사' 형 부정값 (노동조합가입여부 '미가입' 이 존재로 세지던 문제)
    "미가입", "비가입", "미보유", "비보유", "미흡연", "비흡연", "미이용", "미사용",
    "미경험", "무경험", "없다",
})
ABSENCE_TOKENS = MISSING_TOKENS | NEGATIVE_TOKENS      # 하위 호환용 합집합

ZERO_ABSENCE_RATIO = 0.30          # 0 비율이 이 이상이면 0 = '없음'
ABSENCE_OVERRIDES: Dict[str, list] = {}   # {컬럼명: [부재로 볼 값, ...]} (codebook)
# {컬럼명: [결측으로 볼 값, ...]} — '없음'(음성)이 아니라 '값 자체가 없음'.
#   EMD 계산에서 빠진다 (예: 9=무응답, -1=미측정). '없음'은 ABSENCE_OVERRIDES 에.
MISSING_OVERRIDES: Dict[str, list] = {}

# 값이 '범주'인 카테고리: 0 도 하나의 범주 값일 가능성이 높다(인종코드 0=백인 등).
CATEGORY_VALUED = frozenset({
    "RACE_ETHNICITY", "POLITICAL_OPINION", "IDEOLOGY_BELIEF", "SEX_LIFE",
    "EDUCATION", "OCCUPATION", "LOCATION", "DEVICE_ONLINE", "COMMUNICATION",
    "BIOMETRIC", "NON_SENSITIVE",
})

_EN_EVENT_TOKENS = {"count", "counts", "cnt", "num", "number", "times",
                    "recid", "flag", "indicator", "freq",
                    "yn", "has", "is", "ever", "history", "hx"}
_KO_EVENT_TOKENS = ("여부", "유무", "횟수", "건수", "회수")
_EN_YESNO_TOKENS = {"yn", "flag", "indicator", "has", "is", "ever"}
_KO_YESNO_TOKENS = ("여부", "유무")
# 1=있음이 아니라 1=정상(역코딩)일 가능성을 시사하는 이름
_REVERSE_HINTS_EN = {"healthy", "normal", "negative", "clean", "free", "no"}
_REVERSE_HINTS_KO = ("정상", "건강", "음성", "무병", "미보유")
# 숫자 결측코드로 흔히 쓰이는 값 (자동 처리하지 않고 경고만)
_SENTINEL_CODES = (9, 99, 999, 9999, -1, -9, -99, -999)


def _name_words(name: str) -> set:
    """camelCase 를 쪼갠 뒤 영문 단어 집합 (patientHasDiabetes → patient, has, diabetes)."""
    import re as _re
    low = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name or "")).lower()
    return set(w for w in _re.split(r"[^0-9a-z]+", low) if w)


def _name_has_event_token(name: str) -> bool:
    """영문은 단어 단위로(account 의 count 오탐 방지), 한글은 부분일치로 확인."""
    if _name_words(name) & _EN_EVENT_TOKENS:
        return True
    return any(t in str(name or "") for t in _KO_EVENT_TOKENS)


def _name_has_yesno_token(name: str) -> bool:
    if _name_words(name) & _EN_YESNO_TOKENS:
        return True
    return any(t in str(name or "") for t in _KO_YESNO_TOKENS)


def _name_suggests_reverse(name: str) -> bool:
    if _name_words(name) & _REVERSE_HINTS_EN:
        return True
    return any(t in str(name or "") for t in _REVERSE_HINTS_KO)


def _norm_token(v) -> str:
    """codebook 비교용 정규화: 숫자는 1 == 1.0 == '1' 로 맞추고 문자열은 소문자."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v).strip().lower()


def _coerce_numeric_strings(s: pd.Series) -> Optional[pd.Series]:
    """문자열 컬럼의 비결측 값이 전부 숫자면 수치 Series 로 (결측 토큰은 NaN)."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return None
    norm = s.astype(str).str.strip()
    missing = s.isna() | norm.str.lower().isin(MISSING_TOKENS)
    if (~missing).sum() == 0:
        return None
    num = pd.to_numeric(norm.where(~missing), errors="coerce")
    if num[~missing].isna().any():
        return None                                   # 숫자가 아닌 값이 섞임
    return num


def _zero_is_isolated(num: pd.Series) -> bool:
    """0 이 나머지 양수 값들과 동떨어져 있나 (나이 0, 키 0, 혈당 0 → 결측코드 의심).

    양수 최솟값이 2 이상이고 양수 최댓값의 10% 이상이면 0 과 사이에 큰 빈 구간이
    있다고 본다. 횟수(최솟값 1)·소득(최솟값이 최댓값에 비해 아주 작음)은 해당 안 됨.
    """
    pos = num[num > 0]
    if len(pos) == 0 or not (num == 0).any() or (num < 0).any():
        return False
    lo, hi = float(pos.min()), float(pos.max())
    return lo >= 2 and lo >= 0.1 * hi


def _sentinel_codes(num: pd.Series) -> List[float]:
    """9/99/999·-1 처럼 다른 값들과 동떨어진 숫자 결측코드 의심값 목록."""
    num = num.dropna()
    if len(num) < 20:
        return []
    vc = num.value_counts()
    hits = []
    for code in _SENTINEL_CODES:
        if code not in vc.index or vc[code] / len(num) < 0.01:
            continue
        others = num[num != code]
        if len(others) == 0:
            continue
        if code > 0 and float(others.max()) < code * 0.5:     # 최댓값보다 훨씬 큼 (예: 1~5 + 99)
            hits.append(code)
        elif code < 0 and (others >= 0).all():                # 나머지는 전부 0 이상
            hits.append(code)
    return hits


def _sentinel_warning(num: pd.Series) -> str:
    """결측코드 의심값 경고 문구. EMD 에서는 자동 제외, Coverage 는 코드북으로 확정."""
    hits = _sentinel_codes(num)
    if not hits:
        return ""
    return (f"값 {', '.join(str(h) for h in hits)} 이(가) 결측코드일 수 있음 "
            "(EMD 에서는 제외 · Coverage 는 자동 처리 안 함 → 코드북으로 지정)")


@dataclass
class PresenceInfo:
    """한 컬럼의 존재 판정 결과 (지표 계산에서 1회만 만들어 재사용)."""
    mask: pd.Series            # 레코드별 존재 여부 (bool)
    rule: str                  # 적용된 규칙 라벨 (감사용)
    warning: str = ""          # 사람이 확인할 의심 신호 (없으면 "")
    values: Optional[pd.Series] = None   # 판정에 쓴 값 (숫자형 문자열은 수치로 변환됨)


def _numeric_presence(num: pd.Series, name: str, category: Optional[str]
                      ) -> Tuple[pd.Series, str, List[str]]:
    """수치 Series → (부재 mask, 규칙, 경고들). mask=True 가 '부재'."""
    warns: List[str] = []
    vals = num.dropna()
    absent = pd.Series(False, index=num.index)
    if len(vals) == 0:
        return absent, "none", warns
    sentinel = _sentinel_warning(vals)
    if sentinel:
        warns.append(sentinel)

    event_name = _name_has_event_token(name)
    category_valued = (category in CATEGORY_VALUED) and not event_name
    uniq = set(vals.unique())

    if category_valued:
        if _zero_is_isolated(vals):
            warns.append("0 이 다른 값과 동떨어져 결측코드일 수 있음 "
                         "(범주형 카테고리라 0 을 실제 값으로 둠 → 필요하면 코드북)")
        return absent, "category-code", warns

    if uniq <= {0, 1}:
        if _name_suggests_reverse(name):
            warns.append("이름상 1=정상(역코딩)일 수 있음 → 0 을 부재로 본 판정 확인 필요")
        return num.eq(0), "binary", warns
    if uniq == {1, 2} and _name_has_yesno_token(name):
        warns.append("1=예/2=아니오 설문 코딩으로 가정해 2 를 부재로 봄 → 코드북으로 확정 권장")
        return num.eq(2), "binary-12", warns
    if _zero_is_isolated(vals):
        warns.append("0 이 다른 값과 동떨어져 결측코드로 추정해 부재로 봄")
        return num.eq(0), "zero-gap", warns
    if (vals >= 0).all() and (vals == 0).mean() >= ZERO_ABSENCE_RATIO:
        return num.eq(0), f"zero>={ZERO_ABSENCE_RATIO:.2f}", warns
    if event_name:
        return num.eq(0), "name-token", warns
    return absent, "none", warns


def presence_info(sensitive: pd.Series, category: Optional[str] = None) -> PresenceInfo:
    """민감속성 i 가 각 레코드에 '실제로 존재'하는지 판정한다.

    category : 탐지 카테고리 코드(예: "HEALTH", "RACE_ETHNICITY"). None 이면 카테고리를
               모르는 것으로 보고 사건형 규칙을 쓴다(구버전 호환).
    """
    s = _as_series(sensitive)
    name = str(getattr(s, "name", "") or "")
    present = s.notna()

    if name in ABSENCE_OVERRIDES:
        absent = {_norm_token(v) for v in ABSENCE_OVERRIDES[name]}
        norm = s.map(lambda v: _norm_token(v) if pd.notna(v) else None)
        return PresenceInfo((present & ~norm.isin(absent)).astype(bool), "codebook", "", s)

    if pd.api.types.is_bool_dtype(s):
        return PresenceInfo((present & s.fillna(False).astype(bool)).astype(bool), "bool", "", s)

    prefix = ""
    if pd.api.types.is_numeric_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
    else:
        num = _coerce_numeric_strings(s)
        prefix = "str→num:" if num is not None else ""

    if num is not None:
        absent, rule, warns = _numeric_presence(num, name, category)
        mask = (num.notna() & ~absent).astype(bool)
        return PresenceInfo(mask, prefix + rule, " / ".join(warns), num)

    # 문자열
    norm = s.astype(str).str.strip().str.lower()
    tokens = (MISSING_TOKENS if (category in CATEGORY_VALUED
                                 and not _name_has_event_token(name))
              else ABSENCE_TOKENS)
    mask = (present & ~norm.isin(tokens)).astype(bool)
    rule = "string-missing" if tokens is MISSING_TOKENS else "string-tokens"
    return PresenceInfo(mask, rule, "", s)


def presence_rule(sensitive: pd.Series, category: Optional[str] = None) -> str:
    """이 컬럼에 어떤 부재 판정 규칙이 적용되는지 (검토·감사용 라벨)."""
    return presence_info(sensitive, category).rule


def sensitive_presence_mask(sensitive: pd.Series, category: Optional[str] = None) -> pd.Series:
    """민감속성 i가 각 레코드에 '실제로 존재'하는지 나타내는 bool mask.

    Coverage · EMD · 세분성이 모두 이 하나의 정의를 공유한다(분모 기준 통일).
    단순 non-null 이 아니다 — 결측·무정보·음성값, 그리고 규칙상 '없음'을 뜻하는
    값(0, 설문식 2 등)을 부재로 센다. 규칙은 presence_info() 참고.
    """
    return presence_info(sensitive, category).mask


def coverage_risk(sensitive: pd.Series, category: Optional[str] = None,
                  _info: Optional[PresenceInfo] = None) -> float:
    """
    Coverage — 민감속성 i가 실제로 존재하는 레코드 비율.

        C_i = (민감속성 i가 존재하는 레코드 수) / N

    단순 non-null 채움률이 아니라 ``presence_info``의 존재 정의를 쓴다.
    반환 [0,1].
    """
    s = _as_series(sensitive)
    n = len(s)
    if n == 0:
        return 0.0
    info = _info or presence_info(s, category)
    return float(np.clip(info.mask.sum() / n, 0.0, 1.0))


def identifier_like_columns(frame: pd.DataFrame,
                            max_unique_ratio: float = 0.9) -> List[str]:
    """near-unique(식별자형) 컬럼 목록.

    고유값 비율(nunique/N)이 임계 이상인 컬럼. 이런 컬럼은 각 값이 레코드를
    사실상 특정하므로 nMI가 1.0으로 포화된다(고유키는 무엇이든 '결정'한다).
    → 복합 민감도(combo)의 얽힘 계산에서 상대로도 대상으로도 제외해야 한다.
    이들의 실제 위험은 '얽힘'이 아니라 '재식별'(granularity·EE)로 잡힌다.
    """
    n = len(frame)
    if n == 0:
        return []
    return [c for c in frame.columns if _is_identifier_like(frame[c], max_unique_ratio)]


def _is_identifier_like(s: pd.Series, max_unique_ratio: float = 0.9) -> bool:
    n = len(s)
    return (n > 0
            and not _is_numeric(s)                       # 숫자형은 구간화되므로 제외 안 함
            and _date_to_days(s) is None                 # 날짜형도 구간화되므로 제외 안 함
            and s.nunique(dropna=False) / n > max_unique_ratio)


def target_na_reason(sensitive: pd.Series,
                     category: Optional[str] = None,
                     _info: Optional["PresenceInfo"] = None,
                     max_unique_ratio: float = 0.9) -> str:
    """[v9.6] 목표 속성 자체 때문에 모든 성분이 비적용인지 — 사유 문자열("" 이면 아님).

    본문: "유효값이 없거나 … 계산할 수 없는 경우와 식별자형 목표는 비적용".
      · no-valid   : 유효 관측 집합 V 가 비어 있음 (결측·결측코드·이상치 제외 후)
      · identifier : 식별자형 목표 — 전체 기준 고유값 비율 > max_unique_ratio
                     (identifier_like_columns 와 같은 규칙) 또는 V 기준 near-unique
                     (EMD 와 같은 규칙: 고유값 ≥ 20 이고 고유값/|V| ≥ 0.9)
    Coverage 는 분모가 N 이라 항상 정의되므로 이 규칙과 무관하게 수치로 둔다.
    """
    s = _as_series(sensitive)
    info = _info or presence_info(s, category)
    v, _ = emd_valid_values(s, category, _info=info)
    n = len(v)
    if n == 0:
        return "no-valid: 유효 레코드 없음"
    if _is_identifier_like(s, max_unique_ratio):
        return f"식별자형 목표: 고유값 비율 > {max_unique_ratio:g}"
    vv = _date_to_days(v)
    vv = v if vv is None else vv
    if not _is_numeric(vv):
        k = vv.astype(str).nunique()
        if k >= EMD_NEAR_UNIQUE_MIN_K and k / n >= EMD_NEAR_UNIQUE_RATIO:
            return f"식별자형 목표: 유효값 고유 {k}/{n}"
    return ""


def composite_sensitivity(sensitive_frame: pd.DataFrame,
                          bins: int = 10,
                          max_unique_ratio: float = 0.9,
                          method: Optional[str] = None,
                          top_k: Optional[int] = None) -> Dict[str, float]:
    """
    복합 민감도 — 민감속성 i가 다른 민감속성들과 얽혀 연쇄 노출되는 정도.

    Combo_i = 1 − Π_{j≠i} (1 − AMI(민감_i; 민감_j))   ← noisy-OR
      (쌍 값은 COMBO_PAIR_METRIC: "ami"=우연 보정(기본) / "nmi"=기존)
      · 하나와만 강하게 얽혀도 높음(OR 보존)
      · 여러 민감속성과 얽히면 누적되어 더 높음

    ⚠️ near-unique(식별자형) 컬럼 방어(max_unique_ratio):
      access_ip·ADID·지문템플릿처럼 값이 거의 고유한 컬럼은 nMI가 1.0으로
      포화돼 다른 모든 속성의 combo까지 1.0으로 오염시킨다(고유키는 무엇이든
      '결정'하므로). 그래서 고유값 비율이 임계(기본 0.9)를 넘는 컬럼은 얽힘
      계산에서 제외하고 combo=0으로 둔다. 이들의 위험은 재식별(granularity·
      EE) 축에서 별도로 잡힌다.

    민감속성별 값 dict 반환. [v9.6] 얽힘 상대가 없으면(민감속성 1개, 또는 나머지가
    전부 식별자형) 0 이 아니라 NaN(비적용). 상태·사유는 composite_sensitivity_detail.
    그 밖의 경우(식별자형 목표 등)는 기존 수치 그대로다.

    [수정] 통합 방식 선택(method) + 쌍 중복 계산 제거
      · nMI = I/min(H_i,H_j) 는 대칭 → (i,j)와 (j,i)는 같은 값. 쌍 행렬을
        상삼각만 계산해 재사용한다(combo_pair_matrix).
      · 속성별 Combo는 자기가 낀 쌍을 전부 반영한다(순서 무관).
      · method : "noisy_or"  = 1 − Π(1 − nMI)   (기존, 쌍 많으면 1로 포화)
                 "topk_mean" = 상위 top_k 개 nMI 평균 (external_nmi와 같은 집계)
                 "max"       = 가장 강한 1쌍만 (누적 없음)
      · 구간화는 external_nmi와 동일한 bin_for_nmi 사용(날짜·문자열도 축소).
    """
    return {c: r.value for c, r in composite_sensitivity_detail(
        sensitive_frame, bins=bins, max_unique_ratio=max_unique_ratio,
        method=method, top_k=top_k).items()}


def composite_sensitivity_detail(sensitive_frame: pd.DataFrame,
                                 bins: int = 10,
                                 max_unique_ratio: float = 0.9,
                                 method: Optional[str] = None,
                                 top_k: Optional[int] = None,
                                 target_na: Optional[Dict[str, str]] = None,
                                 category_map: Optional[Dict[str, str]] = None
                                 ) -> Dict[str, MetricResult]:
    """composite_sensitivity 의 상태 포함 버전.

    비적용:
      · 목표 자체가 비적용(target_na_reason: 유효값 없음 / 식별자형 목표)
      · 상대 민감속성 없음 — 민감속성 1개, 또는 나머지가 모두 식별자형·유효값 없음
    target_na : {컬럼: 사유}. 없으면 여기서 target_na_reason 으로 판정한다.
    """
    method = method or COMBO_METHOD
    top_k = top_k or COMBO_TOP_K
    cols = list(sensitive_frame.columns)
    category_map = category_map or {}
    if target_na is None:
        target_na = {c: target_na_reason(sensitive_frame[c], category_map.get(c),
                                         max_unique_ratio=max_unique_ratio)
                     for c in cols}
    target_na = {c: r for c, r in target_na.items() if r}

    out: Dict[str, MetricResult] = {c: MetricResult.not_applicable(target_na[c])
                                    for c in cols if c in target_na}
    if len(cols) <= 1:
        for c in cols:
            out.setdefault(c, MetricResult.not_applicable("상대 민감속성 없음 (민감속성 1개)"))
        return out

    P = combo_pair_matrix(sensitive_frame, bins=bins,
                          max_unique_ratio=max_unique_ratio)
    partners_na = [c for c in target_na if c in P.columns]   # 유효값 없는 상대도 상대가 아님
    for i in cols:
        if i in out:
            continue
        if i not in P.index:                         # identifier_like_columns 로 빠진 목표
            out[i] = MetricResult.not_applicable("식별자형 목표")
            continue
        vals = (P.loc[i].drop(labels=[i] + [c for c in partners_na if c != i])
                .dropna().to_numpy(dtype=float))
        if len(vals) == 0:
            out[i] = MetricResult.not_applicable(
                "상대 민감속성 없음 (식별자형·유효값 없음·쌍별 유효 표본<2 제외 후)")
            continue
        out[i] = MetricResult.numeric(_aggregate_combo(vals, method, top_k))
    return out


COMBO_METHOD = "noisy_or"     # "noisy_or" | "topk_mean" | "max"
COMBO_TOP_K = 3
COMBO_PAIR_METRIC = "ami"     # "ami"(우연 보정, 기본) | "nmi"(기존)


# ---------------------------------------------------------------------------
# 지표 설정 묶음 — 모듈 전역 상수를 '안전하게' 바꾸는 방법
#   예전에는 CLI/비교 스크립트가 M.COMBO_METHOD = ... 처럼 모듈 전역을 직접 덮어써
#   (1) 한 번 바꾸면 같은 프로세스의 이후 호출(RiskService 등)에 계속 새어 나가고
#   (2) 어떤 설정으로 계산됐는지 결과만 보고 알 수 없었다.
#   metrics_config() 로 감싸면 블록을 벗어날 때 원래 값으로 복원된다.
#   ※ 전역을 쓰는 구조 자체는 유지(하위 호환)하므로 스레드 간 동시 실행에는
#     안전하지 않다. 서버에서 설정이 다른 요청을 병렬 처리하려면 프로세스를 나눌 것.
# ---------------------------------------------------------------------------
@dataclass
class MetricsConfig:
    combo_method: str = "noisy_or"          # noisy_or | topk_mean | max
    combo_top_k: int = 3
    combo_pair_metric: str = "ami"          # ami | nmi
    zero_absence_ratio: float = 0.30
    absence_overrides: Dict[str, list] = field(default_factory=dict)
    missing_overrides: Dict[str, list] = field(default_factory=dict)

    _GLOBALS = {"combo_method": "COMBO_METHOD", "combo_top_k": "COMBO_TOP_K",
                "combo_pair_metric": "COMBO_PAIR_METRIC",
                "zero_absence_ratio": "ZERO_ABSENCE_RATIO",
                "absence_overrides": "ABSENCE_OVERRIDES",
                "missing_overrides": "MISSING_OVERRIDES"}

    def validate(self) -> "MetricsConfig":
        if self.combo_method not in ("noisy_or", "topk_mean", "max"):
            raise ValueError(f"unknown combo_method: {self.combo_method}")
        if self.combo_pair_metric not in ("ami", "nmi"):
            raise ValueError(f"unknown combo_pair_metric: {self.combo_pair_metric}")
        if int(self.combo_top_k) < 1:
            raise ValueError("combo_top_k 는 1 이상이어야 합니다")
        if not 0.0 <= float(self.zero_absence_ratio) <= 1.0:
            raise ValueError("zero_absence_ratio 는 0~1 이어야 합니다")
        return self

    @classmethod
    def current(cls) -> "MetricsConfig":
        """지금 모듈 전역에 걸려 있는 설정을 스냅샷으로."""
        g = globals()
        return cls(**{f: (dict(g[name]) if f.endswith("_overrides") else g[name])
                      for f, name in cls._GLOBALS.items()})


@contextmanager
def metrics_config(config: Optional[MetricsConfig] = None, **overrides):
    """블록 안에서만 지표 설정을 바꾸고, 끝나면 원래 값으로 되돌린다.

        with metrics_config(combo_pair_metric="nmi"):
            m = compute_metric_matrix(...)
    """
    base = config or MetricsConfig.current()
    cfg = replace(base, **overrides).validate() if overrides else base.validate()
    g = globals()
    saved = {name: g[name] for name in MetricsConfig._GLOBALS.values()}
    try:
        for f, name in MetricsConfig._GLOBALS.items():
            g[name] = getattr(cfg, f)
        yield cfg
    finally:
        g.update(saved)


def _aggregate_combo(vals: np.ndarray, method: str, top_k: int) -> float:
    vals = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    if method == "noisy_or":
        v = 1.0 - np.prod(1.0 - vals)
    elif method == "topk_mean":
        k = max(1, min(int(top_k), len(vals)))
        v = np.sort(vals)[::-1][:k].mean()
    elif method == "max":
        v = vals.max()
    else:
        raise ValueError(f"unknown combo method: {method}")
    return float(np.clip(v, 0.0, 1.0))


def combo_pair_matrix(sensitive_frame: pd.DataFrame,
                      bins: int = 10,
                      max_unique_ratio: float = 0.9) -> pd.DataFrame:
    """민감속성 쌍별 AMI(기본)/nMI 행렬 (대칭, 대각=NaN). 식별자형 컬럼은 제외.
    쌍 값 지표는 COMBO_PAIR_METRIC("ami"|"nmi")로 선택.

    상삼각(i<j)만 계산해 양쪽에 채운다 → 쌍당 1회 계산(중복 제거).
    히트맵·논문 표(쌍별 기여도)에 그대로 쓸 수 있다.
    """
    pair_fn = ami if COMBO_PAIR_METRIC == "ami" else nmi
    ident = set(identifier_like_columns(sensitive_frame, max_unique_ratio))
    pool = [c for c in sensitive_frame.columns if c not in ident]
    binned = {c: bin_for_nmi(sensitive_frame[c], bins=bins) for c in pool}
    P = pd.DataFrame(np.nan, index=pool, columns=pool, dtype=float)
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            xa, xb = binned[pool[a]], binned[pool[b]]
            if pair_valid_count(xa, xb) < PAIR_MIN_VALID:
                continue              # 쌍별 유효 표본 < 2 → J_s 에서 제외 (NaN 유지)
            v = pair_fn(xa, xb)
            P.iat[a, b] = P.iat[b, a] = v
    return P


# ===========================================================================
# 외부 노출 지표
# ===========================================================================

def prepare_ee_bins(ee_frame: pd.DataFrame, bins: int = 10,
                    max_unique_ratio: float = 0.9) -> Dict[str, pd.Series]:
    """식별자형을 뺀 EE 컬럼들의 구간화 결과. 민감속성과 무관하므로 1회만 계산."""
    if ee_frame is None or ee_frame.shape[1] == 0:
        return {}
    ident = set(identifier_like_columns(ee_frame, max_unique_ratio))
    return {c: bin_for_nmi(ee_frame[c], bins=bins)
            for c in ee_frame.columns if c not in ident}


def external_nmi(sensitive: pd.Series,
                 ee_frame: pd.DataFrame,
                 bins: int = 10,
                 top_k: int = 3,
                 max_unique_ratio: float = 0.9,
                 _ee_bins: Optional[Dict[str, pd.Series]] = None) -> float:
    """
    외부 nMI — EE 속성이 민감속성을 얼마나 담는가 (top-cluster 집계).

    각 EE_e 와 민감_s 의 nMI 를 구한 뒤, **그 속성이 낀 상위 top_k 개의 평균**
    으로 속성 단위 셀을 만든다(문서 F: 속성축 top-cluster 집계). max(top-1)를
    쓰지 않는 이유:
      · near-unique(email·IP·adid 같은 식별자형) EE 하나가 nMI=1.0 을 만들어
        max를 1.0으로 포화시킨다. 먼저 그런 컬럼을 제외하고(max_unique_ratio),
        남은 것들의 top_k 평균을 써서 한 컬럼이 셀을 독점하지 못하게 한다.
    이 셀 값은 히트맵·개별뷰·그래프·스칼라가 모두 공유한다(두 벌로 만들지 않음).
    [v9.6] 보조속성(EE)이 없으면(식별자형 제외 후 포함) 0 이 아니라 NaN(비적용).
    """
    # near-unique(식별자형) EE 제외 — combo와 동일한 가드 재사용
    ee_bins = (_ee_bins if _ee_bins is not None
               else prepare_ee_bins(ee_frame, bins, max_unique_ratio))
    return external_nmi_detail(sensitive, ee_bins, bins=bins, top_k=top_k).value


def external_nmi_detail(sensitive: pd.Series,
                        ee_bins: Dict[str, pd.Series],
                        bins: int = 10,
                        top_k: int = 3) -> MetricResult:
    """[v9.6] L_s + 상태. 논문: "보조속성이 없거나 분모가 0이면 비적용".

    ν_se 가 정의되지 않는 쌍(분모 min(H)=0 또는 쌍별 유효 표본 < 2)은 평균에서 뺀다.
    정의된 쌍이 하나도 없으면 L_s 는 비적용 — 목표값이 하나뿐(H(X_s)=0)이면 모든
    쌍의 분모가 0 이므로 여기에 해당한다. m = min(top_k, 정의된 쌍 수).
    """
    if not ee_bins:
        return MetricResult.not_applicable("보조속성(EE) 없음")
    s = bin_for_nmi(_as_series(sensitive), bins=bins)   # 민감속성도 카디널리티 축소
    vals = [v for v in (nmi_defined(s, ev) for ev in ee_bins.values()) if v is not None]
    if not vals:
        return MetricResult.not_applicable(
            "분모 0 (목표 또는 모든 보조속성의 엔트로피 0 / 쌍별 유효 표본<2)")
    vals.sort(reverse=True)                      # 상위부터
    k = max(1, min(top_k, len(vals)))
    return MetricResult.numeric(float(np.clip(np.mean(vals[:k]), 0.0, 1.0)))   # top-k 평균


def qi_violation_mask(qi_frame: pd.DataFrame, k: int = 5) -> pd.Series:
    """각 레코드가 QI 결합 동치류 크기 < k (k-익명 위반)인지 bool mask.

    QI 결합 키는 민감속성과 무관하므로 compute_metric_matrix 에서 한 번만
    계산해 모든 민감속성이 재사용한다(예전엔 민감속성마다 문자열 결합을 다시 함).
    """
    key = qi_frame.astype(str).fillna("nan").agg("\x1f".join, axis=1)  # pandas 3: NaN 유지 → join TypeError 방지
    freq = key.map(key.value_counts())
    return freq.lt(max(2, int(k)))


def granularity_risk(qi_frame: pd.DataFrame,
                     sensitive: Optional[pd.Series] = None,
                     k: int = 5,
                     _violating: Optional[pd.Series] = None,
                     category: Optional[str] = None,
                     _info: Optional["PresenceInfo"] = None) -> float:
    """
    민감속성별 세분성 G_i — 해당 민감속성을 가진 레코드 중 QI 결합으로
    k-익명성을 위반하는 레코드의 비율.

        G_i = #{r : sensitive_i(r) 존재 AND |EC_QI(r)| < k}
              / #{r : sensitive_i(r) 존재}

    여기서 동치류 크기 |EC_QI(r)| 는 전체 데이터셋의 QI 조합을 기준으로 계산하고,
    마지막 분자/분모만 민감속성 i 존재 레코드로 제한한다. 따라서 같은 QI를 쓰더라도
    민감속성별 존재 위치가 다르면 G_i도 달라진다.

    sensitive=None이면 하위 호환을 위해 전체 레코드의 k-익명 위반율을 반환한다.
    [v9.6] 논문: "준식별자나 존재값이 없으면 비적용" — QI 가 없거나 존재 집합 P 가
    비어 있으면(분모 0) 0 이 아니라 NaN(비적용).
    """
    if qi_frame is None or qi_frame.shape[1] == 0:
        return float("nan")
    n = len(qi_frame)
    if n == 0:
        return 0.0

    violating = _violating if _violating is not None else qi_violation_mask(qi_frame, k)

    if sensitive is None:
        return float(np.clip(violating.mean(), 0.0, 1.0))

    s = _as_series(sensitive)
    # positional alignment 보장
    if len(s) != n:
        raise ValueError("sensitive와 qi_frame의 레코드 수가 다릅니다.")
    s = s.copy()
    s.index = qi_frame.index
    if _info is not None:
        present = _info.mask.copy()
        present.index = qi_frame.index
    else:
        present = sensitive_presence_mask(s, category)
    denom = int(present.sum())
    if denom == 0:
        return float("nan")                    # 존재값 없음 → 비적용

    numer = int((violating & present).sum())
    return float(np.clip(numer / denom, 0.0, 1.0))


def external_exposure(sensitive: pd.Series,
                      ee_frame: pd.DataFrame,
                      qi_frame: pd.DataFrame,
                      ee_scores: Sequence[float],
                      ee_threshold: float = 0.7,
                      bins: int = 10,
                      _ee_bins: Optional[Dict[str, pd.Series]] = None,
                      _violating: Optional[pd.Series] = None,
                      category: Optional[str] = None,
                      _info: Optional["PresenceInfo"] = None,
                      target_na: Optional[str] = None
                      ) -> Tuple[float, Dict[str, float]]:
    """
    외부 노출 위험 = 𝟙[max ee_score ≥ 0.7] × √(외부nMI × 세분성)

    ee_score는 게이트(EE/ER 판정), 실제 크기는 nMI·세분성의 기하평균.
    세분성(granularity_risk)은 민감속성 i 존재 레코드에 조건부인 k-익명 위반율 G_i다.
    반환: (외부노출값, 성분 dict).

    [v9.6] 논문 3.4절 규칙:
      · L_s : 보조속성(EE)이 없거나 분모가 0 이면 비적용 (external_nmi_detail)
      · G_s : 준식별자(QI)나 존재값이 없으면(P 공집합) 비적용
      · 목표 자체가 비적용(target_na_reason: 유효값 없음 / 식별자형 목표)이면 셋 다 비적용
      · L_s 또는 G_s 가 비적용이면 외부노출값도 비적용
    성분 dict 에 *_status / *_reason 을 함께 담는다.
    target_na : 목표 비적용 사유. None 이면 여기서 판정, "" 이면 해당 없음.
    """
    if target_na is None:
        target_na = target_na_reason(sensitive, category, _info=_info)
    max_ee = max(ee_scores) if len(ee_scores) else 0.0
    gate = 1.0 if max_ee >= ee_threshold else 0.0
    ee_bins = (_ee_bins if _ee_bins is not None
               else prepare_ee_bins(ee_frame, bins))
    l_r = external_nmi_detail(sensitive, ee_bins, bins=bins)
    ext_nmi = l_r.value
    rare = granularity_risk(qi_frame, sensitive=sensitive,  # 민감속성별 k-익명 위반율 G_i
                            _violating=_violating, category=category, _info=_info)

    l_st, l_why = l_r.status, l_r.reason
    if not ee_bins and ee_frame is not None and ee_frame.shape[1] > 0:
        l_why = "보조속성(EE) 없음 (모두 식별자형으로 제외)"
    if qi_frame is None or qi_frame.shape[1] == 0:
        g_st, g_why = STATUS_NA, "준식별자(QI) 없음"
    elif np.isnan(rare):
        g_st, g_why = STATUS_NA, "존재값 없음 (존재 집합 P 공집합)"
    else:
        g_st, g_why = STATUS_NUMERIC, ""
    if target_na:                            # 목표 자체가 비적용 → L, G 모두
        l_st, l_why = STATUS_NA, target_na
        g_st, g_why = STATUS_NA, target_na
        ext_nmi, rare = float("nan"), float("nan")

    if l_st == STATUS_NUMERIC and g_st == STATUS_NUMERIC:
        magnitude = float(np.sqrt(max(0.0, ext_nmi) * max(0.0, rare)))
        value, e_st, e_why = gate * magnitude, STATUS_NUMERIC, ""
    else:
        magnitude = float("nan")
        value, e_st = float("nan"), STATUS_NA
        e_why = " / ".join(dict.fromkeys(w for w in (l_why, g_why) if w))
    return value, {
        "gate": gate, "max_ee_score": max_ee,
        "external_nmi": ext_nmi,
        "granularity": rare,               # 민감속성별 G_i
        "magnitude_geomean": magnitude,
        "external_nmi_status": l_st, "external_nmi_reason": l_why,
        "granularity_status": g_st, "granularity_reason": g_why,
        "ee_match_status": e_st, "ee_match_reason": e_why,
    }


# ===========================================================================
# 지표 통합 계산 — (속성 × 지표) 행렬
# ===========================================================================

@dataclass
class MetricRow:
    """한 민감속성에 대한 5개 지표 값 (하나로 합치지 않음).

    [v9.6] 각 성분은 *_status(numeric|determined|not_applicable)를 가진다.
    status 가 numeric 이 아니면 값은 NaN (0 으로 대체하지 않음).
    """
    attribute: str
    category: str
    emd: float = 0.0
    coverage: float = 0.0
    combo: float = 0.0            # 복합 민감도 (noisy-OR)
    ee_match: float = 0.0         # 외부 노출 = gate × √(nMI×세분성)
    granularity: float = 0.0      # 민감속성별 k-익명 위반율 G_i
    # 참고용 성분
    external_nmi: float = 0.0
    ee_gate: float = 0.0
    max_ee_score: float = 0.0
    n_present: int = 0            # 존재('있음') 레코드 수 (Coverage, 세분성 분자)
    n_valid: int = 0              # 유효 레코드 수 (EMD 표본: 결측, 결측코드, 이상치 제외)
    absence_rule: str = ""        # 적용된 부재 판정 규칙 (검토용)
    absence_warning: str = ""     # 부재 판정에서 사람이 확인할 의심 신호
    emd_note: str = ""            # EMD 가 확정, 비적용이거나 해석 주의일 때 사유
    # 산출 상태 (numeric | determined | not_applicable)
    emd_status: str = STATUS_NUMERIC
    coverage_status: str = STATUS_NUMERIC
    combo_status: str = STATUS_NUMERIC
    ee_match_status: str = STATUS_NUMERIC
    granularity_status: str = STATUS_NUMERIC
    external_nmi_status: str = STATUS_NUMERIC
    status_reason: str = ""       # 확정, 비적용 사유 ("지표: 사유; ...")


# 상태를 갖는 성분 열 (행렬의 '{key}_status' 열과 짝)
STATUS_METRICS: Tuple[str, ...] = ("emd", "coverage", "combo",
                                   "ee_match", "granularity", "external_nmi")


def compute_metric_matrix(df: pd.DataFrame,
                          sensitive_cols: Sequence[str],
                          qi_cols: Sequence[str],
                          ee_cols: Sequence[str],
                          ee_score_map: Optional[Dict[str, float]] = None,
                          ee_threshold: float = 0.7,
                          bins: int = 10,
                          category_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    탐지 결과를 받아 (민감속성 × 지표) 위험 행렬을 만든다.

    Parameters
    ----------
    df : 원본 데이터프레임
    sensitive_cols : 탐지된 민감속성 컬럼명들
    qi_cols        : 준식별자(QI) 후보 컬럼명들 (동치류·희귀도 계산용)
    ee_cols        : EE 사전에 매칭된 외부 공개 속성 컬럼명들
    ee_score_map   : {ee컬럼: ee_score}. 게이트 판정용
    ee_threshold   : EE/ER 경계 (기본 0.7, 사전과 일치)
    category_map   : {민감컬럼: 탐지 카테고리}. 부재(0 의 의미) 판정에 쓴다.
                     없으면 카테고리 모름으로 판정(구버전 동작에 가까움).

    Returns
    -------
    DataFrame : index=민감속성, columns=[emd, coverage, combo, ee_match,
                granularity, ..., {지표}_status, status_reason].
                각 셀 [0,1], 클수록 위험. 합치지 않음.
                [v9.6] 확정·비적용 셀은 NaN 이고 {지표}_status 로 구분한다.
    """
    ee_score_map = ee_score_map or {}
    qi_frame = df[list(qi_cols)] if qi_cols else pd.DataFrame(index=df.index)
    ee_frame = df[list(ee_cols)] if ee_cols else pd.DataFrame(index=df.index)
    ee_scores = [ee_score_map.get(c, 0.0) for c in ee_cols]

    # 복합 민감도는 민감속성 전체를 한 번에 (쌍 계산)
    sens_frame = df[list(sensitive_cols)] if sensitive_cols else pd.DataFrame(index=df.index)
    category_map = category_map or {}
    # 존재 판정 1회 + 목표 비적용 사유(유효값 없음 / 식별자형 목표) 1회 → 모든 성분이 공유
    infos = {c: presence_info(df[c], category_map.get(c) or None) for c in sensitive_cols}
    target_na = {c: target_na_reason(df[c], category_map.get(c) or None, _info=infos[c])
                 for c in sensitive_cols}
    combo_map = composite_sensitivity_detail(sens_frame, bins=bins, target_na=target_na)

    # 세분성(granularity)은 민감속성별 G_i. 전체 QI 동치류 크기를 구한 뒤
    # 각 민감속성이 실제 존재하는 레코드에 조건부로 k-익명 위반 비율을 계산한다.
    # [리팩토링] 민감속성과 무관한 것(QI 위반 mask, EE 구간화)은 루프 밖에서 1회만.
    violating = (qi_violation_mask(qi_frame) if qi_frame.shape[1] and len(qi_frame)
                 else None)
    ee_bins = prepare_ee_bins(ee_frame, bins=bins)

    rows: List[MetricRow] = []
    for col in sensitive_cols:
        s = df[col]
        cat = category_map.get(col) or None
        info = infos[col]
        emd_r = emd_result(s, bins=bins, _info=info)
        if target_na[col] and emd_r.status == STATUS_NUMERIC:    # 식별자형 목표 → EMD 도 비적용
            emd_r = MetricResult.not_applicable(
                target_na[col] + (f" | {emd_r.reason}" if emd_r.reason else ""))
        n_valid = int(len(emd_valid_values(s, cat, _info=info)[0]))
        ext_val, comp = external_exposure(
            s, ee_frame, qi_frame, ee_scores,
            ee_threshold=ee_threshold, bins=bins,
            _ee_bins=ee_bins, _violating=violating, category=cat, _info=info,
            target_na=target_na[col])
        combo_r = combo_map.get(col, MetricResult.not_applicable("상대 민감속성 없음"))
        reasons = []
        if target_na[col]:                            # 목표 자체 비적용 → 사유 한 번만
            reasons.append(f"전 성분(coverage 제외): {target_na[col]}")
        else:
            if emd_r.status != STATUS_NUMERIC:
                reasons.append(f"emd: {emd_r.reason.split(' | ')[0]}")
            if combo_r.status != STATUS_NUMERIC:
                reasons.append(f"combo: {combo_r.reason}")
            for k in ("external_nmi", "granularity"):
                if comp[f"{k}_status"] != STATUS_NUMERIC:
                    reasons.append(f"{k}: {comp[f'{k}_reason']}")
        rows.append(MetricRow(
            attribute=col,
            category=cat or "",
            emd=emd_r.value,                         # 개별 속성 분포 편향(QI 무관)
            coverage=coverage_risk(s, _info=info),   # 실제 존재율 C_i=존재수/N (QI 무관)
            combo=combo_r.value,
            ee_match=ext_val,
            granularity=comp["granularity"],
            external_nmi=comp["external_nmi"],
            ee_gate=comp["gate"],
            max_ee_score=comp["max_ee_score"],
            n_present=int(info.mask.sum()),
            n_valid=n_valid,
            absence_rule=info.rule,
            absence_warning=info.warning,
            emd_note=emd_r.reason,
            emd_status=emd_r.status,
            combo_status=combo_r.status,
            ee_match_status=comp["ee_match_status"],
            granularity_status=comp["granularity_status"],
            external_nmi_status=comp["external_nmi_status"],
            status_reason="; ".join(reasons),
        ))

    out = pd.DataFrame([r.__dict__ for r in rows])
    if not out.empty:
        out = out.set_index("attribute")
    return out


# ===========================================================================
# 지표 프레임워크 — 지표를 하나로 합치지 않는다.
#   각 지표를 '따로' 두고, 이용자가 목적에 맞게 선택, 해석하도록 메타데이터를
#   제공한다. 여기엔 (1) 무엇을 재는가(지표 설명), (2) 대략적 참고 임계값을
#   담는다. 참고 임계값은 데이터, 맥락에 따라 조정할 실무적 '주목 기준선'이다.
# ===========================================================================

@dataclass(frozen=True)
class MetricSpec:
    key: str                 # 행렬 컬럼명
    name_ko: str
    name_en: str
    axis: str                # "internal"(내부 분포) | "external"(외부 분포)
    measures: str            # 무엇을 재는가 (지표 설명, 한 줄)
    ref_watch: float         # 이 값 이상이면 '한 번 더 볼' 실무 기준선(참고용)
    ref_note: str            # 그 기준선을 어떻게 읽나
    direction: str = "높을수록 위험"


# 순서 = 첨부 히트맵의 열 순서 (내부 3 + 외부 2)
METRIC_SPECS: Dict[str, MetricSpec] = {
    "emd": MetricSpec(
        key="emd", name_ko="EMD", name_en="Earth Mover's Distance",
        axis="internal",
        measures="민감속성 값 분포가 균등분포에서 얼마나 쏠려 있는가(분포 편향, 범주형은 TVD). '없음' 값 포함",
        ref_watch=0.50,
        ref_note="유효 레코드(결측·결측코드·이상치 제외, '없음' 포함)의 분포. 연속·날짜=등폭 EMD/EMD_max, 범주=TVD/(1−1/K). 0/1 여부는 |2p−1|. 0.5+면 값이 한쪽으로 뚜렷이 쏠림.",
    ),
    "coverage": MetricSpec(
        key="coverage", name_ko="Coverage", name_en="Value Coverage",
        axis="internal",
        measures="민감속성이 실제로 존재하는 레코드 비율(존재수/N, 무정보·음성값 제외) — 노출 가능 물량",
        ref_watch=0.40,
        ref_note="실제 존재율. 1이면 모든 레코드가 실제 민감값 보유, 0이면 존재 레코드 없음(전부 결측·무정보·음성).",
    ),
    "combo": MetricSpec(
        key="combo", name_ko="Combo_Bonus", name_en="Composite Sensitivity",
        axis="internal",
        measures="민감속성끼리 얼마나 얽혀 서로를 추론·연쇄 노출시키는가(쌍 AMI, noisy-OR)",
        ref_watch=0.50,
        ref_note="noisy-OR라 여러 속성과 얽히면 빠르게 커짐. 0.5+면 다른 민감속성으로 추론 가능성 큼.",
    ),
    "ee_match": MetricSpec(
        key="ee_match", name_ko="EE 연결성", name_en="External-Exposure Match",
        axis="external",
        measures="외부 공개(EE) 속성과 결합해 민감속성 노출이 증폭되는가",
        ref_watch=0.30,
        ref_note="게이트 통과 시점에 이미 결합 가능. √(외부nMI×세분성) 0.3+면 외부결합 노출이 실질적. "
                 "외부nMI는 near-unique EE 제외 후 top-k 평균(top-cluster).",
    ),
    "granularity": MetricSpec(
        key="granularity", name_ko="세분성", name_en="Granularity (k-anon)",
        axis="external",
        measures="민감속성 i 보유 레코드 중 QI 결합으로 k-익명성을 위반하는 비율 G_i",
        ref_watch=0.50,
        ref_note="전체 데이터의 QI 동치류 크기를 기준으로, 해당 민감속성이 실제 존재하는 레코드만 분모에 둠. "
                 "0=해당 민감속성 보유자가 모두 k명 이상과 겹침, 1=모두 k-익명 위반.",
    ),
}

METRIC_ORDER: List[str] = list(METRIC_SPECS.keys())


def internal_metrics() -> List[str]:
    """내부 분포 위험 축 지표 키."""
    return [k for k, s in METRIC_SPECS.items() if s.axis == "internal"]


def external_metrics() -> List[str]:
    """외부 분포 위험 축 지표 키."""
    return [k for k, s in METRIC_SPECS.items() if s.axis == "external"]
