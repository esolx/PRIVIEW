# -*- coding: utf-8 -*-
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from .config import (CATEGORIES, NEGATIVE_CODE, NON_LEGAL_CODES,
                     REGEX_PATTERNS, VALUE_VALIDATORS)
from .fp_control import value_evidence
from .risk_weight import RiskWeightConfig, attribute_weight

if TYPE_CHECKING:  # 문자열 애노테이션 전용 (런타임 미평가, 순환참조 방지)
    from .pipeline import RiskInput

"""cassed_lite.py — Track B 탐지 강화(하이브리드).

배경
  Track B(비법정)의 약점은 알고리즘이 아니라 신호 부재다. 소득·학력·직업·
  소비 등 일부 비법정 속성은 값이 일반적이라 이름·정규식만으로는 자주 놓친다. 특히 실제
  기업 DB처럼 컬럼명이 축약·오도되면(CASSED가 지적한 문제) 0단계 이름 매핑이
  통째로 실패한다. 그때 **값의 의미**로 건져야 한다.

이 모듈이 하는 일 (추천 순서의 1단계 = 무학습, 즉시 적용)
  · 닫힌/준-닫힌 어휘 가제티어(CASSED의 lookup-table 아이디어)로 값을 스캔.
  · config의 정규식 값검증기(위치·통신·단말·신용)를 재사용.
  · 둘을 합쳐, 이름이 실패해도 **값만으로 Track B 카테고리를 복구**한다.
  · 한 컬럼이 여러 카테고리에 해당하면 **다중라벨**로 기록(CASSED와 동일 철학).

설계 원칙
  · **가산(additive) 후처리.** 기존 0/1/2단계 캐스케이드·5층 FP 방어·검수 큐를
    건드리지 않는다. pipe.run(df) 결과 위에 value_rescue()를 덧대는 구조라
    회귀 위험이 낮다.
  · 값만으로 살린 컬럼은 tier를 보수적으로(대개 PROBABLE/WEAK) 매기고
    review_required=True 로 검수 큐에 넣는다. Recall 우선, 사람이 최종 확인.
  · 가제티어로도 안 잡히는 잔여(예: 신용점수 같은 맨 정수)는
    3단계(파인튜닝 헤드)의 몫으로 남긴다 → residual_gap 로 보고.

3단계(파인튜닝 CASSED 헤드) 자리
  Stage2Rescuer 프로토콜 + FinetunedCassedRescuer 스캐폴드를 둔다. torch/
  sentence-transformers와 학습 가중치가 있으면 그걸 쓰고, 없으면 자동으로
  가제티어+코사인 폴백. 지금 환경엔 학습 스택이 없어 폴백으로 동작한다.
"""


# ===========================================================================
# 1) Track B 가제티어 (닫힌/준-닫힌 어휘)  — CASSED의 lookup table 대응
# ===========================================================================
#   값을 소문자, 공백정리한 뒤: exact 집합은 완전일치, sub 집합은 부분일치로 스캔.
def _norm(v) -> str:
    return " ".join(str(v).strip().lower().split())


# 완전일치용(닫힌 집합): 값 전체가 이 집합에 있으면 히트
EXACT_GAZETTEERS: Dict[str, frozenset] = {
    "EDUCATION": frozenset({
        "초졸", "중졸", "고졸", "전문학사", "학사", "석사", "박사", "수료",
        "고등학교", "대학교", "대학원",
        "elementary", "middle school", "high school", "highschool",
        "associate", "bachelor", "master", "phd", "doctorate", "diploma",
    }),
    "OCCUPATION": frozenset({
        "회사원", "공무원", "자영업", "전문직", "학생", "주부", "무직",
        "engineer", "analyst", "manager", "designer", "teacher", "doctor",
        "nurse", "developer", "researcher", "clerk", "student", "unemployed",
    }),
    "CREDIT_FINANCE": frozenset({
        "정상", "연체", "부도", "파산", "회생",
        "normal", "delinquent", "default", "bankrupt",
    }),
}

# 부분일치용(준-닫힌/접두형): 값 안에 이 토큰이 들어가면 히트
SUB_GAZETTEERS: Dict[str, Tuple[str, ...]] = {
    # 소득 구간 표기: "<3천", "3~5천", ">5천", "저소득", "고소득" 등
    "INCOME_ASSET": ("천만", "억", "소득", "연봉", "만원",
                     "<", "~", "이상", "미만", "저소득", "고소득", "중위"),
    # 소비/구매 상태, 이력
    "CONSUMPTION": ("구매", "결제", "환불", "주문", "구독", "쿠폰",
                    "purchase", "payment", "refund", "order", "coupon"),
    # 위치(주소 텍스트) — 단음절(시/구/동)은 오탐이 커서 제외, 다자, 영문만.
    "LOCATION": ("특별시", "광역시", "번지",
                 "seoul", "busan", "gangnam", "-si", "-gu", "-dong", "street"),
}


def gazetteer_score(category: str, values: Sequence,
                    sample: int = 300) -> float:
    """값이 해당 카테고리 어휘에 맞는 비율(0~1)."""
    vals = [_norm(v) for v in values if v is not None][:sample]
    if not vals:
        return 0.0
    exact = EXACT_GAZETTEERS.get(category)
    subs = SUB_GAZETTEERS.get(category)
    hit = 0
    for v in vals:
        if exact and v in exact:
            hit += 1
        elif subs and any(s in v for s in subs):
            hit += 1
    return hit / len(vals)


# ===========================================================================
# 2) 값 기반 Track B 탐지 (가제티어 + config 정규식 값검증기)
# ===========================================================================
def detect_track_b_by_values(values: Sequence,
                             min_gazetteer: float = 0.5,
                             min_regex: float = 0.30
                             ) -> List[Tuple[str, float, str]]:
    """값만으로 가능한 Track B 카테고리 후보를 점수순으로 반환(다중라벨).

    returns [(category, score, evidence_kind), ...]  score 내림차순.
      evidence_kind ∈ {"gazetteer", "regex"}
    """
    cand: Dict[str, Tuple[float, str]] = {}

    # (a) 가제티어
    for cat in list(EXACT_GAZETTEERS) + list(SUB_GAZETTEERS):
        if cat not in NON_LEGAL_CODES:
            continue
        s = gazetteer_score(cat, values)
        if s >= min_gazetteer:
            cand[cat] = max(cand.get(cat, (0.0, "")), (s, "gazetteer"))

    # 날짜 방어: 값이 대부분 날짜면(YYYY-MM-DD 등) 계좌번호 정규식과 충돌하므로
    #   regex 근거를 신뢰하지 않는다. (2024-05-13 이 account 패턴에 걸리는 문제)
    vlist = [str(v) for v in values if v is not None][:300]
    date_rate = (sum(1 for v in vlist if REGEX_PATTERNS["date"].search(v))
                 / len(vlist)) if vlist else 0.0
    date_heavy = date_rate >= 0.5

    # (b) config 정규식 값검증기 재사용 (LOCATION/COMMUNICATION/DEVICE_ONLINE/CREDIT_FINANCE 등)
    for cat in NON_LEGAL_CODES:
        keys = VALUE_VALIDATORS.get(cat)
        if not keys:
            continue
        if date_heavy and cat == "CREDIT_FINANCE":
            continue   # 날짜 컬럼이 계좌로 오인되는 것 차단
        ok, rate, _ = value_evidence(cat, list(values), min_hit_rate=min_regex)
        if ok and rate >= min_regex:
            cand[cat] = max(cand.get(cat, (0.0, "")), (float(rate), "regex"))

    out = [(c, s, k) for c, (s, k) in cand.items()]
    out.sort(key=lambda t: -t[1])
    return out


# ===========================================================================
# 3) 가산 후처리 — 이름이 실패해도 값으로 Track B 복구 + 다중라벨
# ===========================================================================
@dataclasses.dataclass
class RescueChange:
    column: str
    before: str
    after: str
    score: float
    evidence: str
    secondary: List[str]


def value_rescue(df, result: "RiskInput",
                 cfg: Optional["RiskWeightConfig"] = None,
                 min_gazetteer: float = 0.5,
                 min_regex: float = 0.30) -> Dict:
    """pipe.run(df) 결과에 값 기반 Track B 탐지를 덧대는 가산 후처리.

    · NON_SENSITIVE로 남은 컬럼을 값 근거로 Track B로 승격(검수 큐로).
    · 이미 민감인 컬럼에는 추가 카테고리를 secondary(다중라벨)로 기록.
    반환: {"decisions": 새 결정, "multilabel": {col:[cats]}, "changes":[...]}
    """
    cfg = cfg or RiskWeightConfig()
    decisions = list(result.decisions)
    multilabel: Dict[str, List[str]] = {}
    changes: List[RescueChange] = []

    for i, d in enumerate(decisions):
        if d.column not in df.columns:
            continue
        vals = df[d.column].dropna().tolist()
        hits = detect_track_b_by_values(vals, min_gazetteer, min_regex)
        if not hits:
            continue
        hit_cats = [c for c, _, _ in hits]

        if not d.is_sensitive:
            # ---- 값으로 복구: NON_SENSITIVE → Track B -------------------
            cat, score, kind = hits[0]
            aw = attribute_weight(d.column, cat, vals, cfg=cfg)
            tier = ("CONFIRMED" if score >= 0.8 else
                    "PROBABLE" if score >= 0.5 else "WEAK")
            meta = CATEGORIES.get(cat, CATEGORIES[NEGATIVE_CODE])
            decisions[i] = dataclasses.replace(
                d, category=cat, category_ko=meta.name_ko,
                is_sensitive=True, is_legal=False, track="B",
                stage="Bv(value rescue)", tier=tier,
                review_required=True,           # 값만 근거 → 사람 확인
                value_hit_rate=float(score),
                weight=aw.weight, weight_basis=aw.basis,
                entropy_norm=aw.h_norm,
                reasons=f"값 기반 복구({kind}, 히트율 {score:.2f})")
            changes.append(RescueChange(d.column, "NON_SENSITIVE", cat,
                                        score, kind, hit_cats[1:]))
            if len(hit_cats) > 1:
                multilabel[d.column] = hit_cats
        else:
            # ---- 이미 민감: 추가 라벨을 다중라벨로 -----------------------
            extra = [c for c in hit_cats if c != d.category]
            if extra:
                multilabel[d.column] = [d.category] + extra

    return {"decisions": decisions, "multilabel": multilabel,
            "changes": changes}


# ===========================================================================
# 4) 3단계 자리 — 파인튜닝 CASSED 헤드 스캐폴드 (graceful fallback)
# ===========================================================================
class Stage2Rescuer:
    """2단계 구제기 인터페이스. score(texts)->np.ndarray[len(texts), n_cat]."""
    def available(self) -> bool: ...
    def score(self, texts): ...


class FinetunedCassedRescuer(Stage2Rescuer):
    """mpnet 위에 다중라벨 헤드를 얹은 CASSED-lite 분류기(3단계).

    학습 가중치(model_dir)와 torch/sentence-transformers가 있으면 그걸 쓰고,
    없으면 available()=False 를 반환 → 파이프라인이 기존 코사인 프로토타입
    +가제티어 폴백을 그대로 쓴다. 지금 환경엔 학습 스택이 없어 폴백된다.

    학습 절차(문서용):
      1) synth.make_* 로 (헤더⊕값) 텍스트 + Track B 다중라벨 데이터셋 생성.
      2) SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') 인코더
         위에 sigmoid 다중라벨 헤드. recall 우선 임계값(~0.4, CASSED와 동일).
      3) 학습 후 model_dir 에 저장 → 여기서 로드해 2단계 구제를 교체.
    """
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self._model = None
        try:
            if model_dir:
                from sentence_transformers import SentenceTransformer  # noqa
                # self._model = load_head(model_dir)  # 학습 산출물 로드 지점
        except Exception:
            self._model = None

    def available(self) -> bool:
        return self._model is not None

    def score(self, texts):
        if not self.available():
            raise RuntimeError("파인튜닝 헤드 미탑재 — 폴백 사용")
        return self._model.predict(texts)   # 학습 후 구현
