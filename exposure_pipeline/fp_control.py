# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

"""fp_control.py — 5층 오탐 방어 (제외·값검증·구조가드·증거등급·피드백)"""

from .config import (LEGAL_CODES, NAME_REQUIRED_VALIDATORS, NEGATIVE_CODE,
                     REGEX_PATTERNS, VALUE_VALIDATORS, exclusion_terms)
from .features import profile_to_text

# ===  fp_control.py

"""
과다 탐지(False Positive) 억제 모듈.

Recall을 밀면 FP가 늘어난다. 그건 설계상 감수한 것이고, 문제는
**늘어난 FP를 어디서 어떻게 걷어내느냐**다. 임계값을 다시 올리면
Recall이 무너지므로, 임계값이 아닌 다른 축에서 잡아야 한다.

5층 방어 (위에서 아래로 순서대로 적용)
  L1 제외 규칙   : 명시적 비민감 어휘/패턴에 걸리면 즉시 비민감 확정
  L2 값 검증     : 컬럼명이 민감해도 값이 그 유형이 아니면 강등
  L3 구조 가드   : 카디널리티·자릿수·dtype이 카테고리와 모순이면 강등
  L4 증거 등급   : 독립 신호 개수로 CONFIRMED/PROBABLE/WEAK 구분,
                   WEAK는 자동 확정 대신 검수 큐로
  L5 피드백 루프 : 검수에서 FP로 판정된 컬럼을 제외 규칙·학습 데이터로 환류

핵심 원칙: **FP를 없애지 말고 격리하라.**
임계값을 올려 FP를 지우면 그 아래 있던 TP도 같이 지워진다.
증거 등급으로 나눠 자동 확정 범위만 좁히면 Recall은 그대로 두고
검수 부하만 줄일 수 있다.
"""


# 증거 등급
CONFIRMED = "CONFIRMED"   # 이름 + 값 둘 다 일치 → 자동 확정
PROBABLE = "PROBABLE"     # 신호 2개 → 자동 확정 (법정은 검수 권장)
WEAK = "WEAK"             # 신호 1개 → 검수 큐
NONE = "NONE"


# ---------------------------------------------------------------------------
# L1. 제외 규칙
# ---------------------------------------------------------------------------
@dataclass
class ExclusionRules:
    """비민감으로 확정할 컬럼명 규칙.

    terms   : 부분 문자열 (영문은 단어 경계 적용)
    regexes : 정규표현식 (예: r"^tmp_", r"_bak$")
    exact   : 완전 일치 컬럼명 (검수 피드백으로 쌓이는 곳)
    override_legal : True면 법정 카테고리 판정도 뒤집는다. 기본 False —
                     법정 오탐을 규칙으로 지우는 건 위험하므로 명시적 선택 사항.
    """
    terms: Set[str] = field(default_factory=lambda: set(exclusion_terms()))
    regexes: List[str] = field(default_factory=lambda: [
        r"^(tmp|temp|test|dummy|sample|bak|backup)[_\-]",
        r"[_\-](id|seq|no|idx|key|cd|flag|yn)$",
        r"^(created|updated|deleted|modified)[_\-]?(at|on|date|time)?$",
        r"^(row|record|line)[_\-]?(num|no|id)$",
    ])
    exact: Set[str] = field(default_factory=set)
    override_legal: bool = False

    def __post_init__(self):
        self._compiled = [re.compile(r, re.IGNORECASE) for r in self.regexes]
        self._term_pat = self._build_terms(self.terms)

    @staticmethod
    def _build_terms(terms: Set[str]):
        if not terms:
            return None
        parts = [(r"\b" + re.escape(t) + r"\b") if t.isascii() else re.escape(t)
                 for t in sorted(terms, key=len, reverse=True)]
        return re.compile("|".join(parts), re.IGNORECASE)

    def matches(self, column: str) -> Optional[str]:
        """걸리면 사유 문자열, 아니면 None."""
        name = str(column)
        norm = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
        norm = re.sub(r"[_\-.]+", " ", norm).strip().lower()
        if name in self.exact or norm in self.exact:
            return "exact:검수에서 FP로 확정된 컬럼"
        for rx in self._compiled:
            if rx.search(name) or rx.search(norm):
                return f"regex:{rx.pattern}"
        if self._term_pat:
            m = self._term_pat.search(norm)
            if m:
                return f"term:{m.group()}"
        return None

    def add_false_positive(self, column: str) -> None:
        self.exact.add(str(column))

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps({
            "terms": sorted(self.terms), "regexes": self.regexes,
            "exact": sorted(self.exact), "override_legal": self.override_legal,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ExclusionRules":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(terms=set(d.get("terms", [])), regexes=d.get("regexes", []),
                   exact=set(d.get("exact", [])),
                   override_legal=bool(d.get("override_legal", False)))


# ---------------------------------------------------------------------------
# L2. 값 수준 검증
# ---------------------------------------------------------------------------
def value_evidence(category: str, values: Sequence[str],
                   min_hit_rate: float = 0.30,
                   name_supports: bool = True) -> Tuple[bool, float, str]:
    """값이 해당 카테고리 형식과 맞는가.

    returns (통과여부, 히트율, 사용한 검증기)
    검증기가 정의되지 않은 카테고리는 (True, nan, "n/a") — 판단 보류(통과 처리).
    형식이 없는 카테고리(종교, 정치성향 등)까지 값 검증을 강제하면
    오히려 TP를 죽인다.

    name_supports : 컬럼명이 이 카테고리를 지지하는가.
        NAME_REQUIRED_VALIDATORS(예: GENETIC의 hugo)에 든 헐거운 검증기는
        name_supports=True 일 때만 값 증거로 인정한다. 기본 True 라
        이름 게이트가 없는 카테고리·검증기의 동작은 종전과 동일하다.
    """
    keys = VALUE_VALIDATORS.get(category)
    if not keys:
        return True, float("nan"), "n/a"
    # 이름 게이트: 이름이 카테고리를 지지하지 않으면 헐거운 검증기를 제외
    gated = set(NAME_REQUIRED_VALIDATORS.get(category, ()))
    active_keys = [k for k in keys if (k not in gated) or name_supports]
    if not active_keys:
        # 남은 검증기가 없다(모두 이름 게이트인데 이름 미지지) → 값 증거 없음
        return False, 0.0, ",".join(keys)
    vals = [str(v).strip() for v in values if v is not None][:300]
    if not vals:
        return False, 0.0, ",".join(active_keys)

    # INCOME_ASSET은 capital.gain/loss처럼 값이 단순 정수여도 컬럼명 의미가
    # 강한 경우가 많다. 통화기호 정규식만 강제하면 실제 소득/자산 컬럼을 미탐한다.
    if category == "INCOME_ASSET" and name_supports:
        def _numeric(v):
            v = v.replace(",", "").replace(" ", "")
            try:
                float(v)
                return True
            except Exception:
                return False
        numeric_rate = sum(_numeric(v) for v in vals) / len(vals)
        if numeric_rate >= 0.70:
            return True, numeric_rate, "numeric+name"
    best, best_key = 0.0, active_keys[0]
    for k in active_keys:
        pat = REGEX_PATTERNS[k]
        rate = sum(1 for v in vals if pat.search(v)) / len(vals)
        if rate > best:
            best, best_key = rate, k
    return best >= min_hit_rate, best, best_key


# ---------------------------------------------------------------------------
# 값 사전 근거 — 값이 법정 카테고리 어휘로 채워져 있는가
# ---------------------------------------------------------------------------
_UNINFORMATIVE_VALUES = frozenset({
    "", "nan", "none", "null", "na", "n/a", "unknown", "other", "others", "etc",
    "미상", "모름", "무응답", "기타", "없음", "해당없음", "-", "--",
})


def _norm_value(v) -> str:
    import re as _re
    t = str(v).strip().lower()
    t = _re.sub(r"[\-_/·,]+", " ", t)
    return _re.sub(r"\s+", " ", t).strip()


_LEX_CACHE: Dict[str, Tuple] = {}


def _compile_terms(terms):
    import re as _re
    t_ascii = sorted({_norm_value(t) for t in terms if t.isascii()}, key=len, reverse=True)
    pat = (_re.compile(r"(?<![a-z])(" + "|".join(_re.escape(t) for t in t_ascii) + r")(?![a-z])")
           if t_ascii else None)
    return pat, tuple(t for t in terms if not t.isascii())


def _lexicon_matcher(category: str, with_context: bool = False):
    """카테고리별 (영문 결합 정규식, 한글 단어들, 값 형식 정규식들) — 컴파일 후 캐시.
    with_context=True 면 다의어 값(context_terms: left/right, positive …)도 포함."""
    import re as _re
    from .config import LEXICON_VERSION, VALUE_CONTEXT_TERMS, VALUE_LEXICONS, VALUE_PATTERNS
    terms = tuple(VALUE_LEXICONS.get(category, ()))
    if with_context:
        terms += tuple(VALUE_CONTEXT_TERMS.get(category, ()))
    pats = VALUE_PATTERNS.get(category, ())
    if not terms and not pats:
        return None
    key = f"{category}|{with_context}|{LEXICON_VERSION[0]}"
    if key not in _LEX_CACHE:
        pat, t_ko = _compile_terms(terms) if terms else (None, ())
        _LEX_CACHE[key] = (pat, t_ko, tuple(_re.compile(p) for p in pats))
    return _LEX_CACHE[key]


def lexicon_value_counts(values: Sequence) -> Dict[str, int]:
    """정규화한 '정보 있는 값' → 등장 수 (여러 카테고리 검사에 1회만 계산해 재사용)."""
    counts: Dict[str, int] = {}
    for v in list(values)[:300]:
        if v is None:
            continue
        nv = _norm_value(v)
        if nv in _UNINFORMATIVE_VALUES:
            continue
        counts[nv] = counts.get(nv, 0) + 1
    return counts


def value_lexicon_evidence(category: str, values: Sequence = (),
                           min_rate: float = 0.60, min_distinct: int = 2,
                           _counts: Optional[Dict[str, int]] = None,
                           name: Optional[str] = None
                           ) -> Tuple[bool, float, int]:
    """값이 카테고리 값 사전 단어로 채워져 있는가.

    returns (근거여부, 히트율, 서로 다른 매칭 단어 수)
      · 히트율 = 사전 단어와 맞는 값 / 정보 있는 값  ('기타'·'없음'·결측 등 제외)
      · 우연 일치를 막으려고 서로 다른 단어가 min_distinct 개 이상 맞아야 근거로 본다
        (값 하나가 'white' 인 색상 컬럼 같은 경우 제외).
      · 영문: 단어 경계 일치 / 한글: 2글자 이상 부분 포함, 1글자는 값 전체 일치.
      · name 을 주고 컬럼명이 이 카테고리를 (약하게라도) 가리키면 다의어 값
        (Left/Right, positive/negative, W/B/A …)도 센다 — pol_lean, respondent_race 류.
      · 값 형식 정규식(VALUE_PATTERNS: HGVS 변이 표기 등)과 맞는 값은 값마다 별개로 센다.
    """
    with_ctx = False
    if name:
        from .config import name_evidence
        with_ctx = name_evidence(name, category) is not None
    m = _lexicon_matcher(category, with_context=with_ctx)
    if m is None:
        return False, float("nan"), 0
    pat, t_ko, regexes = m
    counts = _counts if _counts is not None else lexicon_value_counts(values)
    total = sum(counts.values())
    if total == 0:
        return False, 0.0, 0
    hit_n, matched = 0, set()
    for nv, k in counts.items():             # 고유값 단위로 한 번만 검사
        term = None
        if pat is not None:
            mm = pat.search(nv)
            term = mm.group(1) if mm else None
        if term is None:
            for t in t_ko:
                if (len(t) >= 2 and t in nv) or nv == t:
                    term = t
                    break
        if term is None:
            for rx in regexes:
                if rx.search(nv):
                    term = "pattern:" + nv          # 형식 일치는 값마다 다른 근거로 셈
                    break
        if term:
            hit_n += k
            matched.add(term)
    rate = hit_n / total
    return (rate >= min_rate and len(matched) >= min_distinct), rate, len(matched)


# ---------------------------------------------------------------------------
# L3. 구조 가드
# ---------------------------------------------------------------------------
#  카테고리별로 "이럴 리 없다"는 구조 조건
STRUCTURE_GUARDS: Dict[str, Dict[str, float]] = {
    # 자유서술/고유값 성격이라 유니크 값이 극소수면 의심
    "GENETIC":       {"min_unique": 3},
    "BIOMETRIC":     {"min_unique": 10, "min_mean_len": 8},
    "COMMUNICATION": {"min_unique": 5},
    "DEVICE_ONLINE": {"min_unique": 10, "min_mean_len": 8},
    "LOCATION":      {"min_unique": 3},
    "SEX_LIFE":      {"min_unique": 2},
    "CRIMINAL_RECORD": {"min_unique": 2},
    # ── 코사인 오탐 보강 ──
    #   값 형식(정규식)이 없어 L2 값검증을 통과 처리하는 카테고리들.
    #   코사인이 이름만으로 오탐하면 걸러낼 축이 없으므로 최소한의
    #   카디널리티 가드를 둔다. 법정은 위반 시에도 강등이 아니라 WEAK+검수.
    "IDEOLOGY_BELIEF":   {"min_unique": 2},
    "POLITICAL_OPINION": {"min_unique": 2},
    "UNION_PARTY":       {"min_unique": 2},
    "RACE_ETHNICITY":    {"min_unique": 2},
    "HEALTH":            {"min_unique": 2},
}


def structure_ok(category: str, n_unique: int, mean_len: float
                 ) -> Tuple[bool, str]:
    g = STRUCTURE_GUARDS.get(category)
    if not g:
        return True, ""
    if "min_unique" in g and n_unique < g["min_unique"]:
        return False, f"고유값 {n_unique} < {int(g['min_unique'])}"
    if "min_mean_len" in g and mean_len < g["min_mean_len"]:
        return False, f"평균 길이 {mean_len:.1f} < {g['min_mean_len']}"
    return True, ""


# ---------------------------------------------------------------------------
# L4. 증거 등급
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    category: str
    tier: str                 # CONFIRMED | PROBABLE | WEAK | NONE
    auto_confirm: bool        # True면 자동 확정, False면 검수 큐
    reasons: List[str] = field(default_factory=list)
    demoted_from: Optional[str] = None


def adjudicate(category: str,
               name_hit: bool,
               value_ok: bool,
               value_rate: float,
               rf_hit: bool,
               sbert_hit: bool,
               n_unique: int,
               mean_len: float,
               exclusion: Optional[str] = None,
               rules: Optional[ExclusionRules] = None,
               strong_rule: bool = False) -> Verdict:
    """5층 방어를 순서대로 적용해 최종 판정.

    strong_rule : 0단계 강한 칼럼명 매핑 앵커 또는 2단계 값 정규식처럼
        결정적 규칙으로 확정된 단계인지. 이 경우 값 검증기가 없는
        카테고리(HEALTH/정치/종교 등)라도 CONFIRMED로 확정한다.
        단, 값 검증기가 있는데 값이 형식과 어긋나면 확정하지 않고 검수로 보낸다.
        (법정 민감정보 미탐 방지의 핵심 경로)
    """
    reasons: List[str] = []

    # L1
    if exclusion:
        if category in LEGAL_CODES and not (rules and rules.override_legal):
            reasons.append(f"제외규칙({exclusion}) 적중했으나 법정 카테고리라 "
                           f"강등 대신 검수로 보냄")
            return Verdict(category, WEAK, False, reasons)
        return Verdict(NEGATIVE_CODE, NONE, True,
                       [f"제외규칙 적중: {exclusion}"], demoted_from=category)

    if category == NEGATIVE_CODE:
        return Verdict(NEGATIVE_CODE, NONE, True, ["모델이 비민감으로 판정"])

    # L3
    ok, why = structure_ok(category, n_unique, mean_len)
    if not ok:
        reasons.append(f"구조 가드 위반({why})")
        if category in LEGAL_CODES:
            return Verdict(category, WEAK, False, reasons)
        return Verdict(NEGATIVE_CODE, NONE, True, reasons, demoted_from=category)

    # 강한 규칙 앵커(0단계 강한 매핑 / 2단계 값 정규식)는 결정적 신호이므로
    # 값 검증기가 없는 카테고리라도 CONFIRMED로 확정한다. 값 검증기가 있는데
    # 값이 형식과 어긋나는 경우(name-value 모순)만 확정 보류(검수)한다.
    if strong_rule:
        if not np.isnan(value_rate) and not value_ok:
            reasons.append("강한 앵커이나 이름-값 형식 불일치 → 자동 확정 보류(검수)")
            return Verdict(category, WEAK, False, reasons)
        reasons.append("강한 규칙 앵커(0단계 매핑/2단계 정규식) → 확정")
        return Verdict(category, CONFIRMED, True, reasons)

    # L2 + L4
    signals = sum([bool(name_hit), bool(value_ok and not np.isnan(value_rate)),
                   bool(rf_hit), bool(sbert_hit)])
    if name_hit:
        reasons.append("컬럼명 사전 일치")
    if not np.isnan(value_rate):
        reasons.append(f"값 검증 히트율 {value_rate:.2f}"
                       + ("" if value_ok else " (미달)"))
    if rf_hit:
        reasons.append("RF 예측 일치")
    if sbert_hit:
        reasons.append("SBERT 유사도 일치")

    if name_hit and value_ok and not np.isnan(value_rate):
        return Verdict(category, CONFIRMED, True, reasons)
    if not np.isnan(value_rate) and not value_ok:
        # 이름은 민감한데 값이 형식에 안 맞음 → 대표적 FP 패턴
        reasons.append("이름-값 불일치 → 자동 확정 보류")
        return Verdict(category, WEAK, False, reasons)
    if signals >= 2:
        auto = category not in LEGAL_CODES   # 법정은 2신호여도 검수 권장
        if not auto:
            reasons.append("법정 카테고리는 PROBABLE도 검수 대상")
        return Verdict(category, PROBABLE, auto, reasons)
    return Verdict(category, WEAK, False, reasons)


# ---------------------------------------------------------------------------
# L5. 피드백 루프
# ---------------------------------------------------------------------------
@dataclass
class ReviewFeedback:
    """검수 결과를 받아 제외 규칙과 재학습 데이터를 갱신한다."""
    rules: ExclusionRules = field(default_factory=ExclusionRules)
    corrections: Dict[str, str] = field(default_factory=dict)  # column -> 정답 코드

    def record(self, column: str, true_category: str) -> None:
        self.corrections[str(column)] = true_category
        if true_category == NEGATIVE_CODE:
            self.rules.add_false_positive(column)

    def to_training_pairs(self, profiles) -> Tuple[List[str], List[str]]:
        """검수 라벨이 붙은 컬럼을 재학습 코퍼스로 변환."""
        texts, labels = [], []
        for p in profiles:
            if p.name in self.corrections:
                texts.append(profile_to_text(p))
                labels.append(self.corrections[p.name])
        return texts, labels

    def fp_rate_report(self, decisions) -> Dict[str, float]:
        checked = [d for d in decisions if d.column in self.corrections]
        if not checked:
            return {}
        fp = sum(1 for d in checked
                 if d.is_sensitive and self.corrections[d.column] == NEGATIVE_CODE)
        wrong_cat = sum(1 for d in checked
                        if d.is_sensitive and
                        self.corrections[d.column] not in (NEGATIVE_CODE, d.category))
        return {
            "reviewed": len(checked),
            "false_positive": fp,
            "fp_rate": fp / len(checked),
            "wrong_category": wrong_cat,
        }



