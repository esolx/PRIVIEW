# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

"""column_mapping.py — 0단계 칼럼명 매핑 테이블 (가장 강한 신호)"""

from .config import CATEGORIES, LEGAL_CODES, NEGATIVE_CODE

# ===  column_mapping.py

"""
[0단계] 칼럼명 매핑 테이블 — 다른 모든 단계보다 먼저 실행된다.

왜 0단계인가
  칼럼명은 스키마 설계자가 의미를 명시적으로 적어 넣은 유일한 지점이다.
  값은 가명처리로 뭉개져도 이름은 남는다. 신호 대 잡음비가 가장 높으므로
  ML·임베딩을 돌리기 전에 여기서 먼저 확정 지어야 한다.
  0단계에서 확정된 컬럼은 뒤 단계가 뒤집지 않는다(= 앵커 역할).

매칭 우선순위 (위에서 아래로, 먼저 걸리면 종료)
  1. override   : 사람이 직접 지정한 컬럼명 → 카테고리 (최우선, 무조건 신뢰)
  2. exact      : 정규화된 컬럼명이 키워드와 완전 일치      confidence 1.00
  3. token      : 정규화 후 토큰 단위 완전 일치            confidence 0.90
  4. prefix/suffix : 토큰이 키워드로 시작/끝남             confidence 0.75
  5. substring  : 부분 문자열 포함 (최장일치)              confidence 0.60
  6. none       : 매칭 실패 → 1단계(ML)로 넘김

정규화가 하는 일
  camelCase / snake_case / kebab-case / 한글+영문 혼용 / 접두·접미 노이즈
  (tbl_, col_, _cd, _yn, _id, _nm ...) 제거 → 비교 가능한 형태로

  patientDiagnosisCd → "patient diagnosis"
  TB_USR_DSES_NM     → "usr dses"      (약어는 abbreviations로 확장)
  환자_진단명         → "환자 진단"
"""


# 매칭 유형별 신뢰도
CONFIDENCE = {"override": 1.00, "exact": 1.00, "token": 0.90,
              "affix": 0.75, "substring": 0.60, "context_neg": 1.00, "none": 0.0}

# HEALTH는 일반 업무/검진 메타 컬럼에도 자주 등장하므로 이름 하나만으로
# 법정 민감정보를 자동 확정하면 FP가 커진다. 아래 일반어는 후보 신호로만 사용한다.
_WEAK_HEALTH_NAME_TERMS = {
    "health", "medical", "hospital", "patient", "clinical", "treatment",
    "건강", "의료", "병원", "환자", "임상", "검진", "건강검진", "문진",
    "사후관리", "유해인자", "흡연", "음주", "운동여부", "직무스트레스",
}

# 반대로 진단/질병/처방처럼 개인의 실제 건강상태, 진료내용을 직접 가리키는
# 용어는 접미사가 붙어도 강한 이름 근거로 인정한다.
_STRONG_HEALTH_NAME_TERMS = {
    "disease", "diagnosis", "illness", "symptom", "medication",
    "prescription", "pathology", "disorder", "icd", "chronic", "acute",
    "병력", "진단", "질병", "처방", "투약", "증상", "병명", "복용",
    "만성", "급성", "상병", "상병코드",
}


# '민감 속성'과 '민감 주제에 대한 질문'을 분리하는 0단계 문맥 가드.
# 예: BMI -> HEALTH, BMI에 대한 의견 -> NON_SENSITIVE.
_TOPIC_PHRASES = (
    "에 대한 의견", "에대한의견", "대한 인식", "대한인식", "대한 태도", "대한태도",
    "대한 평가", "대한평가", "에 대한 신뢰", "에대한신뢰", "찬반", "공정성",
    "격차에 대한", "차별에 대한", "정책에 대한", "제도에 대한", "서비스에 대한",
    "지도층에 대한", "성공 요인", "성공요인", "성공적인", "이미지",
    "주관적 평가", "주관적인 평가", "평가 정도", "평가정도", "문제점",
    "개선방안", "개선 방안", "미칠 영향", "미치는 영향", "영향에 대한", "효과에 대한",
    "혐오적 발언", "차별적 발언", "보호되어야 하는 정도", "보호 정도",
    "발생 빈도", "발생빈도",
    "이용 빈도", "이용빈도", "사용 빈도", "사용빈도", "제도 만족도", "제도 불만족",
    "opinion about", "opinion on", "views on", "attitude toward", "perception of",
    "trust in", "satisfaction with", "fairness of", "policy on", "thoughts on",
)
_EXPLICIT_SELF = (
    "본인", "응답자", "나의", "내 ", "내가", "자신의", "respondent", "my ",
    "self reported", "self-reported",
)
_META_PHRASES = (
    "법원의 범죄자", "기업에 벌금", "부당하게 대우", "대우 공정", "집단 간", "집단간",
    "종교단체", "종교계 지도층", "노동조합 지도층", "의료계 지도층",
)
_HARD_TOPIC_PHRASES = (
    "에 대한 의견", "에대한의견", "에 대한 인식", "에대한인식", "대한 인식", "대한인식",
    "에 대한 태도", "에대한태도", "대한 태도", "대한태도",
    "에 대한 평가", "에대한평가", "대한 평가", "대한평가", "주관적 평가", "주관적인 평가",
    "주관적 가구소득", "주관적 소득 수준", "주관적 소득수준",
    "에 대한 신뢰", "에대한신뢰", "신뢰도", "신뢰 수준", "신뢰수준",
    "만족도", "불만족도", "자긍심", "느끼는 정도", "느끼는정도",
    "생각하는 정도", "생각하는정도", "필요한 정도", "필요한정도", "필요 정도", "필요정도",
    "원인:", "원인：", "원인에 대한", "원인에대한",
    "성공 요인", "성공요인", "성공적인", "이미지", "공정성", "격차에 대한", "차별에 대한",
    "정책에 대한", "제도에 대한", "서비스에 대한", "기관에 대한", "지도층에 대한",
    "문제점", "개선방안", "개선 방안", "미칠 영향", "미치는 영향", "영향에 대한", "효과에 대한",
    "찬반", "찬성 여부", "찬성여부", "반대 여부", "반대여부", "동의 정도", "동의정도",
    "opinion about", "opinion on", "views on", "attitude toward", "perception of",
    "trust in", "fairness of", "satisfaction with", "thoughts on",
)
_DIRECT_POLITICAL = (
    "지지정당", "지지 정당", "정당지지", "정당 지지", "정치성향", "정치 성향",
    "정치적견해", "정치적 견해", "투표성향", "투표 성향", "투표의향", "투표 의향",
    "party affiliation", "party preference", "political ideology", "political leaning", "voting preference",
)

def _is_topic_only_context(column: str, category: str) -> bool:
    """키워드가 실제 개인 속성이 아니라 질문의 주제로만 등장하면 True.

    POLITICAL_OPINION은 '의견 자체'가 민감할 수 있으므로 일반 가드에서 제외한다.
    override는 이 함수를 거치지 않아 사람이 지정한 정답이 항상 우선한다.
    """
    if category == NEGATIVE_CODE:
        return False
    t = re.sub(r"\s+", " ", str(column).strip().lower())
    if category == "POLITICAL_OPINION" and any(x in t for x in _DIRECT_POLITICAL):
        return False
    # 의견/평가/인식/신뢰/성공요인/이미지/정책, 제도 평가는 개인의 직접 속성이 아니다.
    if any(x in t for x in _HARD_TOPIC_PHRASES):
        return True
    explicit_self = any(x in t for x in _EXPLICIT_SELF)
    if not explicit_self and any(x in t for x in _TOPIC_PHRASES):
        return True
    if not explicit_self and any(x in t for x in _META_PHRASES):
        return True
    # 적정/필요 생활비를 묻는 문항은 실제 개인의 소득, 자산 금액이 아니라 평가 문항.
    if category == "INCOME_ASSET" and any(x in t for x in ("필요한 최소 생활비", "필요 최소 생활비", "적정 생활비", "필요 생활비")) \
            and not any(x in t for x in ("실제 생활비", "생활비 지출액", "월 생활비", "월평균 생활비")):
        return True
    if category == "CRIMINAL_RECORD" and "피해" in t and not any(x in t for x in ("전과", "범죄경력", "가해", "기소", "체포", "수형")):
        return True
    if category == "LOCATION" and "방문" in t and any(x in t for x in ("사업", "서비스", "지원", "간병")):
        return True
    if category == "CREDIT_FINANCE" and any(x in t for x in ("산재보험", "고용보험")) and any(x in t for x in ("가입여부", "가입 여부")):
        return True
    return False


def _insurance_benefit_priority_label(column: str) -> Optional[str]:
    """보험·연금·복지급여 최우선 판정. None이면 이 도메인이 아님."""
    t = re.sub(r"\s+", " ", str(column).strip().lower())
    domain_terms = (
        "보험", "연금", "의료급여", "생계급여", "주거급여", "교육급여", "자활급여",
        "부모급여", "기초생활보장급여", "국민기초생활보장급여",
        "근로장려금", "자녀장려금", "근로(자녀)장려금",
        "재가급여", "시설급여", "특별현금급여", "장기요양급여",
    )
    if not any(x in t for x in domain_terms):
        return None

    # 보험료는 지출/납부 항목이므로 '금액/총액'이 붙어도 NON_SENSITIVE.
    if any(x in t for x in ("보험료", "보험 료")):
        return NEGATIVE_CODE

    admin_markers = (
        "가입여부", "가입 여부", "가입종류", "가입 종류", "가입형태", "가입 형태",
        "수급여부", "수급 여부", "수령여부", "수령 여부", "이용여부", "이용 여부",
        "신청여부", "신청 여부", "신청", "경험여부", "경험 여부", "경험",
        "급여종류", "급여 종류", "보험종류", "보험 종류", "종류", "형태",
        "체납여부", "체납 여부", "체납", "미납", "납부여부", "납부 여부",
        "수급개월", "수급 개월", "수급기간", "수급 기간", "받은 기간", "받은 년",
        "기간", "개월", "이용방법", "이용 방법", "해결방법", "해결 방법",
        "대상여부", "대상 여부", "가구원이 있는지", "받은 가구원이 있는지",
    )
    if any(x in t for x in admin_markers):
        return NEGATIVE_CODE

    direct_income_terms = (
        "보험금", "일시금", "해약금", "해약환급금", "환급금",
        "연금소득", "연금 소득", "연금액", "연금 액",
    )
    amount_markers = (
        "수령액", "수령 액", "지급액", "지급 액", "소득액", "소득 액",
        "연간총액", "연간 총액", "월총액", "월 총액", "총액", "금액",
    )
    if any(x in t for x in direct_income_terms):
        return "INCOME_ASSET"
    if any(x in t for x in amount_markers):
        return "INCOME_ASSET"
    return NEGATIVE_CODE


def _context_guard(column: str, category: str) -> str:
    return NEGATIVE_CODE if _is_topic_only_context(column, category) else category


# 영문 약어형 범죄기록 컬럼과 일반 메타데이터 날짜를 SBERT rescue 전에 확정한다.
# 목적: compas_screening_date 같은 단순 평가일자가 CREDIT_FINANCE 등으로 구조되는 FP를 막고,
#       priors_count / c_charge_degree 같은 명백한 범죄기록 약어는 0단계에서 바로 확정한다.
_STRONG_EN_CRIMINAL_COLUMNS = {
    "priors_count", "juv_fel_count", "juv_misd_count", "two_year_recid",
    "c_jail_in", "c_jail_out",
}

# 현재 기소 혐의의 등급은 '범죄 경력' 자체가 아니므로 강한 비민감 앵커로 둔다.
_STRONG_EN_NONSENSITIVE_COLUMNS = {
    "c_charge_degree",
}

_GENERIC_DATE_CONTEXT_TERMS = (
    "screening", "survey", "assessment", "evaluation", "record",
    "created", "updated", "extract", "load", "snapshot",
)
_SENSITIVE_EVENT_DATE_TERMS = (
    # 범죄 사건/처분 자체의 일자는 범죄경력에 직접 연결되므로 음성 가드에서 제외
    "arrest", "conviction", "charge", "offense", "crime", "jail", "prison",
    "incarceration", "sentence", "probation", "parole", "recid",
    # 건강 사건 자체의 일자는 건강/진료 이력에 직접 연결될 수 있으므로 제외
    "diagnosis", "disease", "treatment", "admission", "discharge",
)


def _english_strong_priority_label(column: str,
                                   dataset_specific: bool = True) -> Optional[str]:
    """새 영문 컬럼에서 의미가 명백한 약어/메타데이터를 0단계에서 확정.

    dataset_specific=False 면 특정 데이터셋(COMPAS) 컬럼명을 그대로 적은 exact
    규칙(_STRONG_EN_*_COLUMNS)은 건너뛰고, 일반 규칙(메타데이터 날짜)만 적용한다.
    ⚠ compas_800 은 평가(test) 데이터이므로, 이 exact 규칙이 켜진 상태의 채점
      결과는 '규칙이 정답을 외운' 값이다. 일반화 성능은 False 로 측정할 것.
    """
    raw = str(column).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")

    if dataset_specific:
        if compact in _STRONG_EN_NONSENSITIVE_COLUMNS:
            return NEGATIVE_CODE
        if compact in _STRONG_EN_CRIMINAL_COLUMNS:
            return "CRIMINAL_RECORD"

    # 단순 시스템/조사/평가 날짜는 민감 속성 자체가 아니다.
    # 다만 arrest_date, diagnosis_date처럼 사건/상태에 직접 연결되는 날짜는 여기서 막지 않는다.
    is_date_like = bool(re.search(r"(?:^|_)(?:date|dt|timestamp|time)(?:$|_)", compact)) \
        or compact.endswith(("_date", "_dt", "_timestamp"))
    generic_context = any(t in compact for t in _GENERIC_DATE_CONTEXT_TERMS)
    sensitive_event = any(t in compact for t in _SENSITIVE_EVENT_DATE_TERMS)
    if is_date_like and generic_context and not sensitive_event:
        return NEGATIVE_CODE

    return None

# 제거할 접두/접미 노이즈
_PREFIXES = ("tbl", "tb", "t", "col", "fld", "f", "dw", "stg", "src", "tmp")
_SUFFIXES = ("cd", "code", "id", "no", "nm", "name", "yn", "fg", "flag",
             "val", "value", "dt", "date", "ymd", "tm", "time", "seq",
             "cnt", "count", "amt", "qty", "key", "idx", "num", "txt", "desc")

# 흔한 DB 약어 확장 (필요에 맞게 추가)
DEFAULT_ABBREVIATIONS: Dict[str, str] = {
    "usr": "user", "cust": "customer", "emp": "employee", "pat": "patient",
    "dses": "disease", "diag": "diagnosis", "dgns": "diagnosis",
    "med": "medication", "presc": "prescription", "trt": "treatment",
    "hosp": "hospital", "clin": "clinical", "symp": "symptom",
    "gen": "genetic", "bio": "biometric", "fngr": "fingerprint",
    "relig": "religion", "polit": "political", "crim": "criminal",
    "juv": "juvenile", "fel": "felony", "misd": "misdemeanor",
    "recid": "recidivism",
    "eth": "ethnicity", "orient": "orientation", "membshp": "membership",
    "addr": "address", "loc": "location", "lat": "latitude",
    "lon": "longitude", "lng": "longitude", "acct": "account",
    "amt": "amount", "sal": "salary", "inc": "income", "edu": "education",
    "occ": "occupation",
}


@dataclass
class MappingHit:
    column: str
    normalized: str
    category: str
    match_type: str            # override | exact | token | affix | substring | none
    matched_term: str
    confidence: float
    legal: bool = False
    # (보강) 매칭된 어휘가 정규화 컬럼명 '전체'를 차지하는지.
    #   True면 이름 신호가 강함(예: '진단명'→'진단'), False면 부분 토큰
    #   (예: 'health_check_flag'→'health check' 중 'health'만 일치).
    #   부분 토큰 매칭은 0단계에서 확정하지 않고 뒷단계로 넘기는 데 쓴다.
    full_name_match: bool = True

    @property
    def hit(self) -> bool:
        return self.match_type != "none"

    @property
    def is_strong(self) -> bool:
        """0단계에서 바로 확정해도 되는 이름 앵커인지.

        핵심 규칙
        - override: 사람이 지정했으므로 항상 강함
        - exact: 원칙적으로 강함. 단, HEALTH의 일반 의료어는 후보로만 사용
        - token: 컬럼명 전체가 그 토큰일 때만 강함
          (health_check_flag 안의 health 같은 부분 토큰은 약함)
        - affix/substring: 항상 약함
        """
        if self.match_type in ("override", "context_neg"):
            return True
        term = str(self.matched_term).strip().lower()
        if self.category == "HEALTH":
            if term in _WEAK_HEALTH_NAME_TERMS:
                return False
            if term in _STRONG_HEALTH_NAME_TERMS:
                return True
        if self.match_type == "exact":
            return True
        if self.match_type == "token":
            return bool(self.full_name_match)
        return False


def normalize_column(name: str,
                     abbreviations: Optional[Dict[str, str]] = None,
                     strip_affix: bool = True) -> str:
    """컬럼명을 비교 가능한 정규형으로."""
    abbr = abbreviations if abbreviations is not None else DEFAULT_ABBREVIATIONS
    s = str(name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)      # camelCase
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)   # HTTPServer → HTTP Server
    s = re.sub(r"([가-힣])([A-Za-z])", r"\1 \2", s)     # 한글+영문
    s = re.sub(r"([A-Za-z])([가-힣])", r"\1 \2", s)
    s = re.sub(r"[_\-.\[\]()/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()

    toks = [t for t in s.split(" ") if t]
    if strip_affix:
        while toks and toks[0] in _PREFIXES:
            toks.pop(0)
        while toks and toks[-1] in _SUFFIXES:
            toks.pop()
        if not toks:                       # 전부 깎이면 원형 복구
            toks = [t for t in s.split(" ") if t]
    toks = [abbr.get(t, t) for t in toks]
    return " ".join(toks)


class ColumnMappingTable:
    """0단계 매핑 테이블.

    기본 사전은 config.CATEGORIES의 keywords에서 자동 생성된다.
    override / 사용자 사전은 CSV·JSON으로 관리하는 것을 권장한다.
    """

    def __init__(self,
                 lang: str = "both",
                 overrides: Optional[Dict[str, str]] = None,
                 extra_terms: Optional[Dict[str, str]] = None,
                 abbreviations: Optional[Dict[str, str]] = None,
                 include_negative: bool = True,
                 dataset_specific_rules: bool = True):
        # dataset_specific_rules: 특정 데이터셋 컬럼명을 하드코딩한 exact 규칙
        #   (membership_grade → CREDIT_FINANCE, COMPAS 약어 컬럼) 사용 여부.
        #   기본 True = 기존 동작. 평가 데이터 누수 없이 측정하려면 False.
        self.dataset_specific_rules = dataset_specific_rules
        self.abbr = dict(DEFAULT_ABBREVIATIONS)
        if abbreviations:
            self.abbr.update({k.lower(): v for k, v in abbreviations.items()})

        self.overrides: Dict[str, str] = {
            normalize_column(k, self.abbr): v
            for k, v in (overrides or {}).items()
        }
        self._raw_overrides: Dict[str, str] = dict(overrides or {})

        terms: Dict[str, str] = {}
        for c in CATEGORIES.values():
            if c.code == NEGATIVE_CODE and not include_negative:
                continue
            kws = ((c.keywords_ko if lang in ("ko", "both") else ()) +
                   (c.keywords_en if lang in ("en", "both") else ()))
            for kw in kws:
                key = normalize_column(kw, self.abbr, strip_affix=False)
                terms.setdefault(key, c.code)
                terms.setdefault(kw.lower(), c.code)
        if extra_terms:
            for k, v in extra_terms.items():
                terms[normalize_column(k, self.abbr, strip_affix=False)] = v
        self.terms = terms
        # 최장일치 순
        self._ordered = sorted(terms.items(), key=lambda kv: -len(kv[0]))

    # -- 조회 -------------------------------------------------------------
    def lookup(self, column: str) -> MappingHit:
        norm = normalize_column(column, self.abbr)
        raw = str(column).strip().lower()

        # 데이터셋 정의 우선 규칙: membership_grade는 조합/정당 가입 여부가 아니라
        # 개인의 신용등급(credit grade)을 뜻하므로 Track B CREDIT_FINANCE로 확정한다.
        # 일반 membership 토큰이 UNION_PARTY로 부분 매칭되는 오탐보다 먼저 적용한다.
        if self.dataset_specific_rules and (
                raw == "membership_grade" or norm == "membership grade"):
            return MappingHit(column, norm, "CREDIT_FINANCE", "exact", "membership_grade_credit",
                              CONFIDENCE["exact"], "CREDIT_FINANCE" in LEGAL_CODES, full_name_match=True)

        # 영문 강제 규칙도 일반 keyword/TF-IDF/SBERT보다 먼저 적용한다.
        _en = _english_strong_priority_label(column, self.dataset_specific_rules)
        if _en is not None:
            if _en == NEGATIVE_CODE:
                return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "english_metadata_date_priority",
                                  CONFIDENCE["context_neg"], False, full_name_match=True)
            return MappingHit(column, norm, _en, "exact", "english_criminal_priority",
                              CONFIDENCE["exact"], _en in LEGAL_CODES, full_name_match=True)

        # 보험/연금/복지급여의 최종 기준을 일반 exact/token/substring keyword보다 먼저 적용.
        _ib = _insurance_benefit_priority_label(column)
        if _ib is not None:
            if _ib == NEGATIVE_CODE:
                return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "insurance_benefit_priority",
                                  CONFIDENCE["context_neg"], False, full_name_match=True)
            return MappingHit(column, norm, _ib, "exact", "insurance_benefit_priority",
                              CONFIDENCE["exact"], _ib in LEGAL_CODES, full_name_match=True)

        # 최종 검수 기준: 보험/의료보장/복지, 지원 이용 정보는 질환, 진단 자체가 아니다.
        # 단, 민간보험/연금에서 실제 소득액을 기록한 경우는 INCOME_ASSET으로 유지한다.
        if any(x in raw for x in ("건강보험", "건강 보험", "의료보험", "의료 보험", "의료보장", "의료 보장", "의료급여", "의료 급여", "민간의료보험", "민간 의료보험", "민간보험", "민간 보험", "사적보험", "사적 보험", "국민연금보험료", "국민연금 보험료", "고용보험료", "고용 보험료")):
            if any(x in raw for x in ("소득액", "연금소득", "소득 합계", "소득합계")):
                return MappingHit(column, norm, "INCOME_ASSET", "exact", "insurance_income_amount",
                                  CONFIDENCE["exact"], "INCOME_ASSET" in LEGAL_CODES, full_name_match=True)
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "insurance_welfare_admin",
                              CONFIDENCE["context_neg"], False, full_name_match=True)
        # 복지/사회보험 급여의 신청, 수급, 경험, 종류는 행정/이용 정보. 실제 금액만 INCOME_ASSET.
        benefit_terms = ("국민기초생활보장급여", "기초생활보장급여", "근로장려금", "자녀장려금",
                         "근로(자녀)장려금", "부모급여", "생계급여", "주거급여", "교육급여",
                         "고용보험 급여", "고용보험급여", "산재보험 급여", "산재보험급여")
        if any(x in raw for x in benefit_terms):
            if any(x in raw for x in ("금액", "수령액", "지급액", "총액", "소득액", "연간총액", "연간 총액", "월액", "연금액")):
                return MappingHit(column, norm, "INCOME_ASSET", "exact", "benefit_income_amount",
                                  CONFIDENCE["exact"], "INCOME_ASSET" in LEGAL_CODES, full_name_match=True)
            if any(x in raw for x in ("신청", "수급", "수령 여부", "수령여부", "경험", "급여 종류", "급여종류", "종류", "대상 여부", "대상여부")):
                return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "benefit_admin_status",
                                  CONFIDENCE["context_neg"], False, full_name_match=True)

        if any(x in raw for x in ("장애인연금", "장애수당", "장애아동수당", "의료재활서비스")):
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "welfare_benefit_service",
                              CONFIDENCE["context_neg"], False, full_name_match=True)
        if "지원사업" in raw or ("지원" in raw and "경험여부" in raw) \
                or (("프로그램" in raw or "서비스" in raw) and "경험여부" in raw):
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "support_program_experience",
                              CONFIDENCE["context_neg"], False, full_name_match=True)
        if any(x in raw for x in ("직장체험", "직업훈련")) and any(x in raw for x in ("프로그램", "경험여부", "참여여부")):
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "employment_program_experience",
                              CONFIDENCE["context_neg"], False, full_name_match=True)
        if any(x in raw for x in ("의료비에 대한 가계 부담", "의료비 충당 방식", "의료비 충당방식")):
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "medical_cost_evaluation",
                              CONFIDENCE["context_neg"], False, full_name_match=True)
        if any(x in raw for x in ("보건의료비", "의료비 비율", "월평균 보건의료비")):
            return MappingHit(column, norm, "CONSUMPTION", "exact", "actual_medical_spending",
                              CONFIDENCE["exact"], "CONSUMPTION" in LEGAL_CODES, full_name_match=True)
        if any(x in raw for x in ("구매행동", "구매 행동", "윤리적 소비행동", "친환경 구매")):
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "purchase_attitude_behavior",
                              CONFIDENCE["context_neg"], False, full_name_match=True)
        if any(x in raw for x in ("정치 참여 경험", "정치참여 경험", "정치 참여경험", "정치참여경험",
                                     "투표 여부", "투표여부", "선거 투표 여부", "선거투표여부",
                                     "정치 뉴스 매체", "정치뉴스 매체", "주로 접하는 정치 뉴스", "주로 접하는 정치뉴스",
                                     "온라인 연락 빈도", "온라인 연락빈도", "정치에 대한 관심도", "정치 관심도", "정치관심도")):
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "political_behavior_media_contact",
                              CONFIDENCE["context_neg"], False, full_name_match=True)
        if any(x in raw for x in ("병원에 못간 경험", "병원에 못 간 경험", "병원에 가지 못한 경험",
                                   "치료 못한 경험", "치료를 못한 경험", "치료받지 못한 경험", "치료 받지 못한 경험",
                                   "의료 이용 못한 경험", "의료이용 못한 경험",
                                   "병원 방문 못하거나 중도 치료 포기 경험", "병원 방문 못하거나", "중도 치료 포기 경험")):
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "healthcare_access_failure_experience",
                              CONFIDENCE["context_neg"], False, full_name_match=True)
        if "이유" in raw:
            if any(x in raw for x in ("병원에 입원한 이유", "입원한 이유", "치료 포기 이유")):
                return MappingHit(column, norm, "HEALTH", "exact", "direct_health_reason",
                                  CONFIDENCE["exact"], "HEALTH" in LEGAL_CODES, full_name_match=True)
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "reason_question",
                              CONFIDENCE["context_neg"], False, full_name_match=True)

        # 개인의 실제 교육수준/학력/졸업 상태는 EDUCATION. 의견, 평가, 인식 문맥이면 NON_SENSITIVE.
        if any(x in raw for x in ("교육수준", "교육 수준", "최종학력", "최종 학력", "학력코드", "학력 코드",
                                  "졸업여부", "졸업 여부", "학교급", "교육연수", "교육 연수")):
            if _is_topic_only_context(column, "EDUCATION"):
                return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "education_topic_only",
                                  CONFIDENCE["context_neg"], False, full_name_match=True)
            return MappingHit(column, norm, "EDUCATION", "exact", "direct_education_level",
                              CONFIDENCE["exact"], "EDUCATION" in LEGAL_CODES, full_name_match=True)

        # 검증된 고정 규칙: 개인 산업분류는 OCCUPATION, 카드 포인트 활동은 CREDIT_FINANCE,
        # 주관적 가구소득 평가는 실제 소득액이 아니므로 NON_SENSITIVE.
        if any(x in raw for x in ("산업분류코드", "산업분류 코드", "산업분류", "isic_", "isic ",
                                  "ksic_", "ksic ", "응답자 산업", "배우자 산업", "10차산업분류")):
            return MappingHit(column, norm, "OCCUPATION", "exact", "industry_classification",
                              CONFIDENCE["exact"], "OCCUPATION" in LEGAL_CODES, full_name_match=True)
        if ("포인트_포인트_" in raw) or any(x in raw for x in ("포인트_이용포인트", "포인트_잔여포인트",
                                  "마일_이용포인트", "마일_잔여포인트", "혜택수혜금액",
                                  "할인건수")):
            return MappingHit(column, norm, "CREDIT_FINANCE", "exact", "card_point_activity",
                              CONFIDENCE["exact"], "CREDIT_FINANCE" in LEGAL_CODES, full_name_match=True)
        if any(x in raw for x in ("주관적 가구소득", "주관적 소득 수준", "주관적 소득수준")):
            return MappingHit(column, norm, NEGATIVE_CODE, "context_neg", "subjective_income_evaluation",
                              CONFIDENCE["context_neg"], False, full_name_match=True)

        # 1. override
        for key in (norm, raw):
            if key in self.overrides:
                cat = self.overrides[key]
                return MappingHit(column, norm, cat, "override", key,
                                  CONFIDENCE["override"], cat in LEGAL_CODES,
                                  full_name_match=True)

        # 2. exact
        for key in (norm, raw):
            if key in self.terms:
                raw_cat = self.terms[key]
                cat = _context_guard(column, raw_cat)
                if cat == NEGATIVE_CODE and raw_cat != NEGATIVE_CODE:
                    return MappingHit(column, norm, cat, "context_neg", key,
                                      CONFIDENCE["context_neg"], False, full_name_match=True)
                return MappingHit(column, norm, cat, "exact", key,
                                  CONFIDENCE["exact"], cat in LEGAL_CODES,
                                  full_name_match=True)

        toks = norm.split()
        # 3. token 완전 일치
        for t in toks:
            if t in self.terms:
                raw_cat = self.terms[t]
                cat = _context_guard(column, raw_cat)
                if cat == NEGATIVE_CODE and raw_cat != NEGATIVE_CODE:
                    return MappingHit(column, norm, cat, "context_neg", t,
                                      CONFIDENCE["context_neg"], False, full_name_match=True)
                # 매칭 토큰이 이름 전체(단일 토큰)를 덮으면 강함, 아니면 부분
                full = (len(toks) == 1)
                return MappingHit(column, norm, cat, "token", t,
                                  CONFIDENCE["token"], cat in LEGAL_CODES,
                                  full_name_match=full)

        # 4. 접두/접미 일치
        for term, cat in self._ordered:
            if len(term) < 3:
                continue
            for t in toks:
                if t.startswith(term) or t.endswith(term):
                    guarded = _context_guard(column, cat)
                    if guarded == NEGATIVE_CODE and cat != NEGATIVE_CODE:
                        return MappingHit(column, norm, guarded, "context_neg", term,
                                          CONFIDENCE["context_neg"], False, full_name_match=True)
                    return MappingHit(column, norm, guarded, "affix", term,
                                      CONFIDENCE["affix"], guarded in LEGAL_CODES,
                                      full_name_match=False)

        # 5. 부분 문자열 (최장일치)
        for term, cat in self._ordered:
            if len(term) < 2:
                continue
            if term in norm:
                # 부분 문자열이 이름 전체와 같으면(정규화 후 동일) 강함
                full = (term == norm)
                guarded = _context_guard(column, cat)
                if guarded == NEGATIVE_CODE and cat != NEGATIVE_CODE:
                    return MappingHit(column, norm, guarded, "context_neg", term,
                                      CONFIDENCE["context_neg"], False, full_name_match=True)
                return MappingHit(column, norm, guarded, "substring", term,
                                  CONFIDENCE["substring"], guarded in LEGAL_CODES,
                                  full_name_match=full)

        return MappingHit(column, norm, NEGATIVE_CODE, "none", "", 0.0, False,
                          full_name_match=False)

    def map_columns(self, columns: Iterable[str]) -> Dict[str, MappingHit]:
        return {str(c): self.lookup(c) for c in columns}

    # -- 큐레이션 ---------------------------------------------------------
    def add_override(self, column: str, category: str) -> None:
        if category not in CATEGORIES:
            raise ValueError(f"알 수 없는 카테고리: {category}")
        self._raw_overrides[column] = category
        self.overrides[normalize_column(column, self.abbr)] = category

    def to_csv(self, path: str, columns: Optional[Sequence[str]] = None) -> None:
        """검수용 CSV 내보내기. 사람이 category 열을 고쳐서 다시 넣으면 된다."""
        rows = []
        if columns:
            for c in columns:
                h = self.lookup(c)
                rows.append({"column": c, "normalized": h.normalized,
                             "category": h.category, "match_type": h.match_type,
                             "matched_term": h.matched_term,
                             "confidence": h.confidence,
                             "legal": h.legal, "reviewed": ""})
        else:
            for col, cat in self._raw_overrides.items():
                rows.append({"column": col, "normalized": normalize_column(col, self.abbr),
                             "category": cat, "match_type": "override",
                             "matched_term": "", "confidence": 1.0,
                             "legal": cat in LEGAL_CODES, "reviewed": "Y"})
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                               ["column", "normalized", "category", "match_type",
                                "matched_term", "confidence", "legal", "reviewed"])
            w.writeheader()
            w.writerows(rows)

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "ColumnMappingTable":
        """검수 완료 CSV를 override로 적재. reviewed 열이 비어있지 않은 행만."""
        ov: Dict[str, str] = {}
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if str(row.get("reviewed", "")).strip():
                    ov[row["column"]] = row["category"]
        return cls(overrides=ov, **kwargs)

    def save_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"overrides": self._raw_overrides, "abbreviations": self.abbr},
            ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str, **kwargs) -> "ColumnMappingTable":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(overrides=d.get("overrides"),
                   abbreviations=d.get("abbreviations"), **kwargs)

    # -- 진단 -------------------------------------------------------------
    def coverage(self, columns: Sequence[str]) -> Dict[str, object]:
        hits = self.map_columns(columns)
        by_type: Dict[str, int] = {}
        for h in hits.values():
            by_type[h.match_type] = by_type.get(h.match_type, 0) + 1
        matched = sum(v for k, v in by_type.items() if k != "none")
        return {
            "total": len(columns),
            "matched": matched,
            "coverage": matched / max(len(columns), 1),
            "by_match_type": by_type,
            "unmatched": [c for c, h in hits.items() if not h.hit],
        }



