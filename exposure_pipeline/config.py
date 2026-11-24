# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

"""config.py — 카테고리 정의 · 정규식 · 민감도 가중치 (단일 진실 공급원)"""

# ===  config.py

"""
민감정보 탐지 테이블 — 사용자 제공 스펙(민감정보_탐지_테이블.txt)을 코드로 옮긴 것.

이 파일이 파이프라인의 **단일 진실 공급원(single source of truth)** 이다.
칼럼명 매핑 · 값 정규표현식 · SBERT 대표문장 · 민감도 가중치가 모두 여기서 나온다.
스펙이 바뀌면 이 파일만 고치면 된다.

가중치 3종을 혼동하지 말 것
  ① RF class_weight        : 학습 손실의 클래스 불균형 보정  → tfidf_rf.py
  ② LEGAL_WEIGHT = 1.5     : 법정 민감정보 위험도 가중치 (일괄 고정)
                             → risk_weight.py
  ③ 비법정 민감도 5단계     : 비법정 속성 위험도 가중치 {0,.25,.5,.75,1}
                             → risk_weight.py
                             SV_adjusted = w1·SV_entropy + w2·SV_reference
                               SV_entropy   = 1 − H/H_max   (논문1)
                               SV_reference = 1 − score/10  (논문3 참조표)
                             를 민감도 방향(높을수록 위험)으로 5단계화.

설계 결정 기록
  · Category.preference(임의 하드코딩 프라이버시 선호값)는 제거했다. 논문1의
    SV_preference는 사용자 100명 설문 평균이라 재현 불가라서, 그 자리에 논문3
    (Schomakers 2019) 지각 민감도 참조표를 넣는다 → sensitivity_ref.py.
  · 테이블의 카테고리별 민감도 가중치(3/2/1.5)는 제거했다. 법정 민감정보는
    전부 LEGAL_WEIGHT = 1.5 를 쓴다. 카테고리 간 경중은 위험도 가중치가 아니라
    뒤이어 계산되는 EMD·Coverage·PMI가 반영한다.
  · 고유식별정보(주민등록번호·여권번호 등, 법 제24조)는 이 파이프라인의
    탐지 대상이 아니다. 탐지 대상은 법 제23조 민감정보뿐이므로 관련 카테고리·
    정규표현식·약어를 모두 제거했다.
  · 인종·민족의 근거를 테이블은 GDPR Art.9(1)로 적었으나, 국내법 근거는
    개인정보 보호법 시행령 제18조 제4호다. 둘 다 병기해 두었다.
"""

@dataclass(frozen=True)
class Category:
    code: str
    name_ko: str
    name_en: str
    legal: bool                       # True = 법정(Track A) / False = 비법정(Track B)
    basis: str                        # 근거 조문
    keywords_en: Tuple[str, ...] = ()
    keywords_ko: Tuple[str, ...] = ()
    prototypes: Tuple[str, ...] = ()  # SBERT 대표 문장 (한/영 혼합, 테이블 원문)
    regexes: Tuple[Tuple[str, str], ...] = ()   # (이름, 패턴) — 값 스캔용
    regex_needs_name: bool = False    # True면 정규식 단독 불가, 칼럼명 키워드와 AND
    rationale: str = ""               # 분류 근거 메모

    @property
    def keywords(self) -> Tuple[str, ...]:
        return self.keywords_ko + self.keywords_en

NEGATIVE_CODE = "NON_SENSITIVE"

# 법정 민감정보 (Track A) — 테이블 1~9 + 고유식별정보
LEGAL_CATEGORIES: List[Category] = [
    # 1 ---------------------------------------------------------------
    Category(
        code="HEALTH", name_ko="건강·의료 정보", name_en="Health / Medical",
        legal=True, basis="개인정보 보호법 제23조제1항 (건강)",

        rationale="GDPR·Harvard DSL·Frontiers 2022 모두 최고 등급. "
                  "Harvard DSL Level 4~5, HIPAA·GDPR 최고 수준 보호 요구",
        keywords_en=("disease", "diagnosis", "condition", "medical", "health", "bmi",
                     "weight_kg", "disability",
                     "illness", "symptom", "medication", "prescription",
                     "treatment", "hospital", "patient", "clinical",
                     "pathology", "disorder", "icd", "drug", "dose",
                     "chronic", "acute"),
        keywords_ko=("병력", "진단", "질병", "건강", "의료", "처방", "투약",
                     "증상", "치료", "병원", "환자", "임상", "병명", "복용",
                     "만성", "급성",
                     # 아래 2개는 테이블에 없던 것을 추가 (KCD 상병코드 대응)
                     "상병", "상병코드",
                     # ── 건강검진 동의서(대한산업보건협회 서식)에서 보강 ──
                     # 검진 자체/종합
                     "건강검진", "검진", "검진일자", "종합소견", "문진", "문진표",
                     "사후관리", "유해인자",
                     # 신체계측 (신장/키는 단독으로 신체정보라 HEALTH에서 제외)
                     "체중", "체질량지수", "비만도", "허리둘레", "시력",
                     # 혈액, 대사 지표
                     "공복혈당", "혈당", "콜레스테롤", "총콜레스테롤",
                     "hdl콜레스테롤", "ldl콜레스테롤", "트리글리세라이드",
                     "중성지방", "신사구체여과율", "요단백",
                     # 감염, 질환
                     "폐결핵", "결핵", "흉부질환", "b형간염", "간염",
                     # 병력/생활습관 문진
                     "가족력", "과거력", "흡연", "음주", "음주량", "운동여부",
                     "직무스트레스", "직무스트레스개인표",
                     "장애", "장애종류", "장애등급"),
        prototypes=(
            "patient's diagnosed medical condition",
            "health status and disease history of individual",
            "clinical diagnosis record",
            "prescription and medication data",
            "chronic illness information",
            "treatment history of a person",
            "병력 및 진단 정보",
            "환자의 건강 상태",
            "처방 및 투약 기록",
            "질병 이력 데이터",
        ),
        regexes=(
            ("icd10", r"^[A-Z]\d{2}(\.[\dA-Z]{1,4})?$"),
            ("kcd",   r"^[A-Z]\d{2}(?:\.[A-Z0-9]{1,2})?$"),
            ("atc",   r"^[A-Z]\d{2}[A-Z]{2}\d{2}$"),
        ),
    ),
    # 2 ---------------------------------------------------------------
    Category(
        code="GENETIC", name_ko="유전 정보", name_en="Genetic information",
        legal=True, basis="개인정보 보호법 시행령 제18조제1호 (유전정보)",

        rationale="본인뿐 아니라 가족 프라이버시에 영향. 노출 시 취소 불가. "
                  "Harvard DSL Level 4~5 별도 명시",
        keywords_en=("genetic", "genome", "dna", "rna", "snp", "allele",
                     "mutation", "chromosome", "genotype", "phenotype",
                     "sequencing"),
        keywords_ko=("유전", "유전자", "게놈", "염색체", "유전형", "표현형",
                     "염기서열", "돌연변이"),
        prototypes=(
            "genetic test result of individual",
            "DNA sequence information",
            "genome sequencing data",
            "SNP and mutation profile",
            "inherited genetic characteristic",
            "유전자 검사 결과",
            "DNA 염기서열 데이터",
            "유전적 특성 정보",
        ),
        regexes=(
            ("dbsnp",    r"^rs\d+$"),
            ("dna_seq",  r"^[ATCG]{10,}$"),
            ("hugo",     r"^[A-Z][A-Z0-9]{1,7}$"),   # 약한 패턴
        ),
        regex_needs_name=True,   # ← 테이블 지시: HUGO는 칼럼명 키워드와 AND 조건
    ),
    # 3 ---------------------------------------------------------------
    Category(
        code="BIOMETRIC", name_ko="생체 인식 정보", name_en="Biometric data",
        legal=True, basis="개인정보 보호법 시행령 제18조제3호 (신체적·생리적·행동적 특징)",

        rationale="비밀번호와 달리 변경 불가. 탈취 시 생애 전반 재식별 수단",
        keywords_en=("fingerprint", "iris", "facial", "biometric", "retina",
                     "voice", "vein", "palm", "gait", "keystroke"),
        keywords_ko=("지문", "홍채", "안면", "생체", "망막", "음성", "정맥",
                     "손바닥", "보행패턴"),
        prototypes=(
            "biometric data for person identification",
            "fingerprint or iris scan data",
            "facial recognition feature vector",
            "unique physical characteristic for authentication",
            "생체 인식 데이터",
            "지문 및 홍채 정보",
            "얼굴 인식 특징값",
            "신체적 특징 기반 식별 정보",
        ),
        regexes=(),   # 테이블: 해당 없음 → 키워드 + SBERT로만 탐지
    ),
    # 4 ---------------------------------------------------------------
    Category(
        code="SEX_LIFE", name_ko="성생활·성적 지향", name_en="Sex life / Orientation",
        legal=True, basis="개인정보 보호법 제23조제1항 (성생활)",

        rationale="노출 시 차별·폭력·법적 불이익. 일부 국가에서는 형사처벌 대상",
        keywords_en=("sexual", "orientation", "lgbtq", "sexuality",
                     "gender_identity", "gender identity", "intersex",
                     "queer", "homosexual", "heterosexual", "bisexual"),
        keywords_ko=("성적", "성지향", "성생활", "성정체성"),
        prototypes=(
            "sexual orientation of a person",
            "gender identity information",
            "data about person's sex life",
            "LGBTQ+ related attribute",
            "성적 지향 정보",
            "성별 정체성 데이터",
            "성생활 관련 개인정보",
        ),
        regexes=(),
    ),
    # 5 ---------------------------------------------------------------
    Category(
        code="RACE_ETHNICITY", name_ko="인종·민족", name_en="Race / Ethnicity",
        legal=True,
        basis="개인정보 보호법 시행령 제18조제4호 / GDPR Art.9(1)",

        rationale="차별 근거가 될 수 있으나 공중보건·통계 목적의 합법 처리가 있어 "
                  "Critical보다 한 단계 낮음. Harvard DSL Level 3~4",
        keywords_en=("race", "ethnicity", "ethnic", "racial", "ancestry",
                     "descent", "indigenous", "tribe"),
        keywords_ko=("인종", "민족", "혈통", "종족", "원주민"),
        prototypes=(
            "racial or ethnic background of individual",
            "ethnic origin data",
            "ancestry and descent information",
            "인종 및 민족 정보",
            "혈통 및 출신 민족 데이터",
            "race classification attribute",
            "ethnic group membership",
        ),
        regexes=(),
    ),
    # 6 ---------------------------------------------------------------
    Category(
        code="POLITICAL_OPINION", name_ko="정치적 견해", name_en="Political opinion",
        legal=True, basis="개인정보 보호법 제23조제1항 (정치적 견해)",

        keywords_en=("political", "party", "vote", "voting", "ideology",
                     "affiliation", "liberal", "conservative", "leftist"),
        keywords_ko=("정치", "정당", "투표", "정치적견해", "이념", "지지정당"),
        prototypes=(
            "political opinion or affiliation",
            "voting behavior and party membership",
            "political ideology of person",
            "정치적 견해 정보",
            "지지 정당 및 투표 성향",
            "political party affiliation data",
            "ideological stance attribute",
        ),
        regexes=(),
    ),
    # 7 ---------------------------------------------------------------
    Category(
        code="IDEOLOGY_BELIEF", name_ko="종교·사상·신념",
        name_en="Religion / Belief",
        legal=True, basis="개인정보 보호법 제23조제1항 (사상·신념)",

        keywords_en=("religion", "religious", "belief", "faith", "sect",
                     "church", "mosque", "temple", "buddhist", "christian",
                     "muslim", "atheist", "spiritual"),
        keywords_ko=("종교", "신앙", "사상", "신념", "교회", "사찰", "성당",
                     "불교", "기독교", "이슬람", "무신론"),
        prototypes=(
            "religious belief or faith of individual",
            "philosophical worldview data",
            "religious affiliation attribute",
            "사상 및 신념 정보",
            "종교 소속 및 신앙 데이터",
            "belief system of person",
            "religious practice information",
            "종교적 신념 관련 속성",
        ),
        regexes=(),
    ),
    # 8 ---------------------------------------------------------------
    Category(
        code="UNION_PARTY", name_ko="노동조합·정당 가입",
        name_en="Union / Party membership",
        legal=True, basis="개인정보 보호법 제23조제1항 (노동조합·정당의 가입·탈퇴)",

        rationale="GDPR Art.9 특별범주이나 Frontiers(2022)에서 실제 피해 사례가 "
                  "상대적으로 적은 것으로 분류. 정치적 견해보다 한 단계 낮음",
        keywords_en=("union", "labor_union", "labor union", "trade_union",
                     "trade union", "guild"),
        keywords_ko=("노동조합", "노조", "정당가입", "조합원", "조합"),
        prototypes=(
            "trade union affiliation status",
            "labor union affiliation",
            "노동조합 가입 여부",
            "정당 가입 및 탈퇴 정보",
            "labor union enrollment",
            "labor organization participation",
        ),
        regexes=(),
    ),
    # 9 ---------------------------------------------------------------
    Category(
        code="CRIMINAL_RECORD", name_ko="범죄 경력", name_en="Criminal record",
        legal=True,
        basis="개인정보 보호법 시행령 제18조제2호 "
              "(형의 실효 등에 관한 법률 제2조제5호) / GDPR Art.10",

        rationale="취업·대출·거주 등에서 직접적 차별로 연결. "
                  "형실효법 제2조제5호 가목 '벌금 이상의 형의 선고'에 형의 종류와 "
                  "그 양(형기·형량)이 내포되므로 형량·형기도 범죄경력자료에 포함된다 "
                  "→ 관련 키워드·정규식 보강.",
        keywords_en=("criminal", "conviction", "offense", "arrest", "crime",
                     "felony", "misdemeanor", "sentence", "probation",
                     "parole",
                     # 형량/형기 보강
                     "imprisonment", "incarceration", "prison_term", "jail",
                     "penal",
                     # ── 범죄경력 회보서에서 보강 ──
                     "criminal record", "record check", "disposition",
                     "prosecution", "indictment", "acquittal", "detention",
                     # ── 형사 카운트/약어형 컬럼 보강(priors_count, juv_fel_count,
                     #    juv_misd_count, two_year_recid, c_charge_degree 등) ──
                     "priors", "juvenile", "juvenile felony",
                     "juvenile misdemeanor", "recidivism", "recidivist",
                     "reoffend"),
        keywords_ko=("범죄", "전과", "범죄경력", "피의자", "수형", "형사",
                     "집행유예", "구속", "체포",
                     # 형량/형기 보강
                     "형량", "형기", "징역", "금고", "실형", "선고", "수감",
                     "복역", "벌금", "선고유예", "보호관찰",
                     # ── 범죄경력 회보서(경찰청 서식)에서 보강 ──
                     "범죄경력회보서", "범죄경력조회", "회보서", "범죄경력자료",
                     "수사경력", "수사경력자료", "조회결과", "조회목적",
                     "범죄사실", "죄명", "처분결과", "처분사항", "처분관서",
                     "선고일자", "자격정지", "자격상실", "과료", "구류",
                     "기소", "불기소", "무혐의", "형실효", "실효"),
        prototypes=(
            "criminal record or conviction history",
            "arrest and offense data",
            "criminal background information",
            "범죄 경력 데이터",
            "전과 기록 정보",
            "criminal justice involvement attribute",
            "법적 처분 이력",
            # 형량/형기 보강
            "sentencing and prison term information",
            "형량 및 선고 형기 정보",
        ),
        # 형량 값은 자유 텍스트지만 '징역/금고/벌금 + 수량'이라는 리터럴 앵커가 있어
        # 오탐이 거의 없다(HUGO 기호와 달리 단독으로도 강한 신호). regex_needs_name 불필요.
        regexes=(
            ("sentence_term", r"(?:징역|금고|자격정지|집행유예)\s?\d+\s?년(?:\s?\d+\s?개?월)?"),
            ("criminal_fine", r"벌금\s?[\d,]+\s?(?:원|만원)"),
        ),
    )
]

# 비법정 민감정보 (Track B) — ⚠️ 테이블에 없음. Track B 산출을 위해 별도 정의했다.
#     위험도 가중치는 엔트로피 민감도 5단계로 계산한다.
NON_LEGAL_CATEGORIES: List[Category] = [
    Category(code="INCOME_ASSET", name_ko="소득·자산", name_en="Income / Assets",
             legal=False, basis="-",
             keywords_en=("income", "salary", "wage", "earnings", "bonus",
                          "asset", "networth", "property", "deposit",
                          "balance", "taxable"),
             keywords_ko=("소득", "연봉", "급여", "임금", "상여", "자산",
                          "재산", "부동산", "예금", "잔액", "과세표준"),
             prototypes=("annual income and salary level",
                         "held assets and deposit balance",
                         "연간 소득과 급여 수준", "보유 자산 및 예금 잔액")),
    Category(code="CREDIT_FINANCE", name_ko="신용·금융거래", name_en="Credit / Finance",
             legal=False, basis="-",
             keywords_en=("credit score", "credit rating", "loan",
                          "delinquency", "debt", "creditcard", "account number",
                          "iban", "insurance", "bankruptcy"),
             keywords_ko=("신용등급", "신용점수", "대출", "연체", "채무",
                          "카드번호", "계좌번호", "보험", "파산", "상환", "선결제",
                          "청구금액", "대표결제방법", "대표결제일", "적립포인트",
                          "이용포인트", "잔여포인트", "포인트잔액", "포인트_포인트_건별",
                          "포인트_포인트_월적립", "마일리지", "혜택수혜금액", "할인금액", "할인건수"),
             prototypes=("credit score or credit rating",
                         "loan and delinquency history",
                         "신용등급 정보", "대출 및 연체 이력"),
             regexes=(("card", r"^(?:\d{4}[-\s]?){3}\d{4}$"),
                      ("account", r"^\d{2,6}-\d{2,6}-\d{2,8}$"))),
    Category(code="EDUCATION", name_ko="학력·교육", name_en="Education",
             legal=False, basis="-",
             keywords_en=("education", "school", "university", "college",
                          "major", "degree", "gpa", "graduation", "enrollment"),
             keywords_ko=("학력", "교육수준", "교육 수준", "최종학력", "학교", "학교급", "전공", "학위", "학점", "졸업", "졸업여부", "재학", "교육연수"),
             prototypes=("education level, school and major",
                         "academic transcript and gpa",
                         "최종 학력 및 전공", "학업 성적 이력")),
    Category(code="OCCUPATION", name_ko="직업·고용", name_en="Occupation",
             legal=False, basis="-",
             keywords_en=("occupation", "job title", "employer", "company",
                          "department", "position", "employment", "tenure",
                          "hire date", "termination"),
             keywords_ko=("직업", "직장", "근무처", "부서", "직급", "직위",
                          "고용형태", "근속", "재직", "퇴사", "종사상지위", "고용상지위",
                          "산업분류", "산업분류코드", "응답자 산업", "배우자 산업", "ISIC", "KSIC"),
             prototypes=("occupation, employer and department",
                         "employment type and tenure",
                         "직업 및 근무처 정보", "고용 형태와 근속 기간")),
    Category(code="LOCATION", name_ko="위치·이동", name_en="Location / Mobility",
             legal=False, basis="-",
             keywords_en=("address", "residence", "location", "latitude",
                          "longitude", "geolocation", "geohash", "zipcode",
                          "postal code", "trajectory", "checkin"),
             keywords_ko=("주소", "거주지", "거주지역", "지역구분", "권역별 지역구분",
                          "위치", "좌표", "위도", "경도", "행정동", "방문지", "이동경로", "동선"),
             prototypes=("residential address and district",
                         "location history as latitude and longitude",
                         "거주지 주소 정보", "위치 이력 데이터"),
             regexes=(("latlon", r"^[-+]?\d{1,3}\.\d{4,}$"),)),
    Category(code="COMMUNICATION", name_ko="통신·접속기록",
             name_en="Communication logs",
             legal=False, basis="-",
             keywords_en=("call log", "sms", "message", "sender", "recipient",
                          "access log", "login", "session id", "ip address",
                          "url", "referrer", "useragent", "search query"),
             keywords_ko=("통화기록", "문자", "메시지", "발신", "수신",
                          "접속기록", "로그인", "세션", "검색어"),
             prototypes=("call and message history",
                         "service access log and login history",
                         "통화 및 문자 기록", "접속 로그 이력"),
             regexes=(("ipv4", r"^(?:\d{1,3}\.){3}\d{1,3}$"),
                      ("url", r"^https?://"))),
    Category(code="CONSUMPTION", name_ko="소비·구매이력", name_en="Consumption",
             legal=False, basis="-",
             keywords_en=("purchase", "payment", "order", "cart", "sku",
                          "subscription", "refund", "coupon", "loyalty"),
             keywords_ko=("구매", "결제", "주문", "장바구니", "구독", "환불",
                          "쿠폰", "멤버십"),
             prototypes=("purchase history and payment record",
                         "ordered product and amount",
                         "구매 이력 및 결제 내역", "주문 상품과 금액")),
    Category(code="CHILDREN_COUNT", name_ko="자녀 수", name_en="Number of Children",
             legal=False, basis="Schomakers et al. (2019) - NUMBER OF CHILDREN",
             keywords_en=("number of children", "num children", "num_children",
                          "children count", "child count", "number_of_children",
                          "children_count", "num_kids"),
             keywords_ko=("자녀 수", "자녀수", "미혼자녀수"),
             prototypes=("number of children a person has",
                         "respondent number of children",
                         "응답자의 자녀 수", "개인이 가진 자녀의 수")),
    Category(code="DEVICE_ONLINE", name_ko="단말·온라인 식별자",
             name_en="Device / Online identifier",
             legal=False, basis="-",
             keywords_en=("cookie", "adid", "idfa", "gaid", "advertising id",
                          "device id", "mac address", "imei", "udid",
                          "push token", "fingerprint"),
             keywords_ko=("쿠키", "광고식별자", "디바이스", "단말기", "푸시토큰"),
             prototypes=("advertising and device identifier",
                         "cookie and device fingerprint",
                         "광고 식별자 및 단말 식별자", "쿠키 및 기기 지문"),
             regexes=(("mac", r"^(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}$"),
                      ("uuid", r"^[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$"))),
]

NEGATIVE_CATEGORY = Category(
    code=NEGATIVE_CODE, name_ko="비민감(일반)", name_en="Non-sensitive",
    legal=False, basis="-",
    keywords_en=("id", "seq", "sequence", "index", "rownum", "row number",
                 "record id", "uuid", "guid", "hash", "token",
                 "surrogate key", "pseudonym key", "created at", "updated at",
                 "timestamp", "datetime", "version", "status", "state",
                 "flag", "type", "code", "count", "quantity", "qty",
                 "amount", "price", "unit price", "total", "sum", "average",
                 "rate", "ratio", "score", "level", "priority", "sort order",
                 "note", "remark", "description", "age", "age group",
                 "agegroup", "gender", "sex", "region", "province"),
    keywords_ko=("일련번호", "순번", "레코드", "행번호", "구분코드", "분류코드",
                 "등록일자", "수정일자", "생성일시", "갱신일시", "버전",
                 "플래그", "상태", "상태코드", "처리결과", "비고", "메모",
                 "설명", "수량", "단가", "합계", "평균", "건수", "횟수",
                 "비율", "지수", "점수", "등급코드", "레벨", "우선순위",
                 "정렬순서", "가명키", "대체키", "해시값", "토큰값", "시퀀스",
                 "연령", "연령대", "나이", "성별", "성별구분",
                 "시도", "권역", "지역구분"),
    prototypes=("record sequence number and row identifier",
                "system metadata such as created and updated timestamp",
                "generic numeric field like quantity, price and total",
                "surrogate key or hash token from pseudonymization",
                "generic demographic bucket such as age group and gender",
                "레코드 일련번호와 순번 등 관리용 항목",
                "등록 일자와 수정 일자 등 시스템 메타데이터",
                "수량, 단가, 합계 등 일반 수치 항목",
                "가명처리로 생성된 대체키와 해시 토큰 값",
                "연령대와 성별 등 일반적인 인구통계 구분 항목",
                # ── 코사인 오탐 보강: 흔한 비민감 컬럼 표현을 더 채워
                #    비민감 유사도를 끌어올려 상대 마진(sim_margin)을 정확하게 ──
                "internal management code or classification code value",
                "boolean flag or yes-no indicator column",
                "processing status and state code of a record",
                "sort order, priority and level number",
                "foreign key reference to another table",
                "free text note, remark and description field",
                "ratio, rate, percentage and index score value",
                "count of events or occurrences over a period",
                "batch id, job id and run identifier for a pipeline",
                "업무 처리용 내부 코드와 분류 코드 값",
                "예/아니오를 나타내는 플래그 또는 여부 컬럼",
                "레코드의 처리 상태와 상태 코드 항목",
                "정렬 순서, 우선순위, 등급 레벨 숫자",
                "다른 테이블을 참조하는 외래키 값",
                "자유 서술 비고, 메모, 설명 텍스트 항목",
                "비율, 백분율, 지수, 점수 같은 계산된 수치",
                "기간 동안의 이벤트 발생 건수와 횟수",
                "배치 작업 식별자와 실행 회차 번호"),
)

CATEGORIES: Dict[str, Category] = {
    c.code: c for c in (LEGAL_CATEGORIES + NON_LEGAL_CATEGORIES + [NEGATIVE_CATEGORY])
}
LEGAL_CODES = frozenset(c.code for c in LEGAL_CATEGORIES)
NON_LEGAL_CODES = frozenset(c.code for c in NON_LEGAL_CATEGORIES)
SENSITIVE_CODES = LEGAL_CODES | NON_LEGAL_CODES

# ② 법정 민감정보 위험도 가중치 — 카테고리 구분 없이 일괄 적용
LEGAL_WEIGHT = 1.5

def is_legal(code: str) -> bool:
    return code in LEGAL_CODES

def category_weight(code: str) -> float:
    """② 법정 민감정보 위험도 가중치. 법정이면 1.5, 아니면 0.

    비법정(Track B)은 이 함수를 쓰지 않는다. risk_weight.grade5()로
    엔트로피 민감도 5단계 {0,.25,.5,.75,1}를 계산해 쓴다.
    """
    c = CATEGORIES.get(code)
    return LEGAL_WEIGHT if (c is not None and c.legal) else 0.0

def keyword_index(lang: str = "both", include_negative: bool = False
                  ) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for c in CATEGORIES.values():
        if c.code == NEGATIVE_CODE and not include_negative:
            continue
        kws = ((c.keywords_ko if lang in ("ko", "both") else ()) +
               (c.keywords_en if lang in ("en", "both") else ()))
        for kw in kws:
            idx.setdefault(kw.lower(), c.code)
    return idx

def exclusion_terms(lang: str = "both") -> Tuple[str, ...]:
    c = NEGATIVE_CATEGORY
    return ((c.keywords_ko if lang in ("ko", "both") else ()) +
            (c.keywords_en if lang in ("en", "both") else ()))

# ---------------------------------------------------------------------------
# 카테고리별 값 정규표현식 (테이블 원문)  — 값 스캔 전용
# ---------------------------------------------------------------------------
CATEGORY_REGEX: Dict[str, Tuple[Tuple[str, re.Pattern], ...]] = {
    c.code: tuple((name, re.compile(pat)) for name, pat in c.regexes)
    for c in CATEGORIES.values() if c.regexes
}
REGEX_NEEDS_NAME = frozenset(c.code for c in CATEGORIES.values()
                             if c.regex_needs_name)

# ---------------------------------------------------------------------------
# 일반 식별자 패턴 — 구조적 피처용 (카테고리 라벨 아님)
# ---------------------------------------------------------------------------
# ※ 주민등록번호, 여권번호 등 고유식별정보(법 제24조) 패턴은 탐지 대상이
#    아니므로 의도적으로 제외했다.
REGEX_PATTERNS: Dict[str, re.Pattern] = {
    "kr_phone":  re.compile(r"\b01[016-9][-\s]?\d{3,4}[-\s]?\d{4}\b"),
    "us_phone":  re.compile(r"\b(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b"),
    "email":     re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "card":      re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "account":   re.compile(r"\b\d{2,6}-\d{2,6}-\d{2,8}\b"),
    "ipv4":      re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "ipv6":      re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b"),
    "mac":       re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    "uuid":      re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"),
    "date":      re.compile(r"\b(19|20)\d{2}[-./]\d{1,2}[-./]\d{1,2}\b"),
    # (수정) 전체 셀 앵커. 기존 \b...\b + .search 는 브라우저 UA의 'X11',
    #   'X86' 같은 [A-Z]+2자리 토큰을 ICD 상병코드로 오인해 HEALTH 값히트를
    #   부풀렸다(예: ecommerce 'Browser Info' 히트율 0.25). 실제 상병코드
    #   컬럼은 셀 전체가 코드('E11.9','I10')이므로 앵커가 정탐을 해치지 않는다.
    "icd":       re.compile(r"^[A-TV-Z]\d{2}(?:\.\d{1,2})?$"),
    "url":       re.compile(r"https?://[^\s]+"),
    "latlon":    re.compile(r"[-+]?\d{1,3}\.\d{4,}"),
    "hex_blob":  re.compile(r"\b[0-9a-fA-F]{16,}\b"),
    "dna_seq":   re.compile(r"\b[ACGT]{10,}\b"),
    "dbsnp":     re.compile(r"\brs\d+\b"),
    # HUGO 유전자 기호 — 매우 헐거운 패턴(A01, USD, 부서코드까지 매칭)이라
    # 단독으로는 못 쓴다. 반드시 컬럼명 게이트(NAME_REQUIRED_VALIDATORS)와
    # AND로만 발동한다. 앵커 문자열이 없어 오탐 위험이 커서다.
    "hugo":      re.compile(r"^[A-Z][A-Z0-9]{1,7}$"),
    "pseudo_id": re.compile(r"\b[A-Za-z]{1,4}[-_]?[0-9A-Fa-f]{6,}\b"),
    # 범죄 경력 — 형량/형기(자유형 기간) 및 벌금액. 리터럴 앵커(징역/금고/벌금)로
    # 오탐이 거의 없어 값 단독 확정도 안전하다. → VALUE_VALIDATORS 에 연결.
    "sentence_term": re.compile(r"(?:징역|금고|자격정지|집행유예)\s?\d+\s?년(?:\s?\d+\s?개?월)?"),
    "criminal_fine": re.compile(r"벌금\s?[\d,]+\s?(?:원|만원)"),
    # 자유형 기간 or 벌금액 — 한 컬럼에 형종이 섞여도(예: 징역, 벌금 혼재) 잡히도록.
    "criminal_disposition": re.compile(
        r"(?:징역|금고|자격정지|집행유예)\s?\d+\s?년|벌금\s?[\d,]+\s?(?:원|만원)"),
}
REGEX_KEYS = tuple(REGEX_PATTERNS.keys())

# fp_control 호환용 — 카테고리별 값 검증기 키
VALUE_VALIDATORS: Dict[str, Tuple[str, ...]] = {
    "HEALTH":         ("icd",),
    # 기본은 강한 앵커(dna_seq, dbsnp)만. hugo 는 오탐 위험이 커서 기본 비활성.
    #   필요하면 enable_hugo_genetic(True) 로 켜면 이름 게이트가 걸린 채 추가된다.
    "GENETIC":        ("dna_seq", "dbsnp"),
    "LOCATION":       ("latlon",),
    "COMMUNICATION":  ("ipv4", "url", "kr_phone", "us_phone"),
    "DEVICE_ONLINE":  ("mac", "uuid", "hex_blob"),
    "CREDIT_FINANCE": ("card", "account"),
    "CRIMINAL_RECORD": ("sentence_term", "criminal_fine", "criminal_disposition"),
}

# 이름 게이트가 필요한 (헐거운) 값 검증기.
#   여기에 든 (카테고리, 정규식키)는 컬럼명이 그 카테고리를 지지할 때만
#   값 증거로 인정된다. Category.regex_needs_name 스펙을 실제로 배선한 것.
#   예: GENETIC 의 hugo 는 'gene', '유전자' 등 이름 신호가 있을 때만 발동 →
#       'dept_code'(A01,B02...)가 유전자로 오인되는 것을 막는다.
NAME_REQUIRED_VALIDATORS: Dict[str, Tuple[str, ...]] = {
    "GENETIC": ("hugo",),
}

# HUGO 유전자 기호 검증기 — 기본 OFF.
#   실질 기여가 좁고(강한 이름은 앞단이 먼저 잡음) 오탐 위험이 있어 옵션으로 둔다.
#   켜면 반드시 이름 게이트(NAME_REQUIRED_VALIDATORS)와 함께 동작한다.
_HUGO_ENABLED = False


def enable_hugo_genetic(on: bool = True) -> None:
    """GENETIC 값 검증에 HUGO 기호 패턴을 켜거나 끈다(기본 OFF).

    on=True 여도 hugo 는 NAME_REQUIRED_VALIDATORS 로 이름 게이트가 걸리므로,
    컬럼명이 유전자를 지지할 때만 발동한다(dept_code 등은 여전히 안전).
    """
    global _HUGO_ENABLED
    base = ("dna_seq", "dbsnp")
    VALUE_VALIDATORS["GENETIC"] = base + (("hugo",) if on else ())
    _HUGO_ENABLED = bool(on)


# ---------------------------------------------------------------------------
# 컬럼명 근거 — 단어 단위 매칭 + 다의어 키워드
#   예전 name_supports_category 는 키워드 '부분 문자열 포함'이라
#   trace_id, grace_period → race, reunion_date → union, healthy_snack → health 처럼
#   철자만 겹쳐도 법정 카테고리 근거가 됐다.
#   ,  영문 키워드 : 단어(토큰) 단위. 복수형(s/es)과 5자 이상 키워드의 접두 확장
#                  (priors→prior, diagnoses→diagnos…)만 허용. camelCase 는 분리.
#   ,  한글 키워드 : 2글자 이상 부분 포함. 한 글자 키워드는 단어 전체 일치만.
#   ,  다의어 키워드: 다른 뜻이 흔한 단어(party, union, 성적, 조합, condition …)는
#                  맥락 단어가 같이 있을 때만 '강한' 근거, 아니면 '약한' 근거.
# ---------------------------------------------------------------------------
import re as _re

# 다의어 키워드 → 이 단어가 컬럼명에 같이 있으면 민감 뜻으로 확정 (맥락 단어)
AMBIGUOUS_NAME_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    # 영문
    "party": ("political", "politics", "member", "membership", "affiliation", "support",
              "supported", "preferred", "preference", "vote", "voting", "yn"),
    "union": ("member", "membership", "labor", "labour", "trade", "worker", "workers",
              "join", "joined", "yn", "flag", "status"),
    "sect": ("religion", "religious", "faith", "belief"),
    "temple": ("religion", "religious", "faith", "attendance", "member"),
    "condition": ("medical", "health", "chronic", "preexisting", "existing", "disease", "patient"),
    "treatment": ("medical", "drug", "hospital", "cancer", "therapy", "patient", "clinical"),
    "orientation": ("sexual", "sex", "gender"),
    "sentence": ("prison", "jail", "criminal", "term", "length", "months", "years", "court"),
    "voice": ("print", "biometric", "sample", "recognition", "template"),
    "palm": ("print", "vein", "biometric", "template"),
    "drug": ("use", "abuse", "prescription", "name", "dose", "dosage", "medication", "test"),
    "dose": ("drug", "medication", "prescription", "mg"),
    "affiliation": ("political", "party", "religious", "union"),
    "vote": ("political", "party", "candidate", "election"),
    "disposition": ("criminal", "court", "case", "charge", "sentence"),
    "detention": ("criminal", "jail", "prison", "police"),
    "descent": ("ethnic", "ethnicity", "race", "racial", "national"),
    "mutation": ("gene", "genetic", "dna", "variant"),
    "sequencing": ("gene", "genetic", "dna", "rna", "genome"),
    "patient": ("diagnosis", "disease", "condition", "medical", "dx", "icd"),
    # 한글
    "성적": ("지향", "취향", "소수자", "정체성"),
    "조합": ("노동", "노조", "조합원", "가입"),
    "음성": ("인식", "지문", "샘플", "생체", "템플릿"),
    "사상": ("이념", "정치", "신념"),
    "금고": ("징역", "형", "선고", "수형"),
    "장애": ("등급", "종류", "유형", "인정", "판정", "장애인", "여부"),
    "실효": ("형", "범죄", "전과"),
}

_HANGUL_RE = _re.compile(r"[가-힣]")
# 접두 확장으로 인정할 어미/붙은 약어 (diagnoses, criminals, diagnosiscd …). healthy, generation ✗
_KW_SUFFIXES = frozenset({"al", "ic", "ed", "ing", "cd", "yn", "dx"})


def _name_tokens(name: str) -> Tuple[List[str], str]:
    """컬럼명 → (토큰 목록, 원문 소문자). camelCase 분리, 영숫자/한글 덩어리 단위."""
    raw = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name or "")).lower()
    return [t for t in _re.split(r"[^0-9a-z가-힣]+", raw) if t], raw


def _kw_in_name(kw: str, tokens: List[str], raw: str) -> bool:
    kw = kw.lower().strip()
    if not kw:
        return False
    if _HANGUL_RE.search(kw):
        if len(kw) >= 2:
            return kw in raw
        return kw in tokens                       # 한 글자 한글 키워드: 단어 전체 일치만
    k_tok = [t for t in _re.split(r"[^0-9a-z]+", kw) if t]
    m = len(k_tok)
    for i in range(len(tokens) - m + 1):
        win = tokens[i:i + m]
        if win[:-1] != k_tok[:-1]:
            continue
        last, k_last = win[-1], k_tok[-1]
        if last == k_last or last in (k_last + "s", k_last + "es"):
            return True
        if (len(k_last) >= 5 and last.startswith(k_last)
                and last[len(k_last):] in _KW_SUFFIXES):
            return True                           # 어미, 약어가 붙은 형태만 (healthy ✗)
    return False


def name_keyword_hits(name: str, code: str) -> List[str]:
    """컬럼명에 걸린 해당 카테고리 키워드 목록 (단어 단위).
    카테고리 기본 키워드 + 사전 파일의 이름 어휘(NAME_TERMS: 약어·동의어·한영)."""
    c = CATEGORIES.get(code)
    if c is None:
        return []
    tokens, raw = _name_tokens(name)
    vocab = tuple(c.keywords) + NAME_TERMS.get(code, ())
    return [kw for kw in vocab if _kw_in_name(kw, tokens, raw)]


# {단어: (반대 맥락 단어들)} — 같이 있으면 다른 뜻 (race_finish_time 의 race = 경주)
NEGATIVE_NAME_CONTEXT: Dict[str, Tuple[str, ...]] = {}


def _ambiguous_without_context(kw: str, tokens: List[str], raw: str) -> bool:
    neg = NEGATIVE_NAME_CONTEXT.get(kw.lower())
    if neg and any(_kw_in_name(cw, tokens, raw) for cw in neg):
        return True                                    # 반대 맥락 → 약한 근거
    ctx = AMBIGUOUS_NAME_KEYWORDS.get(kw.lower())
    if ctx is None:
        return False
    return not any(_kw_in_name(cw, tokens, raw) for cw in ctx)


def name_evidence(name: str, code: str) -> Optional[str]:
    """컬럼명이 카테고리를 얼마나 지지하나 → "strong" | "weak" | None.

    strong : 다의어가 아닌 키워드, 또는 맥락 단어와 함께 나온 다의어 키워드
    weak   : 맥락 없이 나온 다의어 키워드만 걸림 (party_size, 성적, union_station …)
    """
    hits = name_keyword_hits(name, code)
    if not hits:
        return None
    tokens, raw = _name_tokens(name)
    if any(not _ambiguous_without_context(kw, tokens, raw) for kw in hits):
        return "strong"
    return "weak"


def is_ambiguous_term(name: str, term: str) -> bool:
    """매핑이 잡은 용어가 맥락 없는 다의어인가 (0단계 강한 앵커 강등 판단용)."""
    tokens, raw = _name_tokens(name)
    return _ambiguous_without_context(str(term or ""), tokens, raw)


def name_supports_category(name: str, code: str) -> bool:
    """컬럼명이 해당 카테고리를 지지하는가 (약한 근거 포함, 이름 게이트용)."""
    return bool(name_keyword_hits(name, code))


# ---------------------------------------------------------------------------
# 법정 카테고리 값 사전 — 컬럼명을 몰라도 값으로 근거를 찾는다
#   형식(정규식)이 없는 카테고리(인종, 종교, 정치, 노조, 성적지향)와, 정규식은 있으나
#   자연어 값이 흔한 카테고리(질병명, 죄명)를 위한 어휘. 값이 이 사전 단어들로
#   채워져 있으면 '직접 근거'가 된다 (예: q17 = 기독교/불교/천주교 → 종교).
# ---------------------------------------------------------------------------
VALUE_LEXICONS: Dict[str, Tuple[str, ...]] = {
    "RACE_ETHNICITY": (
        "white", "black", "asian", "hispanic", "latino", "latina", "latinx", "caucasian",
        "african american", "native american", "american indian", "alaska native",
        "pacific islander", "native hawaiian", "multiracial", "biracial", "mixed race",
        "middle eastern", "arab", "백인", "흑인", "황인", "아시아계", "히스패닉", "라틴계",
        "아프리카계", "혼혈"),
    "IDEOLOGY_BELIEF": (
        "christian", "christianity", "catholic", "protestant", "orthodox", "buddhist",
        "buddhism", "muslim", "islam", "islamic", "hindu", "hinduism", "jewish", "judaism",
        "sikh", "atheist", "agnostic", "mormon", "기독교", "개신교", "천주교", "가톨릭", "불교",
        "원불교", "이슬람", "이슬람교", "힌두교", "유대교", "무교", "무신론", "천도교", "대종교"),
    "POLITICAL_OPINION": (
        "democrat", "democratic", "republican", "liberal", "conservative", "libertarian",
        "socialist", "progressive", "centrist", "green party", "labour party", "tory",
        "진보", "보수", "중도", "더불어민주당", "민주당", "국민의힘", "정의당", "국민의당",
        "조국혁신당", "개혁신당", "진보당", "녹색당", "무당층"),
    "UNION_PARTY": (
        "union member", "non member", "nonmember", "조합원", "비조합원", "노조원", "비노조원"),
    "SEX_LIFE": (
        "heterosexual", "homosexual", "bisexual", "gay", "lesbian", "asexual", "pansexual",
        "queer", "이성애", "동성애", "양성애", "무성애", "이성애자", "동성애자", "양성애자"),
    "HEALTH": (
        "diabetes", "hypertension", "asthma", "cancer", "depression", "anxiety", "dementia",
        "alzheimer", "stroke", "hiv", "aids", "hepatitis", "tuberculosis", "copd",
        "arthritis", "obesity", "schizophrenia", "bipolar", "epilepsy", "heart disease",
        "kidney disease", "당뇨", "당뇨병", "고혈압", "천식", "암", "우울증", "불안장애", "치매",
        "뇌졸중", "간염", "결핵", "관절염", "비만", "조현병", "조울증", "뇌전증", "심장질환",
        "신장질환", "고지혈증"),
    "CRIMINAL_RECORD": (
        "theft", "larceny", "burglary", "robbery", "assault", "fraud", "homicide", "murder",
        "manslaughter", "dui", "dwi", "drug possession", "trafficking", "vandalism", "arson",
        "felony", "misdemeanor", "절도", "폭행", "상해", "사기", "강도", "살인", "횡령",
        "배임", "음주운전", "마약", "성범죄", "강간", "방화", "협박", "공갈", "도박"),
}


# ---------------------------------------------------------------------------
# 사전 파일 로더 — 값 사전, 이름 어휘를 JSON 데이터로 관리
#   ,  패키지 기본: exposure_pipeline/lexicons/*.json
#   ,  사용자 표준 목록 / 검수 학습 결과: load_lexicon_dir(경로) 로 추가 병합
#     (runtime.build_detector 가 --lexicon-dir, learned/ 폴더를 자동으로 읽음)
#   형식: value_lexicons → {카테고리: {"terms":[], "context_terms":[], "patterns":[]}}
#         name_terms     → {카테고리: {"terms":[], "ambiguous": {단어: [맥락 단어]}}}
#   두 형식을 한 파일에 섞지 않는다. 키가 "_" 로 시작하면 주석으로 무시.
# ---------------------------------------------------------------------------
NAME_TERMS: Dict[str, Tuple[str, ...]] = {}
VALUE_CONTEXT_TERMS: Dict[str, Tuple[str, ...]] = {}
VALUE_PATTERNS: Dict[str, Tuple[str, ...]] = {}
LEXICON_VERSION = [0]          # 값이 바뀌면 증가 → fp_control 캐시 무효화
_LOADED_FILES: List[str] = []


def _merge(dst: Dict[str, Tuple[str, ...]], code: str, items) -> None:
    if not items:
        return
    cur = list(dst.get(code, ()))
    seen = {str(x).lower() for x in cur}
    for x in items:
        if str(x).lower() not in seen:
            cur.append(str(x))
            seen.add(str(x).lower())
    dst[code] = tuple(cur)


def load_lexicon_file(path: str) -> int:
    """JSON 사전 파일 하나를 읽어 전역 사전에 합친다. 합친 카테고리 수를 반환."""
    import json
    import os
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    base = os.path.basename(path).lower()
    # 파일 종류: 이름 규칙(value*/learned_value* ,  name*/learned_name*) 우선, 아니면 키로 추정
    if base.startswith(("value", "learned_value")):
        kind = "value"
    elif base.startswith(("name", "learned_name")):
        kind = "name"
    else:
        specs = [v for k, v in data.items() if isinstance(v, dict) and not k.startswith("_")]
        kind = ("name" if any("ambiguous" in v for v in specs) else "value")
    n = 0
    for code, spec in data.items():
        if code.startswith("_") or code not in CATEGORIES or not isinstance(spec, dict):
            continue
        if kind == "value":
            _merge(VALUE_LEXICONS, code, spec.get("terms", ()))
            _merge(VALUE_CONTEXT_TERMS, code, spec.get("context_terms", ()))
            _merge(VALUE_PATTERNS, code, spec.get("patterns", ()))
        else:
            _merge(NAME_TERMS, code, spec.get("terms", ()))
            _merge(NAME_TERMS, code, tuple((spec.get("ambiguous") or {}).keys()))   # 다의어도 어휘에 포함
            for kw, ctx in (spec.get("ambiguous") or {}).items():
                key = str(kw).lower()
                AMBIGUOUS_NAME_KEYWORDS[key] = tuple(
                    dict.fromkeys(tuple(AMBIGUOUS_NAME_KEYWORDS.get(key, ())) + tuple(ctx)))
            for kw, ctx in (spec.get("negative_context") or {}).items():
                key = str(kw).lower()
                NEGATIVE_NAME_CONTEXT[key] = tuple(
                    dict.fromkeys(tuple(NEGATIVE_NAME_CONTEXT.get(key, ())) + tuple(ctx)))
        n += 1
    LEXICON_VERSION[0] += 1
    _LOADED_FILES.append(path)
    return n


def load_lexicon_dir(directory: str) -> List[str]:
    """폴더 안의 *.json 사전을 모두 병합 (이미 읽은 파일은 건너뜀)."""
    import glob
    import os
    out = []
    if not directory or not os.path.isdir(directory):
        return out
    for fp in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.abspath(fp) in {os.path.abspath(x) for x in _LOADED_FILES}:
            continue
        load_lexicon_file(fp)
        out.append(fp)
    return out


def _load_default_lexicons() -> None:
    import os
    load_lexicon_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lexicons"))


VALUE_LEXICONS = {k: tuple(v) for k, v in VALUE_LEXICONS.items()}   # 병합 가능한 형태로
_load_default_lexicons()
