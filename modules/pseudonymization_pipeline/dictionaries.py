# -*- coding: utf-8 -*-
"""도메인 사전 로더 — 질병(KCD-8), 직업(KSCO/ISCO-08), 지역(법정동)."""
from __future__ import annotations

import csv

import pandas as pd

# 0-A. 클러스터 사전 라벨링 로직 (원래 cluster_labeling.py, 이 파일 하나로 통합)
# =============================================================================
# 실제 데이터의 짧은 표기(콜론/괄호 없는 구어체) -> 189개 사전 공식표기 별칭
ALIASES = {
    "고혈압": "본태성(원발성) 고혈압",
    "당뇨": "제2형 당뇨병",
    "당뇨병": "제2형 당뇨병",
    "고지혈증": "고지혈증(이상지질혈증)",
    "역류성식도염": "위식도역류질환(GERD)",
    "감기": "감기(급성비인두염)",
    "목아픔": "급성인두염",
    "인후통": "급성인두염",
    "코로나19": "코로나바이러스감염증-19(COVID-19)",
    "코로나": "코로나바이러스감염증-19(COVID-19)",
    "독감": "인플루엔자(독감)",
    "골관절염": "골관절염(퇴행성관절염)",
    "허리디스크": "추간판탈출증(허리디스크)",
    "요통": "요통(허리통증)",
    "치질": "치질(치핵)",
    "뇌경색": "뇌졸중(뇌경색)",
}

# 189개 사전에 아직 없던 질병 보강 (KCD-8 3단위코드 기준)
EXTRA_DISEASES = {
    "변비": "소화계통의 질환",      # K59.0 - 소화계통(챕터XI)
    "설사": "소화계통의 질환",      # K52 비감염성위장염 - 소화계통(챕터XI)
    "장염": "소화계통의 질환",      # K52 - 소화계통(챕터XI)
}

def load_kcd8_mapping(path: str = "189개질환_KCD8_검증매핑.csv") -> dict:
    mapping: dict = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mapping[row["질병_한글명(원래표기)"]] = row["대분류_한글명"]

    # 별칭 등록: 짧은 표기가 정식 표기의 대분류를 그대로 가리키게 함
    for alias, official in ALIASES.items():
        if official in mapping:
            mapping[alias] = mapping[official]

    # 신규 질병 보강
    mapping.update(EXTRA_DISEASES)
    return mapping


def load_region_mapping_ko(path: str, target: str = "시도명") -> dict:
    """법정동 지역분류 매핑테이블(한글판)에서 상세값 → 목표 레벨 사전을 만든다.
    계층: 읍면동명 → 시군구명 → 시도명. target을 "시군구명"으로 주면 한 단계만
    올리고, "시도명"(기본값)이면 최상위까지 올린다.

    읍면동명/시군구명은 전국적으로 유일하지 않다(예: '신교동'이 서울과 경산시에
    둘 다 있음, '중구'가 5개 시도에 있음). 같은 이름이 서로 다른 target 값으로
    갈리는 경우는 이름만으로 어느 지역인지 알 수 없으므로 매핑에서 제외한다
    (그런 값은 사전 매칭 없이 일반 클러스터링으로 넘어감)."""
    df = pd.read_csv(path, dtype=str)
    mapping: dict = {}
    for level in ["읍면동명", "시군구명"]:
        unambiguous = df.groupby(level)[target].nunique()
        safe_names = set(unambiguous[unambiguous == 1].index)
        sub = df[df[level].isin(safe_names)].drop_duplicates(level)
        mapping.update(dict(zip(sub[level], sub[target])))
    return mapping


def load_region_mapping_en(path: str, target: str = "sido_en") -> dict:
    """법정동 지역분류 매핑테이블(영문판) 버전. target: eupmyeondong_en/sigungu_en/sido_en.
    한글판과 마찬가지로 이름이 겹치면서 target이 갈리는 값은 매핑에서 제외한다."""
    df = pd.read_csv(path, dtype=str)
    mapping: dict = {}
    for level in ["eupmyeondong_en", "sigungu_en"]:
        unambiguous = df.groupby(level)[target].nunique()
        safe_names = set(unambiguous[unambiguous == 1].index)
        sub = df[df[level].isin(safe_names)].drop_duplicates(level)
        mapping.update(dict(zip(sub[level], sub[target])))
    return mapping


def load_occupation_mapping_ko(path: str, target: str = "대분류명") -> dict:
    """직업분류 매핑테이블(한국표준직업분류, 한글판). 계층: 세세분류명 → 세분류명 →
    소분류명 → 중분류명 → 대분류명. target으로 목표 레벨을 지정(기본 최상위)."""
    df = pd.read_csv(path, dtype=str)
    mapping: dict = {}
    for _, row in df.iterrows():
        for col in ["세세분류명", "세분류명", "소분류명", "중분류명"]:
            mapping[row[col]] = row[target]
    return mapping


def load_occupation_mapping_en(path: str, target: str = "major_label") -> dict:
    """ISCO-08 직업분류 매핑테이블(영문판). 계층: unit_label → minor_label →
    sub_major_label → major_label. target으로 목표 레벨을 지정(기본 최상위)."""
    df = pd.read_csv(path, dtype=str)
    mapping: dict = {}
    for _, row in df.iterrows():
        for col in ["unit_label", "minor_label", "sub_major_label"]:
            mapping[row[col]] = row[target]
    return mapping


def load_combined_region_mapping(ko_path: str, en_path: str, ko_target: str = "시도명", en_target: str = "sido_en") -> dict:
    """한글판+영문판을 하나로 합친 지역 사전. 데이터가 한글/영문 어느 쪽이든 매핑됨."""
    mapping = load_region_mapping_ko(ko_path, target=ko_target)
    mapping.update(load_region_mapping_en(en_path, target=en_target))
    return mapping


def load_combined_occupation_mapping(ko_path: str, en_path: str, ko_target: str = "대분류명", en_target: str = "major_label") -> dict:
    """한글판(KSCO)+영문판(ISCO-08)을 하나로 합친 직업 사전. 두 표준은 값 자체가
    서로 다른 분류체계라 코드/행이 대응되진 않지만, 각자 자기 목표 레벨로
    올리는 사전을 그냥 합쳐서 데이터가 한글/영문 어느 쪽이든 매핑되게 한다."""
    mapping = load_occupation_mapping_ko(ko_path, target=ko_target)
    mapping.update(load_occupation_mapping_en(en_path, target=en_target))
    return mapping


