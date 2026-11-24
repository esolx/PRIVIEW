# -*- coding: utf-8 -*-
"""범주형 처리 — PRAM, ClustEm4Ano 클러스터링, 사전 라벨링, 로컬억제, 스와핑, 부분마스킹, 도메인 컬럼 자동감지."""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

def apply_invariant_pram(series: pd.Series, p_keep: float = 0.90) -> tuple[pd.Series, str, dict]:
    """
    Invariant PRAM — 고유값 2개 이하인 저카디널리티 범주형 속성(모든 위험축)과,
    집계 위험축의 범주형 후보(카디널리티 무관)에 쓰인다. 수식은 범주 개수 k에
    상관없이 그대로 성립한다.
    원본 분포를 유지하면서 각 레코드 값을 확률적으로 교란한다.
    p_keep: 원값 유지 확률(기본 90%). theta=1-p_keep이 교란확률.
    다른 값으로 옮겨갈 확률은 그 값의 전체 분포 비율을 따르므로, 교란 후에도
    전체 분포는 원본과 통계적으로 동일하게 유지된다(Invariant).
    """
    counts = series.value_counts(normalize=True)
    categories = counts.index.tolist()
    pi = counts.values
    k = len(categories)
    theta = 1.0 - p_keep

    P = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                P[i][j] = 1.0 - theta * (1.0 - pi[i])
            else:
                P[i][j] = theta * pi[j]

    cat_index = {c: i for i, c in enumerate(categories)}
    result = []
    for val in series:
        if pd.isna(val):
            result.append(val)
            continue
        idx = cat_index[val]
        new_val = np.random.choice(categories, p=P[idx])
        result.append(new_val)

    note = f"Invariant PRAM (유지율 {p_keep*100:.0f}%, 교란확률 {theta*100:.0f}%)"
    extra_info = {"categories": categories, "transition_matrix": np.round(P, 3).tolist()}
    return pd.Series(result, index=series.index), note, extra_info


# ── 3-B. ClustEm4Ano 클러스터링 + 사전 라벨링 (고유값 3개 이상 범주형용) ──

def default_embedding_fn(values: list) -> np.ndarray:
    """
    ClustEm4Ano 임베딩 단계의 기본 구현 — 문자 n-gram TF-IDF로 근사.
    word2vec/BERT 등으로 바꾸려면 embedding_fn 파라미터에 주입하면 된다.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    return vec.fit_transform(values).toarray()


def cluster_categorical_clustem4ano(
    values: list,
    k_anonymity: int = 3,
    embedding_fn: Callable[[list], np.ndarray] = default_embedding_fn,
    target_n_clusters: Optional[int] = None,
) -> dict:
    """
    ClustEm4Ano 방식: 값 임베딩 → 유사 벡터 자동 클러스터링.

    target_n_clusters가 없으면(기본), 모든 클러스터의 레코드 수 합이 k_anonymity
    이상이 될 때까지 클러스터 개수를 줄여가며 재시도한다. k-익명성은 레코드 수
    기준이지 고유 카테고리 개수 기준이 아니다 — 고유값 개수만으로 체크하면
    카테고리가 몇 개 안 된다는 이유만으로 불필요하게 전부 하나로 뭉쳐버린다.

    target_n_clusters를 지정하면 k-익명성 체크 없이 그 개수로 곧장 클러스터링한다
    (직접 호출용 옵션 — 파이프라인 기본 경로는 k-익명성 기준만 사용).

    반환: {cluster_id: [원본값들]}
    """
    freq = Counter(str(v) for v in values)
    uniq = sorted(freq.keys())
    if len(uniq) <= 1:
        return {"c0": uniq}

    X = default_embedding_fn(uniq) if embedding_fn is None else embedding_fn(uniq)

    if target_n_clusters is not None:
        n_target = max(1, min(target_n_clusters, len(uniq)))
        if n_target == len(uniq):
            labels = np.arange(n_target)
        else:
            model = AgglomerativeClustering(n_clusters=n_target, metric="euclidean", linkage="average")
            labels = model.fit_predict(X)
    else:
        labels = np.arange(len(uniq))
        for n_target in range(len(uniq), 0, -1):
            if n_target == len(uniq) and n_target > 1:
                labels = np.arange(n_target)
            else:
                model = AgglomerativeClustering(n_clusters=n_target, metric="euclidean", linkage="average")
                labels = model.fit_predict(X)
            # 레코드 수(고유값의 실제 등장 빈도 합) 기준 k-익명성 체크
            record_counts: dict = {}
            for val, lab in zip(uniq, labels):
                record_counts[lab] = record_counts.get(lab, 0) + freq[val]
            if min(record_counts.values()) >= k_anonymity or n_target == 1:
                break

    clusters: dict = {}
    for val, lab in zip(uniq, labels):
        clusters.setdefault(f"c{lab}", []).append(val)
    return clusters


def generalize_categorical_series(
    series: pd.Series,
    domain_mapping: dict,
    k_anonymity: int = 3,
    purity_threshold: float = 0.6,
    embedding_fn: Callable[[list], np.ndarray] = default_embedding_fn,
    target_n_clusters: Optional[int] = None,
    dict_first: bool = False,
) -> tuple[pd.Series, dict]:
    """
    dict_first=False(기본): ClustEm4Ano로 전체 값을 클러스터링 후, 클러스터를
        '/'로 join해서 label_cluster()로 사전 매핑해 단일 라벨을 만든다.

    dict_first=True (질병 컬럼 전용): 사전에 이미 있는 값은 클러스터링 없이 곧장
        해당 대분류로 매핑하고, 사전에 없는 값들끼리만 클러스터링한다. 그러지 않으면
        레코드 수가 적은 희귀값이 텍스트 유사도만으로 무관한 값과 묶일 수 있다.
    """
    raw = series.astype(str)

    if dict_first:
        uniq_vals = raw.unique().tolist()
        known_vals = [v for v in uniq_vals if v in domain_mapping]
        unknown_vals = [v for v in uniq_vals if v not in domain_mapping]

        value_to_label: dict = {}
        cluster_info: dict = {}

        # 사전에 있는 값 -> 클러스터링 없이 곧장 매핑
        for v in known_vals:
            label = domain_mapping[v]
            value_to_label[v] = label
            cluster_info[f"dict:{v}"] = {
                "label": label, "purity": 1.0, "n_members": 1, "n_mapped": 1,
                "category_counts": {label: 1}, "unmapped": [], "status": "사전직접매핑",
            }

        # 사전에 없는 값들만 ClustEm4Ano로 클러스터링 (레코드 수 기준 k-익명성)
        if unknown_vals:
            unknown_raw = raw[raw.isin(unknown_vals)].tolist()  # 빈도 유지를 위해 반복값 그대로 전달
            clusters = cluster_categorical_clustem4ano(
                unknown_raw, k_anonymity=k_anonymity, embedding_fn=embedding_fn,
                target_n_clusters=target_n_clusters,
            )
            for cid, members in clusters.items():
                cluster_value = "/".join(members)
                result = label_cluster(cluster_value, domain_mapping, purity_threshold=purity_threshold)
                for m in members:
                    value_to_label[m] = result["label"]
                cluster_info[cid] = result

        generalized = raw.map(value_to_label)
        return generalized, cluster_info

    clusters = cluster_categorical_clustem4ano(
        raw.tolist(), k_anonymity=k_anonymity, embedding_fn=embedding_fn,
        target_n_clusters=target_n_clusters,
    )

    value_to_label = {}
    cluster_info = {}
    for cid, members in clusters.items():
        cluster_value = "/".join(members)
        result = label_cluster(cluster_value, domain_mapping, purity_threshold=purity_threshold)
        for m in members:
            value_to_label[m] = result["label"]
        cluster_info[cid] = result

    generalized = raw.map(value_to_label)
    return generalized, cluster_info

def label_cluster(cluster_value: str, mapping: dict, purity_threshold: float = 0.6, sep: str = "/") -> dict:
    """
    cluster_value: ClustEm4Ano 결과값. 예) "감기/목아픔/폐렴"
    반환: dict(label, purity, n_members, n_mapped, category_counts, status)
    """
    members = [m.strip() for m in cluster_value.split(sep) if m.strip()]
    n_members = len(members)

    categories = []
    unmapped = []
    for m in members:
        cat = mapping.get(m)
        if cat is None:
            unmapped.append(m)
        else:
            categories.append(cat)

    n_mapped = len(categories)

    if n_mapped == 0:
        return {
            "label": cluster_value,           # 매핑 정보가 전혀 없으면 원본 그대로 유지
            "purity": None,
            "n_members": n_members,
            "n_mapped": 0,
            "category_counts": {},
            "unmapped": unmapped,
            "status": "미매핑(사전에없는질병뿐)",
        }

    counts = Counter(categories)
    top_category, top_count = counts.most_common(1)[0]
    purity = top_count / n_mapped   # 매핑된 것들 중에서의 순도 (unmapped는 분모에서 제외)

    if purity >= purity_threshold:
        status = "단일범주" if purity == 1.0 else "다수결라벨"
        label = top_category
        if unmapped:
            status += "(일부미매핑항목포함)"
    else:
        status = "혼재클러스터"
        # 라벨은 원본 값 유지 (섣불리 하나로 뭉개지 않음) + 참고용 상위 2개 카테고리 표기
        top2 = ", ".join(f"{c}({n}/{n_mapped})" for c, n in counts.most_common(2))
        label = f"[혼재:{top2}] {cluster_value}"

    return {
        "label": label,
        "purity": round(purity, 3),
        "n_members": n_members,
        "n_mapped": n_mapped,
        "category_counts": dict(counts),
        "unmapped": unmapped,
        "status": status,
    }



def local_suppress_categorical(series: pd.Series, k_anonymity: int = 3) -> tuple[pd.Series, str, dict]:
    """로컬 억제 — 속성 전체가 아니라, 등가클래스 크기(해당 값의 빈도)가
    k_anonymity 미만인 희귀값 레코드의 셀만 '*'로 억제하고 나머지는 원값 유지.
    k_anonymity=1이면 빈도<1인 값이 없으므로 아무것도 억제되지 않는다(최약 상태)."""
    counts = series.value_counts()
    rare_values = set(counts[counts < k_anonymity].index)
    result = series.apply(lambda v: "*" if v in rare_values else v)
    n_suppressed = int(series.isin(rare_values).sum())
    note = f"로컬 억제 (등가클래스<{k_anonymity}인 {n_suppressed}/{len(series)}건만 억제)"
    return result, note, {"n_suppressed": n_suppressed, "rare_values": list(rare_values)}


def permute_values(series: pd.Series, swap_fraction: float = 0.3) -> tuple[pd.Series, str, float]:
    """스와핑/비상관화 — 레코드 일부의 값을 서로 무작위 순열(permutation) 교환해
    개별 레코드와 속성값의 연결(매칭·상관관계)만 끊고 전체 분포는 그대로 유지.
    수치형·범주형 어느 dtype이든 그대로 동작(값 자체는 안 건드리고 레코드 간
    재배치만 하므로). exposure_link(외부연결 차단)와 aggregation(비상관화) 양쪽
    모두 이 함수를 그대로 재사용한다."""
    n = len(series)
    n_swap = int(round(n * swap_fraction))
    if n_swap < 2:
        return series.copy(), f"스와핑 (교환 대상 {n_swap}건 미만 — 변경 없음)", swap_fraction
    idx = series.index.to_numpy()
    swap_idx = np.random.choice(idx, size=n_swap, replace=False)
    shuffled = np.random.permutation(swap_idx)
    result = series.copy()
    result.loc[swap_idx] = series.loc[shuffled].values
    note = f"스와핑 (레코드 {n_swap}/{n}건 값 순열 교환, 분포 보존)"
    return result, note, swap_fraction


def partial_mask(series: pd.Series, mask_ratio: float = 0.5, mask_token: str = "***") -> tuple[pd.Series, str, dict]:
    """부분마스킹 — 값의 뒷부분을 가려서 앞부분(상위 정보)만 남긴다.

    - 공백으로 나뉘는 값(주소 등, 토큰 2개 이상): 뒤쪽 토큰을 ceil(토큰수 × mask_ratio)개
      가린다. 예) "서울시 강남구 역삼동" → "서울시 강남구 ***"
    - 그 외(이름·코드 등): 뒤쪽 문자를 ceil(길이 × mask_ratio)개 가린다.
      예) "홍길동" → "홍***"
    어느 경우든 최소 1토큰/1글자는 남긴다(전부 가리면 부분마스킹이 아니라 억제).
    길이 1짜리 값은 부분적으로 가릴 수 없어 그대로 둔다.
    가린 부분은 길이와 무관하게 mask_token 하나로 대체한다 — 원래 길이를 남기면
    그 자체가 식별 단서가 되고, 서로 다른 값이 같은 마스킹 결과로 합쳐지지도 않는다.

    반환 info의 cluster_info는 "마스킹 결과가 같은 원본값들"을 하나의 일반화 그룹으로
    보고 만든 것이라, 일반화와 같은 NCP(taxonomy) 지표로 정보손실을 비교할 수 있다.
    """
    import math

    def _mask(v):
        if pd.isna(v):
            return v
        s = str(v)
        tokens = s.split()
        if len(tokens) >= 2:
            n_mask = min(len(tokens) - 1, max(1, math.ceil(len(tokens) * mask_ratio)))
            return " ".join(tokens[: len(tokens) - n_mask] + [mask_token])
        if len(s) <= 1:
            return s
        n_mask = min(len(s) - 1, max(1, math.ceil(len(s) * mask_ratio)))
        return s[: len(s) - n_mask] + mask_token

    uniq = pd.Series(series.dropna().unique())
    value_map = {v: _mask(v) for v in uniq}
    result = series.map(lambda v: value_map.get(v, v) if pd.notna(v) else v)

    groups: dict = {}
    for orig, masked in value_map.items():
        groups.setdefault(masked, []).append(str(orig))
    cluster_info = {
        f"mask:{m}": {"label": m, "n_members": len(members), "members": members}
        for m, members in groups.items()
    }
    n_before, n_after = len(value_map), len(groups)
    note = f"부분마스킹 (뒷부분 {mask_ratio*100:.0f}% 마스킹, 고유값 {n_before}→{n_after}개)"
    return result, note, {"cluster_info": cluster_info, "n_unique_before": n_before, "n_unique_after": n_after}


# 부분마스킹이 "일반화"로서 의미를 갖는지 판단하는 구조 감지.
# 앞부분이 상위 개념을 담고 있는 값(주소, 계층 코드)이어야 뒷부분을 가린 결과가
# 상위 분류와 일치한다. 자유 텍스트 범주(고혈압 → 고***)는 글자만 잘린 우연한
# 묶음이 되므로 준식별자 셀의 후보에서 제외한다.
_ADMIN_SUFFIXES = ("시", "도", "군", "구")


def detect_hierarchical_structure(
    series: pd.Series,
    region_hint: bool = False,
    min_share: float = 0.8,
) -> tuple[Optional[str], dict]:
    """값의 앞→뒤 계층 구조를 감지한다. 반환: ("address" | "code" | None, 판단근거).

    - address: 결측 제외 값의 min_share 이상이 공백 기준 2토큰 이상이고,
      첫 토큰이 행정구역 접미사(시/도/군/구)로 끝나는 비율이 min_share 이상
      (region_hint=True인 지역 컬럼은 접미사 조건 생략). 그리고 첫 토큰을
      공유하는 값들이 있어야 한다(첫 토큰 고유값 수 < 전체 고유값 수).
      → "세무 및 관세 사무원" 같은 여러 단어 직업명은 접미사 조건에서 걸러진다.
    - code: 값의 90% 이상이 공백 없는 영숫자(., - 허용)이고 숫자를 포함하며,
      길이 3 이상·길이 편차 2 이하, 앞 1자리~절반 중 공유되는 접두어가 있어야 한다.
      → KCD(I10.9), 우편번호, 법정동코드 등.
    """
    import math

    vals = pd.Series(series.dropna().astype(str).unique())
    info = {"n_unique": len(vals)}
    if len(vals) < 2:
        return None, {**info, "reason": "고유값 2개 미만"}

    # ── address ──
    tokens = vals.str.split()
    multi = tokens.str.len() >= 2
    multi_share = float(multi.mean())
    info["multi_token_share"] = round(multi_share, 3)
    if multi_share >= min_share:
        first = tokens[multi].str[0]
        admin_share = float(first.str.endswith(_ADMIN_SUFFIXES).mean())
        info["admin_suffix_share"] = round(admin_share, 3)
        prefix_shared = first.nunique() < multi.sum()
        if (region_hint or admin_share >= min_share) and prefix_shared:
            return "address", {**info, "reason": "공백 구분 행정구역 계층"}

    # ── code ──
    code_like = vals.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]*") & vals.str.contains(r"\d")
    code_share = float(code_like.mean())
    info["code_like_share"] = round(code_share, 3)
    if code_share >= 0.9:
        lens = vals[code_like].str.len()
        if lens.min() >= 3 and (lens.max() - lens.min()) <= 2:
            # 앞 1자리 ~ 앞 절반 중 어느 길이에서든 공유되는 접두어가 있으면 계층 코드로 본다
            # (KCD는 앞 1자리가 장(chapter), 우편번호, 법정동코드는 앞자리가 상위 행정구역)
            half = math.ceil(lens.median() / 2)
            codes = vals[code_like]
            prefix_shared = any(codes.str[:p].nunique() < len(codes) for p in range(1, half + 1))
            if prefix_shared:
                return "code", {**info, "reason": "고정 길이 계층 코드"}

    return None, {**info, "reason": "앞→뒤 계층 구조 없음(자유 텍스트 범주)"}


DISEASE_COLUMN_KEYWORDS = ["질병", "진단", "질환", "diagnosis", "disease"]
OCCUPATION_COLUMN_KEYWORDS = ["직업", "업종", "직종", "occupation", "job"]
REGION_COLUMN_KEYWORDS = ["지역", "행정구역", "거주지", "주소", "시군구", "읍면동", "시도", "region", "address", "city"]


def is_disease_column(name: str) -> bool:
    lname = name.lower()
    return any(kw in name or kw in lname for kw in DISEASE_COLUMN_KEYWORDS)


def is_occupation_column(name: str) -> bool:
    lname = name.lower()
    return any(kw in name or kw in lname for kw in OCCUPATION_COLUMN_KEYWORDS)


def is_region_column(name: str) -> bool:
    lname = name.lower()
    return any(kw in name or kw in lname for kw in REGION_COLUMN_KEYWORDS)

