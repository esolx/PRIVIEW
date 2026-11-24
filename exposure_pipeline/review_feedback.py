# -*- coding: utf-8 -*-
"""
review_feedback.py — 검수 결과를 탐지 근거로 되먹이기 (재현율 누적 개선)
=====================================================================

검수자가 탐지 결과 CSV(run_integrated 의 {out}_detection.csv)에 ``reviewed_label``
열을 채우면, 그 판정을 다음 실행부터 '직접 근거'로 쓰도록 learned/ 폴더에 저장한다.

  learned/learned_name_overrides.csv   컬럼명 → 카테고리 (다음부터 0단계 강한 매핑)
  learned/learned_value_lexicons.json  법정 컬럼의 범주형 값 → 값 사전 (이름이 달라도 값으로 잡힘)
  learned/learned_labels.csv           labels_review.csv 형식 → 2_train_from_labels.py 재학습용

· 비민감(NON_SENSITIVE)으로 확정한 컬럼도 컬럼명 오버라이드로 저장 → 같은 오탐 재발 방지
· 값 사전에는 '정보 있는 범주형 텍스트 값'만 넣는다 (숫자·Y/N·고유값 50개 초과·'기타' 제외)
· 같은 컬럼을 다시 검수하면 최신 판정으로 덮어쓴다

⚠ 평가(벤치마크)를 돌릴 때는 learned/ 를 쓰지 말 것 — 평가 데이터의 검수 결과가
  섞이면 누수가 된다 (build_detector(learned_dir=None)).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from .config import CATEGORIES, LEGAL_CODES, NEGATIVE_CODE
from .features import profile_dataframe, profile_to_text
from .fp_control import _UNINFORMATIVE_VALUES, _norm_value

LEARNED_OVERRIDES = "learned_name_overrides.csv"
LEARNED_VALUES = "learned_value_lexicons.json"
LEARNED_LABELS = "learned_labels.csv"
MAX_LEXICON_VALUES = 50


def _informative_text_values(series: pd.Series) -> List[str]:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty or pd.to_numeric(vals, errors="coerce").notna().all():
        return []                                   # 숫자 코드는 값 사전으로 쓰지 않음
    uniq = list(dict.fromkeys(vals))
    if len(uniq) > MAX_LEXICON_VALUES:
        return []                                   # 자유 텍스트, 식별자형
    out = []
    for v in uniq:
        nv = _norm_value(v)
        if nv in _UNINFORMATIVE_VALUES or nv in {"y", "n", "yes", "no", "true", "false", "0", "1"}:
            continue
        if len(nv) < 2:
            continue
        out.append(v)
    return out


def load_overrides(learned_dir: str) -> Dict[str, str]:
    p = os.path.join(learned_dir or "", LEARNED_OVERRIDES)
    if not learned_dir or not os.path.exists(p):
        return {}
    df = pd.read_csv(p, dtype=str).fillna("")
    return {r.column: r.category for r in df.itertuples() if r.column and r.category in CATEGORIES}


def apply_review(detection_csv: str, data_csv: Optional[str], learned_dir: str = "learned",
                 label_col: str = "reviewed_label", source: Optional[str] = None) -> Dict[str, int]:
    """검수 완료 탐지 CSV → learned/ 갱신. 반환: 반영 건수 요약."""
    det = pd.read_csv(detection_csv, dtype=str).fillna("")
    if label_col not in det.columns:
        raise ValueError(f"{detection_csv} 에 '{label_col}' 열이 없습니다. 검수 판정을 이 열에 적어 주세요.")
    rev = det[det[label_col].str.strip() != ""].copy()
    bad = sorted(set(rev[label_col]) - set(CATEGORIES))
    if bad:
        raise ValueError(f"알 수 없는 카테고리 코드: {bad} (config.CATEGORIES 의 코드를 쓰세요)")
    os.makedirs(learned_dir, exist_ok=True)
    data = None
    if data_csv:
        from .runtime import read_csv_any
        data = read_csv_any(data_csv)
        data.columns = [str(c) for c in data.columns]
    src = source or (os.path.basename(data_csv) if data_csv else os.path.basename(detection_csv))

    # 1) 컬럼명 오버라이드
    p_ov = os.path.join(learned_dir, LEARNED_OVERRIDES)
    ov = load_overrides(learned_dir)
    for r in rev.itertuples():
        ov[getattr(r, "column")] = getattr(r, label_col)
    pd.DataFrame({"column": list(ov), "category": list(ov.values()), "reviewed": "1"}
                 ).to_csv(p_ov, index=False, encoding="utf-8-sig")

    # 2) 값 사전 (법정 컬럼, 범주형 텍스트 값)
    p_val = os.path.join(learned_dir, LEARNED_VALUES)
    lex = json.load(open(p_val, encoding="utf-8")) if os.path.exists(p_val) else {
        "_about": "검수로 확정된 법정 컬럼의 값 (review_feedback.apply_review 가 자동 생성)"}
    n_val = 0
    if data is not None:
        for r in rev.itertuples():
            cat, col = getattr(r, label_col), getattr(r, "column")
            if cat not in LEGAL_CODES or col not in data.columns:
                continue
            vals = _informative_text_values(data[col])
            if len(vals) < 2:
                continue
            terms = lex.setdefault(cat, {"terms": []})["terms"]
            for v in vals:
                if v not in terms:
                    terms.append(v)
                    n_val += 1
    json.dump(lex, open(p_val, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 3) 재학습 라벨 (labels_review.csv 형식)
    n_lab = 0
    if data is not None:
        p_lab = os.path.join(learned_dir, LEARNED_LABELS)
        old = (pd.read_csv(p_lab, dtype=str).fillna("") if os.path.exists(p_lab)
               else pd.DataFrame(columns=["source", "column", "auto_label", "corrected_label", "use", "text"]))
        cols = [c for c in rev["column"] if c in data.columns]
        profiles = {p.name: p for p in profile_dataframe(data[cols])} if cols else {}
        rows = []
        for r in rev.itertuples():
            col = getattr(r, "column")
            if col not in profiles:
                continue
            rows.append({"source": src, "column": col, "auto_label": getattr(r, "category", ""),
                         "corrected_label": getattr(r, label_col), "use": "1",
                         "text": profile_to_text(profiles[col])})
        new = pd.DataFrame(rows)
        if not new.empty:
            key = set(zip(new.source, new.column))
            old = old[[(s, c) not in key for s, c in zip(old.source, old.column)]]
            pd.concat([old, new], ignore_index=True).to_csv(p_lab, index=False, encoding="utf-8-sig")
            n_lab = len(new)
    return {"reviewed": len(rev), "overrides": len(ov), "new_value_terms": n_val, "labels": n_lab,
            "legal_confirmed": int(rev[label_col].isin(LEGAL_CODES).sum()),
            "negative_confirmed": int((rev[label_col] == NEGATIVE_CODE).sum())}
