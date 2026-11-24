# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LEGAL_CODES, NEGATIVE_CODE, SENSITIVE_CODES
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # 문자열 애노테이션 전용 (런타임 미평가, 순환참조 방지)
    from .pipeline import ExposurePipeline

"""synth.py — 민감속성 라벨이 달린 합성 데이터셋 생성·내보내기.

목적
  · 논문의 18개 카테고리(법정 9 + 비법정 9)를 모두 한 번씩 태우는 테스트셋.
  · 오탐 방어를 실제로 자극하는 '함정' 칼럼 포함
      - benign_score : 'score'라는 민감해 보이는 이름 + 비민감 값
      - refund_flag  : 'amount'류 오인 유도용 불리언
      - dept_code    : 부서코드(비민감)인데 상병코드 형식과 헷갈릴 수 있는 값
  · 결과를 CSV로 떨궈 리뷰/재현이 가능하게 한다.

정답(TRUTH)은 사람이 라벨링한 것으로 취급한다. 파이프라인 출력과 대조해
Recall/Precision/카테고리 정확도를 계산한다.
"""


def make_synthetic_df(n: int = 800, seed: int = 20260730):
    """민감속성 칼럼이 포함된 합성 데이터프레임과 정답 라벨을 반환."""
    r = np.random.default_rng(seed)

    df = pd.DataFrame({
        # ================= 비민감 (일반 컬럼) =========================
        "member_no":        [f"M{i:08d}" for i in range(n)],
        "signup_date":      pd.to_datetime("2024-01-01")
                            + pd.to_timedelta(r.integers(0, 600, n), unit="D"),
        "age_band":         r.choice(["10대", "20대", "30대", "40대", "50대+"], n),
        "gender":           r.choice(["M", "F"], n),
        "nickname":         r.choice(["hoya", "sunny", "kkiri", "bluesky", "mint"], n),
        "row_id":           np.arange(n),

        # ---- 오탐 함정(비민감) : 이름이 민감어를 흉내 ----------------
        "benign_score":     r.integers(0, 100, n),          # 게임 점수. 신용점수 아님
        "refund_flag":      r.choice([True, False], n),      # amount 아님(불리언)
        "dept_code":        r.choice(["A01", "B02", "C03"], n),  # 부서코드(상병코드 형식과 유사)

        # ================= 법정 민감정보 (Track A) ====================
        # 건강, 의료
        "상병코드":          r.choice(["E11.9", "I10", "F32.1", "J45.0", "C34.9"], n),
        "처방약품명":        r.choice(["메트포르민", "암로디핀", "세르트랄린", "아토르바스타틴"], n),
        "patientDiagnosisCd": r.choice(["E119", "I10", "F321", "J450"], n),
        # 유전
        "유전자검사결과":     r.choice(["rs334", "rs1801133", "rs429358", "rs7412"], n),
        # 생체
        "fngr_print_tmpl":  [f"{r.integers(0, 16**15):020x}" for _ in range(n)],
        # 성생활, 지향
        "SEXUAL_ORIENTATION": r.choice(["hetero", "homo", "bi", "undisclosed"], n),
        # 인종, 민족
        "출신민족":          r.choice(["한민족", "조선족", "기타"], n),
        # 정치적 견해
        "지지정당":          r.choice(["A당", "B당", "C당", "무응답"], n),
        # 종교, 사상
        "종교":              r.choice(["개신교", "불교", "천주교", "원불교", "무교"], n),
        # 노조, 정당 가입
        "노동조합가입여부":   r.choice(["가입", "미가입", "탈퇴"], n),
        # 범죄경력
        "범죄경력조회결과":   r.choice(["해당없음", "벌금", "집행유예", "실형"], n),

        # ================= 비법정 민감정보 (Track B) ==================
        "연소득구간":        r.choice(["<3천", "3~5천", "5~7천", ">7천"], n),
        "credit_score":     r.integers(300, 950, n),
        "최종학력":          r.choice(["고졸", "전문학사", "학사", "석사", "박사"], n),
        "job_title":        r.choice(["engineer", "analyst", "manager", "designer"], n),
        "home_address":     [f"서울 강남구 역삼동 {r.integers(1, 400)}" for _ in range(n)],
        "access_ip":        [f"10.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(0,255)}"
                             for _ in range(n)],
        "ADID":             [f"{r.integers(0, 16**15):016x}" for _ in range(n)],
        "purchase_amount":  r.integers(0, 500, n) * 1000,
        "가구원수":          r.integers(1, 7, n),
        "자녀수":            r.integers(0, 5, n),
    })

    truth = {
        "member_no": "NON_SENSITIVE", "signup_date": "NON_SENSITIVE",
        "age_band": "NON_SENSITIVE", "gender": "NON_SENSITIVE",
        "nickname": "NON_SENSITIVE", "row_id": "NON_SENSITIVE",
        "benign_score": "NON_SENSITIVE", "refund_flag": "NON_SENSITIVE",
        "dept_code": "NON_SENSITIVE",
        # Track A
        "상병코드": "HEALTH", "처방약품명": "HEALTH", "patientDiagnosisCd": "HEALTH",
        "유전자검사결과": "GENETIC", "fngr_print_tmpl": "BIOMETRIC",
        "SEXUAL_ORIENTATION": "SEX_LIFE", "출신민족": "RACE_ETHNICITY",
        "지지정당": "POLITICAL_OPINION", "종교": "IDEOLOGY_BELIEF",
        "노동조합가입여부": "UNION_PARTY", "범죄경력조회결과": "CRIMINAL_RECORD",
        # Track B
        "연소득구간": "INCOME_ASSET", "credit_score": "CREDIT_FINANCE",
        "최종학력": "EDUCATION", "job_title": "OCCUPATION",
        "home_address": "LOCATION", "access_ip": "COMMUNICATION",
        "ADID": "DEVICE_ONLINE", "purchase_amount": "CONSUMPTION",
        "가구원수": "NON_SENSITIVE",
        "자녀수": "CHILDREN_COUNT",
    }
    return df, truth


def export_dataset(df: pd.DataFrame, truth: dict, csv_path: str,
                   truth_path: str) -> None:
    """합성 데이터셋과 정답 라벨을 각각 CSV로 저장."""
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"column": c, "true_category": truth[c],
          "track": ("A" if truth[c] in LEGAL_CODES
                    else "B" if truth[c] in SENSITIVE_CODES else "-")}
         for c in df.columns]
    ).to_csv(truth_path, index=False, encoding="utf-8-sig")


def results_frame(pipe: "ExposurePipeline", df: pd.DataFrame,
                  truth: dict) -> pd.DataFrame:
    """파이프라인을 돌려 정답 대조까지 붙인 결과 프레임."""
    res = pipe.run(df)
    out = pipe.to_frame(res)
    out["true_category"] = out["column"].map(truth)
    out["category_match"] = np.where(
        out["category"] == out["true_category"], "○", "×")
    cols = ["column", "category", "true_category", "category_match",
            "track", "stage", "tier", "is_sensitive", "review_required",
            "weight", "entropy_norm", "sbert_sim", "value_hit_rate",
            "weight_basis"]
    return out[[c for c in cols if c in out.columns]], res


def metrics(res, truth: dict) -> dict:
    """탐지 Recall/Precision + 카테고리 정확도."""
    yt = np.array([1 if truth[d.column] != NEGATIVE_CODE else 0
                   for d in res.decisions])
    yp = np.array([1 if d.is_sensitive else 0 for d in res.decisions])
    tp = int(((yt == 1) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    acc = float(np.mean([d.category == truth[d.column] for d in res.decisions]))
    return {"category_accuracy": round(acc, 4),
            "detect_recall": round(tp / max(tp + fn, 1), 4),
            "detect_precision": round(tp / max(tp + fp, 1), 4),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}
