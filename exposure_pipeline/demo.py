# -*- coding: utf-8 -*-
from __future__ import annotations


import numpy as np

"""demo.py — 데모 진입점 (python -m exposure_pipeline)"""

from .column_mapping import ColumnMappingTable
from .features import (bootstrap_training_set, profile_dataframe,
                       profile_to_text)
from .pipeline import ExposurePipeline, PipelineConfig
from .preprocess import make_baselines
from .risk_weight import RiskWeightConfig

# ===  데모 (python exposure_pipeline_all.py 로 실행)

"""데모 실행 진입점.

    python -m exposure_pipeline

가명 예시 데이터프레임을 만들어 0~2단계 + FP방어 + 위험도 가중치까지
전체 파이프라인을 한 번 돌리고 결과를 출력한다.
"""

import pandas as pd



def make_df(n=500, seed=7):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        # --- 비민감 -------------------------------------------------
        "psn_key":            [f"PSN-{i:07d}" for i in range(n)],
        "created_at":         pd.to_datetime("2025-01-01") + pd.to_timedelta(r.integers(0,400,n), unit="D"),
        "age_group":          r.choice(["20s","30s","40s","50s"], n),
        "Driver_Gender":      r.choice(["M","F"], n),
        "row_num":            np.arange(n),
        # --- 법정 (한글) ---------------------------------------------
        "상병코드":            r.choice(["E11.9","I10","F32.1","J45.0"], n),
        "처방약품명":          r.choice(["메트포르민","암로디핀","세르트랄린"], n),
        "유전자검사결과":       r.choice(["rs334","rs1801133","rs429358"], n),
        "노동조합가입여부":     r.choice(["가입","미가입","탈퇴"], n),
        "지지정당":            r.choice(["A당","B당","무응답"], n),
        "종교":                r.choice(["개신교","불교","천주교","무교"], n),
        "범죄경력조회결과":     r.choice(["해당없음","벌금","집행유예"], n),
        "출신민족":            r.choice(["한민족","조선족","기타"], n),
        # --- 법정 (영문) ---------------------------------------------
        "patientDiagnosisCd": r.choice(["E119","I10","F321"], n),
        "fngr_print_tmpl":    [f"{r.integers(0,16**15):020x}" for _ in range(n)],
        "SEXUAL_ORIENTATION": r.choice(["hetero","homo","bi","undisclosed"], n),
        # --- 비법정 (한/영 혼합) --------------------------------------
        "연소득구간":          r.choice(["<3천","3~5천",">5천"], n),
        "credit_score":       r.integers(300,900,n),
        "최종학력":            r.choice(["고졸","대졸","석사","박사"], n),
        "job_title":          r.choice(["engineer","analyst","manager"], n),
        "home_address":       [f"seoul gangnam {r.integers(1,300)}" for _ in range(n)],
        "access_ip":          [f"10.{r.integers(0,255)}.{r.integers(0,255)}.{r.integers(0,255)}" for _ in range(n)],
        "ADID":               [f"{r.integers(0,16**15):016x}" for _ in range(n)],
        "purchase_amount":    r.integers(0,300,n)*1000,
        "가구원수":            r.integers(1,6,n),
        "자녀수":              r.integers(0,5,n),
    })


TRUTH = {
    "psn_key":"NON_SENSITIVE","created_at":"NON_SENSITIVE","age_group":"NON_SENSITIVE",
    "Driver_Gender":"NON_SENSITIVE","row_num":"NON_SENSITIVE",
    "상병코드":"HEALTH","처방약품명":"HEALTH","유전자검사결과":"GENETIC",
    "노동조합가입여부":"UNION_PARTY","지지정당":"POLITICAL_OPINION",
    "종교":"IDEOLOGY_BELIEF","범죄경력조회결과":"CRIMINAL_RECORD",
    "출신민족":"RACE_ETHNICITY","patientDiagnosisCd":"HEALTH",
    "fngr_print_tmpl":"BIOMETRIC","SEXUAL_ORIENTATION":"SEX_LIFE",
"연소득구간":"INCOME_ASSET","credit_score":"CREDIT_FINANCE",
    "최종학력":"EDUCATION","job_title":"OCCUPATION","home_address":"LOCATION",
    "access_ip":"COMMUNICATION","ADID":"DEVICE_ONLINE","purchase_amount":"CONSUMPTION",
    "가구원수":"NON_SENSITIVE",
    "자녀수":"CHILDREN_COUNT",
}


def main():
    df = make_df()

    print("="*76); print("[0단계] 칼럼명 매핑 테이블 — 다른 모든 것보다 먼저"); print("="*76)
    mapper = ColumnMappingTable(lang="both")
    cov = mapper.coverage(list(df.columns))
    print(f"  커버리지 {cov['matched']}/{cov['total']} = {cov['coverage']:.1%}   {cov['by_match_type']}")
    print(f"  {'컬럼':22s}{'정규형':24s}{'카테고리':18s}{'유형':10s}conf")
    for c in df.columns:
        h = mapper.lookup(c)
        mark = "○" if h.category == TRUTH[c] else "×"
        print(f"  {c:22s}{h.normalized:24s}{h.category:18s}{h.match_type:10s}{h.confidence:.2f} {mark}")
    print(f"  미매칭(→1단계로): {cov['unmatched']}")

    print("\n"+"="*76); print("[전처리] keep vs ignore"); print("="*76)
    bl = make_baselines(lang="both")
    for s in ["노동조합가입여부 가입 탈퇴 조합비", "patientDiagnosisCd icd10 prescription medication"]:
        k, g = bl["keep"].run(s), bl["ignore"].run(s)
        print(f"\n  원문   : {s}")
        print(f"  keep   : {k.tokens}\n           보호토큰={k.protected}")
        print(f"  ignore : {g.tokens}")

    print("\n"+"="*76); print("[1·2단계] 학습 + Recall 우선 임계값"); print("="*76)
    X, y = bootstrap_training_set(n_per_cat=60)
    cfg = PipelineConfig(policy="keep", lang="both", mapping_table=mapper,
                         risk_config=RiskWeightConfig())
    cfg.rf_config.use_struct = False
    pipe = ExposurePipeline(cfg).fit(X, y)

    prof = profile_dataframe(df)
    vt = [profile_to_text(p) for p in prof]; vl = [TRUTH[p.name] for p in prof]
    cal = pipe.calibrate(vt, vl)
    print("  1단계:", cal["stage1"])
    print("  2단계(SBERT 구제):", cal["stage2"],
          f"| 잔여 {cal['residual_n']}개 중 양성 {cal['residual_positives']}개")

    print("\n"+"="*76); print("[결과] Track 라우팅 + 위험도 가중치"); print("="*76)
    res = pipe.run(df)
    out = pipe.to_frame(res)
    out["정답"] = out["column"].map(TRUTH)
    out["일치"] = np.where(out["category"] == out["정답"], "○", "×")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(out[["column","category","정답","일치","track","stage","tier",
               "weight","entropy_norm","review_required"]].to_string(index=False))

    acc = (out["category"] == out["정답"]).mean()
    yt = np.array([1 if TRUTH[d.column]!="NON_SENSITIVE" else 0 for d in res.decisions])
    yp = np.array([1 if d.is_sensitive else 0 for d in res.decisions])
    tp=int(((yt==1)&(yp==1)).sum()); fn=int(((yt==1)&(yp==0)).sum()); fp=int(((yt==0)&(yp==1)).sum())
    print(f"\n  카테고리 정확도 {acc:.3f} | 탐지 Recall {tp/max(tp+fn,1):.3f} "
          f"Precision {tp/max(tp+fp,1):.3f} (FN={fn}, FP={fp})")
    print(f"  Track A ({len(res.track_a)}): {res.track_a}")
    print(f"  Track B ({len(res.track_b)}): {res.track_b}")
    print(f"  법정 가중치: {res.thresholds['legal_weight']} (일괄 고정)")

    print("\n"+"="*76)
    print("[스펙 갭] 매핑 테이블에서 빠진 키워드 찾기"); print("="*76)
    miss = [d.column for d in res.decisions
            if TRUTH[d.column] != "NON_SENSITIVE" and not d.is_sensitive]
    print(f"  놓친 컬럼: {miss}")
    print("  → 새 어휘는 config를 고치지 않고 extra_terms로 즉시 보강할 수 있다:")
    m2 = ColumnMappingTable(lang="both", extra_terms={"진료과": "HEALTH"})
    h = m2.lookup("진료과코드")
    print(f"     진료과코드 -> {h.category} ({h.match_type}, conf {h.confidence})")

    print("\n  --- 가중치 확인 ---")
    print("  Track A(법정) 전부 1.5 고정: " +
          ", ".join(f"{d.column}={d.weight}" for d in res.decisions if d.track=="A"))
    print("  Track B(비법정) 논문1틀+논문3 민감도 5단계(높을수록 민감): " +
          ", ".join(f"{d.column}={d.weight}" for d in res.decisions if d.track=="B"))


if __name__ == "__main__":
    main()
