# -*- coding: utf-8 -*-
"""v9 회귀 테스트 — 부재(0 의 의미) 판정 + EE 사전 부분일치.

실행 (pytest 없이도 됨):
    cd exposure_pipeline_FINAL
    python tests/test_presence_and_ee.py        # 또는  python -m pytest tests
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from exposure_pipeline import ee_dictionary as E      # noqa: E402
from exposure_pipeline import metrics as M            # noqa: E402

RNG = np.random.default_rng(0)
N = 1000


def _cov(series, category):
    info = M.presence_info(series, category)
    return info, M.coverage_risk(series, _info=info)


def test_disease_flag_zero_is_absent():
    s = pd.Series(RNG.choice([0, 1], N, p=[.9, .1]), name="diabetes_yn")
    info, cov = _cov(s, "HEALTH")
    assert info.rule == "binary" and abs(cov - 0.1) < 0.03


def test_category_codes_zero_is_a_value():
    sex = pd.Series(RNG.choice([0, 1], N), name="sex")
    race = pd.Series(RNG.choice([0, 1, 2, 3, 4], N, p=[.4, .2, .2, .1, .1]), name="race")
    assert _cov(sex, "NON_SENSITIVE")[1] == 1.0
    assert _cov(race, "RACE_ETHNICITY")[1] == 1.0


def test_event_token_overrides_category_valued():
    # 범주형 카테고리여도 이름이 '여부'면 0/1 플래그로 본다
    s = pd.Series(RNG.choice([0, 1], N, p=[.8, .2]), name="종교활동여부")
    info, cov = _cov(s, "IDEOLOGY_BELIEF")
    assert info.rule == "binary" and cov < 0.3


def test_isolated_zero_is_missing_code():
    s = pd.Series(np.where(RNG.random(N) < .05, 0, RNG.integers(18, 90, N)), name="age")
    info, cov = _cov(s, None)
    assert info.rule == "zero-gap" and 0.92 < cov < 0.98 and info.warning


def test_survey_12_coding():
    s = pd.Series(RNG.choice([1, 2], N, p=[.2, .8]), name="흡연여부")
    info, cov = _cov(s, "HEALTH")
    assert info.rule == "binary-12" and cov < 0.3 and info.warning


def test_yes_no_strings_and_numeric_strings():
    for vals, name in ((["Y", "N"], "disease"), (["예", "아니오"], "질병"),
                       (["0.0", "1.0"], "disease")):
        s = pd.Series(RNG.choice(vals, N, p=[.1, .9] if vals[0] in ("Y", "예") else [.9, .1]),
                      name=name)
        assert _cov(s, "HEALTH")[1] < 0.2, vals


def test_nonresponse_is_missing_for_category_valued():
    s = pd.Series(RNG.choice(["A당", "B당", "무응답"], N), name="지지정당")
    assert _cov(s, "POLITICAL_OPINION")[1] < 0.75


def test_warnings_only_for_ambiguous():
    healthy = pd.Series(RNG.choice([0, 1], N, p=[.1, .9]), name="healthy")
    assert "역코딩" in M.presence_info(healthy, "HEALTH").warning
    score = pd.Series(np.where(RNG.random(N) < .05, 99, RNG.integers(1, 6, N)), name="depression_score")
    info = M.presence_info(score, "HEALTH")
    assert "99" in info.warning and info.rule == "none"      # 자동 처리 안 함


def test_codebook_wins():
    s = pd.Series(RNG.choice([1, 2, 9], N), name="q17")
    with M.metrics_config(absence_overrides={"q17": [2, 9]}):
        info = M.presence_info(s, "HEALTH")
    assert info.rule == "codebook"


def test_ee_no_false_substring_matches():
    bad = ["usage_count", "message", "average_score", "dosage", "storage_gb", "code", "x",
           "recipient", "shipping", "구매금액", "시험점수", "가구원수", "성적", "성과급"]
    hits = {c: a for c, a in E.match_columns(bad).items() if a}
    assert not hits, hits


def test_ee_true_matches_kept():
    good = {"age": "Age", "age_band": "Age", "patientAge": "Age", "zipCode": "ZIP Code / Postal Code",
            "home_zip_code": "ZIP Code / Postal Code", "birth_year": "Birth Year",
            "연령대": "Age", "성별": "Gender", "나이": "Age"}
    res = E.match_columns(list(good))
    for c, attr in good.items():
        assert res[c] is not None and res[c].attr_en == attr, (c, res[c])


def test_tvd_is_normalized_to_full_range():
    # 2종 95:5 는 극단 쏠림 → 1 에 가까워야 함 (예전엔 최대 0.5)
    s = pd.Series(RNG.choice(["A", "B"], N, p=[.95, .05]), name="a")
    v, note = M.emd_detail(s)
    assert v > 0.85 and note == "", (v, note)


def test_emd_zero_cases_have_reason():
    ident = pd.Series([f"id{i}" for i in range(N)], name="adid")
    assert M.emd_detail(ident, category="DEVICE_ONLINE")[1].startswith("near-unique")
    const = pd.Series(["A"] * N, name="c")
    assert M.emd_detail(const)[1].startswith("single-value")


def test_emd_includes_absent_values():
    # '없음'도 분포에 넣는다: 0/1 여부 → |2p-1|
    rare = pd.Series(RNG.choice([0, 1], N, p=[.95, .05]), name="disease_yn")
    v, _ = M.emd_detail(rare, category="HEALTH")
    assert abs(v - abs(2 * (rare == 1).mean() - 1)) < 1e-9
    even = pd.Series([0, 1] * (N // 2), name="two_year_recid")
    assert M.emd_detail(even, category="CRIMINAL_RECORD")[0] == 0.0


def test_emd_excludes_missing_codes_and_outliers():
    yn = pd.Series(["Y"] * 100 + ["N"] * 600 + ["모름"] * 300, name="disease")
    v, note = M.emd_detail(yn, category="HEALTH")            # p = 100/700
    assert abs(v - abs(2 * 100 / 700 - 1)) < 1e-9 and "missing 300" in note
    age = pd.Series(np.where(RNG.random(N) < .05, 0, RNG.integers(18, 90, N)), name="age")
    vals, ex = M.emd_valid_values(age)
    assert (vals == 0).sum() == 0 and ex["missing-code"] > 0
    score = pd.Series(np.where(RNG.random(N) < .05, 99, RNG.integers(1, 6, N)), name="score")
    vals, ex = M.emd_valid_values(score, "HEALTH")
    assert 99 not in set(vals) and ex["missing-code"] > 0
    inc = pd.Series(np.r_[RNG.lognormal(10, .5, 995), [1e12] * 5], name="income")
    assert M.emd_valid_values(inc, "INCOME_ASSET")[1]["outlier"] == 5
    q = pd.Series(RNG.choice([1, 2, 9], N), name="q9")
    with M.metrics_config(missing_overrides={"q9": [9]}):
        vals, ex = M.emd_valid_values(q, "HEALTH")
    assert 9 not in set(vals) and ex["missing-code"] > 0


def test_dates_use_ordered_emd():
    base = pd.to_datetime("2020-01-01")
    days = np.r_[RNG.integers(0, 30, 900), RNG.integers(30, 1000, 100)]
    s = (base + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d").to_series(name="visit_date")
    assert M.emd_detail(s)[0] > 0.7


def test_korean_negative_values_absent():
    s = pd.Series(RNG.choice(["가입", "미가입"], N, p=[.3, .7]), name="노동조합가입여부")
    assert M.coverage_risk(s, category="UNION_PARTY") < 0.4


# ---------------------------------------------------------------- v9.2 법정 판정
def test_name_evidence_token_level_and_ambiguity():
    from exposure_pipeline.config import name_evidence
    assert name_evidence("trace_id", "RACE_ETHNICITY") is None          # 철자 겹침 X
    assert name_evidence("healthy_snack", "HEALTH") is None
    assert name_evidence("party_size", "POLITICAL_OPINION") == "weak"   # 다의어, 맥락 없음
    assert name_evidence("party_support", "POLITICAL_OPINION") == "strong"
    assert name_evidence("성적", "SEX_LIFE") == "weak"
    assert name_evidence("성적지향", "SEX_LIFE") == "strong"
    assert name_evidence("priors_count", "CRIMINAL_RECORD") == "strong"


def test_value_lexicon_evidence():
    from exposure_pipeline.fp_control import value_lexicon_evidence as vle
    assert vle("IDEOLOGY_BELIEF", ["기독교", "불교", "천주교", "무교"] * 10)[0]
    assert not vle("RACE_ETHNICITY", ["white", "red", "blue", "green"] * 10)[0]   # 색상
    assert not vle("UNION_PARTY", list("ABCDF") * 10)[0]


def test_legal_gate_end_to_end():
    import warnings
    warnings.filterwarnings("ignore")
    from exposure_pipeline.pipeline import ExposurePipeline, PipelineConfig
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "exposure_pipeline")
    pipe = ExposurePipeline(PipelineConfig(lang="both", dataset_specific_rules=False))
    pipe.load_rf(os.path.join(root, "rf_model.joblib"), text_mode="rf")
    cal = pd.read_csv(os.path.join(root, "calibration_holdout.csv")).fillna("")
    pipe.calibrate(cal.text.tolist(), cal.label.tolist())
    df = pd.DataFrame({
        "final_grade": RNG.choice(list("ABCDF"), 300),                       # 오탐 함정
        "trace_id": ["".join(RNG.choice(list("0123456789abcdef"), 10)) for _ in range(300)],
        "q17": RNG.choice(["기독교", "불교", "천주교", "무교"], 300),          # 불투명 이름 + 종교 값
        "two_year_recid": RNG.choice([0, 1], 300),                            # 약어 매핑
    })
    d = pipe.to_frame(pipe.run(df)).set_index("column")
    assert not d.at["final_grade", "is_legal"]
    assert not d.at["trace_id", "is_legal"]
    assert d.at["q17", "category"] == "IDEOLOGY_BELIEF"
    assert d.at["two_year_recid", "is_legal"] and d.at["two_year_recid", "review_required"]


# ---------------------------------------------------------------- v9.4 재현율
def test_extended_name_terms_and_contexts():
    from exposure_pipeline.config import name_evidence as ne
    assert ne("hiv_status", "HEALTH") == "strong"
    assert ne("pol_lean", "POLITICAL_OPINION") == "strong"
    assert ne("lean_body_mass", "POLITICAL_OPINION") == "weak"
    assert ne("c_charge_degree", "CRIMINAL_RECORD") == "strong"
    assert ne("battery_charge_pct", "CRIMINAL_RECORD") == "weak"
    assert ne("race", "RACE_ETHNICITY") == "strong"
    assert ne("race_finish_time", "RACE_ETHNICITY") == "weak"          # 반대 맥락


def test_context_values_need_name_support():
    from exposure_pipeline.fp_control import value_lexicon_evidence as vle
    lr = ["Left", "Right", "Center"] * 20
    assert vle("POLITICAL_OPINION", lr, name="pol_lean")[0]
    assert not vle("POLITICAL_OPINION", ["left", "right"] * 30, name="hand_dominance")[0]
    pn = ["positive", "negative"] * 30
    assert not vle("HEALTH", pn, name="sentiment")[0]


def test_hgvs_pattern_value_evidence():
    from exposure_pipeline.fp_control import value_lexicon_evidence as vle
    vals = ["BRCA1 c.68_69del", "BRCA2 c.5946del", "TP53 p.R175H"] * 10
    assert vle("GENETIC", vals)[0]


def test_review_feedback_roundtrip(tmp_path=None):
    import tempfile
    from exposure_pipeline.review_feedback import apply_review, load_overrides
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    data = pd.DataFrame({"q42": RNG.choice(["Cao Dai", "Tenrikyo", "Rastafari"], 200),
                         "amt": RNG.integers(1, 99, 200)})
    data_p, det_p = os.path.join(d, "data.csv"), os.path.join(d, "det.csv")
    data.to_csv(data_p, index=False)
    pd.DataFrame({"column": ["q42", "amt"], "category": ["NON_SENSITIVE"] * 2,
                  "reviewed_label": ["IDEOLOGY_BELIEF", "NON_SENSITIVE"]}).to_csv(det_p, index=False)
    out = apply_review(det_p, data_p, os.path.join(d, "learned"))
    assert out["new_value_terms"] == 3 and out["labels"] == 2
    assert load_overrides(os.path.join(d, "learned"))["q42"] == "IDEOLOGY_BELIEF"


# ---------------------------------------------------------------- v9.5 조치 목록
def test_action_lists_by_axis():
    from exposure_pipeline import exposure_stage as ES
    m = pd.DataFrame({"category": ["HEALTH"] * 3,
                      "emd": [0.9, 0.1, 0.6], "coverage": [1.0, 1.0, 1.0], "combo": [0.0, 0.0, 0.0],
                      "ee_match": [0.1, 0.8, 0.55], "granularity": [0.1, 0.1, 0.1]},
                     index=["a_int", "b_ext", "c_both"])
    a = ES.assess_dataset(m)
    internal = [x[1] for x in ES.topk_actions_split(a, order_by="internal")]
    external = [x[1] for x in ES.topk_actions_split(a, order_by="external")]
    assert internal == ["a_int", "c_both"]
    assert external == ["b_ext", "c_both"]
    txt = ES.format_assessment(a, action_order="split")
    assert "[내부 기준]" in txt and "[외부 기준]" in txt


# ---------------------------------------------------------------- v9.6 산출 상태
def test_emd_status_determined_and_not_applicable():
    r = M.emd_result(pd.Series([1] * N, name="hiv_yn"), category="HEALTH")
    assert r.status == M.STATUS_DETERMINED and np.isnan(r.value)      # 0 이 아님
    r = M.emd_result(pd.Series([np.nan] * N, name="religion"), category="IDEOLOGY_BELIEF")
    assert r.status == M.STATUS_NA and np.isnan(r.value) and r.reason.startswith("no-valid")
    r = M.emd_result(pd.Series([f"id{i}" for i in range(N)], name="adid"))
    assert r.status == M.STATUS_NA and r.reason.startswith("near-unique")
    r = M.emd_result(pd.Series(RNG.choice([0, 1], N, p=[.9, .1]), name="d_yn"), category="HEALTH")
    assert r.status == M.STATUS_NUMERIC and 0 < r.value <= 1


def test_combo_na_only_without_partner():
    one = pd.DataFrame({"a": RNG.choice([0, 1], N)})
    assert M.composite_sensitivity_detail(one)["a"].status == M.STATUS_NA
    assert np.isnan(M.composite_sensitivity(one)["a"])
    # 상대가 식별자형뿐 → 상대 없음(비적용) / 식별자형 목표 → 비적용
    two = pd.DataFrame({"a": RNG.choice([0, 1], N), "uid": [f"u{i}" for i in range(N)]})
    d = M.composite_sensitivity_detail(two)
    assert d["a"].status == M.STATUS_NA and "상대" in d["a"].reason
    assert d["uid"].status == M.STATUS_NA and "식별자형" in d["uid"].reason
    # 상대가 유효값 없는 속성뿐 → 상대 없음
    two = pd.DataFrame({"a": RNG.choice([0, 1], N), "empty": [np.nan] * N})
    d = M.composite_sensitivity_detail(two)
    assert d["a"].status == M.STATUS_NA and d["empty"].status == M.STATUS_NA
    three = pd.DataFrame({"a": RNG.choice([0, 1], N), "b": RNG.choice([0, 1], N)})
    assert all(r.status == M.STATUS_NUMERIC for r in M.composite_sensitivity_detail(three).values())


def test_external_na_without_ee_or_qi():
    df = pd.DataFrame({"dis": RNG.choice([0, 1], N), "age": RNG.integers(20, 60, N),
                       "sex": RNG.choice(["M", "F"], N)})
    m = M.compute_metric_matrix(df, ["dis"], qi_cols=[], ee_cols=[])
    row = m.loc["dis"]
    assert row.external_nmi_status == M.STATUS_NA and np.isnan(row.external_nmi)
    assert row.granularity_status == M.STATUS_NA and np.isnan(row.granularity)
    assert row.ee_match_status == M.STATUS_NA and np.isnan(row.ee_match)
    assert "준식별자" in row.status_reason and "보조속성" in row.status_reason
    m = M.compute_metric_matrix(df, ["dis"], qi_cols=["age", "sex"], ee_cols=["age"],
                                ee_score_map={"age": 0.9})
    row = m.loc["dis"]
    assert {row.external_nmi_status, row.granularity_status, row.ee_match_status} == {M.STATUS_NUMERIC}
    # QI 만 있고 EE 없음 → G 는 수치, L, 외부노출은 비적용
    m = M.compute_metric_matrix(df, ["dis"], qi_cols=["age", "sex"], ee_cols=[])
    row = m.loc["dis"]
    assert row.granularity_status == M.STATUS_NUMERIC and row.ee_match_status == M.STATUS_NA


def test_assessment_excludes_non_numeric():
    from exposure_pipeline import exposure_stage as ES
    df = pd.DataFrame({"hiv_yn": [1] * N,                         # EMD 확정
                       "religion": [np.nan] * N,                  # EMD 비적용
                       "diag": RNG.choice(["a", "b", "c"], N, p=[.9, .05, .05])})
    m = M.compute_metric_matrix(df, list(df.columns), qi_cols=[], ee_cols=[])
    a = ES.assess_dataset(m)
    assert a.determined == ["hiv_yn"]
    # hiv_yn 은 EMD 가 빠져도 Combo 수치가 있어 순위에 남고, 0 으로 깔리지 않는다
    comp = a.components
    assert not np.isnan(comp.loc["diag", "internal_risk"])
    assert all(np.isnan(comp["external_risk"]))                 # EE, QI 없음 → 외부축 미산출
    assert set(a.unscored) | set(a.scores.index) == set(df.columns)
    assert a.n_total == len(a.scores)
    sc = a.status_counts
    assert sc.loc["emd", "확정"] == 1 and sc.loc["emd", "비적용"] == 1 and sc.loc["emd", "수치"] == 1
    txt = ES.format_assessment(a)
    assert "우선 검토" in txt and "hiv_yn" in txt and "nan" not in txt
    js = ES.matrix_to_view_json(m)
    rec = {r["attribute"]: r for r in js}
    assert rec["hiv_yn"]["emd"] is None and rec["hiv_yn"]["emd_status"] == M.STATUS_DETERMINED


def _edge_frame():
    return pd.DataFrame({
        "dis": RNG.choice([0, 1], N, p=[.8, .2]),
        "empty": [np.nan] * N,                                   # 유효값 없음
        "dna_seq": [f"ACGT{i:05d}" for i in range(N)],           # 식별자형 목표
        "neg_only": [0] * N,                                     # 유효값은 있으나 전부 '없음'
        "age": RNG.integers(20, 60, N), "sex": RNG.choice(["M", "F"], N)})


def test_target_na_all_components_except_coverage():
    df = _edge_frame()
    sens = ["dis", "empty", "dna_seq", "neg_only"]
    m = M.compute_metric_matrix(df, sens, qi_cols=["age", "sex"], ee_cols=["age"],
                                ee_score_map={"age": 0.9},
                                category_map={c: "HEALTH" for c in sens})
    for attr in ("empty", "dna_seq"):
        row = m.loc[attr]
        for k in ("emd", "combo", "external_nmi", "granularity", "ee_match"):
            assert row[f"{k}_status"] == M.STATUS_NA and np.isnan(row[k]), (attr, k)
        assert row.coverage_status == M.STATUS_NUMERIC              # 분모 N → 항상 정의
    assert "유효" in m.at["empty", "status_reason"]
    assert "식별자형" in m.at["dna_seq", "status_reason"]
    assert m.at["dis", "combo_status"] == M.STATUS_NUMERIC        # neg_only 가 상대로 남음


def test_empty_presence_granularity_is_na():
    # 논문: "준식별자나 존재값이 없으면 비적용"
    df = _edge_frame()
    m = M.compute_metric_matrix(df, ["neg_only", "dis"], qi_cols=["age", "sex"], ee_cols=[],
                                category_map={"neg_only": "HEALTH", "dis": "HEALTH"})
    row = m.loc["neg_only"]
    assert row.n_present == 0 and row.n_valid == N
    assert row.granularity_status == M.STATUS_NA and np.isnan(row.granularity)
    assert "존재값 없음" in row.status_reason
    assert row.emd_status == M.STATUS_DETERMINED                 # 유효값 1종(0) → 확정
    assert m.at["dis", "granularity_status"] == M.STATUS_NUMERIC


def test_external_nmi_zero_denominator_is_na():
    # 논문: "보조속성이 없거나 분모가 0이면 비적용"
    df = pd.DataFrame({"examide": ["No"] * N, "dis": RNG.choice([0, 1], N),
                       "age": RNG.integers(20, 60, N), "const_ee": ["x"] * N})
    r = M.external_nmi_detail(df["examide"], M.prepare_ee_bins(df[["age"]]))
    assert r.status == M.STATUS_NA and "분모 0" in r.reason        # 목표 엔트로피 0
    # 상수 보조속성 쌍만 빠지고 나머지로 계산 (예전엔 0 이 평균에 섞임)
    bins = M.prepare_ee_bins(df[["age", "const_ee"]])
    r = M.external_nmi_detail(df["dis"], bins)
    only_age = M.external_nmi_detail(df["dis"], M.prepare_ee_bins(df[["age"]]))
    assert r.status == M.STATUS_NUMERIC and abs(r.value - only_age.value) < 1e-12
    r = M.external_nmi_detail(df["dis"], M.prepare_ee_bins(df[["const_ee"]]))
    assert r.status == M.STATUS_NA
    m = M.compute_metric_matrix(df, ["examide", "dis"], qi_cols=["age"], ee_cols=["age"],
                                ee_score_map={"age": 0.9})
    assert m.at["examide", "external_nmi_status"] == M.STATUS_NA
    assert m.at["examide", "ee_match_status"] == M.STATUS_NA


def test_combo_excludes_partners_with_few_pair_samples():
    # 논문: "쌍별 유효 표본이 2개 미만인 상대는 J_s 에서 제외"
    a = RNG.choice(["p", "q"], N).astype(object)
    b = np.full(N, np.nan, dtype=object)
    a[:N // 2] = np.nan
    b[0] = "u"                                       # a, b 가 함께 관측된 행 없음 → 쌍 표본 0
    b[N // 2] = "v"                                  # b 의 유효값 2개 (b 자체는 유효값 있음)
    b[N // 2 + 1] = "u"                              # → 쌍 표본 2 (경계값: 포함)
    df = pd.DataFrame({"a": a, "b": b})
    assert M.pair_valid_count(df["a"], df["b"]) == 2
    assert not np.isnan(M.combo_pair_matrix(df).at["a", "b"])
    b2 = b.copy()
    b2[N // 2 + 1] = np.nan                          # 쌍 표본 1 → 제외
    df2 = pd.DataFrame({"a": a, "b": b2})
    assert M.pair_valid_count(df2["a"], df2["b"]) == 1
    assert np.isnan(M.combo_pair_matrix(df2).at["a", "b"])
    d = M.composite_sensitivity_detail(df2)
    assert d["a"].status == M.STATUS_NA and d["b"].status == M.STATUS_NA


def test_evaluation_omissions_not_scored_zero():
    from types import SimpleNamespace as NS
    from exposure_pipeline import exposure_stage as ES
    dec = [NS(column="race", category="RACE_ETHNICITY", is_sensitive=True, tier="CONFIRMED"),
           NS(column="hiv", category="NON_SENSITIVE", is_sensitive=False, tier="NONE"),
           NS(column="weak_col", category="HEALTH", is_sensitive=True, tier="WEAK")]
    res = NS(decisions=dec)
    cols = ["race", "hiv", "weak_col", "age"]
    om = ES.evaluation_omissions(cols, ["race"],
                                 expected_sensitive={"race": "RACE_ETHNICITY", "hiv": "HEALTH",
                                                     "missing_col": "HEALTH"},
                                 detection_result=res, min_tier=("CONFIRMED", "PROBABLE"))
    why = dict(zip(om.attribute, om.reason))
    assert set(why) == {"hiv", "missing_col", "weak_col"}
    assert why["hiv"].startswith("탐지되지 않음") and "tier" in why["weak_col"]
    # 기준 목록 없이도 tier 로 빠진 민감속성은 보고
    om2 = ES.evaluation_omissions(cols, ["race"], detection_result=res,
                                  min_tier=("CONFIRMED", "PROBABLE"))
    assert om2.attribute.tolist() == ["weak_col"]
    # 전체 컬럼 행렬: 평가 누락 행은 0 이 아니라 NaN + evaluation=omitted
    df = pd.DataFrame({"race": RNG.choice(["a", "b"], 50), "hiv": RNG.choice([0, 1], 50),
                       "age": RNG.integers(20, 60, 50)})
    m = M.compute_metric_matrix(df, ["race"], qi_cols=["age"], ee_cols=[])
    roles = ES.assign_roles(df.columns, ["race"])
    allm = ES.build_all_columns_matrix(df, roles, sensitive_matrix=m, omitted=["hiv"])
    assert allm.at["hiv", "evaluation"] == "omitted" and allm.at["hiv", "category"] == ES.OMITTED_LABEL
    assert np.isnan(allm.at["hiv", "emd"]) and allm.at["age", "evaluation"] == "non_sensitive"
    a = ES.assess_dataset(m, omitted=["hiv"])
    assert a.omitted == ["hiv"] and a.n_total == 1 and "평가 누락" in ES.format_assessment(a)


def test_unscored_attribute_not_in_rho():
    from exposure_pipeline import exposure_stage as ES
    m = pd.DataFrame({"emd": [0.9, np.nan], "coverage": [1.0, 1.0], "combo": [np.nan, np.nan],
                      "ee_match": [np.nan, np.nan], "granularity": [np.nan, np.nan],
                      "emd_status": ["numeric", "determined"]}, index=["x", "y"])
    a = ES.assess_dataset(m)
    assert a.n_total == 1 and a.rho == 1.0 and a.unscored == ["y"] and a.determined == ["y"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL {name}: {e}")
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
