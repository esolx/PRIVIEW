"""
demo_exposure.py — 탐지 → 노출 지표 통합 데모
=============================================
실제 ExposurePipeline로 민감정보를 탐지한 뒤, 그 결과를 exposure_stage로
넘겨 5개 지표 행렬을 계산하는 전 과정을 보여준다.

실행: python -m exposure_pipeline.demo_exposure
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .pipeline import ExposurePipeline, PipelineConfig
from .features import bootstrap_training_set
from . import exposure_stage as ES


def _make_demo_df(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zipcode = rng.choice(["12345", "67890", "11111", "22222"], n)
    gender = rng.choice(["M", "F"], n)
    age = rng.integers(20, 75, n)
    username = [f"user_{i:04d}" for i in range(n)]
    # 상병코드(민감): 특정 zip에서 동질적으로 쏠림 → coverage↑
    disease = np.where(
        zipcode == "12345",
        rng.choice(["C50", "C50", "C50", "J00"], n),   # 유방암 쏠림
        rng.choice(["J00", "K21", "E11", "C50"], n))
    # 연소득구간(민감): 질병과 얽힘 → 복합 민감도↑
    income = np.where(np.isin(disease, ["C50", "E11"]), "상위", "하위")
    return pd.DataFrame({
        "zipcode": zipcode, "gender": gender, "age": age,
        "username": username, "상병코드": disease, "연소득구간": income,
    })


def main():
    df = _make_demo_df()
    print("데이터:", df.shape, "| 컬럼:", list(df.columns))

    # --- 1) 탐지 파이프라인 학습 + 실행 ---
    cfg = PipelineConfig(lang="both")
    tx, ly = bootstrap_training_set(seed=cfg.random_state)
    # 데모 실행용: calibration과 학습을 분리. 실제 실험에서는 labels_review의
    # 별도 validation split을 사용하고 test/COMPAS/adult로 calibrate하지 말 것.
    from sklearn.model_selection import train_test_split
    tr_x, va_x, tr_y, va_y = train_test_split(
        list(tx), list(ly), test_size=0.25, stratify=list(ly), random_state=cfg.random_state)
    pipe = ExposurePipeline(cfg).fit(tr_x, tr_y)
    pipe.calibrate(va_x, va_y)
    result = pipe.run(df)

    det = pipe.to_frame(result)
    sens_cols = det[det.is_sensitive].column.tolist()
    print("\n[탐지] 민감속성:", sens_cols)

    # --- 2) 탐지 결과 → 노출 지표 단계 ---
    out = ES.run_from_detection(df, result)
    roles = out["roles"]
    print("\n[역할]")
    print("  민감 :", roles.sensitive)
    print("  EE   :", roles.ee, "| ee_score:", roles.ee_score_map)
    print("  QI   :", roles.qi)

    matrix = out["matrix"]
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\n[속성 × 지표 위험 행렬]  (합치지 않음 · 각 셀 0~1 · 클수록 위험)")
    cols = ["category", "emd", "coverage", "combo", "ee_match", "granularity"]
    print(matrix[cols] if not matrix.empty else "  (민감속성 없음)")

    # --- 3) (A) 분석용 — 지표 설명 + 선택 프레임워크 ---
    print("\n" + ES.describe_metrics())

    print("\n[예시] 내부 분포 축만 선택해서 보기")
    print(ES.select_metrics(matrix, axis="internal"))

    print("\n[참고기준선 초과 후보]  (등급이 아니라 '한 번 더 볼' 포인터)")
    for k, attrs in ES.flag_by_reference(matrix).items():
        if attrs:
            print(f"  {k:12s} ≥ {ES.METRIC_SPECS[k].ref_watch:.2f}: {attrs}")

    # granularity(레코드 수준)는 QI 조합만의 함수 → 속성마다 값이 아니라 데이터셋 1개.
    if not matrix.empty and "granularity" in matrix.columns:
        g = float(matrix["granularity"].iloc[0])
        print(f"\n[데이터셋 재식별(k=5 위반율)] {g:.3f}  "
              f"(QI={out['roles'].qi} · 모든 속성 동일한 게 정상)")

    # --- 4) (B) 판정용 — 데이터셋 H/M/L 등급 + 조치 순서(top-k) ---
    #     통합위험 = max(내부 최댓값, 외부 최댓값) → τ 초과 판정 → ρ → 등급
    assess = ES.assess_dataset(matrix)      # τ=0.5, ρ컷=(0.5,0.2) 기본
    print("\n" + ES.format_assessment(assess))

    # --- 5) 시각화 PNG ---
    png1 = ES.render_heatmap(matrix, out_path="exposure_heatmap.png",
                             category_map=out["roles"].category_map)
    views = ES.render_metric_views(matrix, out_prefix="metric_view")  # 지표별 개별 뷰
    png2 = ES.render_risk_curve(assess, out_path="risk_curve.png")
    print(f"\n[통합 히트맵] {png1}" if png1 else "\n[통합 히트맵] matplotlib 미설치 — 생략")
    if views:
        print(f"[개별 뷰]    {views.get('all')}  (+ 지표별 "
              f"{sum(1 for k in views if k != 'all')}장)")
    else:
        print("[개별 뷰]    matplotlib 미설치 — 생략")
    print(f"[위험곡선]   {png2}" if png2 else "[위험곡선]   matplotlib 미설치 — 생략")


if __name__ == "__main__":
    main()
