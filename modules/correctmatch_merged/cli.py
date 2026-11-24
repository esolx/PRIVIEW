"""CLI 실행 스크립트.

사용 예:
    python -m correctmatch_toolkit.cli 데이터셋.csv
    python -m correctmatch_toolkit.cli 데이터셋.csv --method equal_width --n-bins 5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data_io import load_dataset
from .pipeline import analyze_all_combinations, assess_individual_risk, assess_reidentification_risk
from .risk_classification import classify_attribute_risk, dataset_level_risk


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="데이터셋의 연속형 속성을 자동으로 이산화한 뒤 재식별 위험(uniqueness/correctness)을 계산합니다."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="분석할 데이터셋 파일 경로 (csv, xlsx, tsv, json, parquet 지원). 생략하면 예시 랜덤 데이터로 실행됩니다.",
    )
    parser.add_argument("--method", default="quantile", choices=["quantile", "equal_width"], help="이산화 방식")
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="구간 개수. 지정하지 않으면 --bin-rule 규칙으로 컬럼별 자동 계산",
    )
    parser.add_argument(
        "--bin-rule",
        default="fd",
        choices=["sturges", "fd"],
        help="--n-bins 미지정 시 구간 수 자동 계산 규칙. 'fd'(Freedman-Diaconis, 기본, 분산·이상치 반영해 더 세밀함) "
        "또는 'sturges'(표본 수만 반영, 더 성긴 구간)",
    )
    parser.add_argument("--min-unique", type=int, default=20, help="연속형 판단 최소 고유값 개수")
    parser.add_argument(
        "--unique-ratio", type=float, default=0.05, help="연속형 판단 최소 고유값 비율 (0~1)"
    )
    parser.add_argument(
        "--individual-index",
        type=int,
        default=None,
        help="개별 레코드 위험도를 함께 보고 싶다면 해당 레코드의 행 번호(0부터 시작)를 지정",
    )
    parser.add_argument(
        "--no-combinations",
        action="store_true",
        help="기본으로 실행되는 전체 속성 조합 분석을 생략하고 싶을 때 지정 (전체 속성 하나로 합친 결과만 봄)",
    )
    parser.add_argument(
        "--max-combo-size",
        type=int,
        default=4,
        help="--all-combinations 사용 시, 분석할 조합의 최대 속성 개수 제한 (기본 4, 전체 다 보려면 -1 지정)",
    )
    parser.add_argument(
        "--combo-output",
        default="combination_results.csv",
        help="--all-combinations 결과를 저장할 CSV 파일명 (기본: combination_results.csv)",
    )
    return parser


def _example_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=1000),
            "income": rng.normal(4000, 1500, size=1000).round(1),
            "gender": rng.choice(["M", "F"], size=1000),
            "city": rng.choice(["Seoul", "Busan", "Incheon"], size=1000),
            "num_children": rng.integers(0, 4, size=1000),
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.dataset_path is None:
        print("[안내] 파일 경로가 지정되지 않아 예시 랜덤 데이터로 실행합니다.")
        print("       실제 데이터로 실행하려면: python -m correctmatch_toolkit.cli 데이터셋.csv\n")
        df = _example_dataframe()
    else:
        df = load_dataset(args.dataset_path)
        print(f"[로딩 완료] {args.dataset_path} — {len(df)}행 x {len(df.columns)}열\n")

    result = assess_reidentification_risk(
        df,
        method=args.method,
        n_bins=args.n_bins,
        bin_rule=args.bin_rule,
    )
    print("\nUniqueness:", result["uniqueness"])
    print("Correctness:", result["correctness"])
    print("\n이산화 결과 미리보기:")
    print(result["discretized_df"].head())

    if args.individual_index is not None:
        individual_result = assess_individual_risk(
            df,
            individual_index=args.individual_index,
            n=len(df),
            method=args.method,
            n_bins=args.n_bins,
            bin_rule=args.bin_rule,
        )
        print(f"\n[레코드 {args.individual_index}번 개별 위험도]")
        print("Individual uniqueness:", individual_result["individual_uniqueness"])
        print("Individual correctness:", individual_result["individual_correctness"])

    if not args.no_combinations:
        max_size = None if args.max_combo_size == -1 else args.max_combo_size
        combo_df = analyze_all_combinations(
            df,
            method=args.method,
            n_bins=args.n_bins,
            max_size=max_size,
            bin_rule=args.bin_rule,
        )
        for size, group in combo_df.groupby("size"):
            group_sorted = group.sort_values(["uniqueness", "correctness"], ascending=[False, False])
            print(f"\n[속성 {size}개 조합] 상위 5개 (uniqueness 높은 순, 동률이면 correctness 높은 순)")
            print(group_sorted.head(5).to_string(index=False))
        combo_df.to_csv(args.combo_output, index=False)
        print(f"\n전체 {len(combo_df)}개 조합 결과를 '{args.combo_output}'에 저장했습니다.")

        attribute_risk_df = classify_attribute_risk(combo_df)
        print("\n[속성별 위험도 등급 (개인 vs 상위조합 등장빈도 → max)]")
        print(attribute_risk_df.to_string(index=False))
        attribute_risk_df.to_csv("attribute_risk_levels.csv", index=False)
        print("\n속성별 등급 결과를 'attribute_risk_levels.csv'에 저장했습니다.")

        final_dataset_grade = dataset_level_risk(attribute_risk_df)
        print(f"\n[데이터셋 전체 식별 위험 등급] {final_dataset_grade} (속성별 등급의 최빈값 기준)")


if __name__ == "__main__":
    main()
