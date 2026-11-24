# -*- coding: utf-8 -*-
"""
CLI 실행 스크립트.

사용 예:
    python -m aggregation_risk_system.cli \
        --data_file data/features.csv \
        --target_col dementia \
        --out_dir results/aggregation/ \
        --min_combo_size 2 --max_combo_size 4
"""

from __future__ import annotations

import argparse

from .analyzer import AggregationRiskAnalyzer
from .config import AggregationConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='집계 위험도 분석 시스템 (다모델·이중기준)')
    parser.add_argument('--data_file', required=True, help='특성+타깃 CSV 경로')
    parser.add_argument('--target_col', required=True, help='민감정보 타깃 컬럼명')
    parser.add_argument('--out_dir', default='results/aggregation/', help='결과 저장 경로')
    parser.add_argument('--feature_cols', nargs='*', help='분석 특성 컬럼 (기본: 타깃 제외 전체)')
    parser.add_argument('--categorical_cols', nargs='*',
                        help='범주형(인코딩 대상) 컬럼 명시 지정 (기본: 문자열/카테고리 dtype 자동 감지)')
    parser.add_argument('--high_cardinality_threshold', type=int, default=10,
                        help='범주형 컬럼의 카테고리 종류가 이 값을 넘으면 One-Hot 대신 빈도 인코딩 적용 (기본 10)')
    parser.add_argument('--min_combo_size', type=int, default=2)
    parser.add_argument('--max_combo_size', type=int, default=4)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--random_baseline', type=float, default=0.5,
                        help='무작위(우연) 수준 임계값. 이 값 이하 정확도인 조합은 안심으로 분류 (기본 0.5)')
    parser.add_argument('--attribute_top_n', type=int, default=10,
                        help='속성별 등급 산출 시 "상위 조합"으로 볼 개수 (기본 10)')
    parser.add_argument('--attribute_combo_high_cut', type=float, default=0.7,
                        help='상위 조합 중 이 비율 이상 등장하면 해당 속성을 H로 분류 (기본 0.7)')
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = AggregationConfig(
        data_file=args.data_file,
        target_col=args.target_col,
        out_dir=args.out_dir,
        feature_cols=args.feature_cols or [],
        categorical_cols=args.categorical_cols or [],
        high_cardinality_threshold=args.high_cardinality_threshold,
        min_combo_size=args.min_combo_size,
        max_combo_size=args.max_combo_size,
        test_size=args.test_size,
        seed=args.seed,
        n_jobs=args.n_jobs,
        save_every=args.save_every,
        random_baseline=args.random_baseline,
        attribute_top_n=args.attribute_top_n,
        attribute_combo_high_cut=args.attribute_combo_high_cut
    )

    analyzer = AggregationRiskAnalyzer(config)
    analyzer.run()


if __name__ == "__main__":
    main()
