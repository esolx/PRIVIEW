"""대시보드에서 최신 집계 모듈의 시험/확정 재평가 API를 연결한다."""
from __future__ import annotations

import pandas as pd

from modules.aggregation_risk_system import AggregationConfig, AggregationRiskAnalyzer
from modules.aggregation_risk_system.preprocessing import (
    detect_date_columns, convert_date_columns, detect_categorical_columns,
)


class AggregationReassessment:
    """한 번 준비한 기준 결과로 후보를 시험하고, 채택한 값만 확정 반영한다."""

    def __init__(self, df: pd.DataFrame, target_col: str, out_dir,
                 min_combo_size: int, max_combo_size: int, fixed_thresholds: dict = None):
        """fixed_thresholds: [v25] 처리 전(1단계) 집계 경계. 주면 이 데이터로 경계를 다시 뽑지 않는다."""
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / 'baseline.csv'
        df.to_csv(csv_path, index=False)
        config = AggregationConfig(
            data_file=str(csv_path), target_col=target_col, out_dir=str(out_dir),
            min_combo_size=min_combo_size, max_combo_size=max_combo_size,
            fixed_thresholds=fixed_thresholds or None,
        )
        self.analyzer = AggregationRiskAnalyzer(config)
        # 재평가 캐시에 필요한 계산만 수행한다. 화면용 그래프를 반복 생성하지 않는다.
        self.analyzer.load_data()
        self.analyzer.init_models()
        accuracy = self.analyzer.compute_combination_accuracies()
        self.analyzer._baseline_risk_df = self.analyzer.classify_risks(accuracy).copy()

    def evaluate(self, df: pd.DataFrame, attr: str, commit: bool = False) -> dict:
        analyzer = self.analyzer
        if attr not in analyzer.feature_cols:
            # 타깃은 예측 입력이 아니므로 해당 속성의 집계 입력 위험은 없다.
            return {'attribute_grade': 'L', 'attribute_score': 0.0, 'n_recomputed': 0}
        prepared = convert_date_columns(df, detect_date_columns(df, analyzer.feature_cols))
        if not (analyzer.X_train.index.isin(prepared.index).all()
                and analyzer.X_test.index.isin(prepared.index).all()):
            raise ValueError('집계 재평가 데이터의 행 인덱스가 기준 데이터와 다릅니다.')
        old_categorical = analyzer.categorical_cols
        # 수치 구간화/억제로 문자열이 된 입력도 최신 encode_combo가 처리하도록 연결한다.
        trial_train = analyzer.X_train.copy()
        trial_train[attr] = prepared.loc[trial_train.index, attr].values
        analyzer.categorical_cols = detect_categorical_columns(trial_train, analyzer.feature_cols)
        try:
            result = analyzer.reassess_after_pseudonymization(prepared, attr, commit=commit)
            if commit:
                # 다음 속성의 후보는 직전에 확정한 속성값을 함께 사용해야 한다.
                analyzer.X_train[attr] = prepared.loc[analyzer.X_train.index, attr].values
                analyzer.X_test[attr] = prepared.loc[analyzer.X_test.index, attr].values
            return result
        finally:
            if not commit:
                analyzer.categorical_cols = old_categorical
