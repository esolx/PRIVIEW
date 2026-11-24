# -*- coding: utf-8 -*-
"""
집계 위험도 분석 엔진.

비식별 특성 조합 → 민감정보(이진분류) 예측 성능을 다수 모델로 측정하고,
RiskClassifier(이중 기준) × ModeFusion(최빈 통합)으로 조합별 위험 등급을 매긴 뒤
데이터셋 전체의 H/M/L 등급까지 산출하는 오케스트레이션 클래스입니다.
"""

from __future__ import annotations

import gc
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .attribute_risk import classify_attribute_risk, classify_attribute_risk_by_label_share
from .config import AggregationConfig
from .mode_fusion import ModeFusion
from .models import ModelFactory
from .preprocessing import convert_date_columns, detect_categorical_columns, detect_date_columns, encode_combo
from .risk_classifier import RiskClassifier
from .utils import now, write_csv
from .visualization import visualize as visualize_risk_df


class AggregationRiskAnalyzer:
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.target_col: str = config.target_col
        self.X_train = self.X_test = self.y_train = self.y_test = None

        self.models: Dict[str, Any] = {}
        self.risk_classifier = RiskClassifier(config)
        self.threshold_source = "computed"   # "fixed"(처리 전 경계) / "computed" / "partially_fixed"
        self.mode_fusion = ModeFusion(config)

        # ── 재가명처리 루프용 캐시 ───────────────────────────
        # run()이 한 번 끝나면 채워진다. reassess_after_pseudonymization()이
        # 이 두 값을 이용해 "영향받은 콤보만" 재계산한다.
        self._cached_thresholds: Dict[str, Dict[str, tuple]] = {}
        self._baseline_risk_df: Optional[pd.DataFrame] = None

    # ── 데이터 로딩 ─────────────────────────────────────
    def load_data(self):
        print(f"[{now()}] 데이터 로딩: {self.config.data_file}")
        self.df = pd.read_csv(self.config.data_file)

        if self.config.feature_cols:
            self.feature_cols = self.config.feature_cols
        else:
            self.feature_cols = [c for c in self.df.columns if c != self.target_col]

        # 날짜형(YYYY-MM-DD, YYYY-MM) 문자열 컬럼은 범주형으로 오인되면 시간
        # 순서, 간격 정보가 사라지므로, 범주형 감지보다 먼저 걸러내 "최솟값 대비
        # 경과일수"인 연속형으로 변환한다. (categorical_cols를 명시 지정한
        # 경우는 사용자가 이미 처리 방식을 확정한 것으로 보고 건드리지 않는다.)
        if not self.config.categorical_cols:
            date_cols = detect_date_columns(self.df, self.feature_cols)
            if date_cols:
                print(f"  [날짜형 컬럼 감지 - 경과일수로 변환] {date_cols}")
                self.df = convert_date_columns(self.df, date_cols)

        # 범주형(문자열) 컬럼 감지 — 명시적으로 지정 안 했으면 dtype으로 자동 판단
        if self.config.categorical_cols:
            self.categorical_cols = [c for c in self.config.categorical_cols if c in self.feature_cols]
        else:
            self.categorical_cols = detect_categorical_columns(self.df, self.feature_cols)
        if self.categorical_cols:
            high_card = [c for c in self.categorical_cols
                        if self.df[c].nunique() > self.config.high_cardinality_threshold]
            low_card = [c for c in self.categorical_cols if c not in high_card]
            if low_card:
                print(f"  [범주형 컬럼 감지 - One-Hot] {low_card}")
            if high_card:
                print(f"  [범주형 컬럼 감지 - 빈도 인코딩(고카디널리티)] {high_card}")

        # 전처리
        self.df = (self.df[self.feature_cols + [self.target_col]]
                   .replace([np.inf, -np.inf], np.nan)
                   .dropna())

        # 타깃 이진화 (이미 0/1이면 그대로)
        y = self.df[self.target_col]
        if set(y.unique()) - {0, 1}:
            print(f"  [타깃 이진화] 고유값: {sorted(y.unique())[:5]}")
            # 최빈값=0, 나머지=1
            mode_val = y.mode()[0]
            self.df[self.target_col] = (y != mode_val).astype(int)

        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.seed, stratify=y
        )
        print(f"  → {len(self.df)}행, 특성 {len(self.feature_cols)}개, "
              f"타깃: {self.target_col}")
        self._set_random_baseline()

    def _set_random_baseline(self):
        """[v25] 우연 수준 정확도. 고정 경계를 받았으면 처리 전 값, 아니면 다수 클래스 정확도."""
        fixed = (self.config.fixed_thresholds or {}).get("__random_baseline__")
        # 재평가 전용 완화폭: 고정 경계(=재평가)일 때만 안심 기준(우연 수준)을 이만큼 올린다
        self.reassess_safe_margin = float((self.config.fixed_thresholds or {}).get("__reassess_safe_margin__") or 0)
        if fixed is not None:
            value = float(fixed) + self.reassess_safe_margin
        elif self.config.random_baseline_mode == "majority":
            # 학습 데이터의 다수 클래스를 평가 데이터에 그대로 찍었을 때의 정확도
            majority = self.y_train.mode()[0]
            value = float((self.y_test == majority).mean())
        else:
            value = float(self.config.random_baseline)
        self.config.random_baseline = value
        self.risk_classifier.random_baseline = value
        print(f"  → 우연 수준 정확도(random_baseline): {value:.4f} "
              f"({'처리 전 값 고정' if fixed is not None else self.config.random_baseline_mode}"
              f"{f', 재평가 완화 +{self.reassess_safe_margin:.3f}' if self.reassess_safe_margin else ''})")

    # ── 모델 초기화 ─────────────────────────────────────
    def init_models(self):
        factory = ModelFactory(self.config)
        self.models = factory.get_all_models()
        print(f"[{now()}] 모델 {len(self.models)}개 준비: {list(self.models.keys())}")

    # ── 조합별 정확도 계산 ───────────────────────────────
    def compute_combination_accuracies(self) -> pd.DataFrame:
        """
        특성 조합별로 모든 모델에 대한 정확도(+F1) 산출
        - 조합 크기: min_combo_size ~ max_combo_size
        - 결과: combinations × models 정확도 테이블
        """
        print(f"[{now()}] 조합별 정확도 계산 시작...")
        combo_out = self.out_dir / 'combination_accuracies.csv'
        # 같은 폴더에 다시 실행하면 이전 결과 뒤에 이어 붙어 조합이 중복되므로, 시작할 때 지운다
        if combo_out.exists():
            combo_out.unlink()
        all_combos = []
        for r in range(self.config.min_combo_size,
                       min(self.config.max_combo_size + 1, len(self.feature_cols) + 1)):
            all_combos.extend(combinations(self.feature_cols, r))

        print(f"  총 조합 수: {len(all_combos):,}개")

        columns = ['Feature_Combination', 'Num_Features'] + \
                  [f'{m}_Accuracy' for m in self.models] + \
                  [f'{m}_F1' for m in self.models] + \
                  ['Mean_Accuracy', 'Mean_F1']

        buffer = []
        for i, combo in enumerate(tqdm(all_combos, desc='조합 정확도')):
            combo_key = '+'.join(combo)
            X_tr = self.X_train[list(combo)]
            X_te = self.X_test[list(combo)]
            if self.categorical_cols:
                X_tr, X_te = encode_combo(
                    X_tr, X_te, self.categorical_cols,
                    high_cardinality_threshold=self.config.high_cardinality_threshold,
                )
            row = {'Feature_Combination': combo_key, 'Num_Features': len(combo)}

            model_accs, model_f1s = [], []
            for model_name, model in self.models.items():
                try:
                    model.fit(X_tr, self.y_train)
                    y_pred = model.predict(X_te)
                    acc = accuracy_score(self.y_test, y_pred)
                    f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                except Exception as e:
                    acc, f1 = 0.0, 0.0
                row[f'{model_name}_Accuracy'] = acc
                row[f'{model_name}_F1'] = f1
                model_accs.append(acc)
                model_f1s.append(f1)

            row['Mean_Accuracy'] = float(np.mean(model_accs))
            row['Mean_F1'] = float(np.mean(model_f1s))
            buffer.append(row)

            if (i + 1) % self.config.save_every == 0:
                df_buf = pd.DataFrame(buffer)
                write_csv(df_buf, combo_out, mode='a', header=not combo_out.exists())
                buffer.clear()
                gc.collect()

        if buffer:
            df_buf = pd.DataFrame(buffer)
            write_csv(df_buf, combo_out, mode='a', header=not combo_out.exists())

        print(f"  → 저장: {combo_out}")
        return pd.read_csv(combo_out)

    # ── 위험도 분류 (이중 기준 × 10 모델) ──────────────
    def classify_risks(self, acc_df: pd.DataFrame) -> pd.DataFrame:
        """
        각 모델의 정확도 배열에 이중기준 적용 → 20개 라벨 → 최빈 통합
        """
        print(f"[{now()}] 위험도 분류 (이중기준 × 다모델)...")
        model_names = list(self.models.keys())
        n = len(acc_df)

        # 모델별 × 기준별 라벨 수집
        all_label_cols = []
        for model_name in model_names:
            acc_col = f'{model_name}_Accuracy'
            if acc_col not in acc_df.columns:
                continue
            accuracies = acc_df[acc_col].to_numpy()
            fixed = (self.config.fixed_thresholds or {}).get(model_name)

            if fixed:
                # 처리 전 평가에서 뽑은 경계를 그대로 쓴다 (경계를 다시 뽑지 않음)
                inf_th_high, inf_th_mid = map(float, fixed["inflection"])
                grad_th_high, grad_th_mid = map(float, fixed["gradient"])
                self.threshold_source = "fixed"
            else:
                # 기준 1: 변곡점 기반 / 기준 2: 변화율 기반 — threshold도 함께 뽑아서 캐시해둔다
                inf_labels, inf_th_high, inf_th_mid = self.risk_classifier.inflection_based(
                    accuracies, return_thresholds=True
                )
                grad_labels, grad_th_high, grad_th_mid = self.risk_classifier.gradient_based(
                    accuracies, return_thresholds=True
                )
                if self.config.fixed_thresholds:
                    print(f"  ! 고정 경계에 '{model_name}'이 없어 이 모델은 경계를 새로 뽑습니다")
                    self.threshold_source = "partially_fixed"
            if fixed or self.config.label_by_value:
                # 처리 전후에 같은 규칙: 정확도 값 ≥ 경계 비교로 라벨
                inf_labels = self.risk_classifier.classify_with_cached_thresholds(
                    accuracies, inf_th_high, inf_th_mid)
                grad_labels = self.risk_classifier.classify_with_cached_thresholds(
                    accuracies, grad_th_high, grad_th_mid)

            col1 = f'{model_name}_inflection'
            acc_df[col1] = inf_labels
            all_label_cols.append(col1)
            col2 = f'{model_name}_gradient'
            acc_df[col2] = grad_labels
            all_label_cols.append(col2)

            self._cached_thresholds[model_name] = {
                "inflection": (inf_th_high, inf_th_mid),
                "gradient": (grad_th_high, grad_th_mid),
            }

        print(f"  → {len(all_label_cols)}개 라벨 컬럼 생성 (목표: 20개)")

        # 최빈 통합
        acc_df['Final_Risk'] = acc_df[all_label_cols].apply(
            lambda row: self.mode_fusion.fuse(row.tolist()), axis=1
        )

        # Risk Score (최빈 우선순위 기반 수치화)
        priority = self.config.risk_priority
        acc_df['Risk_Score_Numeric'] = acc_df['Final_Risk'].map(priority)

        return acc_df

    # ── 재가명처리 루프용 빠른 재평가 (캐시된 threshold 사용) ─────
    def reassess_after_pseudonymization(
        self, updated_df: pd.DataFrame, attr: str, commit: bool = False
    ) -> Dict:
        """
        run()이 이미 한 번 끝나서 self._baseline_risk_df / self._cached_thresholds가
        채워진 상태에서, attr 하나가 가명처리(강도 사다리의 한 단계)된 후의 영향만
        빠르게 반영한다.

        - attr이 포함된 콤보만 다시 학습/예측해서 정확도를 새로 구한다
          (attr이 안 들어간 콤보는 attr 값이 바뀌어도 정확도가 그대로이므로 스킵).
        - 새 정확도는 스플라인/차분을 다시 돌리지 않고, baseline에서 캐시해 둔
          threshold(self._cached_thresholds)와 직접 비교해서 라벨만 매긴다.
        - 그 외 콤보의 라벨은 baseline 그대로 두고, 데이터셋 전체 H/M/L 비율과
          attr의 속성별 등급만 다시 집계한다 (이 두 집계 자체는 label만 보는
          가벼운 pandas 연산이라 재계산 비용이 거의 없다).

        commit=False(기본값)이면 self._baseline_risk_df를 건드리지 않고 복사본에서만
        계산한다 — 강도 사다리가 실제로 채택할지 모르는 "시험 호출"(재가명처리
        루프의 reassess_fn 자리)에는 반드시 이 기본값을 써야 한다. 특정 속성의
        기법/강도가 최종 확정된 뒤에만 commit=True로 한 번 호출해서, 이후 다른
        속성을 재평가할 때 이번 확정 결과가 baseline에 반영되도록 해야 한다
        (아니면 매 시행마다 baseline이 오염돼, 탈락한 후보의 값이 남는다).

        Args:
            updated_df: attr 컬럼이 이번 강도로 가명처리된 후의 전체 데이터프레임.
                원본과 같은 인덱스를 가져야 한다 (self.X_train/X_test의 인덱스로
                값을 끌어온다).
            attr: 이번에 재처리된 속성명.
            commit: True면 결과를 self._baseline_risk_df에 영구 반영한다.

        Returns:
            {
              "dataset_final_risk": compute_dataset_final_risk()와 동일한 dict,
              "attribute_grade": attr의 H/M/L (classify_attribute_risk 기준),
              "n_recomputed": 실제로 다시 학습/예측한 콤보 수,
            }
        """
        if self._baseline_risk_df is None or not self._cached_thresholds:
            raise RuntimeError(
                "reassess_after_pseudonymization()은 run()으로 baseline을 먼저 "
                "계산해 threshold를 캐시해 둔 뒤에만 호출할 수 있습니다."
            )

        # commit=False면 baseline을 복사해서 그 위에서만 계산 — 원본은 무사
        risk_df = self._baseline_risk_df if commit else self._baseline_risk_df.copy()
        affected_mask = risk_df["Feature_Combination"].apply(
            lambda combo_key: attr in combo_key.split("+")
        )
        affected_rows = risk_df[affected_mask]
        if affected_rows.empty:
            # 이 attr이 어떤 조합에도 없음(예: min_combo_size 이상의 조합에서
            # 아예 안 쓰인 속성) — 데이터셋 전체 결과는 그대로, attr 등급만 재조회
            dataset_final_risk = self.compute_dataset_final_risk(risk_df)
            attribute_risk_df = self.attribute_grades(risk_df)
            grade_row = attribute_risk_df[attribute_risk_df["attribute"] == attr]
            grade = grade_row["grade"].iloc[0] if not grade_row.empty else "L"
            score = float(grade_row[self.attribute_score_col].iloc[0]) if not grade_row.empty else 0.0
            return {
                "dataset_final_risk": dataset_final_risk, "attribute_grade": grade,
                "attribute_score": score, "n_recomputed": 0,
            }

        # updated_df에서 attr의 새 값을, 원래 train/test 분할과 같은 인덱스로 끌어온다
        attr_train = updated_df.loc[self.X_train.index, attr]
        attr_test = updated_df.loc[self.X_test.index, attr]

        model_names = list(self.models.keys())
        n_recomputed = 0

        for _, row in affected_rows.iterrows():
            combo = tuple(row["Feature_Combination"].split("+"))

            X_tr = self.X_train[list(combo)].copy()
            X_te = self.X_test[list(combo)].copy()
            X_tr[attr] = attr_train.values
            X_te[attr] = attr_test.values

            if self.categorical_cols:
                X_tr, X_te = encode_combo(
                    X_tr, X_te, self.categorical_cols,
                    high_cardinality_threshold=self.config.high_cardinality_threshold,
                )

            model_accs = []
            for model_name in model_names:
                model = self.models[model_name]
                try:
                    model.fit(X_tr, self.y_train)
                    y_pred = model.predict(X_te)
                    acc = accuracy_score(self.y_test, y_pred)
                except Exception:
                    acc = 0.0
                model_accs.append(acc)

                # 캐시된 threshold로 즉시 라벨 매기기 (스플라인/차분 재계산 없음)
                th_inf_high, th_inf_mid = self._cached_thresholds[model_name]["inflection"]
                th_grad_high, th_grad_mid = self._cached_thresholds[model_name]["gradient"]
                inf_label = self.risk_classifier.classify_with_cached_thresholds(
                    np.array([acc]), th_inf_high, th_inf_mid
                )[0]
                grad_label = self.risk_classifier.classify_with_cached_thresholds(
                    np.array([acc]), th_grad_high, th_grad_mid
                )[0]

                risk_df.loc[row.name, f"{model_name}_Accuracy"] = acc
                risk_df.loc[row.name, f"{model_name}_inflection"] = inf_label
                risk_df.loc[row.name, f"{model_name}_gradient"] = grad_label

            risk_df.loc[row.name, "Mean_Accuracy"] = float(np.mean(model_accs))

            all_label_cols = [f"{m}_inflection" for m in model_names] + \
                              [f"{m}_gradient" for m in model_names]
            risk_df.loc[row.name, "Final_Risk"] = self.mode_fusion.fuse(
                risk_df.loc[row.name, all_label_cols].tolist()
            )
            n_recomputed += 1

        # 라벨(H/M/L 비율, 속성별 등장빈도)만 보는 가벼운 집계 — 재학습 없음
        dataset_final_risk = self.compute_dataset_final_risk(risk_df)
        attribute_risk_df = self.attribute_grades(risk_df)
        grade_row = attribute_risk_df[attribute_risk_df["attribute"] == attr]
        grade = grade_row["grade"].iloc[0] if not grade_row.empty else "L"
        # 등급의 근거가 된 원값 그대로 (label_share: H 비율, top_n: 상위조합 등장비율) — 집계축끼리
        # 강도만 비교할 때 쓰는 값이라 다른 축과 스케일을 맞출 필요가 없다.
        score = float(grade_row[self.attribute_score_col].iloc[0]) if not grade_row.empty else 0.0

        if commit and not self.config.freeze_thresholds:
            # 확정 시점에만 threshold도 같이 갱신 — 모델 재학습 없이, 스냅샷에
            # 이미 있는(방금 갱신된 것 포함) 전체 정확도 배열에 스플라인/차분만
            # 다시 돌리는 거라 비용이 거의 없다. 이걸 매 확정마다 해두면
            # "N번마다 refresh" 같은 별도 주기 관리 없이 드리프트 문제가 사라진다.
            self._refresh_thresholds_from_snapshot(risk_df)

        return {
            "dataset_final_risk": dataset_final_risk,
            "attribute_grade": grade,
            "attribute_score": score,
            "n_recomputed": n_recomputed,
        }

    # ── v25: 속성 등급 / 경계 내보내기 ──────────────────────
    def attribute_grades(self, risk_df: pd.DataFrame) -> pd.DataFrame:
        if self.config.attribute_grade_method == "top_n":
            return classify_attribute_risk(
                risk_df, top_n=self.config.attribute_top_n,
                combo_high_cut=self.config.attribute_combo_high_cut,
            )
        return classify_attribute_risk_by_label_share(risk_df)

    @property
    def attribute_score_col(self) -> str:
        if self.config.attribute_grade_method == "top_n":
            return f"top{self.config.attribute_top_n}_appearance_ratio"
        return "high_share"

    def export_thresholds(self) -> Dict:
        """모델별 경계 {모델: {"inflection": [high, mid], "gradient": [high, mid]}} (JSON 저장용)"""
        out = {m: {k: [float(v[0]), float(v[1])] for k, v in th.items()}
               for m, th in self._cached_thresholds.items()}
        out["__random_baseline__"] = float(self.risk_classifier.random_baseline)
        return out

    def _refresh_thresholds_from_snapshot(self, risk_df: pd.DataFrame) -> None:
        """스냅샷(risk_df)에 이미 저장된 전체 콤보 정확도로 threshold를 다시 뽑는다.
        모델 재학습은 전혀 필요 없다 — inflection_based/gradient_based를
        return_thresholds=True로 다시 돌리는 것뿐이라 비용이 거의 없다."""
        for model_name in self.models:
            acc_col = f"{model_name}_Accuracy"
            if acc_col not in risk_df.columns:
                continue
            accuracies = risk_df[acc_col].to_numpy()
            _, inf_th_high, inf_th_mid = self.risk_classifier.inflection_based(
                accuracies, return_thresholds=True
            )
            _, grad_th_high, grad_th_mid = self.risk_classifier.gradient_based(
                accuracies, return_thresholds=True
            )
            self._cached_thresholds[model_name] = {
                "inflection": (inf_th_high, inf_th_mid),
                "gradient": (grad_th_high, grad_th_mid),
            }

    # ── 시각화 ──────────────────────────────────────────
    def visualize(self, risk_df: pd.DataFrame):
        print(f"[{now()}] 시각화...")
        visualize_risk_df(risk_df, self.out_dir)

    # ── 전체 실행 ────────────────────────────────────────
    def run(self) -> Dict:
        self.load_data()
        self.init_models()
        acc_df = self.compute_combination_accuracies()
        risk_df = self.classify_risks(acc_df)

        # 재가명처리 루프에서 reassess_after_pseudonymization()이 쓸 baseline 저장
        # (threshold는 classify_risks 안에서 이미 self._cached_thresholds에 채워짐)
        self._baseline_risk_df = risk_df.copy()

        # 결과 저장
        write_csv(risk_df, self.out_dir / 'final_risk_classification.csv')
        self.visualize(risk_df)

        # 요약 리포트
        summary = self._summarize(risk_df)

        # 데이터셋 전체 집계 위험도 (H/M/L) 확정
        dataset_final_risk = self.compute_dataset_final_risk(risk_df)
        summary["dataset_level_hml"] = dataset_final_risk

        # 속성별 위험 등급 (H/M/L) — 조합 위험도(정확도)만으로,
        # correctmatch_toolkit의 classify_attribute_risk와 동일 기준(상위 N개
        # 조합 등장 비율)으로 산출. 2단계(가명처리 적용)로 넘길 때 "어느
        # 조합이 위험한지"가 아니라 "어느 변수를 처리해야 하는지"가 필요하므로
        # 조합 단위 결과를 속성 단위로 환산한다.
        attribute_risk_df = self.attribute_grades(risk_df)
        write_csv(attribute_risk_df, self.out_dir / 'attribute_risk_levels.csv')
        summary["attribute_level_hml"] = attribute_risk_df.set_index("attribute")["grade"].to_dict()
        summary["attribute_level_score"] = (attribute_risk_df.set_index("attribute")[self.attribute_score_col]
                                            .astype(float).to_dict())
        summary["attribute_grade_method"] = self.config.attribute_grade_method
        summary["thresholds"] = self.export_thresholds()
        summary["threshold_source"] = getattr(self, "threshold_source", "computed")
        summary["safe_baseline"] = float(self.risk_classifier.random_baseline)          # 이 평가에서 쓴 안심 기준
        summary["reassess_safe_margin"] = float(getattr(self, "reassess_safe_margin", 0.0))

        self._save_report(summary)

        print(f"\n{'='*60}")
        print(f"집계 위험도 분석 완료")
        print(f"  총 조합: {len(risk_df):,}개")
        for level in ["고위험", "위험", "중간", "안심"]:
            cnt = (risk_df['Final_Risk'] == level).sum()
            print(f"  {level}: {cnt}개 ({cnt/len(risk_df)*100:.1f}%)")
        print(f"  최종 데이터셋 위험도(H/M/L): {dataset_final_risk['dataset_final_risk']}")
        h_attrs = [a for a, g in summary["attribute_level_hml"].items() if g == "H"]
        print(f"  속성별 H 등급: {h_attrs if h_attrs else '없음'}")
        print(f"결과 저장: {self.config.out_dir}")
        print(f"{'='*60}")
        return summary

    def _summarize(self, risk_df: pd.DataFrame) -> Dict:
        total = len(risk_df)
        dist = risk_df['Final_Risk'].value_counts().to_dict()
        above_baseline = (risk_df['Mean_Accuracy'] > self.config.random_baseline).sum()

        top_risky = (risk_df[risk_df['Final_Risk'].isin(["고위험", "위험"])]
                     .sort_values('Mean_Accuracy', ascending=False)
                     .head(10)[['Feature_Combination', 'Mean_Accuracy', 'Final_Risk']]
                     .to_dict('records'))

        return {
            "analysis_type": "집계 위험도 (다모델·이중기준)",
            "timestamp": now(),
            "config": {
                "data_file": self.config.data_file,
                "target_col": self.config.target_col,
                "n_features": len(self.feature_cols),
                "feature_cols": self.feature_cols,
                "n_models": len(self.models),
                "model_names": list(self.models.keys()),
                "combo_range": f"{self.config.min_combo_size}~{self.config.max_combo_size}"
            },
            "summary": {
                "total_combinations": total,
                "above_baseline_pct": round(above_baseline / total * 100, 2),
                "risk_distribution": dist,
                "top25_accuracy_threshold": round(
                    float(np.quantile(risk_df['Mean_Accuracy'], 0.75)), 4),
                "top10_risky_combos": top_risky
            }
        }

    # ── 데이터셋 전체 집계 위험도 (H/M/L) ─────────────────
    def compute_dataset_final_risk(self, risk_df: pd.DataFrame) -> Dict:
        """
        조합별 4단계 위험 라벨(고위험/위험/중간/안심)을 3단계로 재집계하여
        데이터셋 전체의 최종 집계 위험도(H/M/L)를 확정한다.

        규칙:
          1) 각 조합에 대해 이미 산출된 Final_Risk 라벨을 사용
          2) H = 고위험 + 위험 비율, M = 중간 비율, L = 안심 비율
          3) 세 비율 중 가장 높은 비율을 데이터셋 전체의 집계 위험도로 확정
             (비율의 최빈값이며, 사전에 정해진 고정 임계값이 아님)
        """
        total = len(risk_df)
        counts = risk_df['Final_Risk'].value_counts().to_dict()

        n_critical = counts.get("고위험", 0)
        n_high = counts.get("위험", 0)
        n_middle = counts.get("중간", 0)
        n_low = counts.get("안심", 0)

        h_count = n_critical + n_high
        m_count = n_middle
        l_count = n_low

        h_pct = round(h_count / total * 100, 1) if total else 0.0
        m_pct = round(m_count / total * 100, 1) if total else 0.0
        l_pct = round(l_count / total * 100, 1) if total else 0.0

        ratios = {"H": h_pct, "M": m_pct, "L": l_pct}
        final_label = max(ratios, key=ratios.get)

        result = {
            "raw_4label_distribution": {
                "고위험": {"n": int(n_critical),
                        "pct": round(n_critical / total * 100, 1) if total else 0.0},
                "위험":  {"n": int(n_high),
                        "pct": round(n_high / total * 100, 1) if total else 0.0},
                "중간":  {"n": int(n_middle),
                        "pct": round(n_middle / total * 100, 1) if total else 0.0},
                "안심":  {"n": int(n_low),
                        "pct": round(n_low / total * 100, 1) if total else 0.0},
            },
            "hml_aggregation": {
                "H": {"n": int(h_count), "pct": h_pct},
                "M": {"n": int(m_count), "pct": m_pct},
                "L": {"n": int(l_count), "pct": l_pct},
            },
            "dataset_final_risk": final_label
        }

        print(f"\n[{now()}] 데이터셋 전체 집계 위험도 산출")
        print(f"  고위험 {result['raw_4label_distribution']['고위험']['pct']}%"
              f"(n={result['raw_4label_distribution']['고위험']['n']}), "
              f"위험 {result['raw_4label_distribution']['위험']['pct']}%"
              f"(n={result['raw_4label_distribution']['위험']['n']}), "
              f"중간 {result['raw_4label_distribution']['중간']['pct']}%"
              f"(n={result['raw_4label_distribution']['중간']['n']}), "
              f"안심 {result['raw_4label_distribution']['안심']['pct']}%"
              f"(n={result['raw_4label_distribution']['안심']['n']})")
        print(f"  → H({h_pct}) / M({m_pct}) / L({l_pct}) "
              f"→ 최종 데이터셋 위험도 {final_label}")

        return result

    def _save_report(self, summary: Dict):
        report_path = self.out_dir / 'aggregation_risk_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[{now()}] 집계 위험도 리포트 저장: {report_path}")
