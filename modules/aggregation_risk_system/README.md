# aggregation_risk_system

집계 위험도 분석 시스템 (다모델, 이중기준 최빈 통합).
비식별 특성 조합(2개 이상 부분집합)이 민감정보(이진분류)를 얼마나 잘 예측해내는지
10개 ML 모델로 측정하고, 변곡점 기반 + 변화율 기반 이중 기준을 최빈 통합하여
조합별 위험 등급과 데이터셋 전체의 H/M/L 등급을 산출합니다.

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 전체 파이프라인 파라미터 (`AggregationConfig`) |
| `utils.py` | 공통 유틸 — 타임스탬프(`now`), CSV 저장(`write_csv`) |
| `preprocessing.py` | 범주형(문자열) 컬럼 자동 감지 및 조합 단위 One-Hot 인코딩 |
| `attribute_risk.py` | 조합 위험도만으로 속성별 H/M/L 산출 (correctmatch_toolkit의 classify_attribute_risk와 동일 기준) |
| `models.py` | ML 모델 팩토리 (`ModelFactory`) — LR/SVM/KNN/RF/ExtraTrees/MLP/PCA+RF 및 설치돼 있으면 XGBoost/LightGBM/CatBoost |
| `risk_classifier.py` | 이중 기준 위험도 구간화 (`RiskClassifier`) — 변곡점 기반, 변화율 기반, 분위수 폴백 |
| `mode_fusion.py` | 최빈 통합 (`ModeFusion`) — 20개(모델×기준) 라벨을 하나의 최종 등급으로 |
| `visualization.py` | 결과 시각화 — 정확도 분포, 위험도 파이차트, Top20 막대그래프, 모델별 박스플롯 |
| `analyzer.py` | 핵심 분석 엔진 (`AggregationRiskAnalyzer`) — 데이터 로딩→조합별 정확도 계산→위험도 분류→시각화→리포트 저장을 오케스트레이션 |
| `cli.py` | 명령줄 실행 진입점 (`argparse`) |
| `__init__.py` | 패키지 공개 API |

의존 관계 (아래로 갈수록 상위 레벨):

```
config ──┬─▶ models
         ├─▶ risk_classifier
         └─▶ mode_fusion
                  │
visualization ────┴─▶ analyzer ─▶ cli
```

## 설치

```bash
pip install -r requirements.txt
```

XGBoost / LightGBM / CatBoost는 선택 사항입니다 — 설치돼 있지 않으면 경고만 출력하고
나머지 7개 모델로 진행합니다.

## 사용법

### CLI로 실행

```bash
python -m aggregation_risk_system.cli \
    --data_file data/features.csv \
    --target_col dementia \
    --out_dir results/aggregation/ \
    --min_combo_size 2 --max_combo_size 4
```

### 파이썬에서 직접 사용

```python
from aggregation_risk_system import AggregationConfig, AggregationRiskAnalyzer

config = AggregationConfig(
    data_file="data/features.csv",
    target_col="dementia",
    out_dir="results/aggregation/",
    min_combo_size=2,
    max_combo_size=4,
)
analyzer = AggregationRiskAnalyzer(config)
summary = analyzer.run()
```

## 날짜형 속성 처리

`YYYY-MM-DD`, `YYYY-MM` 형식의 문자열 컬럼은 범주형 감지보다 먼저 자동으로
걸러져, **해당 컬럼 내 최솟값(가장 이른 날짜) 대비 경과일수**인 연속형(float)으로
변환됩니다. 그대로 두면 범주형(One-Hot/빈도 인코딩)으로 오인되어 시간적
순서, 간격 정보가 사라지기 때문입니다. 컬럼명은 그대로 유지되므로 조합 탐색에
영향을 주지 않습니다. `--categorical_cols`로 특정 컬럼을 직접 지정한 경우에는
이미 처리 방식을 확정한 것으로 보고 날짜 자동 변환을 건너뜁니다.

## 범주형(문자열) 속성 처리

`region`, `city`처럼 문자열로 된 범주형 컬럼은 자동으로 감지되어, 그 컬럼이 포함된
조합에 한해 인코딩된 뒤 모델에 들어갑니다. 이미 숫자 코드(`0,1,2,3`)로 인코딩된
이산형 컬럼은 범주형으로 자동 감지되지 않고 기존처럼 수치형 그대로 처리됩니다 —
강제로 범주형 취급하고 싶다면 `--categorical_cols` 옵션(또는
`AggregationConfig(categorical_cols=[...])`)으로 직접 지정하세요.

카테고리 종류 개수에 따라 인코딩 방식이 자동으로 갈립니다:

- **저카디널리티**(기본: 카테고리 10개 이하) → **One-Hot 인코딩** (정보 손실 없음)
- **고카디널리티**(카테고리 10개 초과) → **빈도 인코딩** (각 카테고리를 train 내 등장
  비율 숫자로 치환) — One-Hot을 그대로 쓰면 컬럼이 수십~수백 개로 폭증해서
  KNN, SVM, 선형모델, MLP 같은 거리/가중합 기반 모델의 성능이 떨어지고, 그 결과
  실제로는 위험한 조합이 다수결에서 "안심"으로 과소평가될 수 있기 때문입니다.

기준값은 `--high_cardinality_threshold`(기본 10)로 조정할 수 있습니다.

```bash
python -m aggregation_risk_system.cli --data_file data.csv --target_col y \
    --categorical_cols region occupation_type --high_cardinality_threshold 15
```

## 출력 파일

- `combination_accuracies.csv` — 조합별 모델별 정확도/F1
- `final_risk_classification.csv` — 조합별 최종 위험 등급(Final_Risk)까지 포함
- `risk_accuracy_distribution.png`, `risk_distribution_pie.png`, `top20_high_risk_combinations.png`, `model_accuracy_boxplot.png`
- `aggregation_risk_report.json` — 요약 리포트 (데이터셋 전체 H/M/L 등급 포함)
- `attribute_risk_levels.csv`: 속성별 위험 등급 및 조합별 등급 비율. 재가명처리 대상 선정에 활용합니다.
