# correctmatch_toolkit

CorrectMatch(Rocher et al., 2019)를 파이썬에서 사용하기 위한 래퍼에, 연속형 속성
이산화(binning) 및 H/M/L 위험 등급 산출 기능을 더한 재식별 위험 평가 도구입니다.

원본 CorrectMatch.jl은 이산형 속성만 다룰 수 있어서, 이 패키지는 연속형 속성을
자동으로 구간화(binning)한 뒤 동일한 로직을 그대로 태우는 파이프라인을 제공합니다.

## 모듈 구성

기능별로 파일을 나누어, 각 파일이 하나의 책임만 갖도록 구성했습니다.

| 파일 | 역할 |
|---|---|
| `julia_bridge.py` | Julia(CorrectMatch.jl) 런타임 초기화, 파이썬↔줄리아 데이터 변환 |
| `core.py` | 원본 CorrectMatch 로직 — `uniqueness`, `correctness`, 모델 피팅(`fit_model`)/샘플링(`sample_model`), 개별 레코드 위험도 |
| `discretization.py` | 연속형 → 이산형 변환 (binning) — Sturges' rule, 컬럼 자동 탐지, quantile/equal_width/custom 구간화 |
| `pipeline.py` | 이산화 + CorrectMatch 평가를 잇는 상위 파이프라인 — 전체 데이터셋 평가, 개별 레코드 평가, 전체 속성 조합 분석 |
| `risk_classification.py` | 속성별 / 데이터셋 전체 H/M/L 위험 등급 산출 (개인 등급 + 상위 조합 등장 빈도) |
| `data_io.py` | csv/xlsx/tsv/json/parquet 데이터셋 로딩 |
| `cli.py` | 명령줄 실행 진입점 (`argparse`) |
| `__init__.py` | 패키지 공개 API (`from correctmatch_toolkit import ...`) |

의존 관계는 대략 다음과 같습니다 (아래로 갈수록 상위 레벨):

```
julia_bridge  ─┐
                ├─▶ core
                │
discretization ─┴─▶ pipeline ─▶ risk_classification
                                       │
data_io ──────────────────────────────┴─▶ cli
```

## 설치

```bash
pip install -r requirements.txt
```

Julia와 CorrectMatch.jl / DataFrames.jl은 최초 임포트 시 자동으로 설치를 시도합니다
(설치돼 있지 않다면 몇 분 정도 걸릴 수 있습니다).

## 사용법

### 파이썬에서 직접 사용

```python
import pandas as pd
from correctmatch_toolkit import assess_reidentification_risk

df = pd.read_csv("dataset.csv")
result = assess_reidentification_risk(df, method="quantile")

print(result["uniqueness"], result["correctness"])
```

### CLI로 실행

```bash
python -m correctmatch_toolkit.cli dataset.csv --method quantile
```

주요 옵션:
- `--method {quantile,equal_width}` : 이산화 방식
- `--n-bins N` : 구간 개수 (생략 시 Sturges' rule로 컬럼별 자동 계산)
- `--individual-index N` : N번째 레코드의 개별 위험도도 함께 계산
- `--no-combinations` : 전체 속성 조합 분석 생략
- `--max-combo-size N` : 조합 분석 시 최대 속성 개수 (기본 4)

## 원본 저작권

```
CorrectMatch
Copyright © 2019 Université catholique de Louvain, UCLouvain
Copyright © 2019 Imperial College London
by Luc Rocher, Julien Hendrickx, Yves-Alexandre de Montjoye
GNU General Public License v3 (또는 그 이후 버전)
```
