# 재가명처리 실험 실행 (코랩)

## 1. 준비 셀

```python
!pip install -q -r /content/PRIVIEW/requirements.txt
import sys
sys.path.insert(0, "/content/PRIVIEW")
%cd /content/PRIVIEW
```

- 목적 외 사용 모듈을 쓰지 않으므로 Java(konlpy)는 필요 없습니다.
- 식별 위험은 Julia(CorrectMatch)가 필요합니다. `juliacall`이 처음 실행될 때 Julia를 설치하므로 시간이 걸립니다.

## 2. 시드 셀

```python
from modules.pseudonymization_pipeline import set_base_seed
import experiment

set_base_seed(42)
```

## 3. 샘플 확인 셀

```python
out = experiment.run_experiment(
    "experiments/sample_small.csv", target_col="two_year_recid",
    out_dir="runs/sample_seed_42",
    id_max_combo=2, agg_min_combo=1, agg_max_combo=2,
)
out["n_reassessment_errors"]   # 0이어야 정상. 0이 아니면 strength_search_log.csv의 *_error 열 확인
```

## 4. 본 실험 (시드 여러 개)

```python
baseline = experiment.run_baseline("내데이터.csv", "타깃컬럼", "runs/main/baseline")

for s in [42, 0, 1, 2, 3]:
    set_base_seed(s)
    experiment.run_repseudonymization(baseline, f"runs/main/seed_{s}")
```

### 목표 등급 및 반복 횟수

```python
# 저위험 목표, 기본 반복 횟수
experiment.run_repseudonymization(baseline, "runs/main/seed_42_L")
# 중위험 목표, 1회 실행
experiment.run_repseudonymization(baseline, "runs/main/seed_42_M", target_level="M", max_rounds=1)
```

- 라운드 2부터는 `repseudonymization/round_2/`, `round_3/`에 강도별 기록이 따로 저장됩니다.
- `repseudonymization/rounds_summary.json`: 라운드별 처리 컬럼, 통합 등급, 멈춘 이유, TV 유용성(표시용)
- `repseudonymization/utility_tv.csv`: 처리 컬럼별 TV와 허용 폭(재표본 95% × √2)

## 5. 결과 보기

```python
import pandas as pd
log = pd.read_csv("runs/main/seed_42/repseudonymization/strength_search_log.csv")
log[["column", "technique", "rung", "utility_loss", "achieved", "final_adopted", "has_error"]]
```

## 결과 폴더

```
runs/main/
  baseline/                         처리 전 평가 (integrated_result.json)
  seed_42/
    experiment_config.json          기본 시드, 모듈 시드, 설정, 시작/종료 시각
    repseudonymized.csv             최종 재가명처리본
    risk_comparison.json            처리 전후 비교
    after/                          처리 후 평가 원자료
    repseudonymization/
      strength_search_log.csv       시도별 기록 (컬럼 × 기법 × 단계)
      candidate_comparison.csv      기법별 비교
      strength_search_log.json
      summary.json
      trials/<컬럼>/<기법>__<단계>.csv            시도별 처리본
      reassessment/<컬럼>/<기법>__<단계>/         시도별 재평가 결과 (+ error.log)
```
