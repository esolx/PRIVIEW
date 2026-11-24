# 노출 위험 평가 모듈

민감정보 탐지 결과와 내부 및 외부 노출 지표를 바탕으로 속성별 위험과 데이터셋의 위험 수준을 평가합니다.

## 모듈 구성

| 파일 | 역할 |
|---|---|
| `config.py`, `column_mapping.py` | 민감정보 분류체계, 컬럼명 매핑 및 탐지 규칙 |
| `preprocess.py`, `features.py` | 전처리 및 컬럼 특성 추출 |
| `tfidf_rf.py`, `sbert_sim.py` | 텍스트 유사도 및 Random Forest 기반 탐지 |
| `cassed_lite.py`, `fp_control.py` | 값 기반 보완 탐지 및 오탐 제어 |
| `risk_weight.py`, `sensitivity_ref.py` | 민감도 기반 위험 가중치 |
| `metrics.py`, `exposure_stage.py` | EMD, Coverage, 복합 민감도, 외부 노출 및 세분성 평가 |
| `pipeline.py`, `service.py` | 탐지 및 평가 파이프라인 |
| `synth.py`, `demo.py` | 합성데이터 생성 및 예제 실행 |

## 실행

프로젝트 최상위 디렉터리에서 의존성을 설치한 후 실행합니다.

```bash
pip install -r requirements.txt
python -m exposure_pipeline
```

탐지에는 컬럼명, 값 형식, 텍스트 유사도 및 분류 모델을 활용합니다. 법정 및 비법정 민감정보의 분류 결과에 따라 위험 가중치를 적용하며, 신호가 부족한 속성은 값 기반 탐지로 보완합니다. 탐지 결과와 노출 지표는 통합 위험 평가 및 재가명처리 대상 선정에 활용됩니다.

제공된 RF 모델과 보정 데이터는 평가 환경의 재현을 위해 포함되어 있습니다. 모델의 평가 결과는 입력 데이터 특성에 따라 달라질 수 있습니다.
