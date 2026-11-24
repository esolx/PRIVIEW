# -*- coding: utf-8 -*-
"""
runtime.py — 실행 스크립트·서비스가 공유하는 준비 단계
======================================================

[리팩토링] 아래 코드가 run_integrated.py / 5_run_final_pipeline.py / service.py /
compare_combo.py / score_detection.py 에 각각 복사돼 있었고, 복사본끼리 조금씩
달라져 있었다(예: service.py 만 RF 텍스트 모드가 'raw' 라 저장 RF 와 입력 표현이
어긋남, CSV 인코딩 폴백 누락). 한 곳으로 모아 모든 진입점이 같은 경로를 탄다.

    read_csv_any()          utf-8-sig → utf-8 → cp949 → euc-kr 순서로 읽기
    load_calibration()      calibration_holdout.csv → CalibrationSet (+부트스트랩 보충)
    build_detector()        rf_model.joblib 로드 + 보정까지 끝난 ExposurePipeline
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import NEGATIVE_CODE, SENSITIVE_CODES
from .features import bootstrap_training_set, sanitize_rf_text

ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "cp949", "euc-kr")

# 저장 RF(2_train_from_labels.py)는 profile_to_rf_text 표현으로 학습된다.
DEFAULT_RF_TEXT_MODE = "rf"


def read_csv_any(path: str, **kwargs) -> pd.DataFrame:
    """여러 인코딩을 차례로 시도해 CSV 를 읽는다."""
    last: Optional[Exception] = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last = e
    raise last if last else RuntimeError(f"CSV 를 읽지 못했습니다: {path}")


def has_both_binary(labels: Sequence[str]) -> bool:
    """보정에는 민감/비민감 이진 라벨이 둘 다 있어야 한다."""
    s = set(str(l) for l in labels)
    return (NEGATIVE_CODE in s) and any(l in SENSITIVE_CODES for l in s)


@dataclass
class CalibrationSet:
    texts: List[str]                    # RF 입력 형식 (holdout 'text')
    labels: List[str]
    sim_texts: Optional[List[str]]      # 유사도 단계용 원문 프로파일 ('text_raw'), 없으면 None
    supplemented: bool                  # 부트스트랩 어휘로 보충했는가
    source: str

    def __len__(self) -> int:
        return len(self.labels)


def load_calibration(path: Optional[str], required: bool = True,
                     bootstrap_n_per_cat: int = 8, bootstrap_seed: int = 7
                     ) -> CalibrationSet:
    """calibration_holdout.csv 를 읽어 보정 세트를 만든다.

    · 필수 열: text, label. 선택 열: text_raw (유사도 단계용 원문 프로파일).
    · holdout 이 없거나 한쪽 이진 클래스뿐이면 '스키마 어휘' 부트스트랩으로 보충
      (test 데이터가 아니므로 누수 없음).
    · text_raw 가 없는 구버전 holdout 이면 경고 — 유사도 단계 임계값이
      run() 과 다른 텍스트 표현으로 튜닝된다(2_train_from_labels.py 재실행 권장).
    """
    texts: List[str] = []
    labels: List[str] = []
    sim_texts: Optional[List[str]] = None
    if path and os.path.exists(path):
        cal = read_csv_any(path).fillna("")
        if not {"text", "label"}.issubset(cal.columns):
            raise ValueError(f"{path} 에는 text,label 컬럼이 필요합니다")
        texts = cal["text"].astype(str).tolist()
        labels = cal["label"].astype(str).tolist()
        if "text_raw" in cal.columns:
            raw = cal["text_raw"].astype(str).tolist()
            # 빈 칸은 RF 텍스트로 대체(행 단위 하위 호환)
            sim_texts = [r if r.strip() else t for r, t in zip(raw, texts)]
        else:
            warnings.warn(
                f"{path} 에 text_raw 열이 없습니다 → 유사도 단계(TF-IDF 코사인·SBERT) "
                "임계값이 RF 용 텍스트로 튜닝됩니다(run() 입력 표현과 불일치). "
                "2_train_from_labels.py 를 다시 실행하면 text_raw 가 함께 저장됩니다.",
                RuntimeWarning, stacklevel=2)
    elif required:
        raise FileNotFoundError(f"calibration 없음: {path} (2_train_from_labels.py 로 먼저 생성)")

    supplemented = False
    if not texts or not has_both_binary(labels):
        bt, bl = bootstrap_training_set(n_per_cat=bootstrap_n_per_cat, seed=bootstrap_seed)
        texts += [sanitize_rf_text(t) for t in bt]
        if sim_texts is not None:
            sim_texts += list(bt)
        labels += list(bl)
        supplemented = True
    return CalibrationSet(texts, labels, sim_texts, supplemented, str(path))


def build_detector(model_path: Optional[str] = "rf_model.joblib",
                   calibration_path: Optional[str] = "calibration_holdout.csv",
                   lang: str = "both",
                   rf_text_mode: str = DEFAULT_RF_TEXT_MODE,
                   dataset_specific_rules: bool = True,
                   allow_bootstrap_model: bool = False,
                   use_sim_texts: bool = True,
                   learned_dir: Optional[str] = "learned",
                   lexicon_dirs: Sequence[str] = (),
                   ):
    """학습된 RF 를 주입하고 보정까지 마친 ExposurePipeline 을 돌려준다.

    반환: (pipe, calibration_info: dict, meta: dict)
      meta = {"model_source", "calibration": CalibrationSet}
    allow_bootstrap_model=True 면 모델 파일이 없을 때 부트스트랩 학습으로 폴백
    (정확도 낮음 — 대시보드 데모용). False 면 FileNotFoundError.
    use_sim_texts=False 면 text_raw 가 있어도 쓰지 않는다(구버전과 동일한 보정 재현용).
    lexicon_dirs : 사용자 표준 어휘 폴더들 (값 사전·이름 어휘 JSON, config.load_lexicon_dir)
    learned_dir  : 검수 되먹임 결과 폴더 (review_feedback.apply_review 가 만든 것).
                   컬럼명 오버라이드 + 학습된 값 사전을 읽는다. 평가 때는 None 으로 끌 것.
    """
    from .pipeline import ExposurePipeline, PipelineConfig

    from .config import load_lexicon_dir
    from .column_mapping import ColumnMappingTable
    from .review_feedback import load_overrides

    loaded = []
    for d in lexicon_dirs or ():
        loaded += load_lexicon_dir(d)
    overrides = {}
    if learned_dir:
        loaded += load_lexicon_dir(learned_dir)
        overrides = load_overrides(learned_dir)

    cfg = PipelineConfig(lang=lang, dataset_specific_rules=dataset_specific_rules)
    if overrides:
        cfg.mapping_table = ColumnMappingTable(lang=lang, overrides=overrides,
                                               dataset_specific_rules=dataset_specific_rules)
    pipe = ExposurePipeline(cfg)

    if model_path and os.path.exists(model_path):
        pipe.load_rf(model_path, text_mode=rf_text_mode)
        model_source = f"joblib:{model_path}"
    elif allow_bootstrap_model:
        pipe.fit()
        model_source = ("bootstrap — rf_model.joblib 없음. "
                        "2_train_from_labels.py 로 학습 후 경로를 넘겨주세요.")
    else:
        raise FileNotFoundError(f"RF 모델 없음: {model_path} (2_train_from_labels.py 로 먼저 생성)")

    cal = load_calibration(calibration_path, required=not allow_bootstrap_model)
    info = pipe.calibrate(cal.texts, cal.labels,
                          sim_texts=(cal.sim_texts if use_sim_texts else None))
    return pipe, info, {"model_source": model_source, "calibration": cal,
                        "lexicon_files": loaded, "learned_overrides": len(overrides)}


def format_calibration(info: Dict) -> str:
    """calibrate() 결과를 한 줄 요약으로."""
    def _t(r):
        return "None" if r is None else f"{r.threshold:.4f}"
    return (f"tau1={_t(info.get('stage1'))} tau2={_t(info.get('stage2'))} "
            f"tau_legal={_t(info.get('legal'))} "
            f"residual={info.get('residual_n')}(양성 {info.get('residual_positives')})")
