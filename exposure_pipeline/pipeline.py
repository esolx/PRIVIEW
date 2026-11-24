# -*- coding: utf-8 -*-
from __future__ import annotations
import warnings

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

"""pipeline.py — 0~2단계 캐스케이드 오케스트레이션 + Track 라우팅"""

from .column_mapping import ColumnMappingTable, MappingHit
from .config import (CATEGORIES, LEGAL_CODES, LEGAL_WEIGHT, NEGATIVE_CODE,
                     REGEX_PATTERNS, SENSITIVE_CODES, VALUE_LEXICONS,
                     VALUE_VALIDATORS, is_ambiguous_term, name_evidence,
                     name_keyword_hits,
                     name_supports_category)
from .features import (ColumnProfile, bootstrap_training_set,
                       profile_dataframe, profile_to_text, profile_to_rf_text,
                       sanitize_rf_text, struct_matrix, to_binary)
from .fp_control import (ExclusionRules, ReviewFeedback, adjudicate,
                         lexicon_value_counts, value_evidence,
                         value_lexicon_evidence)

# 결정적 규칙으로 후보가 만들어진 단계 (RF, SBERT 가 뒤집지 않음)
STRONG_STAGES = ("0(map)", "2(regex)", "2(value-lexicon)")
from .preprocess import PreprocessConfig, Preprocessor
from .risk_weight import RiskWeightConfig, attribute_weight
from .sbert_sim import (PrototypeSimilarity, SbertConfig,
                        legal_from_similarity, sensitivity_from_similarity)
from .tfidf_rf import RFConfig, TfidfRandomForest
from .threshold import ThresholdResult, dual_threshold, tune_threshold

# ===  pipeline.py

"""
노출 위험도 평가 파이프라인 — 탐지/분류 + 가중치 산출.

  가명 데이터셋
    → 컬럼 프로파일링
    → 전처리 (keep / ignore)
    → [0단계] 칼럼명 매핑 앵커 (가장 강한 신호)
    → [1단계] TF-IDF 코사인 유사도            ← 싸고 결정적
    → [2단계] 정규표현식 값 스캔               ← 싸고 결정적
    → [3단계] Sentence-BERT 구제 (1·2 잔여분만, 벡터연산만)
    → [4단계] FP 방어 5층 (제외규칙 · 값검증 · 구조가드 · 증거등급 · 피드백)
    → [5단계] RandomForest 경계 보조
              · 앞 단계 미확정/경계 사례만 승격
              · 약한 TF-IDF/SBERT 오탐은 고신뢰 비민감 RF가 기각 가능
              · 강한 매핑/제외규칙은 RF가 뒤집지 않음
    → 법정 여부 판정 → Track A / Track B
    → 가중치: Track A = 1.5 고정 / Track B = 엔트로피 민감도 5단계
    → [이후] EMD · Coverage · Combo_bonus · EE연결성/PMI

왜 RF를 맨 뒤로 두나 (효율)
  싸고 결정적인 신호(코사인·정규식)로 대부분을 먼저 확정하고, 비싼
  SBERT는 잔여분에만, 학습 모델 RF는 그래도 안 잡힌 '경계' 컬럼에만
  마지막으로 관여시킨다. RF는 확정 탐지를 뒤집지 못하므로(잔여분 승격
  전용) 노출 면적이 최소화되고, 탐지력은 rf ≥ t1+0.5 조건으로 종전과
  동일하게 보존된다. 순서만 뒤로 뺀 것이라 Recall은 유지되고 RF의
  오탐이 앞단 확정 결과를 오염시키지 않는다.
"""


@dataclass
class PipelineConfig:
    policy: str = "keep"                  # "keep" | "ignore"
    lang: str = "both"                    # "ko" | "en" | "both"
    # 임계값
    min_recall_stage1: float = 0.90       # 1단계는 정밀도 여유를 둔다
    min_recall_stage2: float = 0.95       # 2단계에서 잔여분을 마저 건진다
    min_recall_legal: float = 0.99
    beta: float = 2.0
    sim_margin_min: float = 0.06         # (보강1) 코사인 오탐 방어: 0.02→0.06
    #   법정(Track A) 카테고리는 놓치면 안 되므로 더 관대한(작은) 마진을 쓴다.
    #   비법정에는 sim_margin_min을, 법정에는 sim_margin_legal을 적용해
    #   "오탐은 줄이되 법정 탐지 Recall은 지키는" 비대칭 마진.
    sim_margin_legal: float = 0.02
    require_calibration: bool = True       # 미보정 t1=0 실행을 금지
    # 사람이 검수한 정답은 여기서 exact lookup 하지 않는다.
    # corrected_label/승인 auto_label은 RF supervised training에만 사용한다.
    # RF는 19-class라 0.75~0.85 같은 binary식 확률 문턱이 지나치게 높다.
    # top-class confidence + top1-top2 margin을 함께 사용한다.
    rf_class_min_confidence: float = 0.30
    rf_class_margin_min: float = 0.08
    rf_standalone_confidence: float = 0.50
    rf_standalone_margin: float = 0.20
    # 0단계 칼럼명 매핑
    mapping_table: Optional[ColumnMappingTable] = None
    mapping_min_confidence: float = 0.60   # 이 값 이상 '강한' 매핑이면 0단계 앵커
    #   (보강) 0단계 확정은 '강한 매핑'(exact / 이름 전체를 덮는 token)만.
    #   약한 매핑(부분 token ,  affix ,  substring)은 확정하지 않고 후보로
    #   1~4단계(코사인, 정규식, SBERT, RF)에 넘겨 그들이 판단하게 한다.
    #   강함/약함 판별은 MappingHit.is_strong 이 담당한다.
    negative_anchor_confidence: float = 0.90  # 이 값 이상 비민감 매칭이면 음성 앵커
    # 특정 데이터셋 컬럼명을 하드코딩한 exact 규칙(COMPAS 약어, membership_grade).
    #   True = 기존 동작. 평가 데이터(compas_800)로 일반화 성능을 잴 때는 False.
    dataset_specific_rules: bool = True
    # 값 단위 SBERT 의미 투표 (실제 SBERT backend 일 때만 동작, 간접 근거 → 검수 큐)
    sbert_value_vote: bool = True
    value_vote_margin: float = 0.05     # 투표 1위(법정) − 비민감 유사도가 이 이상이면 검수
    value_vote_k: int = 20              # 컬럼당 임베딩할 최대 고유값 수
    # ── (보강 1, 2) SBERT 양방향 재검증 (cross-check) ──
    #   기존 SBERT는 '구제(rescue)'로 미탐만 복구했다. 여기에 이미 확정된
    #   컬럼도 SBERT로 다시 확인하는 검증 축을 더한다.
    #   ,  오탐 재확인: 민감 확정인데 SBERT가 강하게 반대(비민감)면 검수로.
    #   ,  미탐 재확인: 비민감 확정인데 SBERT가 강하게 민감이면 검수로 승격.
    #   ,  법정 재확인: 법정 legal_score가 높으면 비법정/비민감이라도 검수로.
    #   자동 확정을 '뒤집지'는 않고 검수 큐로만 보낸다(Recall 안전).
    sbert_crosscheck: bool = True
    sbert_disagree_margin: float = 0.10   # 민감확정 vs SBERT비민감 반대 강도
    sbert_missed_margin: float = 0.12     # 비민감확정 vs SBERT민감 강도(미탐)
    sbert_legal_margin: float = 0.10      # 법정 재확인 legal_score 문턱
    # 위험도 가중치 (class_weight와 무관)
    risk_config: RiskWeightConfig = field(default_factory=RiskWeightConfig)
    # 하위 설정
    rf_config: RFConfig = field(default_factory=RFConfig)
    sbert_config: SbertConfig = field(default_factory=SbertConfig)
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)
    exclusion_rules: ExclusionRules = field(default_factory=ExclusionRules)
    sample_size: int = 300
    random_state: int = 42


@dataclass
class ColumnDecision:
    column: str
    category: str
    category_ko: str
    is_sensitive: bool
    is_legal: bool
    track: str                 # "A" | "B" | "-"
    stage: str                 # "0(map)" | "1" | "2(SBERT rescue)" | "-"
    tier: str                  # CONFIRMED | PROBABLE | WEAK | NONE
    review_required: bool
    rf_proba: float
    tfidf_sim: float
    sbert_sim: float
    value_hit_rate: float
    weight: float = 0.0        # 위험도 가중치 (α_legal 또는 엔트로피 5단계)
    weight_basis: str = ""
    entropy_norm: float = 0.0
    legal_basis: str = ""
    ne_evidence: str = ""
    reasons: str = ""
    match_type: str = ""       # 0단계 매칭 유형
    matched_term: str = ""
    candidate_category: str = ""   # FP 방어 전/후 후보. 최종 category와 분리
    candidate_tier: str = ""
    candidate_stage: str = ""
    rf_label: str = ""
    rf_confidence: float = 0.0
    rf_margin: float = 0.0


@dataclass
class RiskInput:
    """다음 단계(EMD/Coverage/PMI)로 넘길 계약."""
    track_a: List[str]
    track_b: List[str]
    review: List[str]
    weights: Dict[str, float]
    decisions: List[ColumnDecision]
    thresholds: Dict[str, object]


@dataclass
class _ColumnScores:
    """run() 내부용: 컬럼별 단계 점수를 한 번에 계산해 담아 두는 묶음."""
    pre_results: list
    S1: np.ndarray            # (n_cols, n_cats) TF-IDF 코사인 유사도
    S2: np.ndarray            # (n_cols, n_cats) SBERT(또는 폴백) 유사도
    codes: List[str]          # S1/S2 의 카테고리 순서
    cs1: np.ndarray           # 1단계 민감 마진
    s2: np.ndarray            # 3단계 민감 마진
    sb_sens: np.ndarray       # SBERT 재검증 민감 마진 (= s2)
    sb_legal: np.ndarray      # SBERT 법정 마진
    rf_P: np.ndarray          # RF 클래스 확률
    rf_classes: List[str]
    rf_pred: np.ndarray       # RF top-1 라벨
    rf_p: np.ndarray          # RF 민감 확률 (= 1 − P[NON])


class ExposurePipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        pc = self.cfg.preprocess_config
        pc.policy, pc.lang = self.cfg.policy, self.cfg.lang
        self.pre = Preprocessor(pc)
        self.mapper = self.cfg.mapping_table or ColumnMappingTable(
            lang=self.cfg.lang,
            dataset_specific_rules=self.cfg.dataset_specific_rules)

        self.rf = TfidfRandomForest(self.cfg.rf_config)
        # 1단계 코사인: 어휘 수준 (TF-IDF)
        self.sim1 = PrototypeSimilarity(SbertConfig(force_backend="tfidf"))
        # 2단계 구제: 의미 수준 (SBERT)
        self.sim2 = PrototypeSimilarity(self.cfg.sbert_config)

        self.tau1: Optional[ThresholdResult] = None
        self.tau2: Optional[ThresholdResult] = None
        self.tau_legal: Optional[ThresholdResult] = None
        self.dual: Optional[Dict] = None
        self.feedback = ReviewFeedback(rules=self.cfg.exclusion_rules)
        self._fitted = False
        # 'rf'=category-hint 토큰을 제거한 RF 전용 텍스트(외부 학습 모델 기본)
        # 'proc'=pipeline.fit 방식의 전처리 텍스트.
        self._rf_text_mode = "proc"

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    def fit(self, texts: Optional[Sequence[str]] = None,
            labels: Optional[Sequence[str]] = None,
            struct: Optional[np.ndarray] = None) -> "ExposurePipeline":
        if texts is None or labels is None:
            texts, labels = bootstrap_training_set(seed=self.cfg.random_state)
            struct = None
            # 라이브러리 코드는 print 대신 warnings — 서비스/대시보드에서 로그로 잡히고
            # 필요하면 호출측이 걸러낼 수 있다.
            warnings.warn("정답셋 미제공 → 카테고리 어휘 부트스트랩으로 학습한다. "
                          "검수 라벨이 모이면 fit(texts, labels)로 재학습할 것.",
                          RuntimeWarning, stacklevel=2)
        proc = self.pre.transform(texts)
        self.rf.fit(proc, labels, struct)
        self.sim1.build()
        self.sim2.build()
        self._fitted = True
        return self

    def load_rf(self, model, text_mode: str = "rf",
                build_prototypes: bool = True) -> "ExposurePipeline":
        """외부에서 학습·저장한 RandomForest(rf_model.joblib)를 주입한다.

        재학습 없이, sim1/sim2 프로토타입만 빌드하고 곧바로 run() 가능 상태로 만든다.
        2_train_from_labels.py 로 만든 rf_model.joblib 을
        내부/외부 위험 파이프라인에 그대로 연결하기 위한 훅.

        model      : 경로(str/Path) 또는 TfidfRandomForest 객체.
        text_mode  : 주입 RF가 학습된 텍스트 표현.
                     'rf'   = category-hint를 제거한 profile_to_rf_text (기본)
                     'raw'  = 예전 모델 호환용 profile_to_text 원문
                     'proc' = 파이프라인 self.pre 전처리 텍스트 (pipeline.fit 방식)
        """
        if isinstance(model, (str, os.PathLike)):
            import joblib
            model = joblib.load(model)
        self.rf = model
        # 저장 RF가 struct(구조피처)를 안 쓰면 scaler가 None → run()에서 struct 자동 무시
        self._rf_text_mode = text_mode
        if build_prototypes:
            self.sim1.build()
            self.sim2.build()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # 점수
    # ------------------------------------------------------------------
    def _rf_sensitive_proba(self, proc, struct=None) -> np.ndarray:
        P = self.rf.predict_proba(proc, struct)
        classes = list(self.rf.classes_)
        if NEGATIVE_CODE in classes:
            return 1.0 - P[:, classes.index(NEGATIVE_CODE)]
        return P.max(axis=1)

    def _rf_legal_proba(self, proc, struct=None) -> np.ndarray:
        P = self.rf.predict_proba(proc, struct)
        idx = [i for i, c in enumerate(self.rf.classes_) if c in LEGAL_CODES]
        return P[:, idx].sum(axis=1) if idx else np.zeros(P.shape[0])

    def _stage1_score(self, proc) -> np.ndarray:
        """1단계 = TF-IDF 코사인 단독. RF는 4단계 보정으로 뒤로 뺐다.

        struct(구조 피처)는 RF 분류기 전용이라 여기서 받지 않는다. TF-IDF
        코사인은 프로파일 '텍스트'만 본다(과거 stage1=RF+코사인 시절의
        struct 인자를 제거).
        """
        return self.sim1.sensitivity_score(proc)

    def _rf_calib_threshold(self) -> float:
        """RF 경계 보정 임계값. 종전 max(rf-0.5, cs)>=t1 의 RF 몫과 동일하게
        rf_proba >= t1+0.5 를 유지 → 탐지력 보존, 순서만 뒤로."""
        if self.tau1 is None:
            raise RuntimeError("RF 보정 임계값을 계산하려면 calibrate()가 먼저 필요합니다.")
        return min(1.0, max(0.75, self.tau1.threshold + 0.5))

    def _regex_value_category(self, values, name: str = "",
                              date_guard: bool = True
                              ) -> Tuple[Optional[str], float]:
        """[2단계] 정규식 값 스캔 — 값이 카테고리 정규식에 강하게 맞으면 그 카테고리.

        name : 컬럼명. 이름 게이트가 걸린 검증기(예: GENETIC의 hugo)는
               컬럼명이 그 카테고리를 지지할 때만 값 증거로 인정된다.
        """
        vlist = [str(v) for v in values if v is not None][:300]
        date_heavy = (date_guard and vlist and
                      sum(1 for v in vlist if REGEX_PATTERNS["date"].search(v))
                      / len(vlist) >= 0.5)
        best_cat, best_rate = None, 0.0
        for cat in SENSITIVE_CODES:
            if cat not in VALUE_VALIDATORS:
                continue
            if date_heavy and cat == "CREDIT_FINANCE":
                continue                     # 날짜가 계좌번호로 오인되는 것 차단
            ns = name_supports_category(name, cat)
            ok, rate, _ = value_evidence(cat, list(values), min_hit_rate=0.6,
                                         name_supports=ns)
            if ok and rate > best_rate:
                best_cat, best_rate = cat, float(rate)
        return best_cat, best_rate

    @staticmethod
    def _lexicon_value_category(values, name: Optional[str] = None
                                ) -> Tuple[Optional[str], float, int]:
        """[2단계 값 사전] 값이 법정 카테고리 어휘로 채워져 있으면 그 카테고리.

        컬럼명이 처음 보는 이름이거나 불투명해도(q17, attr_7) 값으로 근거를 찾는다.
        """
        best = (None, 0.0, 0)
        counts = lexicon_value_counts(values)          # 값 정규화는 1회만
        for cat in VALUE_LEXICONS:
            ok, rate, distinct = value_lexicon_evidence(cat, _counts=counts, name=name)
            if ok and rate > best[1]:
                best = (cat, float(rate), int(distinct))
        return best

    def _category_from_sims(self, sims: np.ndarray, codes: Sequence[str],
                            evidence: set) -> str:
        """단일 유사도 벡터 + NER 증거로 카테고리 결정 (RF 개입 없음)."""
        legal_ev = [c for c in evidence if c in LEGAL_CODES]
        if legal_ev:
            # (수정) 법정 증거를 '무조건' 우선하지 않는다.
            #   자유서술 컬럼(직업명, 주소 등)은 값 몇 개에 들어간 의료/정치 단어
            #   때문에 HEALTH, POLITICAL 같은 법정 카테고리가 증거로 낀다. 예:
            #   'Job' 컬럼에 'Clinical psychologist'가 있어 HEALTH가 증거가 되면
            #   기존 로직은 OCCUPATION(정답, 비법정)을 통째로 버리고 HEALTH를 골랐다.
            #   → 증거 전체 중 '유사도 최상위'를 기본으로 두되, 법정 증거가 그와
            #     근소차(≤ 법정마진)일 때만 recall 보호를 위해 법정을 우선한다.
            best_legal = max(legal_ev, key=lambda c: sims[codes.index(c)])
            best_ev = max(evidence, key=lambda c: sims[codes.index(c)])
            if best_ev in LEGAL_CODES:
                return best_ev
            gap = sims[codes.index(best_ev)] - sims[codes.index(best_legal)]
            return best_legal if gap <= self.cfg.sim_margin_legal else best_ev
        neg_i = codes.index(NEGATIVE_CODE)
        sens_i = [i for i, c in enumerate(codes) if c != NEGATIVE_CODE]
        best = int(sens_i[int(np.argmax(sims[sens_i]))])
        # (보강1, 2) 비대칭 마진: 최상위 후보가 법정이면 관대한 마진,
        #   비법정이면 엄격한 마진. 법정 탐지 Recall을 지키면서 비법정
        #   코사인 오탐만 선별적으로 걷어낸다.
        margin = (self.cfg.sim_margin_legal if codes[best] in LEGAL_CODES
                  else self.cfg.sim_margin_min)
        if not evidence and float(sims[best] - sims[neg_i]) < margin:
            return NEGATIVE_CODE
        label = codes[best]
        if evidence and label not in evidence:
            label = max(evidence, key=lambda c: sims[codes.index(c)])
        return label

    def _stage2_score(self, proc) -> np.ndarray:
        return self.sim2.sensitivity_score(proc)

    # ------------------------------------------------------------------
    # 임계값 보정 — 캐스케이드 구조에 맞춰 단계별로
    # ------------------------------------------------------------------
    def calibrate(self, val_texts: Sequence[str], val_labels: Sequence[str],
                  val_struct: Optional[np.ndarray] = None,
                  sim_texts: Optional[Sequence[str]] = None) -> Dict:
        """단계별 임계값 보정.

        val_texts : RF 입력 형식 텍스트 (calibration_holdout.csv 의 ``text``).
        sim_texts : 유사도 단계(1단계 TF-IDF 코사인·3단계 SBERT)용 텍스트.
                    run() 에서 sim1/sim2 는 ``profile_to_text``(힌트 토큰 포함 원문
                    프로파일)를 받으므로, 임계값도 같은 표현으로 튜닝해야 한다.
                    holdout 의 ``text_raw`` 열을 넘기면 된다. None 이면 val_texts 를
                    그대로 쓴다(구버전 holdout 호환 — 표현 불일치 가능).
        """
        if not self._fitted:
            raise RuntimeError("fit() 먼저 호출할 것")
        if sim_texts is not None and len(sim_texts) != len(val_texts):
            raise ValueError("sim_texts 와 val_texts 의 길이가 다릅니다.")
        proc = self.pre.transform(sim_texts if sim_texts is not None else val_texts)
        # RF 는 run() 과 같은 표현을 받아야 한다('proc' 모드만 전처리 텍스트).
        rf_val = (self.pre.transform(val_texts) if self._rf_text_mode == "proc"
                  else list(val_texts))
        y = to_binary(val_labels)

        s1 = self._stage1_score(proc)
        self.tau1 = tune_threshold(y, s1, mode="recall_constrained",
                                   min_recall=self.cfg.min_recall_stage1,
                                   beta=self.cfg.beta)

        # 2단계는 **1단계 음성(잔여분) 위에서만** 튜닝한다
        residual = s1 < self.tau1.threshold
        res_out: Optional[ThresholdResult] = None
        if residual.any() and y[residual].sum() > 0:
            s2 = self._stage2_score([proc[i] for i in np.where(residual)[0]])
            # 잔여분 안에서 남은 양성을 최대한 건지는 방향
            need = min(self.cfg.min_recall_stage2, 1.0)
            res_out = tune_threshold(y[residual], s2,
                                     mode="recall_constrained",
                                     min_recall=need, beta=self.cfg.beta)
            self.tau2 = res_out
        else:
            self.tau2 = None

        # 최종 결합 점수로 회색지대 계산 (보고용 — run() 판정에는 쓰이지 않음)
        final = self._combined_score(proc)
        self.dual = dual_threshold(y, final,
                                   min_recall=self.cfg.min_recall_stage2,
                                   min_precision=0.90, beta=self.cfg.beta)

        # 법정 임계값 (보고용 — run() 판정에는 쓰이지 않고 thresholds 에만 기록)
        y_legal = np.array([1 if l in LEGAL_CODES else 0 for l in val_labels])
        if y_legal.sum() > 0:
            ls = self.sim2.legal_score(proc) + self._rf_legal_proba(rf_val, val_struct)
            self.tau_legal = tune_threshold(y_legal, ls, mode="recall_constrained",
                                            min_recall=self.cfg.min_recall_legal,
                                            beta=self.cfg.beta)

        return {"stage1": self.tau1, "stage2": self.tau2,
                "legal": self.tau_legal, "dual": self.dual,
                "residual_n": int(residual.sum()),
                "residual_positives": int(y[residual].sum())}

    def _combined_score(self, proc) -> np.ndarray:
        """캐스케이드 결과를 하나의 점수축으로 환산 (회색지대 계산용).

        1단계 통과분은 그대로, 잔여분은 2단계 점수를 1단계 임계값 아래
        구간으로 사상해 순서를 보존한다.
        """
        s1 = self._stage1_score(proc)
        if self.tau1 is None:
            return s1
        out = s1.copy()
        res = np.where(s1 < self.tau1.threshold)[0]
        if len(res) and self.tau2 is not None:
            s2 = self._stage2_score([proc[i] for i in res])
            rescued = s2 >= self.tau2.threshold
            # 구제된 것은 1단계 임계값 바로 위로 올린다
            out[res[rescued]] = self.tau1.threshold + 1e-6
        return out

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------
    def _rf_inputs(self, raw: Sequence[str], rf_docs: Sequence[str],
                   proc: Sequence[str]) -> Sequence[str]:
        """주입 RF 가 학습된 텍스트 표현(_rf_text_mode)에 맞는 입력을 고른다."""
        if self._rf_text_mode == "rf":
            return rf_docs
        if self._rf_text_mode == "raw":
            return raw
        return proc

    def _compute_scores(self, profiles: Sequence[ColumnProfile]) -> "_ColumnScores":
        """모든 컬럼의 단계별 점수를 **한 번씩만** 계산한다.

        [리팩토링] 예전 run() 은 같은 텍스트를
          · sim1: similarity + sensitivity_score           → 2회 인코딩
          · sim2: similarity + stage2 + sensitivity + legal → 4회 인코딩
          · RF  : predict_proba(전체) 2회 + predict 1회 + 컬럼마다 predict_proba 1회
        로 반복 계산했다(실제 SBERT 면 인코딩 비용 4배, RF 는 컬럼 수에 비례).
        여기서는 유사도 행렬 S 와 RF 확률 P 를 한 번 구해 모든 점수를 그 위에서 뽑는다.
        수학적으로 동일한 값이다(RF.predict = classes_[argmax P]).
        """
        raw = [profile_to_text(p) for p in profiles]
        rf_docs = [profile_to_rf_text(p) for p in profiles]
        pre_results = [self.pre.run(t) for t in raw]
        proc = [r.text for r in pre_results]
        rf_in = self._rf_inputs(raw, rf_docs, proc)

        struct, _ = struct_matrix(profiles) if self.cfg.rf_config.use_struct else (None, [])
        if struct is not None and self.rf.scaler is None:
            struct = None        # 저장 RF 가 구조피처 없이 학습됐으면 무시

        S1, codes = self.sim1.similarity(proc)
        S2, codes2 = self.sim2.similarity(proc)
        # sim1/sim2 카테고리 순서가 같다는 가정을 계약으로 고정 (어긋나면 조용히
        # 틀린 카테고리가 나오므로 즉시 에러).
        if list(codes) != list(codes2):
            raise AssertionError(
                "sim1/sim2 카테고리 순서 불일치 — 프로토타입 빌드 순서를 확인하라")
        codes = list(codes)

        P = self.rf.predict_proba(rf_in, struct)
        classes = list(self.rf.classes_)
        if NEGATIVE_CODE in classes:
            rf_p = 1.0 - P[:, classes.index(NEGATIVE_CODE)]
        else:
            rf_p = P.max(axis=1)
        rf_pred = np.asarray(classes, dtype=object)[P.argmax(axis=1)]

        s2 = sensitivity_from_similarity(S2, codes)
        return _ColumnScores(
            pre_results=pre_results, S1=S1, S2=S2, codes=codes,
            cs1=sensitivity_from_similarity(S1, codes),   # 1단계(TF-IDF 코사인)
            s2=s2,                                         # 3단계(SBERT)
            sb_sens=s2,                                    # 재검증용 = 같은 값
            sb_legal=legal_from_similarity(S2, codes),
            rf_P=P, rf_classes=classes, rf_pred=rf_pred, rf_p=rf_p,
        )

    @staticmethod
    def _prepare_frame(df):
        """컬럼명을 문자열로 통일하고 중복 컬럼명을 거부한다.

        profile_dataframe 은 name=str(col) 로 저장하므로 정수형 컬럼명이 있으면
        뒤에서 df[p.name] 이 KeyError 가 났다. 중복 컬럼명이면 매핑 dict 가
        덮어써지고 df[name] 이 DataFrame 을 돌려줘 가중치 계산이 깨진다.
        """
        cols = [str(c) for c in df.columns]
        dup = sorted({c for c in cols if cols.count(c) > 1})
        if dup:
            raise ValueError(f"중복 컬럼명이 있습니다(탐지 불가): {dup}")
        if list(df.columns) != cols:
            df = df.copy()
            df.columns = cols
        return df

    def _candidate_stage(self, i: int, p: ColumnProfile, mh: MappingHit,
                         evidence: set, sc: "_ColumnScores",
                         t1: float, t2: float) -> Tuple[str, str, Optional[str]]:
        """0~3단계 캐스케이드로 후보 (stage, category) 를 만든다.

        반환: (stage, category, map_candidate)
        사람 검수 라벨은 여기서 lookup/override 하지 않는다 — 알려진 컬럼도
        같은 탐지 경로를 통과해야 오탐/미탐 원인을 볼 수 있다.
        """
        neg_anchor = (mh.hit and mh.category == NEGATIVE_CODE
                      and mh.confidence >= self.cfg.negative_anchor_confidence)
        map_candidate = (mh.category if (mh.hit and mh.category != NEGATIVE_CODE)
                         else None)

        # 다의어 용어(성적, sect, party …)로 걸린 매핑은 맥락 단어가 없으면
        #        강한 앵커로 쓰지 않는다 (아래 단계에서 다른 근거로 다시 판단).
        if map_candidate and mh.is_strong \
                and mh.confidence >= self.cfg.mapping_min_confidence \
                and not is_ambiguous_term(p.name, mh.matched_term):
            return "0(map)", mh.category, map_candidate
        # 컬럼명에 강한 법정 어휘가 있으면(hiv_status 의 hiv) 범용 단어(status, id, code)로
        #        걸린 음성 앵커가 그것을 덮지 않는다.
        strong_legal_name = self._strong_name_category(p.name)
        if neg_anchor and not strong_legal_name:
            return "0(map-neg)", NEGATIVE_CODE, map_candidate

        # 값 사전 — 값이 종교명, 정당명, 인종, 질병명, 죄명으로 채워져 있으면
        #        컬럼명과 무관하게 결정적 근거. 철자 유사도(1단계)보다 먼저 본다.
        lcat, _lrate, _ld = self._lexicon_value_category(p.values, name=p.name)
        if lcat:
            return "2(value-lexicon)", lcat, map_candidate

        # 강한 앵커는 아니어도, 법정 카테고리를 가리키는 믿을 만한 매핑
        #        (token/약어: two_year_recid→recidivism, fngr_print_tmpl→fingerprint)은
        #        후보로 올린다. 예전엔 버려져 뒤 단계가 놓치면 조용한 미탐이 됐다.
        #        확정 여부는 FP 방어, 법정 게이트가 판단한다(대개 '법정 유지 + 검수').
        if (map_candidate in LEGAL_CODES
                and self._map_is_direct(p.name, mh, map_candidate)):
            return "0(map-weak)", map_candidate, map_candidate

        # 확장 이름 어휘(약어, 동의어, 한영)가 법정 카테고리를 '강하게' 가리키면 후보
        #        (hiv_status, pol_lean, c_charge_degree, denomination …). 매핑 사전에
        #        없어도 이름 근거로 후보를 올려, 뒤 단계가 놓쳐도 조용히 사라지지 않게.
        if strong_legal_name:
            return "0(name-term)", strong_legal_name, map_candidate

        if sc.cs1[i] >= t1:
            c = self._category_from_sims(sc.S1[i], sc.codes, evidence)
            if c != NEGATIVE_CODE:
                return "1(cosine)", c, map_candidate
        if not self.cfg.exclusion_rules.matches(p.name):
            rcat, _rrate = self._regex_value_category(p.values, name=p.name)
            if rcat:
                return "2(regex)", rcat, map_candidate
        if sc.s2[i] >= t2:
            c = self._category_from_sims(sc.S2[i], sc.codes, evidence)
            if c != NEGATIVE_CODE:
                return "3(SBERT rescue)", c, map_candidate
        return "-", NEGATIVE_CODE, map_candidate

    def _rf_assist(self, i: int, p: ColumnProfile, stage: str, cat: str,
                   verdict, excl, map_candidate, evidence: set,
                   sc: "_ColumnScores", t2: float, v_ok, v_rate,
                   legal_direct: bool = False):
        """[5단계] RF 보조 — FP 방어가 남긴 WEAK/미확정 사례 재심.

        legal_direct : 현재 후보가 법정이고 이름·값 직접 근거가 있는가. 이 경우 RF 의
                       '비민감' 판단만으로 지우지 않는다 (법정 미탐 방지, 검수로 남김).

        강한 map/regex 확정과 제외규칙 적중은 RF 가 뒤집지 않는다.
        반환: (verdict, stage, cat, v_ok, v_rate, rf_label, rf_conf, rf_margin)
        """
        cls = sc.rf_classes
        P_i = sc.rf_P[i]
        rf_label = str(sc.rf_pred[i])
        rf_conf = float(P_i[cls.index(rf_label)]) if rf_label in cls else 0.0
        sorted_p = np.sort(P_i)
        rf_margin = float(sorted_p[-1] - sorted_p[-2]) if len(sorted_p) >= 2 else rf_conf
        rf_reliable = (rf_conf >= self.cfg.rf_class_min_confidence
                       and rf_margin >= self.cfg.rf_class_margin_min)
        rf_standalone = (rf_conf >= self.cfg.rf_standalone_confidence
                         and rf_margin >= self.cfg.rf_standalone_margin)
        rf_info = (rf_label, rf_conf, rf_margin)

        strong_rule = stage in STRONG_STAGES
        if strong_rule or excl is not None:
            return (verdict, stage, cat, v_ok, v_rate) + rf_info

        candidate_cat = verdict.category
        unresolved = (not verdict.auto_confirm) or candidate_cat == NEGATIVE_CODE
        sbert_top = sc.codes[int(np.argmax(sc.S2[i]))]
        Verdict = type(verdict)

        # A) FP 방어가 WEAK/미확정으로 낮춘 후보는 RF 가 재심한다.
        #    19-class top1 확률과 top1-top2 margin 을 함께 본다.
        if unresolved and rf_reliable:
            # 법정 후보에 직접 근거(이름, 값)가 있으면, RF 가 다른 카테고리를 제안해도
            # 후보를 바꾸지 않는다 (간접 근거가 직접 근거를 덮지 않음). 검수로 남김.
            if (legal_direct and rf_label != candidate_cat
                    and rf_label != NEGATIVE_CODE):
                verdict = Verdict(
                    candidate_cat, verdict.tier, False,
                    list(verdict.reasons) +
                    [f"RF={rf_label}({rf_conf:.2f})이나 법정 직접 근거가 있어 {candidate_cat} 유지 → 검수"],
                    demoted_from=None,
                )
                return (verdict, stage, cat, v_ok, v_rate) + rf_info
            if rf_label == NEGATIVE_CODE and legal_direct:
                verdict = Verdict(
                    candidate_cat, verdict.tier, False,
                    list(verdict.reasons) +
                    [f"RF 는 비민감({rf_conf:.2f})이나 법정 직접 근거가 있어 유지 → 검수"],
                    demoted_from=None,
                )
                return (verdict, stage, cat, v_ok, v_rate) + rf_info
            if rf_label == NEGATIVE_CODE:
                verdict = Verdict(
                    NEGATIVE_CODE, "NONE", True,
                    list(verdict.reasons) + [f"RF 보조 비민감 확정({rf_conf:.2f})"],
                    demoted_from=(candidate_cat if candidate_cat != NEGATIVE_CODE else None),
                )
                return (verdict, "5(RF assist-neg)", NEGATIVE_CODE, v_ok, v_rate) + rf_info
            if rf_label in SENSITIVE_CODES:
                name_sup = name_supports_category(p.name, rf_label)
                rv_ok, rv_rate, _ = value_evidence(rf_label, p.values,
                                                   name_supports=name_sup)
                sbert_agrees = bool(sbert_top == rf_label and sc.s2[i] >= t2)
                name_hit = (rf_label in evidence) or (map_candidate == rf_label) or name_sup
                # RF 양성은 단독으로 무조건 확정하지 않는다. 같은 카테고리를 지지하는
                # 이름/매핑/SBERT/값 근거가 하나라도 있으면 확정, 없으면 검수 후보.
                support = name_hit or rv_ok or sbert_agrees
                rv = adjudicate(
                    category=rf_label, name_hit=name_hit,
                    value_ok=rv_ok, value_rate=rv_rate, rf_hit=True,
                    sbert_hit=sbert_agrees,
                    n_unique=p.n_unique, mean_len=p.struct.get("mean_len", 0.0),
                    exclusion=None, rules=self.cfg.exclusion_rules,
                )
                if (support and rv.auto_confirm) or rf_standalone:
                    # 사람이 고친 양성 패턴을 RF 가 매우 강하게 학습한 경우 독립
                    # 근거가 약해도 보조 단계에서 구제할 수 있다.
                    verdict = Verdict(
                        rf_label, "CONFIRMED" if rf_standalone else rv.tier, True,
                        list(rv.reasons) +
                        ([f"RF 고신뢰 단독 구제(conf={rf_conf:.2f}, margin={rf_margin:.2f})"]
                         if rf_standalone and not support else []),
                        demoted_from=None,
                    )
                    return (verdict, "5(RF assist)", rf_label, rv_ok, rv_rate) + rf_info
                verdict = Verdict(
                    rf_label, "WEAK", False,
                    list(rv.reasons) +
                    [f"RF={rf_label}({rf_conf:.2f})이나 독립 근거 부족 → 검수"],
                    demoted_from=None,
                )
                return (verdict, "5(RF candidate)", rf_label, v_ok, v_rate) + rf_info
            return (verdict, stage, cat, v_ok, v_rate) + rf_info

        # B) cosine/SBERT 가 만든 약한 양성은 RF 가 고신뢰 NON 이면 기각.
        if stage in {"1(cosine)", "3(SBERT rescue)"} \
                and rf_label == NEGATIVE_CODE and rf_reliable and not legal_direct:
            verdict = Verdict(
                NEGATIVE_CODE, "NONE", True,
                list(verdict.reasons) +
                [f"RF 비민감 재심(conf={rf_conf:.2f}, margin={rf_margin:.2f}) → 약한 유사도 판정 기각"],
                demoted_from=verdict.category,
            )
            return (verdict, "5(RF reject)", NEGATIVE_CODE, v_ok, v_rate) + rf_info
        return (verdict, stage, cat, v_ok, v_rate) + rf_info

    @staticmethod
    def _strong_name_category(name: str) -> Optional[str]:
        """컬럼명이 '강하게' 가리키는 법정 카테고리 (여럿이면 가장 긴 키워드 기준)."""
        best, best_len = None, 0
        for lc in LEGAL_CODES:
            if name_evidence(name, lc) != "strong":
                continue
            L = max((len(k) for k in name_keyword_hits(name, lc)), default=0)
            if L > best_len:
                best, best_len = lc, L
        return best

    @staticmethod
    def _map_is_direct(name: str, mh: MappingHit, cat: str) -> bool:
        """매핑 결과를 '컬럼명 직접 근거'로 쓸 수 있나.
        exact·token·override 만 인정. affix·substring 은 철자 겹침(trace→race,
        healthy→health)이라 제외 — 단 2글자 이상 한글 부분일치(노조가입→노조)는 인정.
        다의어 용어는 맥락 단어가 있어야 인정."""
        if not (mh.hit and mh.category == cat and mh.matched_term):
            return False
        if is_ambiguous_term(name, mh.matched_term):
            return False
        if mh.match_type in ("exact", "token", "override"):
            return True
        if mh.match_type == "substring":
            t = str(mh.matched_term)
            return (not t.isascii()) and len(t) >= 2
        return False

    def _legal_direct_evidence(self, p: ColumnProfile, cat: str, mh: MappingHit,
                               map_candidate: Optional[str], stage: str
                               ) -> Tuple[bool, str]:
        """[v9.2] 법정 후보에 '직접 근거'가 있는가 → (있음, 근거 설명).

        직접 근거 = 이 컬럼 자체에 대한 사실:
          · 결정적 규칙 단계 (강한 매핑 · 값 정규식 · 값 사전)
          · 컬럼명 키워드 (다의어는 맥락 단어가 있을 때만)
          · 매핑 용어 (다의어 제외)
          · 값 형식 검증 통과 (ICD·형량 등) / 값 사전 일치
        간접 근거(철자 유사도 1단계, SBERT 유사도, RF 예측)만 있는 법정 후보는
        최종 법정으로 내보내지 않고 검수 큐로 보낸다.
        """
        if cat not in LEGAL_CODES:
            return True, ""
        if stage in STRONG_STAGES:
            return True, f"결정적 규칙({stage})"
        if name_evidence(p.name, cat) == "strong":
            return True, "컬럼명 키워드"
        if self._map_is_direct(p.name, mh, cat):
            return True, f"매핑 용어({mh.matched_term})"
        ns = name_supports_category(p.name, cat)
        ok, rate, key = value_evidence(cat, p.values, name_supports=ns)
        if ok and not np.isnan(rate):
            # ICD 형식(A01, B02 …)은 부서코드, 등급코드와 형식이 같다. 이름이 건강을
            # 가리키지 않으면 서로 다른 코드가 충분히 많을 때만 진단코드로 본다.
            if key == "icd" and not ns and p.n_unique < 5:
                pass
            else:
                return True, f"값 형식({key} {rate:.2f})"
        lok, lrate, ld = value_lexicon_evidence(cat, p.values, name=p.name)
        if lok:
            return True, f"값 사전({lrate:.2f}, {ld}종)"
        return False, ""

    @staticmethod
    def _final_category(verdict, legal_direct: bool = True) -> str:
        """candidate 와 final 을 명시적으로 분리한다.

        · 비법정: auto_confirm=False 면 최종 민감으로 내보내지 않고 검수 후보로.
        · 법정(Track A) [v9.2]: 이름·값 '직접 근거'가 있을 때만 최종 법정.
            - 직접 근거 있음 + 오탐 근거로 강등되지 않음 → 법정 유지
              (확정이 아니면 검수 필요 표시) — 미탐 방지
            - 직접 근거 없음(유사도·RF 같은 간접 근거만) → 최종 NON + 검수 후보
              — 철자 유사도(grade≈trade) 같은 오탐 방지
        """
        if verdict.category in LEGAL_CODES:
            if legal_direct and verdict.demoted_from is None:
                return verdict.category
            return NEGATIVE_CODE
        if verdict.auto_confirm:
            return verdict.category
        return NEGATIVE_CODE

    def _sbert_crosscheck(self, i: int, sensitive: bool, legal: bool, v_ok, v_rate,
                          sc: "_ColumnScores", legal_direct: bool = False
                          ) -> Tuple[bool, bool, str]:
        """SBERT 양방향 재검증 (강한 0단계 앵커는 호출하지 않는다).

          · 민감 확정인데 SBERT 가 강하게 반대 → 비법정: 강등 / 법정: 값 근거
            없을 때만 강등, 있으면 유지+검수
          · 비민감인데 SBERT 가 강하게 민감 → 검수
          · 법정 아님인데 SBERT legal_score 높음 → 검수
        '강등'은 실제 SBERT backend 일 때만. TF-IDF 폴백은 코사인과 신호가 겹쳐
        (순환) 강등 없이 검수 플래그만 단다.
        반환: (demote_to_negative, review_flag, note)
        """
        sbert_real = (getattr(self.sim2, "backend", "") == "sbert")
        sb_top = sc.codes[int(np.argmax(sc.S2[i]))]
        sb_sens, sb_legal = sc.sb_sens[i], sc.sb_legal[i]
        demote = flag = False
        note = ""
        if sensitive and sb_sens <= -self.cfg.sbert_disagree_margin:
            if not sbert_real:
                flag = True
                note = f"SBERT(폴백) 반대 margin={sb_sens:.2f} → 강등 보류·검수"
            elif not legal:
                demote = True
                note = f"SBERT 반대(margin={sb_sens:.2f}) → 비법정 자동 강등"
            elif legal_direct:
                flag = True
                note = f"SBERT 반대(margin={sb_sens:.2f})하나 법정 직접 근거 있음 → 유지+검수"
            elif not v_ok:
                demote = True
                note = f"SBERT 반대 ∧ 값근거 없음 → 법정 강등(margin={sb_sens:.2f})"
            else:
                flag = True
                note = f"SBERT 반대하나 값근거 있음(hit={v_rate:.2f}) → 법정 유지+검수"
        elif (not sensitive) and sb_sens >= self.cfg.sbert_missed_margin:
            flag = True
            note = f"SBERT 민감 의심(margin={sb_sens:.2f}, top={sb_top}) → 검수"
        # 법정 의심: 강등되지 않았고 아직 법정이 아닌데 SBERT legal_score 높음
        still_legal = legal and not demote
        if (not demote) and (not still_legal) and sb_legal >= self.cfg.sbert_legal_margin:
            flag = True
            note = ((note + " | " if note else "")
                    + f"SBERT 법정 의심(legal_score={sb_legal:.2f}, top={sb_top}) → 검수")
        return demote, flag, note

    def run(self, df, original_df=None) -> RiskInput:
        if not self._fitted:
            raise RuntimeError("fit() 또는 load_rf() 먼저 호출할 것")
        if self.cfg.require_calibration and self.tau1 is None:
            raise RuntimeError(
                "Pipeline is not calibrated. 검증셋으로 "
                "pipe.calibrate(val_texts, val_labels)를 호출한 뒤 run() 하세요. "
                "최종 test/COMPAS/adult 데이터로 calibrate하면 안 됩니다."
            )
        df = self._prepare_frame(df)
        if original_df is not None:
            original_df = self._prepare_frame(original_df)
        profiles = profile_dataframe(df, sample_size=self.cfg.sample_size,
                                     seed=self.cfg.random_state)
        sc = self._compute_scores(profiles)
        # ---- [0단계] 칼럼명 매핑 테이블 (가장 먼저, 가장 강한 신호) ----
        maps: Dict[str, MappingHit] = self.mapper.map_columns([p.name for p in profiles])

        # 값 단위 의미 투표 — 실제 SBERT 일 때만 (TF-IDF 폴백은 철자 비교라 의미 없음)
        vote, vote_codes = None, None
        if self.cfg.sbert_value_vote and getattr(self.sim2, "backend", "") == "sbert":
            vote, vote_codes = self.sim2.value_vote([p.values for p in profiles],
                                                    k=self.cfg.value_vote_k)

        t1 = self.tau1.threshold
        t2 = self.tau2.threshold if self.tau2 else np.inf
        t_rf = self._rf_calib_threshold()                # 앵커 충돌 판단용 RF 임계값

        decisions: List[ColumnDecision] = []
        for i, p in enumerate(profiles):
            evidence = set(l for l in sc.pre_results[i].ne_labels if l in CATEGORIES)
            mh = maps[p.name]

            # ---- 0~3단계: 후보 생성 (map → 값사전 → cosine → regex → SBERT) ----
            stage, cat, map_candidate = self._candidate_stage(i, p, mh, evidence, sc, t1, t2)
            anchored = (stage == "0(map)")               # 강한 양성 앵커

            # ---- [4단계] FP 방어 5층 ----
            # 제외 규칙은 0단계 '강한' 확정(앵커)만 뒤집지 못한다.
            excl = None if anchored else self.cfg.exclusion_rules.matches(p.name)
            # 강한 법정 이름 어휘가 후보 카테고리를 가리키면 범용 단어 제외규칙은 적용하지 않음
            if excl is not None and cat in LEGAL_CODES and name_evidence(p.name, cat) == "strong":
                excl = None
            v_ok, v_rate, _ = value_evidence(
                cat, p.values, name_supports=name_supports_category(p.name, cat))
            # membership_grade 는 데이터셋 스키마상 신용등급. Gold/Silver 같은 등급값은
            # 금융 정규식과 형식이 달라 0단계 앵커를 값 형식으로 다시 강등하지 않는다.
            if (mh.matched_term == "membership_grade_credit"
                    and cat == "CREDIT_FINANCE" and anchored):
                v_ok, v_rate = True, float("nan")
            # 값 사전으로 잡힌 후보는 값 자체가 근거 → 형식 검증기(ICD 등)로 다시 강등하지 않음
            if stage == "2(value-lexicon)":
                _ok, _lr, _ = value_lexicon_evidence(cat, p.values, name=p.name)
                v_ok, v_rate = True, float(_lr)
            # FP 방어 신호는 '현재 후보 카테고리와 같은 신호'만 센다. SBERT 도 top-1
            # 이면서 calibration 임계값을 통과해야 지지 신호로 인정한다.
            cat_name_hit = ((cat in evidence) or (map_candidate == cat)
                            or name_supports_category(p.name, cat))
            cat_sbert_hit = (cat != NEGATIVE_CODE
                             and sc.codes[int(np.argmax(sc.S2[i]))] == cat
                             and sc.s2[i] >= t2)
            verdict = adjudicate(
                category=cat if stage != "-" else NEGATIVE_CODE,
                name_hit=cat_name_hit,
                value_ok=v_ok, value_rate=v_rate,
                rf_hit=False, sbert_hit=cat_sbert_hit,
                n_unique=p.n_unique, mean_len=p.struct.get("mean_len", 0.0),
                exclusion=excl, rules=self.cfg.exclusion_rules,
                # 0단계 강한 매핑 앵커, 2단계 값 정규식은 결정적 규칙 → 확정 경로
                strong_rule=(stage in STRONG_STAGES),
            )

            # ---- [5단계] RF 보조 (법정 직접 근거가 있으면 RF 비민감만으로 지우지 않음) ----
            pre_direct, _ = self._legal_direct_evidence(p, verdict.category, mh,
                                                        map_candidate, stage)
            (verdict, stage, cat, v_ok, v_rate,
             rf_label, rf_conf, rf_margin) = self._rf_assist(
                i, p, stage, cat, verdict, excl, map_candidate, evidence,
                sc, t2, v_ok, v_rate,
                legal_direct=(verdict.category in LEGAL_CODES and pre_direct))

            candidate_cat, candidate_tier = verdict.category, verdict.tier
            # ---- 법정 최종 게이트: 이름, 값 직접 근거가 있어야 최종 법정 ----
            legal_direct, direct_why = self._legal_direct_evidence(
                p, candidate_cat, mh, map_candidate, stage)
            final_cat = self._final_category(verdict, legal_direct)
            sensitive = final_cat in SENSITIVE_CODES
            legal = sensitive and final_cat in LEGAL_CODES

            # 0단계 음성 앵커인데 코사인/RF 가 강하게 민감 → 사람에게 넘긴다
            anchor_conflict = (stage == "0(map-neg)"
                               and (sc.cs1[i] >= t1 or sc.rf_p[i] >= t_rf))

            # ---- SBERT 재검증 게이트 (강한 앵커는 신뢰, 나머지만) ----
            sbert_flag, sbert_note = False, ""
            if self.cfg.sbert_crosscheck and not anchored:
                demote, sbert_flag, sbert_note = self._sbert_crosscheck(
                    i, sensitive, legal, v_ok, v_rate, sc,
                    legal_direct=(legal and legal_direct))
                if demote:
                    final_cat, sensitive, legal = NEGATIVE_CODE, False, False

            # ---- 위험도 가중치 ----
            vals = df[p.name].dropna().tolist()
            orig = (original_df[p.name].dropna().tolist()
                    if original_df is not None and p.name in original_df.columns
                    else None)
            aw = attribute_weight(p.name, final_cat, vals,
                                  cfg=self.cfg.risk_config, original=orig)

            unresolved_candidate = (not verdict.auto_confirm and candidate_cat != NEGATIVE_CODE)
            # 법정 후보였는데 직접 근거가 없어 최종에서 빠진 경우 → 반드시 검수 (조용한 미탐 방지)
            legal_gate_drop = (candidate_cat in LEGAL_CODES and not legal
                               and verdict.demoted_from is None)
            # 컬럼명이 법정 카테고리를 강하게 가리키는데 최종 비민감 → 검수
            strong_name_conflict = (not sensitive) and any(
                name_evidence(p.name, lc) == "strong" for lc in LEGAL_CODES)
            if legal and not verdict.auto_confirm:
                status_note = f" | 법정 후보(직접 근거: {direct_why}) → 최종 법정 유지, 검수 필요"
            elif legal_gate_drop:
                status_note = (" | 법정 후보이나 이름·값 직접 근거 없음(유사도·RF 등 간접 근거만)"
                               " → 최종 NON_SENSITIVE, 검수 후보")
            elif unresolved_candidate and not sensitive:
                status_note = " | WEAK/미확정 후보 → 최종 NON_SENSITIVE, 검수 후보"
            elif unresolved_candidate:
                status_note = " | 미확정 후보 → 검수 필요"
            else:
                status_note = ""
            if strong_name_conflict:
                status_note += " | 컬럼명이 법정 카테고리를 가리키나 최종 비민감 → 검수"
            vote_flag = False
            if vote is not None and not legal:
                row = vote[i]
                top = vote_codes[int(np.argmax(row))]
                vm = float(row.max() - row[vote_codes.index(NEGATIVE_CODE)])
                if top in LEGAL_CODES and vm >= self.cfg.value_vote_margin:
                    vote_flag = True
                    status_note += f" | SBERT 값 투표: {top} 의심(margin={vm:.2f}) → 검수"
            meta = CATEGORIES.get(final_cat, CATEGORIES[NEGATIVE_CODE])
            decisions.append(ColumnDecision(
                column=p.name,
                category=final_cat,
                category_ko=meta.name_ko,
                is_sensitive=sensitive,
                is_legal=legal,
                track="A" if legal else ("B" if sensitive else "-"),
                stage=stage if sensitive else "-",
                match_type=mh.match_type,
                matched_term=mh.matched_term,
                tier=verdict.tier,
                review_required=(unresolved_candidate or anchor_conflict or sbert_flag
                                 or legal_gate_drop or strong_name_conflict or vote_flag),
                rf_proba=float(sc.rf_p[i]),
                tfidf_sim=float(sc.cs1[i]),
                sbert_sim=float(sc.s2[i]),
                value_hit_rate=float(v_rate),
                weight=aw.weight,
                weight_basis=aw.basis,
                entropy_norm=aw.h_norm,
                legal_basis=meta.basis if legal else "",
                ne_evidence=",".join(sorted(evidence)),
                reasons=" | ".join(verdict.reasons) + status_note
                        + (f" | {sbert_note}" if sbert_note else ""),
                candidate_category=candidate_cat,
                candidate_tier=candidate_tier,
                candidate_stage=stage,
                rf_label=rf_label,
                rf_confidence=rf_conf,
                rf_margin=rf_margin,
            ))

        return RiskInput(
            track_a=[d.column for d in decisions if d.track == "A"],
            track_b=[d.column for d in decisions if d.track == "B"],
            review=[d.column for d in decisions if d.review_required],
            weights={d.column: d.weight for d in decisions if d.is_sensitive},
            decisions=decisions,
            thresholds={"tau1": t1, "tau2": (None if t2 == np.inf else t2),
                        "tau_legal": self.tau_legal.threshold if self.tau_legal else None,
                        "legal_weight": LEGAL_WEIGHT},
        )

    # ------------------------------------------------------------------
    def to_frame(self, result: RiskInput):
        import pandas as pd
        return pd.DataFrame([d.__dict__ for d in result.decisions]) \
                 .sort_values(["track", "weight", "sbert_sim"],
                              ascending=[True, False, False])

    def apply_feedback(self, corrections: Dict[str, str],
                       profiles: Optional[Sequence[ColumnProfile]] = None,
                       refit: bool = False) -> Dict:
        """검수 결과 반영. {컬럼명: 정답카테고리}"""
        for col, cat in corrections.items():
            self.feedback.record(col, cat)
        info = {"exclusions": len(self.feedback.rules.exact),
                "corrections": len(self.feedback.corrections)}
        if refit and profiles:
            tx, ly = self.feedback.to_training_pairs(profiles)
            if tx:
                base_x, base_y = bootstrap_training_set(seed=self.cfg.random_state)
                # 검수 라벨은 부트스트랩보다 5배 가중 (중복 삽입)
                self.fit(list(base_x) + list(tx) * 5, list(base_y) + list(ly) * 5)
                info["refit_on"] = len(tx)
        return info


# ---------------------------------------------------------------------------
def compare_baselines(texts, labels, val_texts, val_labels,
                      min_recall: float = 0.95, lang: str = "both") -> Dict:
    """동일 조건에서 keep / ignore 전처리만 바꿔 성능 차이를 측정."""
    out = {}
    for policy in ("keep", "ignore"):
        cfg = PipelineConfig(policy=policy, lang=lang,
                             min_recall_stage1=min_recall,
                             min_recall_stage2=min_recall)
        cfg.rf_config.use_struct = False
        pipe = ExposurePipeline(cfg).fit(texts, labels)
        cal = pipe.calibrate(val_texts, val_labels)
        cv = pipe.rf.evaluate_cv(pipe.pre.transform(texts), labels)
        out[policy] = {"cv": cv, "stage1": cal["stage1"], "stage2": cal["stage2"],
                       "gray_zone_rate": cal["dual"]["gray_zone_rate"]}
    return out



