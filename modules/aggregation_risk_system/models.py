# -*- coding: utf-8 -*-
"""
ML 모델 팩토리.

LR, SVM, KNN, RF, ExtraTrees, MLP, PCA+RF 는 항상 포함되고,
XGBoost / LightGBM / CatBoost는 설치되어 있을 때만 추가됩니다
(설치 안 돼 있으면 경고만 출력하고 건너뜁니다).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import AggregationConfig

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[경고] xgboost 미설치 → pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[경고] lightgbm 미설치 → pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False
    print("[경고] catboost 미설치 → pip install catboost")


class ModelFactory:
    def __init__(self, config: AggregationConfig):
        self.seed = config.seed
        self.n_jobs = config.n_jobs
        self.n_estimators = config.n_estimators
        self.knn_neighbors = config.knn_neighbors
        self.lr_max_iter = config.lr_max_iter
        self.mlp_hidden_layers = config.mlp_hidden_layers
        self.mlp_max_iter = config.mlp_max_iter
        self.pca_variance_ratio = config.pca_variance_ratio

    def get_all_models(self) -> Dict[str, Any]:
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=self.lr_max_iter, random_state=self.seed),   # n_jobs는 scikit-learn 1.8부터 효과 없음(경고만 남)
            "SVM": SVC(probability=False, random_state=self.seed),
            "KNN": KNeighborsClassifier(n_neighbors=self.knn_neighbors, n_jobs=self.n_jobs),
            "RandomForest": RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=self.seed, n_jobs=self.n_jobs),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=self.n_estimators, random_state=self.seed, n_jobs=self.n_jobs),
            "MLP": MLPClassifier(
                hidden_layer_sizes=self.mlp_hidden_layers, max_iter=self.mlp_max_iter,
                random_state=self.seed),
            "PCA_RandomForest": Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=self.pca_variance_ratio, random_state=self.seed)),
                ('clf', RandomForestClassifier(
                    n_estimators=self.n_estimators, random_state=self.seed, n_jobs=self.n_jobs))
            ])
        }
        if XGB_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=self.n_estimators, random_state=self.seed,
                eval_metric='logloss', verbosity=0)
        if LGB_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=self.n_estimators, random_state=self.seed,
                verbose=-1, n_jobs=self.n_jobs)
        if CAT_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(
                n_estimators=self.n_estimators, random_state=self.seed,
                verbose=0)
        return models
