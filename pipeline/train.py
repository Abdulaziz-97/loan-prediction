from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


@dataclass(frozen=True)
class SplitInfo:
    cutoff: Any
    n_train: int
    n_test: int


def time_split(df, *, date_col: str, test_fraction: float = 0.2) -> tuple[list[int], list[int], SplitInfo]:
    """
    Time-based split by `date_col` using a quantile cutoff.
    Returns train_indices, test_indices, SplitInfo.
    """
    s = df[date_col]
    if str(s.dtype).startswith("datetime") is False:
        raise TypeError(f"{date_col} must be datetime dtype; got {s.dtype}")

    non_null = s.dropna()
    if len(non_null) == 0:
        raise ValueError(f"{date_col} has no non-null values")

    cutoff = non_null.quantile(1.0 - test_fraction)
    train_idx = df.index[s <= cutoff].tolist()
    test_idx = df.index[s > cutoff].tolist()
    return train_idx, test_idx, SplitInfo(cutoff=cutoff, n_train=len(train_idx), n_test=len(test_idx))


class Winsorizer:
    """Train-fit winsorization (two-sided clipping) for numeric columns."""

    def __init__(self, p_low: float = 0.005, p_high: float = 0.995):
        self.p_low = p_low
        self.p_high = p_high
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.quantile(X, self.p_low, axis=0)
        self.upper_ = np.quantile(X, self.p_high, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)


def convert_to_string(X):
    """Convert categorical columns to string to fix mixed int/str types."""
    if isinstance(X, pd.DataFrame):
        return X.astype(str).fillna("Missing")
    else:
        X = np.asarray(X, dtype=str)
        X = np.where(X == "nan", "Missing", X)
        X = np.where(X == "None", "Missing", X)
        return X


def build_model_pipeline(*, numeric_cols: list[str], categorical_cols: list[str], cleaning_spec):
    """Build a leakage-safe sklearn Pipeline with CatBoost."""
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("winsor", Winsorizer(p_low=cleaning_spec.winsor_p_low, p_high=cleaning_spec.winsor_p_high)),
            ("scale", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("to_str", FunctionTransformer(convert_to_string, validate=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    clf = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        verbose=100,
        random_seed=42,
        early_stopping_rounds=50,
        task_type="GPU",
        devices="0",
        max_ctr_complexity=1,
        subsample=0.8,
        bootstrap_type="Bernoulli",
    )

    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def evaluate_binary(y_true, y_proba, *, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

    fpr, tpr, thr = roc_curve(y_true, y_proba)
    for target_fpr in (0.01, 0.05, 0.10):
        mask = fpr <= target_fpr
        if mask.any():
            idx = int(mask.nonzero()[0][-1])
            metrics[f"recall_at_fpr_{int(target_fpr*100):02d}pct"] = float(tpr[idx])
            metrics[f"threshold_at_fpr_{int(target_fpr*100):02d}pct"] = float(thr[idx])
        else:
            metrics[f"recall_at_fpr_{int(target_fpr*100):02d}pct"] = 0.0
            metrics[f"threshold_at_fpr_{int(target_fpr*100):02d}pct"] = 1.0

    return metrics


def prepare_catboost_data(df, numeric_cols, categorical_cols):
    """Prepare dataframe for CatBoost."""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Missing")
            df[col] = df[col].replace({"nan": "Missing", "None": "Missing"})
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def train_catboost_direct(
    X_train, y_train, X_val, y_val,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    custom_params: dict[str, Any] | None = None,
):
    """Memory-efficient CatBoost training using native Pool format with GPU."""
    X_train = prepare_catboost_data(X_train, numeric_cols, categorical_cols)
    X_val = prepare_catboost_data(X_val, numeric_cols, categorical_cols)

    all_cols = list(X_train.columns)
    cat_indices = [all_cols.index(c) for c in categorical_cols if c in all_cols]

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_indices,
    )
    val_pool = Pool(
        data=X_val,
        label=y_val,
        cat_features=cat_indices,
    )

    params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "auto_class_weights": "Balanced",
        "verbose": 100,
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "task_type": "GPU",
        "devices": "0",
        "max_ctr_complexity": 1,
    }

    if not (custom_params and ("bagging_temperature" in custom_params or custom_params.get("bootstrap_type") == "Bayesian")):
        params["bootstrap_type"] = "Bernoulli"
        params["subsample"] = 0.8
    else:
        params["bootstrap_type"] = "Bayesian"

    if custom_params:
        params.update(custom_params)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def train_catboost_memory_optimized(
    X_train, y_train, X_val, y_val,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    custom_params: dict[str, Any] | None = None,
):
    """Memory-optimized CatBoost training using in-memory Pools with aggressive GC."""
    X_train = prepare_catboost_data(X_train, numeric_cols, categorical_cols)
    all_cols = list(X_train.columns)
    cat_indices = [all_cols.index(c) for c in categorical_cols if c in all_cols]

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_indices,
    )
    del X_train
    gc.collect()

    X_val = prepare_catboost_data(X_val, numeric_cols, categorical_cols)
    val_pool = Pool(
        data=X_val,
        label=y_val,
        cat_features=cat_indices,
    )
    del X_val
    gc.collect()

    params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "auto_class_weights": "Balanced",
        "verbose": 100,
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "task_type": "GPU",
        "devices": "0",
        "max_ctr_complexity": 1,
    }

    if not (custom_params and ("bagging_temperature" in custom_params or custom_params.get("bootstrap_type") == "Bayesian")):
        params["bootstrap_type"] = "Bernoulli"
        params["subsample"] = 0.8
    else:
        params["bootstrap_type"] = "Bayesian"

    if custom_params:
        params.update(custom_params)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def train_lightgbm_memory_optimized(
    X_train, y_train, X_val, y_val,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    custom_params: dict[str, Any] | None = None,
):
    """Memory-optimized LightGBM GPU training designed to avoid 'bin size 749' GPU errors."""
    import lightgbm as lgb
    
    all_f = [c for c in numeric_cols + categorical_cols if c in X_train.columns]
    X_train = X_train[all_f].copy()
    X_val = X_val[all_f].copy()
    
    for col in numeric_cols:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').astype('float32').fillna(0)
            X_val[col] = pd.to_numeric(X_val[col], errors='coerce').astype('float32').fillna(0)
            
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    ds_params = {
        "max_bin": 255, 
        "verbose": -1,
        "bin_construct_sample_cnt": 200000
    }
    
    train_data = lgb.Dataset(
        X_train, label=y_train, 
        categorical_feature=categorical_cols, 
        params=ds_params,
        free_raw_data=True
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, 
        reference=train_data, 
        categorical_feature=categorical_cols, 
        params=ds_params,
        free_raw_data=True
    )

    del X_train, X_val
    gc.collect()

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "is_unbalance": True,
        "seed": 42,
        "device": "gpu",
        "max_bin": 255,
        "gpu_use_dp": False,
        "verbose": -1
    }
    
    if custom_params:
        filtered_custom = {k:v for k,v in custom_params.items() if k not in ["device", "max_bin"]}
        params.update(filtered_custom)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['valid'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50), 
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def train_xgboost_memory_optimized(
    X_train, y_train, X_val, y_val,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    custom_params: dict[str, Any] | None = None,
):
    """Memory-optimized XGBoost training wrapper using DMatrix."""
    import xgboost as xgb

    all_f = [c for c in numeric_cols + categorical_cols if c in X_train.columns]
    X_train = X_train[all_f].copy()
    X_val = X_val[all_f].copy()

    for col in numeric_cols:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce').astype('float32').fillna(0)
            X_val[col] = pd.to_numeric(X_val[col], errors='coerce').astype('float32').fillna(0)

    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    dtrain = xgb.QuantileDMatrix(X_train, label=y_train, enable_categorical=True, max_bin=255)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    del X_train, X_val
    gc.collect()

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "scale_pos_weight": float(len(y_train[y_train==0]) / len(y_train[y_train==1])),
        "seed": 42,
        "verbosity": 1
    }
    if custom_params:
        params.update(custom_params)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'valid')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    return model


def train_linear_baseline(
    X_train, y_train, X_val, y_val,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    custom_params: dict[str, Any] | None = None,
):
    """Standard Logistic Regression baseline using sklearn pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    
    categorical_pipe = Pipeline(steps=[
        ("to_str", FunctionTransformer(convert_to_string, validate=False)),
        ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=20))
    ])
    
    pre = ColumnTransformer(transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ], remainder="drop")
    
    params = {
        "max_iter": 100,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }
    if custom_params:
        params.update(custom_params)
        
    model = Pipeline(steps=[
        ("preprocess", pre),
        ("model", LogisticRegression(**params))
    ])
    
    model.fit(X_train, y_train)
    return model

