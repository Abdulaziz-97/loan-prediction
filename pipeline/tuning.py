import gc
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pipeline.train import prepare_catboost_data

def objective(trial, X_train, y_train, X_val, y_val, numeric_cols, categorical_cols):
    """Optuna objective function for CatBoost hyperparameter tuning."""
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 1, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "auto_class_weights": "Balanced",
        "verbose": False,
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "task_type": "GPU",
        "devices": "0",
        "max_ctr_complexity": 1,
    }

    X_train_prep = prepare_catboost_data(X_train, numeric_cols, categorical_cols)
    X_val_prep = prepare_catboost_data(X_val, numeric_cols, categorical_cols)

    all_cols = list(X_train_prep.columns)
    cat_indices = [all_cols.index(c) for c in categorical_cols if c in all_cols]

    train_pool = Pool(data=X_train_prep, label=y_train, cat_features=cat_indices)
    val_pool = Pool(data=X_val_prep, label=y_val, cat_features=cat_indices)

    del X_train_prep, X_val_prep
    gc.collect()

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    return model.get_best_score()["validation"]["Logloss"]

def run_optuna_study(X_train, y_train, X_val, y_val, numeric_cols, categorical_cols, n_trials=20):
    """Executes the Optuna study and returns the best params."""
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, numeric_cols, categorical_cols),
        n_trials=n_trials
    )
    
    print("\n" + "="*30)
    print("Best Trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    return study.best_params
