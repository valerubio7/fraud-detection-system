from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

if TYPE_CHECKING:
    import optuna

    TrialType = optuna.Trial
else:
    TrialType = Any


def run_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    n_trials: int,
    *,
    seed: int,
    timeout: int | None,
    mlflow_enabled: bool,
    tuning_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)

    objective = build_optuna_objective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        scale_pos_weight=scale_pos_weight,
        seed=seed,
        mlflow_enabled=mlflow_enabled,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    if tuning_summary is not None:
        tuning_summary["best_pr_auc_val"] = float(study.best_value)
        tuning_summary["best_trial"] = int(study.best_trial.number)
        tuning_summary["n_trials"] = int(len(study.trials))

    return dict(study.best_params)


def build_optuna_objective(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    seed: int,
    mlflow_enabled: bool,
) -> Callable[[TrialType], float]:
    def objective(trial: TrialType) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", scale_pos_weight * 0.5, scale_pos_weight * 2.0),
            "eval_metric": "aucpr",
            "random_state": seed,
            "early_stopping_rounds": 20,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        pr_auc_val = float(average_precision_score(y_val, proba))

        if mlflow_enabled:
            import mlflow

            with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
                mlflow.log_params({key: params[key] for key in trial.params})
                mlflow.log_metrics({"pr_auc_val": pr_auc_val})

        return pr_auc_val

    return objective
