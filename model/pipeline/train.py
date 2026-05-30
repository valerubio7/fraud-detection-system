import argparse
import importlib.util
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import psycopg2
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.utils.metrics import evaluate_model, find_optimal_threshold
from model.utils.plots import (
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_pr_curve_plot,
    save_roc_curve_plot,
    save_threshold_analysis_plot,
)
from model.utils.selected_features import SELECTED_FEATURES
from model.utils.tuning import run_optuna_study
from offline_features.feature_selection import select_features
from offline_features.featurizer import TransactionFeaturizer
from offline_features.imbalance_strategies import compute_scale_pos_weight

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="artifacts/model", help="Directory to save train artifacts.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows.")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning.")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials to run.")
    parser.add_argument("--optuna-timeout", type=int, default=None, help="Optional Optuna timeout in seconds.")
    parser.add_argument("--cost-fn", type=float, default=100.0, help="Cost of a false negative (missed fraud).")
    parser.add_argument("--cost-fp", type=float, default=5.0, help="Cost of a false positive (blocked legitimate).")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed classification threshold override.")
    return parser.parse_args()


def load_transactions(limit: int | None) -> pd.DataFrame:
    conn = psycopg2.connect(
        host=os.getenv("TIMESCALE_HOST", "timescaledb"),
        port=int(os.getenv("TIMESCALE_PORT", "5432")),
        user=os.getenv("TIMESCALE_USER", "fraud_timeseries_user"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DB", "fraud_transactions_timeseries"),
    )
    query = """
        SELECT
            transaction_id,
            user_id,
            merchant_id,
            merchant_category,
            amount,
            country,
            device_type,
            ip_hash,
            timestamp,
            is_fraud
        FROM public.transactions
        ORDER BY timestamp
    """
    params = None
    if limit is not None:
        query = f"{query}\nLIMIT %s"
        params = (limit,)
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


def _update_selected_features(features: list[str]) -> None:
    Path("/tmp/selected_features.json").write_text(json.dumps(features), encoding="utf-8")
    logger.info(
        "Wrote %d features to /tmp/selected_features.json. "
        "setup.sh will update selected_features.py and rebuild serving.",
        len(features),
    )


def build_features(
    df: pd.DataFrame, y: pd.Series, output_dir: Path, seed: int
) -> tuple[pd.DataFrame, TransactionFeaturizer]:
    featurizer = TransactionFeaturizer(encoders_dir=output_dir)
    X_full = featurizer.fit_transform(df, y)
    report = select_features(X_full, y, use_boruta=False, random_state=seed)
    featurizer.apply_selection(report)
    selected = featurizer.get_feature_names()
    if selected != SELECTED_FEATURES:
        logger.warning("Selected features differ from selected_features.py — updating automatically.")
        _update_selected_features(selected)
    X_full = featurizer.transform(df)
    return X_full, featurizer


def temporal_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    seed: int,
    params: dict[str, object],
) -> XGBClassifier:
    effective_spw = float(params.get("scale_pos_weight", scale_pos_weight))
    base_params = {k: v for k, v in params.items() if k != "scale_pos_weight"}
    model = XGBClassifier(
        **{
            **base_params,
            "eval_metric": "aucpr",
            "scale_pos_weight": effective_spw,
            "random_state": seed,
            "early_stopping_rounds": 20,
        }
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def save_metadata(
    output_dir: Path,
    df: pd.DataFrame,
    features: list[str],
    params: dict[str, object],
    scale_pos_weight: float,
    split_sizes: dict[str, int],
    tuning_metadata: dict[str, object],
    evaluation_metadata: dict[str, object],
) -> Path:
    ts_min = pd.to_datetime(df["timestamp"]).min()
    ts_max = pd.to_datetime(df["timestamp"]).max()
    metadata = {
        "training_date": datetime.now(UTC).isoformat(),
        "training_data_from": ts_min.isoformat() if pd.notna(ts_min) else None,
        "training_data_to": ts_max.isoformat() if pd.notna(ts_max) else None,
        "rows": split_sizes,
        "features": features,
        "hyperparameters": params,
        "scale_pos_weight": scale_pos_weight,
        **tuning_metadata,
        **evaluation_metadata,
    }
    metadata_path = output_dir / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def update_metadata_with_mlflow(metadata_path: Path, run_id: str, experiment_name: str) -> None:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["mlflow_run_id"] = run_id
    metadata["mlflow_experiment_name"] = experiment_name
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def log_summary(y_full: pd.Series, split_sizes: dict[str, int], params: dict[str, object], output_dir: Path) -> None:
    counts = y_full.value_counts().to_dict()
    logger.info("Class distribution — legitimate: %d, fraud: %d", counts.get(0, 0), counts.get(1, 0))
    logger.info(
        "Split sizes — train: %d, validation: %d, test: %d",
        split_sizes["train"],
        split_sizes["validation"],
        split_sizes["test"],
    )
    logger.info("Hyperparameters: %s", params)
    logger.info("Artifacts saved to: %s", output_dir)


def is_tracking_uri_available(tracking_uri: str, timeout_seconds: float = 2.0) -> bool:
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return True

    host = parsed.hostname
    if host is None:
        return False

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def start_mlflow_run() -> tuple[mlflow.ActiveRun, str, str]:
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection-v1")
        if not is_tracking_uri_available(tracking_uri):
            raise RuntimeError(f"MLflow tracking unavailable at {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            _experiment_tags = {
                "project": "fraud-detection-mlops",
                "team": "mlops",
                "data_version": "v1",
                "model_algorithm": "xgboost",
                "task": "binary_classification",
            }
            for key, value in _experiment_tags.items():
                client.set_experiment_tag(experiment.experiment_id, key, value)

        run_name = f"train-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        run = mlflow.start_run(run_name=run_name)

        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            git_commit = "unknown"
        mlflow.set_tag("git_commit", git_commit)

        return run, tracking_uri, experiment_name
    except Exception as exc:
        raise RuntimeError(f"MLflow tracking unavailable: {exc}") from exc


def log_mlflow_outputs(
    model: XGBClassifier,
    X_full: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    params: dict[str, object],
    scale_pos_weight: float,
    seed: int,
    split_sizes: dict[str, int],
    training_data_from: str,
    training_data_to: str,
    features: list[str],
    tuning_summary: dict[str, float] | None,
    tuning_best_params: dict[str, object] | None,
    evaluation_metrics: dict[str, object] | None,
    optimal_threshold: float | None,
    evaluation_artifacts: list[Path],
) -> tuple[str, str]:
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("MLflow run is not active.")

    mlflow.log_params(
        {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "min_child_weight": params["min_child_weight"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "early_stopping_rounds": params["early_stopping_rounds"],
            "scale_pos_weight": float(scale_pos_weight),
            "seed": seed,
            "train_size": split_sizes["train"],
            "val_size": split_sizes["validation"],
            "test_size": split_sizes["test"],
            "training_data_from": training_data_from,
            "training_data_to": training_data_to,
            "n_features": len(features),
            "features": ",".join(features),
        }
    )

    best_iteration = model.best_iteration
    best_iteration_value = int(best_iteration) if best_iteration is not None else -1
    n_total = int(len(y_train))
    n_fraud = int(y_train.sum())
    class_ratio = float(n_fraud / n_total) if n_total else 0.0
    mlflow.log_metrics({"best_iteration": float(best_iteration_value), "class_ratio": class_ratio})

    if tuning_summary is not None and tuning_best_params is not None:
        mlflow.log_metrics(
            {
                "tuning_best_pr_auc_val": tuning_summary["best_pr_auc_val"],
                "tuning_n_trials": float(tuning_summary["n_trials"]),
                "tuning_best_trial": float(tuning_summary["best_trial"]),
            }
        )
        mlflow.log_params({f"best_{key}": value for key, value in tuning_best_params.items()})

    if evaluation_metrics is not None:
        flattened = {f"test_{key}": value for key, value in evaluation_metrics.items() if not isinstance(value, dict)}
        nested_confusion = evaluation_metrics.get("confusion_matrix")
        if isinstance(nested_confusion, dict):
            for key, value in nested_confusion.items():
                flattened[f"test_confusion_{key}"] = value
        mlflow.log_metrics(flattened)
        if optimal_threshold is not None:
            mlflow.log_metrics({"optimal_threshold": float(optimal_threshold)})
        for artifact_path in evaluation_artifacts:
            mlflow.log_artifact(str(artifact_path))

    update_metadata_with_mlflow(
        output_dir / "training_metadata.json",
        active_run.info.run_id,
        mlflow.get_experiment(active_run.info.experiment_id).name,
    )
    reference_path = output_dir / "reference_dataset.parquet"
    X_full[features].to_parquet(reference_path, index=False)
    mlflow.log_artifacts(str(output_dir))

    return active_run.info.run_id, active_run.info.experiment_id


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    np.random.seed(args.seed)

    if args.tune and importlib.util.find_spec("optuna") is None:
        raise SystemExit("Optuna is required for --tune. Install with: uv sync --group model")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        mlflow_run, tracking_uri, _ = start_mlflow_run()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    df = load_transactions(args.limit)
    if df.empty:
        raise RuntimeError("No transactions loaded from TimescaleDB.")

    df = df[df["is_fraud"].notna()].copy()
    if df.empty:
        raise RuntimeError("No labeled transactions available for training.")

    ts_min = pd.to_datetime(df["timestamp"]).min()
    ts_max = pd.to_datetime(df["timestamp"]).max()
    training_data_from = ts_min.isoformat() if pd.notna(ts_min) else ""
    training_data_to = ts_max.isoformat() if pd.notna(ts_max) else ""

    y_full = df["is_fraud"].astype(int)
    X_full, _ = build_features(df, y_full, output_dir, args.seed)
    actual_features = list(X_full.columns)

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X_full, y_full)
    if X_train.empty or X_val.empty or X_test.empty:
        raise RuntimeError("Insufficient data for train/validation/test split.")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    tuning_metadata: dict[str, object] = {
        "tuning_enabled": False,
        "tuning_n_trials": 0,
        "tuning_best_params": None,
        "tuning_best_pr_auc_val": None,
    }
    evaluation_metadata: dict[str, object] = {"evaluation_results": None, "optimal_threshold": None}
    tuning_summary: dict[str, float] | None = None
    tuning_best_params: dict[str, object] | None = None
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if args.tune:
        tuning_summary = {}
        tuning_best_params = run_optuna_study(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            scale_pos_weight=scale_pos_weight,
            n_trials=args.n_trials,
            seed=args.seed,
            timeout=args.optuna_timeout,
            mlflow_enabled=mlflow_run is not None,
            tuning_summary=tuning_summary,
        )
        params = {
            "n_estimators": int(tuning_best_params["n_estimators"]),
            "max_depth": int(tuning_best_params["max_depth"]),
            "learning_rate": float(tuning_best_params["learning_rate"]),
            "min_child_weight": int(tuning_best_params["min_child_weight"]),
            "subsample": float(tuning_best_params["subsample"]),
            "colsample_bytree": float(tuning_best_params["colsample_bytree"]),
            "gamma": float(tuning_best_params["gamma"]),
            "reg_alpha": float(tuning_best_params["reg_alpha"]),
            "reg_lambda": float(tuning_best_params["reg_lambda"]),
            "scale_pos_weight": float(tuning_best_params["scale_pos_weight"]),
        }
        tuning_metadata = {
            "tuning_enabled": True,
            "tuning_n_trials": int(tuning_summary["n_trials"]),
            "tuning_best_params": tuning_best_params,
            "tuning_best_pr_auc_val": float(tuning_summary["best_pr_auc_val"]),
        }
        logger.info("Optuna best trial: %s", int(tuning_summary["best_trial"]))
        logger.info("Optuna best params: %s", tuning_best_params)
        logger.info("Optuna best PR-AUC: %.6f", tuning_summary["best_pr_auc_val"])

    model = train_model(X_train, y_train, X_val, y_val, scale_pos_weight, args.seed, params)

    val_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.round(np.arange(0.1, 0.91, 0.01), 2)
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_val.to_numpy(),
        val_proba,
        thresholds,
        cost_false_negative=args.cost_fn,
        cost_false_positive=args.cost_fp,
    )
    if args.threshold is not None:
        optimal_threshold = float(args.threshold)

    test_metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=optimal_threshold,
        cost_false_negative=args.cost_fn,
        cost_false_positive=args.cost_fp,
    )
    evaluation_metadata = {"evaluation_results": test_metrics, "optimal_threshold": optimal_threshold}

    model_path = output_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)

    split_sizes = {"train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test))}
    params = {**params, "eval_metric": "aucpr", "early_stopping_rounds": 20}
    evaluation_results_path = output_dir / "evaluation_results.json"
    with evaluation_results_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metrics": test_metrics,
                "optimal_threshold": optimal_threshold,
                "cost_false_negative": args.cost_fn,
                "cost_false_positive": args.cost_fp,
            },
            handle,
            indent=2,
        )

    confusion_path = output_dir / "confusion_matrix.png"
    roc_path = output_dir / "roc_curve.png"
    pr_path = output_dir / "pr_curve.png"
    feature_importance_path = output_dir / "feature_importance.png"
    threshold_path = output_dir / "threshold_analysis.png"

    save_confusion_matrix_plot(test_metrics["confusion_matrix"], confusion_path)
    save_roc_curve_plot(y_test.to_numpy(), model.predict_proba(X_test)[:, 1], roc_path)
    save_pr_curve_plot(y_test.to_numpy(), model.predict_proba(X_test)[:, 1], pr_path)
    save_feature_importance_plot(model, actual_features, feature_importance_path)
    save_threshold_analysis_plot(threshold_metrics, optimal_threshold, threshold_path)
    evaluation_artifacts = [
        evaluation_results_path,
        confusion_path,
        roc_path,
        pr_path,
        feature_importance_path,
        threshold_path,
    ]
    effective_spw = float(params.get("scale_pos_weight", scale_pos_weight))
    save_metadata(
        output_dir=output_dir,
        df=df,
        features=actual_features,
        params=params,
        scale_pos_weight=effective_spw,
        split_sizes=split_sizes,
        tuning_metadata=tuning_metadata,
        evaluation_metadata=evaluation_metadata,
    )
    try:
        run_id, experiment_id = log_mlflow_outputs(
            model=model,
            X_full=X_full,
            y_train=y_train,
            output_dir=output_dir,
            params=params,
            scale_pos_weight=effective_spw,
            seed=args.seed,
            split_sizes=split_sizes,
            training_data_from=training_data_from,
            training_data_to=training_data_to,
            features=actual_features,
            tuning_summary=tuning_summary,
            tuning_best_params=tuning_best_params,
            evaluation_metrics=test_metrics,
            optimal_threshold=optimal_threshold,
            evaluation_artifacts=evaluation_artifacts,
        )
        signature = infer_signature(X_train, model.predict(X_train))
        model_name = os.getenv("MODEL_NAME", "FraudDetectionModel")
        if not hasattr(model, "_estimator_type"):
            model._estimator_type = "classifier"
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=X_train.head(5),
        )

        client = MlflowClient()
        if model_info.registered_model_version is None:
            raise SystemExit("Failed to register model in MLflow Registry.")
        registered_version = model_info.registered_model_version
        client.transition_model_version_stage(
            name=model_name,
            version=registered_version,
            stage="Staging",
        )
        client.update_model_version(
            name=model_name,
            version=registered_version,
            description=(
                f"Trained {datetime.now(UTC).strftime('%Y-%m-%d')}"
                f" on {split_sizes['train']} samples. "
                f"F1={test_metrics['f1_score']:.4f}"
                f", AUC-ROC={test_metrics['roc_auc']:.4f}."
            ),
        )
        client.set_model_version_tag(model_name, registered_version, "stage_lifecycle", "Staging")
        client.set_model_version_tag(model_name, registered_version, "quality_gates", "pending")
        if tracking_uri is not None:
            run_url = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
            logger.info("MLflow run_id: %s", run_id)
            logger.info("MLflow run URL: %s", run_url)
    except Exception as exc:
        logger.warning("MLflow logging failed: %s — artifacts saved to %s", exc, output_dir)
    log_summary(y_full, split_sizes, params, output_dir)
    logger.info(
        "Evaluation — F1: %.4f, PR-AUC: %.4f, ROC-AUC: %.4f",
        test_metrics["f1_score"],
        test_metrics["pr_auc"],
        test_metrics["roc_auc"],
    )
    logger.info("Threshold used: %.2f", test_metrics["threshold"])
    logger.info("Estimated total cost: %.2f", test_metrics["total_cost"])
    logger.info("Fraud detected: %.2f%%", test_metrics["fraud_detected_pct"] * 100)
    logger.info("Model saved: %s", model_path)
    logger.info("Encoder saved: %s", output_dir / "categorical_encoder.joblib")
    logger.info("Metadata saved: %s", output_dir / "training_metadata.json")

    try:
        mlflow.end_run()
    except Exception as exc:
        logger.warning("Failed to finalize MLflow run: %s", exc)


if __name__ == "__main__":
    main()
