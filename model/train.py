"""Train the fraud detection XGBoost model."""

from __future__ import annotations

import argparse
import json
import logging
import socket
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
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from feature_engineering.offline.featurizer import TransactionFeaturizer
from feature_engineering.offline.imbalance import compute_scale_pos_weight
from feature_engineering.offline.selection import select_features
from model.features import SELECTED_FEATURES

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/model",
        help="Directory to save training artifacts.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows.")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking.",
    )
    return parser.parse_args()


def load_transactions(limit: int | None) -> pd.DataFrame:
    settings = config.timescaledb_settings
    conn = psycopg2.connect(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        dbname=settings.db,
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


def build_features(
    df: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, TransactionFeaturizer]:
    featurizer = TransactionFeaturizer(encoders_dir=output_dir)
    X_full = featurizer.fit_transform(df, y)
    report = select_features(X_full, y, use_boruta=False, random_state=seed)
    featurizer.apply_selection(report)
    selected = featurizer.get_feature_names()
    if selected != SELECTED_FEATURES:
        raise RuntimeError(
            "Selected features do not match model/features.py. "
            "Run feature selection on the seed dataset and update SELECTED_FEATURES."
        )
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
) -> XGBClassifier:
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
        "early_stopping_rounds": 20,
    }
    model = XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def save_metadata(
    output_dir: Path,
    df: pd.DataFrame,
    features: list[str],
    params: dict[str, object],
    scale_pos_weight: float,
    split_sizes: dict[str, int],
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
    }
    metadata_path = output_dir / "training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def update_metadata_with_mlflow(
    metadata_path: Path,
    run_id: str,
    experiment_name: str,
) -> None:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["mlflow_run_id"] = run_id
    metadata["mlflow_experiment_name"] = experiment_name
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def log_summary(
    y_full: pd.Series,
    split_sizes: dict[str, int],
    params: dict[str, object],
    output_dir: Path,
) -> None:
    counts = y_full.value_counts().to_dict()
    logger.info(
        "Class distribution — legitimate: %d, fraud: %d",
        counts.get(0, 0),
        counts.get(1, 0),
    )
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


def start_mlflow_run(
    disable_mlflow: bool,
) -> tuple[mlflow.ActiveRun | None, str | None, str | None]:
    if disable_mlflow:
        return None, None, None

    try:
        mlflow_settings = config.mlflow_settings
        tracking_uri = mlflow_settings.tracking_uri
        if not is_tracking_uri_available(tracking_uri):
            logger.warning("MLflow tracking unavailable at %s", tracking_uri)
            return None, None, None
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_settings.experiment_name)
        run_name = f"train-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        run = mlflow.start_run(run_name=run_name)
        return run, tracking_uri, mlflow_settings.experiment_name
    except Exception as exc:
        logger.warning("MLflow tracking unavailable: %s", exc)
        return None, None, None


def log_mlflow_outputs(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path,
    params: dict[str, object],
    scale_pos_weight: float,
    seed: int,
    split_sizes: dict[str, int],
    training_data_from: str,
    training_data_to: str,
    features: list[str],
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
    mlflow.log_metrics(
        {
            "best_iteration": float(best_iteration_value),
            "class_ratio": class_ratio,
        }
    )

    update_metadata_with_mlflow(
        output_dir / "training_metadata.json",
        active_run.info.run_id,
        mlflow.get_experiment(active_run.info.experiment_id).name,
    )
    mlflow.log_artifacts(str(output_dir))

    signature = infer_signature(X_val, model.predict(X_val))
    input_example = X_val.head(5)
    model_name = config.model_settings.model_name
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example,
    )

    return active_run.info.run_id, active_run.info.experiment_id


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow_run, tracking_uri, _ = start_mlflow_run(args.no_mlflow)

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

    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X_full, y_full)
    if X_train.empty or X_val.empty or X_test.empty:
        raise RuntimeError("Insufficient data for train/validation/test split.")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    model = train_model(X_train, y_train, X_val, y_val, scale_pos_weight, args.seed)

    model_path = output_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)

    split_sizes = {
        "train": int(len(X_train)),
        "validation": int(len(X_val)),
        "test": int(len(X_test)),
    }
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 20,
    }
    save_metadata(
        output_dir=output_dir,
        df=df,
        features=SELECTED_FEATURES,
        params=params,
        scale_pos_weight=scale_pos_weight,
        split_sizes=split_sizes,
    )
    if mlflow_run is not None:
        try:
            run_id, experiment_id = log_mlflow_outputs(
                model=model,
                X_val=X_val,
                y_train=y_train,
                output_dir=output_dir,
                params=params,
                scale_pos_weight=scale_pos_weight,
                seed=args.seed,
                split_sizes=split_sizes,
                training_data_from=training_data_from,
                training_data_to=training_data_to,
                features=SELECTED_FEATURES,
            )
            if tracking_uri is not None:
                run_url = f"{tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
                print(f"MLflow run_id: {run_id}")
                print(f"MLflow run URL: {run_url}")
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)
    log_summary(y_full, split_sizes, params, output_dir)
    logger.info("Model saved: %s", model_path)
    logger.info("Encoder saved: %s", output_dir / "categorical_encoder.joblib")
    logger.info("Metadata saved: %s", output_dir / "training_metadata.json")

    if mlflow_run is not None:
        try:
            mlflow.end_run()
        except Exception as exc:
            logger.warning("Failed to finalize MLflow run: %s", exc)


if __name__ == "__main__":
    main()
