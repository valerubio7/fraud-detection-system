"""Train the fraud detection XGBoost model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_transactions(args.limit)
    if df.empty:
        raise RuntimeError("No transactions loaded from TimescaleDB.")

    df = df[df["is_fraud"].notna()].copy()
    if df.empty:
        raise RuntimeError("No labeled transactions available for training.")

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
    log_summary(y_full, split_sizes, params, output_dir)
    logger.info("Model saved: %s", model_path)
    logger.info("Encoder saved: %s", output_dir / "categorical_encoder.joblib")
    logger.info("Metadata saved: %s", output_dir / "training_metadata.json")


if __name__ == "__main__":
    main()
