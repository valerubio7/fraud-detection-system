from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from mlflow.tracking import MlflowClient

import config
from model.selected_features import SELECTED_FEATURES

_ARTIFACTS_DIR = Path("/tmp/fraud_model/artifacts/model")


class ModelLoader:
    def __init__(self) -> None:
        self._model = None
        self._encoder = None
        self.model_name: str | None = None
        self.model_version: str | None = None
        self.model_stage: str | None = None
        self.deployment_id: int | None = None
        self.loaded_at: datetime | None = None

    def load(self) -> None:
        client = MlflowClient(tracking_uri=config.mlflow_settings.tracking_uri)

        versions = client.get_latest_versions(
            config.model_settings.model_name,
            stages=[config.model_settings.model_stage],
        )
        if not versions:
            raise RuntimeError(
                f"No model version found in stage '{config.model_settings.model_stage}' "
                f"for model '{config.model_settings.model_name}'"
            )
        version = versions[0]

        client.download_artifacts(version.run_id, "", dst_path="/tmp/fraud_model")
        self._model = joblib.load(_ARTIFACTS_DIR / "xgboost_model.joblib")
        self._encoder = joblib.load(_ARTIFACTS_DIR / "categorical_encoder.joblib")

        pg = config.postgres_settings
        conn = psycopg2.connect(
            host=pg.host,
            port=pg.port,
            user=pg.user,
            password=pg.password,
            dbname=pg.db,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM public.model_deployments WHERE is_active = TRUE LIMIT 1"
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            raise RuntimeError("No active deployment found in model_deployments")
        self.deployment_id = row[0]

        self.model_name = config.model_settings.model_name
        self.model_version = str(version.version)
        self.model_stage = config.model_settings.model_stage
        self.loaded_at = datetime.now(UTC)

    def prepare_features(self, raw: dict, window_features: dict[str, float]) -> np.ndarray:
        log_amount = np.log1p(raw["amount"])
        hour_of_day = raw["timestamp"].hour
        day_of_week = raw["timestamp"].weekday()

        df = pd.DataFrame(
            [
                {
                    "merchant_category": raw["merchant_category"],
                    "country": raw["country"],
                    "device_type": raw["device_type"],
                }
            ]
        )
        encoded = self._encoder.transform(df)

        features: dict[str, float] = {
            "log_amount": log_amount,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "merchant_category_encoded": encoded["merchant_category_encoded"].iloc[0],
            "country_encoded": encoded["country_encoded"].iloc[0],
            **window_features,
        }

        values = [features[f] for f in SELECTED_FEATURES]
        return np.array(values, dtype=np.float64).reshape(1, -1)
