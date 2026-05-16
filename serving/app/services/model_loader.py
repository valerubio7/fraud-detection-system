from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import psycopg2
from mlflow.tracking import MlflowClient

import config

_ARTIFACTS_DIR = Path("/tmp/fraud_model/artifacts/model")


class ModelLoader:
    def __init__(self) -> None:
        self._model = None
        self._encoder = None
        self._mc_map: dict[str, float] = {}
        self._mc_global: float = 0.0
        self._country_map: dict[str, float] = {}
        self._country_global: float = 0.0
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

        self._mc_map = self._encoder._merchant_category_enc.mapping_
        self._mc_global = self._encoder._merchant_category_enc.global_mean_
        self._country_map = self._encoder._country_enc.mapping_
        self._country_global = self._encoder._country_enc.global_mean_

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
        mc_encoded = self._mc_map.get(str(raw["merchant_category"]), self._mc_global)
        country_encoded = self._country_map.get(str(raw["country"]), self._country_global)

        return np.array(
            [
                [
                    np.log1p(raw["amount"]),
                    raw["timestamp"].hour,
                    raw["timestamp"].weekday(),
                    mc_encoded,
                    country_encoded,
                    window_features["tx_count_1h"],
                    window_features["tx_count_24h"],
                    window_features["tx_count_7d"],
                    window_features["amount_sum_1h"],
                    window_features["amount_sum_24h"],
                    window_features["seconds_since_last_tx"],
                    window_features["amount_ratio_vs_user_avg"],
                    window_features["is_country_new"],
                    window_features["distinct_countries_seen"],
                    window_features["is_merchant_new"],
                    window_features["distinct_merchants_seen"],
                ]
            ],
            dtype=np.float64,
        )
