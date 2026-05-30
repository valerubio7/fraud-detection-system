import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import psycopg2
from mlflow.tracking import MlflowClient

_ARTIFACTS_DIR = Path("/tmp/fraud_model")


class ModelLoader:
    def __init__(self) -> None:
        self._model = None
        self._encoder = None
        self._mc_map: dict[str, float] = {}
        self._mc_global: float = 0.0
        self._country_map: dict[str, float] = {}
        self._country_global: float = 0.0
        self._device_type_map: dict[str, int] = {}
        self.model_name: str | None = None
        self.model_version: str | None = None
        self.model_stage: str | None = None
        self.deployment_id: int | None = None
        self.loaded_at: datetime | None = None

    def load(self) -> None:
        client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        versions = client.get_latest_versions(
            os.getenv("MODEL_NAME", "FraudDetectionModel"), stages=[os.getenv("MODEL_STAGE", "Production")]
        )
        if not versions:
            raise RuntimeError(
                f"No model version found in stage '{os.getenv('MODEL_STAGE', 'Production')}' "
                f"for model '{os.getenv('MODEL_NAME', 'FraudDetectionModel')}'"
            )
        version = versions[0]

        Path("/tmp/fraud_model").mkdir(parents=True, exist_ok=True)
        client.download_artifacts(version.run_id, "", dst_path="/tmp/fraud_model")
        self._model = joblib.load(_ARTIFACTS_DIR / "xgboost_model.joblib")
        self._encoder = joblib.load(_ARTIFACTS_DIR / "categorical_encoder.joblib")

        self._mc_map = self._encoder._merchant_category_enc.mapping_
        self._mc_global = self._encoder._merchant_category_enc.global_mean_
        self._country_map = self._encoder._country_enc.mapping_
        self._country_global = self._encoder._country_enc.global_mean_
        self._device_type_map = self._encoder._device_type_enc.mapping_

        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgresql"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "fraud_metadata_user"),
            password=os.getenv("POSTGRES_PASSWORD"),
            dbname=os.getenv("POSTGRES_DB", "fraud_metadata"),
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM public.model_deployments WHERE is_active = TRUE LIMIT 1")
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None:
            raise RuntimeError("No active deployment found in model_deployments")
        self.deployment_id = row[0]

        self.model_name = os.getenv("MODEL_NAME", "FraudDetectionModel")
        self.model_version = str(version.version)
        self.model_stage = os.getenv("MODEL_STAGE", "Production")
        self.loaded_at = datetime.now(UTC)

    def prepare_features(self, raw: dict, window_features: dict[str, float]) -> np.ndarray:
        mc_encoded = self._mc_map.get(str(raw["merchant_category"]), self._mc_global)
        country_encoded = self._country_map.get(str(raw["country"]), self._country_global)
        device_type_encoded = self._device_type_map.get(str(raw["device_type"]), -1)

        return np.array(
            [
                [
                    np.log1p(raw["amount"]),
                    raw["timestamp"].hour,
                    raw["timestamp"].weekday(),
                    mc_encoded,
                    country_encoded,
                    device_type_encoded,
                    window_features["tx_count_1h"],
                    window_features["tx_count_24h"],
                    window_features["tx_count_7d"],
                    window_features["amount_sum_1h"],
                    window_features["amount_sum_24h"],
                    window_features["seconds_since_last_tx"],
                    window_features["amount_ratio_vs_user_avg"],
                    window_features["is_country_new"],
                    window_features["is_merchant_new"],
                    window_features["distinct_merchants_seen"],
                ]
            ],
            dtype=np.float64,
        )
