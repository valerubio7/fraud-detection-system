import os
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from serving.app.services.model_loader import ModelLoader


@pytest.fixture
def loader_with_mock_encoder():
    """ModelLoader con mocks de _model y encoding — no necesita MLflow ni PostgreSQL."""
    loader = ModelLoader()

    # prepare_features usa lookups de dict directamente (no _encoder.transform).
    # Se asignan valores conocidos para que los tests sean deterministas.
    loader._mc_map = {"grocery": 0.05}
    loader._mc_global = 0.0
    loader._country_map = {"AR": 0.10}
    loader._country_global = 0.0

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
    loader._model = mock_model

    loader.model_version = "3"
    loader.model_stage = "Production"
    loader.deployment_id = 42
    loader.loaded_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
    return loader


class TestPrepareFeatures:
    WINDOW_FEATURES = {
        "tx_count_1h": 3,
        "tx_count_24h": 10,
        "tx_count_7d": 50,
        "amount_sum_1h": 300.0,
        "amount_sum_24h": 1000.0,
        "seconds_since_last_tx": 600.0,
        "amount_ratio_vs_user_avg": 1.5,
        "is_country_new": 0.0,
        "is_merchant_new": 1.0,
        "distinct_merchants_seen": 7,
    }

    def test_output_shape_is_1x16(self, loader_with_mock_encoder):
        raw = {
            "amount": 150.0,
            "timestamp": datetime(2025, 1, 15, 14, 30, 0, tzinfo=UTC),
            "merchant_category": "grocery",
            "country": "AR",
            "device_type": "mobile",
        }
        result = loader_with_mock_encoder.prepare_features(raw, self.WINDOW_FEATURES)
        assert result.shape == (1, 16)

    def test_log_amount_is_correct(self, loader_with_mock_encoder):
        raw = {
            "amount": 150.0,
            "timestamp": datetime(2025, 1, 15, 14, 30, 0, tzinfo=UTC),
            "merchant_category": "grocery",
            "country": "AR",
            "device_type": "mobile",
        }
        result = loader_with_mock_encoder.prepare_features(raw, self.WINDOW_FEATURES)
        assert result[0, 0] == pytest.approx(np.log1p(150.0))

    def test_hour_of_day_is_correct(self, loader_with_mock_encoder):
        raw = {
            "amount": 100.0,
            "timestamp": datetime(2025, 1, 15, 14, 30, 0, tzinfo=UTC),
            "merchant_category": "grocery",
            "country": "AR",
            "device_type": "mobile",
        }
        result = loader_with_mock_encoder.prepare_features(raw, self.WINDOW_FEATURES)
        assert result[0, 1] == 14

    def test_day_of_week_is_correct(self, loader_with_mock_encoder):
        # 2025-01-15 es miércoles → weekday() = 2
        raw = {
            "amount": 100.0,
            "timestamp": datetime(2025, 1, 15, 14, 30, 0, tzinfo=UTC),
            "merchant_category": "grocery",
            "country": "AR",
            "device_type": "mobile",
        }
        result = loader_with_mock_encoder.prepare_features(raw, self.WINDOW_FEATURES)
        assert result[0, 2] == 2

    def test_encoded_features_use_maps(self, loader_with_mock_encoder):
        raw = {
            "amount": 100.0,
            "timestamp": datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC),
            "merchant_category": "grocery",
            "country": "AR",
            "device_type": "mobile",
        }
        result = loader_with_mock_encoder.prepare_features(raw, self.WINDOW_FEATURES)
        assert result[0, 3] == pytest.approx(0.05)  # merchant_category_encoded
        assert result[0, 4] == pytest.approx(0.10)  # country_encoded

    def test_window_features_are_in_output(self, loader_with_mock_encoder):
        raw = {
            "amount": 100.0,
            "timestamp": datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC),
            "merchant_category": "grocery",
            "country": "AR",
            "device_type": "mobile",
        }
        result = loader_with_mock_encoder.prepare_features(raw, self.WINDOW_FEATURES)
        # Layout: log_amount[0], hour[1], weekday[2], mc_enc[3], country_enc[4],
        #         device_type_enc[5], tx_count_1h[6], ..., seconds_since_last_tx[11]
        assert result[0, 6] == pytest.approx(3.0)  # tx_count_1h
        assert result[0, 11] == pytest.approx(600.0)  # seconds_since_last_tx

    def test_zero_amount_uses_log1p(self, loader_with_mock_encoder):
        raw = {
            "amount": 0.0,
            "timestamp": datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC),
            "merchant_category": "grocery",
            "country": "AR",
            "device_type": "mobile",
        }
        result = loader_with_mock_encoder.prepare_features(raw, self.WINDOW_FEATURES)
        assert result[0, 0] == pytest.approx(np.log1p(0.0))  # = 0.0

    def test_unknown_merchant_and_country_fall_back_to_global_mean(self, loader_with_mock_encoder):
        raw = {
            "amount": 100.0,
            "timestamp": datetime(2025, 1, 15, 14, 0, 0, tzinfo=UTC),
            "merchant_category": "unknown_cat",
            "country": "XX",
            "device_type": "mobile",
        }
        result = loader_with_mock_encoder.prepare_features(raw, self.WINDOW_FEATURES)
        assert result[0, 3] == pytest.approx(0.0)  # _mc_global
        assert result[0, 4] == pytest.approx(0.0)  # _country_global


class TestModelLoaderLoad:
    def _setup_env(self):
        self._saved_env = {}
        env_vars = {
            "MLFLOW_TRACKING_URI": "http://test-mlflow:5000",
            "MODEL_NAME": "TestModel",
            "MODEL_STAGE": "Production",
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_USER": "test",
            "POSTGRES_PASSWORD": "test",
            "POSTGRES_DB": "test",
        }
        for k, v in env_vars.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v

    def _teardown_env(self):
        for k, old_v in self._saved_env.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v

    def test_load_raises_if_no_production_model(self):
        self._setup_env()
        try:
            loader = ModelLoader()
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = []

            with patch("serving.app.services.model_loader.MlflowClient", return_value=mock_client):
                with pytest.raises(RuntimeError, match="[Pp]roduction"):
                    loader.load()
        finally:
            self._teardown_env()

    def test_load_raises_if_no_active_deployment(self):
        self._setup_env()
        try:
            loader = ModelLoader()
            mock_version = MagicMock()
            mock_version.run_id = "run_abc"
            mock_version.version = "5"
            mock_client = MagicMock()
            mock_client.get_latest_versions.return_value = [mock_version]

            mock_conn = MagicMock()
            mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
            mock_cursor.fetchone.return_value = None

            with (
                patch("serving.app.services.model_loader.MlflowClient", return_value=mock_client),
                patch("serving.app.services.model_loader.joblib.load", return_value=MagicMock()),
                patch("serving.app.services.model_loader.psycopg2.connect", return_value=mock_conn),
            ):
                with pytest.raises(RuntimeError, match="[Dd]eployment"):
                    loader.load()
        finally:
            self._teardown_env()
