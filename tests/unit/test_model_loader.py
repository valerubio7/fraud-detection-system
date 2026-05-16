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
        "distinct_countries_seen": 3,
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
        assert result[0, 5] == pytest.approx(3.0)  # tx_count_1h
        assert result[0, 10] == pytest.approx(600.0)  # seconds_since_last_tx

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
    def _make_mock_config(self):
        mock_config = MagicMock()
        mock_config.mlflow_settings.tracking_uri = "http://test-mlflow:5000"
        mock_config.model_settings.model_name = "TestModel"
        mock_config.model_settings.model_stage = "Production"
        mock_config.postgres_settings.host = "localhost"
        mock_config.postgres_settings.port = 5432
        mock_config.postgres_settings.user = "test"
        mock_config.postgres_settings.password = "test"
        mock_config.postgres_settings.db = "test"
        return mock_config

    def test_load_raises_if_no_production_model(self):
        loader = ModelLoader()
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []

        with (
            patch("serving.app.services.model_loader.MlflowClient", return_value=mock_client),
            patch("serving.app.services.model_loader.config", self._make_mock_config()),
        ):
            with pytest.raises(RuntimeError, match="[Pp]roduction"):
                loader.load()

    def test_load_raises_if_no_active_deployment(self):
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
            patch("serving.app.services.model_loader.config", self._make_mock_config()),
        ):
            with pytest.raises(RuntimeError, match="[Dd]eployment"):
                loader.load()
