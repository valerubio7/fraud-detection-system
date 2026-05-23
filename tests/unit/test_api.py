from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from serving.app.main import app

VALID_REQUEST = {
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user_123",
    "merchant_id": "merchant_456",
    "merchant_category": "grocery",
    "amount": 150.0,
    "country": "AR",
    "timestamp": "2025-01-15T14:30:00Z",
    "device_type": "mobile",
    "ip_hash": "abc123",
    "features": {
        "tx_count_1h": 3.0,
        "tx_count_24h": 10.0,
        "tx_count_7d": 50.0,
        "amount_sum_1h": 300.0,
        "amount_sum_24h": 1000.0,
        "seconds_since_last_tx": 600.0,
        "amount_ratio_vs_user_avg": 1.5,
        "is_country_new": 0.0,
        "distinct_countries_seen": 3.0,
        "is_merchant_new": 1.0,
        "distinct_merchants_seen": 7.0,
    },
}


@pytest.fixture
def mock_model_loader():
    loader = MagicMock()
    loader._model = MagicMock()
    loader._model.predict_proba.return_value = np.array([[0.2, 0.8]])
    loader._encoder = MagicMock()
    loader.model_version = "3"
    loader.model_stage = "Production"
    loader.deployment_id = 42
    loader.loaded_at = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
    loader.prepare_features.return_value = np.zeros((1, 16))
    return loader


@pytest.fixture(autouse=True)
def inject_mock_loader(mock_model_loader):
    """Inyecta el mock en app.state antes de cada test.

    ASGITransport no dispara el lifespan, por lo que app.state.prediction_store
    y prediction_cache tampoco se inicializan — se inyectan aquí como mocks.
    """
    app.state.model_loader = mock_model_loader

    mock_cache = MagicMock()
    mock_cache.get.return_value = None  # siempre cache miss → ejecuta lógica real
    app.state.prediction_cache = mock_cache
    app.state.prediction_store = AsyncMock()

    yield

    app.state.model_loader = None
    app.state.prediction_cache = None
    app.state.prediction_store = None


class TestHealthEndpoint:
    async def test_health_ok_when_model_loaded(self, mock_model_loader):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    async def test_health_degraded_when_model_not_loaded(self):
        app.state.model_loader = MagicMock(_model=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"


class TestModelInfoEndpoint:
    async def test_model_info_returns_expected_fields(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["model_version"] == "3"
        assert body["deployment_id"] == 42
        assert "fraud_score_threshold" in body
        assert "loaded_at" in body

    async def test_model_info_returns_503_when_model_not_loaded(self):
        app.state.model_loader = MagicMock(_model=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/model/info")
        assert response.status_code == 503


class TestPredictEndpoint:
    async def test_predict_valid_request_returns_200(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200
        body = response.json()
        assert "prediction_score" in body
        assert "prediction_label" in body
        assert body["transaction_id"] == VALID_REQUEST["transaction_id"]
        assert 0.0 <= body["prediction_score"] <= 1.0
        assert isinstance(body["prediction_label"], bool)

    async def test_predict_with_high_score_returns_fraud(self, mock_model_loader):
        mock_model_loader._model.predict_proba.return_value = np.array([[0.05, 0.95]])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json=VALID_REQUEST)
        assert response.json()["prediction_label"] is True

    async def test_predict_with_low_score_returns_not_fraud(self, mock_model_loader):
        mock_model_loader._model.predict_proba.return_value = np.array([[0.9, 0.1]])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json=VALID_REQUEST)
        assert response.json()["prediction_label"] is False

    async def test_predict_invalid_amount_returns_422(self):
        payload = {**VALID_REQUEST, "amount": -50.0}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json=payload)
        assert response.status_code == 422

    async def test_predict_missing_features_returns_422(self):
        payload = {k: v for k, v in VALID_REQUEST.items() if k != "features"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json=payload)
        assert response.status_code == 422

    async def test_predict_returns_cached_result_on_cache_hit(self, mock_model_loader):
        cached = {
            "transaction_id": VALID_REQUEST["transaction_id"],
            "prediction_score": 0.99,
            "prediction_label": True,
            "model_version": "3",
            "latency_ms": 1.0,
        }
        app.state.prediction_cache.get.return_value = cached
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200
        assert response.json()["prediction_score"] == pytest.approx(0.99)
        mock_model_loader._model.predict_proba.assert_not_called()

    async def test_predict_503_when_model_not_loaded(self):
        app.state.model_loader = MagicMock(_model=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 503


class TestPredictBatchEndpoint:
    async def test_batch_predict_returns_all_predictions(self, mock_model_loader):
        mock_model_loader._model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        payload = {"items": [VALID_REQUEST, VALID_REQUEST]}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert len(body["predictions"]) == 2

    async def test_batch_predict_empty_items_returns_422(self):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict/batch", json={"items": []})
        assert response.status_code == 422

    async def test_batch_predict_latency_ms_is_present(self):
        payload = {"items": [VALID_REQUEST]}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict/batch", json=payload)
        assert "latency_ms" in response.json()
        assert response.json()["latency_ms"] >= 0.0
