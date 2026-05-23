from unittest.mock import MagicMock, patch

import pytest

from serving.app.services.prediction_cache import PredictionCache


@pytest.fixture
def mock_cache():
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    with patch("serving.app.services.prediction_cache.Redis", return_value=mock_client):
        cache = PredictionCache(host="localhost", port=6379)
    cache._client = mock_client
    return cache


class TestPredictionCacheAvailability:
    def test_is_available_when_redis_pings(self):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        with patch("serving.app.services.prediction_cache.Redis", return_value=mock_client):
            cache = PredictionCache(host="localhost", port=6379)
        assert cache.is_available is True

    def test_is_not_available_when_redis_raises(self):
        mock_client = MagicMock()
        mock_client.ping.side_effect = Exception("Connection refused")
        with patch("serving.app.services.prediction_cache.Redis", return_value=mock_client):
            cache = PredictionCache(host="localhost", port=6379)
        assert cache.is_available is False


class TestPredictionCacheGet:
    def test_get_returns_none_on_cache_miss(self, mock_cache):
        mock_cache._client.get.return_value = None
        result = mock_cache.get("tx_123")
        assert result is None

    def test_get_returns_dict_on_cache_hit(self, mock_cache):
        import json

        data = {"transaction_id": "tx_123", "prediction_score": 0.9}
        mock_cache._client.get.return_value = json.dumps(data)
        result = mock_cache.get("tx_123")
        assert result == data

    def test_get_uses_correct_key_format(self, mock_cache):
        mock_cache._client.get.return_value = None
        mock_cache.get("tx_abc")
        mock_cache._client.get.assert_called_once_with("prediction:tx_abc")

    def test_get_returns_none_when_client_returns_none(self, mock_cache):
        mock_cache._client.get.return_value = None
        assert mock_cache.get("unknown_tx") is None


class TestPredictionCacheSet:
    def test_set_stores_json_serialized_data(self, mock_cache):
        import json

        data = {"score": 0.8, "label": True}
        mock_cache.set("tx_999", data, ttl_seconds=120)
        mock_cache._client.set.assert_called_once_with("prediction:tx_999", json.dumps(data), ex=120)

    def test_set_uses_default_ttl_of_60(self, mock_cache):
        mock_cache.set("tx_ttl", {"score": 0.5})
        call_kwargs = mock_cache._client.set.call_args
        assert call_kwargs[1].get("ex") == 60 or (call_kwargs[0][2:] and call_kwargs[0][-1] == 60)
