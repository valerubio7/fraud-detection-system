from unittest.mock import AsyncMock, MagicMock

from serving.app.services.prediction_store import PredictionStore


def _make_store():
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return PredictionStore(pool=mock_pool, deployment_id=42), mock_conn


class TestPredictionStoreInit:
    def test_stores_pool_and_deployment_id(self):
        mock_pool = MagicMock()
        store = PredictionStore(pool=mock_pool, deployment_id=7)
        assert store._pool is mock_pool
        assert store._deployment_id == 7


class TestPredictionStoreSave:
    async def test_save_executes_insert(self):
        store, mock_conn = _make_store()
        await store.save("550e8400-e29b-41d4-a716-446655440000", 0.8, True, 12.5)
        mock_conn.execute.assert_called_once()

    async def test_save_passes_correct_uuid(self):
        import uuid

        store, mock_conn = _make_store()
        tx_id = "550e8400-e29b-41d4-a716-446655440000"
        await store.save(tx_id, 0.75, False, 5.0)
        args = mock_conn.execute.call_args[0]
        assert args[1] == uuid.UUID(tx_id)

    async def test_save_does_not_raise_when_pool_fails(self):
        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(side_effect=Exception("db down"))
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        store = PredictionStore(pool=mock_pool, deployment_id=1)
        await store.save("550e8400-e29b-41d4-a716-446655440000", 0.9, True, 3.0)
