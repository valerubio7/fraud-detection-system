"""
Integration tests for the serving pipeline:
  - PredictionStore.save() → asyncpg + real PostgreSQL
  - fetch_labeled_predictions() → psycopg2 + real PostgreSQL
  - /predict endpoint with real Redis cache + real PostgreSQL persistence
"""

import asyncio
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

POSTGRESQL_MIGRATION = Path("database/postgresql/migrations/001_initial_schema.sql")
POSTGRES_IMAGE = "postgres:15"

VALID_REQUEST = {
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user_123",
    "merchant_id": "merchant_456",
    "merchant_category": "grocery",
    "amount": 150.0,
    "country": "AR",
    "device_type": "mobile",
    "timestamp": "2025-01-15T14:30:00Z",
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


def _setup_pg(pg: PostgresContainer) -> dict:
    import psycopg2
    import psycopg2.extras

    psycopg2.extras.register_uuid()

    pg_params = {
        "host": pg.get_container_host_ip(),
        "port": int(pg.get_exposed_port(5432)),
        "user": "test_user",
        "password": "test_pass",
        "dbname": "test_pg",
    }
    conn = psycopg2.connect(**pg_params)
    with conn.cursor() as cur:
        cur.execute(POSTGRESQL_MIGRATION.read_text())
    conn.commit()

    # Insert a model_deployments row so predictions_history FK is satisfiable.
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.model_deployments
                (model_name, version, mlflow_run_id,
                 training_data_from, training_data_to)
            VALUES (%s, %s, %s, NOW() - INTERVAL '30 days', NOW())
            RETURNING id
            """,
            ("FraudModel", "v1", f"run_{uuid.uuid4().hex}"),
        )
        deployment_id = cur.fetchone()[0]
    conn.commit()
    conn.close()

    return pg_params, deployment_id


# ---------------------------------------------------------------------------
# 1. PredictionStore (asyncpg)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPredictionStoreIntegration:
    _pg_params: dict | None = None
    _deployment_id: int | None = None

    @pytest.fixture(autouse=True, scope="class")
    def setup_postgres(self):
        with PostgresContainer(
            image=POSTGRES_IMAGE,
            username="test_user",
            password="test_pass",
            dbname="test_pg",
        ) as pg:
            pg_params, deployment_id = _setup_pg(pg)
            TestPredictionStoreIntegration._pg_params = pg_params
            TestPredictionStoreIntegration._deployment_id = deployment_id
            yield

    async def _make_pool(self):
        import asyncpg

        p = self.__class__._pg_params
        return await asyncpg.create_pool(
            host=p["host"],
            port=p["port"],
            user=p["user"],
            password=p["password"],
            database=p["dbname"],
            min_size=1,
            max_size=2,
        )

    async def test_save_persists_row_to_db(self):
        import psycopg2

        from serving.app.services.prediction_store import PredictionStore

        pool = await self._make_pool()
        store = PredictionStore(pool=pool, deployment_id=self.__class__._deployment_id)
        tx_id = str(uuid.uuid4())

        await store.save(tx_id, 0.85, True, 12.3)
        await pool.close()

        p = self.__class__._pg_params
        conn = psycopg2.connect(**p)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT prediction_score, prediction_label, latency_ms "
                "FROM public.predictions_history WHERE transaction_id = %s",
                (uuid.UUID(tx_id),),
            )
            row = cur.fetchone()
        conn.close()

        assert row is not None
        assert abs(row[0] - 0.85) < 1e-6
        assert row[1] is True
        assert abs(row[2] - 12.3) < 1e-3

    async def test_save_silently_absorbs_constraint_violation(self):
        """prediction_score outside [0,1] violates CHECK → error is swallowed."""

        from serving.app.services.prediction_store import PredictionStore

        pool = await self._make_pool()
        store = PredictionStore(pool=pool, deployment_id=self.__class__._deployment_id)

        # score = 1.5 violates predictions_history_prediction_score_ck
        await store.save(str(uuid.uuid4()), 1.5, True, 5.0)  # must not raise
        await pool.close()

    async def test_two_saves_produce_two_rows(self):
        import psycopg2

        from serving.app.services.prediction_store import PredictionStore

        pool = await self._make_pool()
        store = PredictionStore(pool=pool, deployment_id=self.__class__._deployment_id)
        tx1, tx2 = str(uuid.uuid4()), str(uuid.uuid4())

        await store.save(tx1, 0.3, False, 8.0)
        await store.save(tx2, 0.7, True, 9.0)
        await pool.close()

        p = self.__class__._pg_params
        conn = psycopg2.connect(**p)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM public.predictions_history WHERE transaction_id IN (%s, %s)",
                (uuid.UUID(tx1), uuid.UUID(tx2)),
            )
            count = cur.fetchone()[0]
        conn.close()

        assert count == 2


# ---------------------------------------------------------------------------
# 2. fetch_labeled_predictions (psycopg2)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFetchLabeledPredictionsIntegration:
    _pg_params: dict | None = None
    _deployment_id: int | None = None

    @pytest.fixture(autouse=True, scope="class")
    def setup_postgres(self):
        with PostgresContainer(
            image=POSTGRES_IMAGE,
            username="test_user",
            password="test_pass",
            dbname="test_pg",
        ) as pg:
            pg_params, deployment_id = _setup_pg(pg)
            TestFetchLabeledPredictionsIntegration._pg_params = pg_params
            TestFetchLabeledPredictionsIntegration._deployment_id = deployment_id
            yield

    def _seed(self, rows: list[tuple]) -> None:
        """Insert (prediction_score, prediction_label, actual_label) rows."""
        import psycopg2

        p = self.__class__._pg_params
        conn = psycopg2.connect(**p)
        with conn.cursor() as cur:
            for score, label, actual in rows:
                cur.execute(
                    """
                    INSERT INTO public.predictions_history
                        (transaction_id, model_version_id, prediction_score,
                         prediction_label, actual_label, latency_ms)
                    VALUES (gen_random_uuid(), %s, %s, %s, %s, 5.0)
                    """,
                    (self.__class__._deployment_id, score, label, actual),
                )
        conn.commit()
        conn.close()

    def _set_env(self):
        p = self.__class__._pg_params
        return {
            "POSTGRES_HOST": p["host"],
            "POSTGRES_PORT": str(p["port"]),
            "POSTGRES_USER": p["user"],
            "POSTGRES_PASSWORD": p["password"],
            "POSTGRES_DB": p["dbname"],
        }

    def test_returns_empty_dataframe_when_no_labeled_rows(self):
        from mlops.evidently.model_drift import fetch_labeled_predictions

        saved_env = {k: os.environ.get(k) for k in self._set_env()}
        os.environ.update(self._set_env())
        try:
            df = fetch_labeled_predictions(deployment_id=self.__class__._deployment_id)
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        assert len(df) == 0
        assert "prediction_proba" in df.columns

    def test_returns_only_labeled_rows(self):
        from mlops.evidently.model_drift import fetch_labeled_predictions

        self._seed([(0.8, True, True), (0.2, False, False)])

        saved_env = {k: os.environ.get(k) for k in self._set_env()}
        os.environ.update(self._set_env())
        try:
            df = fetch_labeled_predictions(deployment_id=self.__class__._deployment_id)
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        assert len(df) >= 2
        assert df["target"].notna().all()

    def test_excludes_rows_without_actual_label(self):
        import psycopg2

        from mlops.evidently.model_drift import fetch_labeled_predictions

        p = self.__class__._pg_params
        conn = psycopg2.connect(**p)
        with conn.cursor() as cur:
            # Insert a row WITHOUT actual_label (NULL)
            cur.execute(
                """
                INSERT INTO public.predictions_history
                    (transaction_id, model_version_id, prediction_score,
                     prediction_label, latency_ms)
                VALUES (gen_random_uuid(), %s, 0.6, TRUE, 5.0)
                """,
                (self.__class__._deployment_id,),
            )
        conn.commit()
        conn.close()

        saved_env = {k: os.environ.get(k) for k in self._set_env()}
        os.environ.update(self._set_env())
        try:
            df = fetch_labeled_predictions(deployment_id=self.__class__._deployment_id)
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # All returned rows must have a non-null target
        if len(df) > 0:
            assert df["target"].notna().all()


# ---------------------------------------------------------------------------
# 3. /predict endpoint — real Redis cache + real PostgreSQL persistence
# ---------------------------------------------------------------------------


def _mock_model_loader():
    loader = MagicMock()
    loader._model = MagicMock()
    loader._model.predict_proba.return_value = np.array([[0.2, 0.8]])
    loader.model_version = "42"
    loader.prepare_features.return_value = np.zeros((1, 17))
    return loader


@pytest.mark.integration
class TestServingApiIntegration:
    _pg_params: dict | None = None
    _deployment_id: int | None = None
    _redis_host: str | None = None
    _redis_port: int | None = None

    @pytest.fixture(autouse=True, scope="class")
    def setup_containers(self):
        with (
            PostgresContainer(
                image=POSTGRES_IMAGE,
                username="test_user",
                password="test_pass",
                dbname="test_pg",
            ) as pg,
            RedisContainer(image="redis:7.2-alpine") as redis_c,
        ):
            pg_params, deployment_id = _setup_pg(pg)
            TestServingApiIntegration._pg_params = pg_params
            TestServingApiIntegration._deployment_id = deployment_id
            TestServingApiIntegration._redis_host = redis_c.get_container_host_ip()
            TestServingApiIntegration._redis_port = int(redis_c.get_exposed_port(6379))
            yield

    def _make_app_state(self, pool):
        from serving.app.services.prediction_cache import PredictionCache
        from serving.app.services.prediction_store import PredictionStore

        cache = PredictionCache(
            host=self.__class__._redis_host,
            port=self.__class__._redis_port,
        )
        store = PredictionStore(pool=pool, deployment_id=self.__class__._deployment_id)
        return cache, store

    async def _make_pool(self):
        import asyncpg

        p = self.__class__._pg_params
        return await asyncpg.create_pool(
            host=p["host"],
            port=p["port"],
            user=p["user"],
            password=p["password"],
            database=p["dbname"],
            min_size=1,
            max_size=2,
        )

    async def test_predict_stores_result_in_redis_cache(self):
        """First /predict call stores result; second call returns from cache without running model."""
        from serving.app.main import app

        pool = await self._make_pool()
        cache, store = self._make_app_state(pool)
        loader = _mock_model_loader()

        app.state.model_loader = loader
        app.state.prediction_store = store
        app.state.prediction_cache = cache

        tx_id = str(uuid.uuid4())
        request_body = {**VALID_REQUEST, "transaction_id": tx_id}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r1 = await client.post("/predict", json=request_body)
            assert r1.status_code == 200
            assert r1.json()["prediction_score"] == pytest.approx(0.8)

            # Second call: Redis cache should be hit, model NOT called again
            loader._model.predict_proba.reset_mock()
            r2 = await client.post("/predict", json=request_body)

        await pool.close()

        assert r2.status_code == 200
        assert r2.json()["prediction_score"] == pytest.approx(0.8)
        loader._model.predict_proba.assert_not_called()

    async def test_predict_persists_prediction_to_postgresql(self):
        """After /predict the background task inserts a row into predictions_history."""

        import psycopg2

        from serving.app.main import app

        pool = await self._make_pool()
        cache, store = self._make_app_state(pool)
        loader = _mock_model_loader()

        app.state.model_loader = loader
        app.state.prediction_store = store
        app.state.prediction_cache = cache

        tx_id = str(uuid.uuid4())
        request_body = {**VALID_REQUEST, "transaction_id": tx_id}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/predict", json=request_body)
            assert r.status_code == 200
            # Give the background task a moment to execute.
            await asyncio.sleep(0.1)

        await pool.close()

        p = self.__class__._pg_params
        conn = psycopg2.connect(**p)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT prediction_score FROM public.predictions_history WHERE transaction_id = %s",
                (uuid.UUID(tx_id),),
            )
            row = cur.fetchone()
        conn.close()

        assert row is not None
        assert abs(row[0] - 0.8) < 1e-6

    async def test_predict_cache_miss_runs_model(self):
        """Fresh transaction_id hits the model (not cache)."""
        from serving.app.main import app

        pool = await self._make_pool()
        cache, store = self._make_app_state(pool)
        loader = _mock_model_loader()

        app.state.model_loader = loader
        app.state.prediction_store = store
        app.state.prediction_cache = cache

        request_body = {**VALID_REQUEST, "transaction_id": str(uuid.uuid4())}

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post("/predict", json=request_body)

        await pool.close()

        assert r.status_code == 200
        loader._model.predict_proba.assert_called_once()
