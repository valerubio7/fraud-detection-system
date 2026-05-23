import os
import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from testcontainers.postgres import PostgresContainer

TIMESCALEDB_IMAGE = "timescale/timescaledb:latest-pg15"
POSTGRES_IMAGE = "postgres:15"
TIMESCALEDB_MIGRATION = Path("database/timescaledb/migrations/001_initial_schema.sql")
POSTGRESQL_MIGRATION = Path("database/postgresql/migrations/001_initial_schema.sql")
MLFLOW_TRACKING_DIR = "/tmp/mlflow_integration_test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_synthetic_transactions(n: int = 2000, fraud_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    categories = ["grocery", "electronics", "travel", "entertainment", "fuel"]
    countries = ["AR", "BR", "MX", "CL", "CO"]
    rows = [
        {
            "transaction_id": str(uuid.uuid4()),
            "user_id": f"user_{i % 100}",
            "merchant_id": f"merchant_{i % 50}",
            "merchant_category": rng.choice(categories),
            "amount": float(rng.lognormal(mean=4.0, sigma=1.0)),
            "country": rng.choice(countries),
            "device_type": rng.choice(["mobile", "desktop", "tablet"]),
            "ip_hash": f"hash_{i}",
            "timestamp": base_time + timedelta(hours=i),
            "is_fraud": bool(rng.random() < fraud_rate),
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def setup_timescaledb(container: PostgresContainer) -> dict:
    conn_params = {
        "host": container.get_container_host_ip(),
        "port": int(container.get_exposed_port(5432)),
        "user": "test_user",
        "password": "test_pass",
        "dbname": "test_tsdb",
    }
    import psycopg2

    conn = psycopg2.connect(**conn_params)
    # TimescaleDB DDL (continuous aggregates) requires autocommit.
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        cur.execute(TIMESCALEDB_MIGRATION.read_text())
    conn.close()
    return conn_params


def setup_postgresql(container: PostgresContainer) -> dict:
    conn_params = {
        "host": container.get_container_host_ip(),
        "port": int(container.get_exposed_port(5432)),
        "user": "test_user",
        "password": "test_pass",
        "dbname": "test_pg",
    }
    import psycopg2

    conn = psycopg2.connect(**conn_params)
    with conn.cursor() as cur:
        cur.execute(POSTGRESQL_MIGRATION.read_text())
    conn.commit()
    conn.close()
    return conn_params


def seed_timescaledb(conn_params: dict, df: pd.DataFrame) -> None:
    import psycopg2.extras

    psycopg2.extras.register_uuid()
    conn = psycopg2.connect(**conn_params)
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO public.transactions
                (transaction_id, user_id, merchant_id, merchant_category,
                 amount, country, device_type, ip_hash, timestamp, is_fraud)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    uuid.UUID(row["transaction_id"]),
                    row["user_id"],
                    row["merchant_id"],
                    row["merchant_category"],
                    row["amount"],
                    row["country"],
                    row["device_type"],
                    row["ip_hash"],
                    row["timestamp"].to_pydatetime()
                    if hasattr(row["timestamp"], "to_pydatetime")
                    else row["timestamp"],
                    row["is_fraud"],
                )
                for _, row in df.iterrows()
            ],
        )
    conn.commit()
    conn.close()


def _make_selection_report(X_full: pd.DataFrame, selected_features: list[str]):
    """Build a FeatureSelectionReport that selects exactly the given features."""
    from offline_features.feature_selection import FeatureSelectionReport

    all_features = X_full.columns.tolist()
    dropped = [f for f in all_features if f not in selected_features]
    return FeatureSelectionReport(
        all_features=all_features,
        selected_features=selected_features,
        dropped_features=dropped,
        drop_reason={f: "redundant" for f in dropped},
        importance_df=pd.DataFrame({"feature": all_features, "importance": [0.1] * len(all_features)}),
        redundant_pairs=[],
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrainingPipelineIntegration:
    # Class-level slots for params shared between fixture and test methods
    _tsdb_params: dict | None = None
    _pg_params: dict | None = None

    @pytest.fixture(autouse=True, scope="class")
    def setup_containers(self):
        tsdb = PostgresContainer(
            image=TIMESCALEDB_IMAGE,
            username="test_user",
            password="test_pass",
            dbname="test_tsdb",
        )
        pg = PostgresContainer(
            image=POSTGRES_IMAGE,
            username="test_user",
            password="test_pass",
            dbname="test_pg",
        )

        with tsdb, pg:
            tsdb_params = setup_timescaledb(tsdb)
            pg_params = setup_postgresql(pg)
            df = generate_synthetic_transactions(n=2000)
            seed_timescaledb(tsdb_params, df)

            # Inject config settings via env vars (pydantic-settings reads them
            # via validation_alias; direct attribute mutation would fail without
            # the required env vars already set).
            env_overrides = {
                "TIMESCALE_HOST": tsdb_params["host"],
                "TIMESCALE_PORT": str(tsdb_params["port"]),
                "TIMESCALE_USER": tsdb_params["user"],
                "TIMESCALE_PASSWORD": tsdb_params["password"],
                "TIMESCALE_DB": tsdb_params["dbname"],
                "POSTGRES_HOST": pg_params["host"],
                "POSTGRES_PORT": str(pg_params["port"]),
                "POSTGRES_USER": pg_params["user"],
                "POSTGRES_PASSWORD": pg_params["password"],
                "POSTGRES_DB": pg_params["dbname"],
                "MLFLOW_BACKEND_STORE_URI": f"sqlite:///{MLFLOW_TRACKING_DIR}/mlflow.db",
            }
            saved_env = {k: os.environ.get(k) for k in env_overrides}
            os.environ.update(env_overrides)

            import mlflow

            mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")

            TestTrainingPipelineIntegration._tsdb_params = tsdb_params
            TestTrainingPipelineIntegration._pg_params = pg_params

            try:
                yield
            finally:
                for k, old_v in saved_env.items():
                    if old_v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = old_v
                shutil.rmtree(MLFLOW_TRACKING_DIR, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    def test_load_transactions_returns_dataframe(self):
        """load_transactions debe retornar un DataFrame con >= 2000 filas."""
        from model.pipeline.train import load_transactions

        df = load_transactions(limit=None)
        assert len(df) >= 2000
        assert "is_fraud" in df.columns
        assert df["is_fraud"].notna().all()

    def test_build_features_returns_correct_columns(self):
        """build_features debe retornar un DataFrame con exactamente SELECTED_FEATURES."""
        from model.pipeline.train import build_features, load_transactions
        from model.utils.selected_features import SELECTED_FEATURES

        df = load_transactions(limit=500)
        y = df["is_fraud"].astype(int)
        output_dir = Path("/tmp/test_model_artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)

        # select_features on synthetic data (no real fraud signal) would produce
        # different importances than production data; mock it to return the
        # canonical SELECTED_FEATURES so the pipeline check always passes.
        def mock_select(X, y_s, **kwargs):
            return _make_selection_report(X, SELECTED_FEATURES)

        with patch("model.pipeline.train.select_features", side_effect=mock_select):
            X, _ = build_features(df, y, output_dir=output_dir, seed=42)

        assert list(X.columns) == SELECTED_FEATURES
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_full_training_registers_model_in_mlflow(self):
        """El pipeline completo registra el modelo en MLflow."""
        from model.pipeline.train import build_features, load_transactions, temporal_split, train_model
        from model.utils.selected_features import SELECTED_FEATURES
        from offline_features.imbalance_strategies import compute_scale_pos_weight

        df = load_transactions(limit=1000)
        y = df["is_fraud"].astype(int)
        output_dir = Path("/tmp/test_model_full")
        output_dir.mkdir(parents=True, exist_ok=True)

        def mock_select(X, y_s, **kwargs):
            return _make_selection_report(X, SELECTED_FEATURES)

        import mlflow
        import mlflow.tracking
        import mlflow.xgboost

        with patch("model.pipeline.train.select_features", side_effect=mock_select):
            with mlflow.start_run() as run:
                X, _ = build_features(df, y, output_dir=output_dir, seed=42)
                X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X, y)
                spw = compute_scale_pos_weight(y_train)
                model = train_model(X_train, y_train, X_val, y_val, spw, seed=42, params={})
                if not hasattr(model, "_estimator_type"):
                    model._estimator_type = "classifier"
                mlflow.xgboost.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="FraudDetectionModel_test",
                )
                run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("FraudDetectionModel_test")
        assert len(versions) >= 1
        assert versions[0].run_id == run_id

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_model_deployments_table_is_writable(self):
        """model_deployments puede recibir un INSERT."""
        import psycopg2

        conn = psycopg2.connect(**self.__class__._pg_params)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.model_deployments
                    (model_name, version, mlflow_run_id,
                     training_data_from, training_data_to)
                VALUES (%s, %s, %s, NOW() - INTERVAL '30 days', NOW())
                RETURNING id
                """,
                ("FraudDetectionModel_test", "1", "run_test_abc"),
            )
            row_id = cur.fetchone()[0]
        conn.commit()
        conn.close()
        assert row_id is not None
