# ruff: noqa: E402
import sys

# The root conftest stubs heavy packages (redis, psycopg2, pandas, mlflow, …) as
# MagicMocks so unit tests can import serving/consumer modules without installing
# them. Remove those stubs here so integration tests import the real packages.
_STUB_ROOTS = {
    "asyncpg",
    "confluent_kafka",
    "fastavro",
    "imblearn",
    "joblib",
    "mlflow",
    "pandas",
    # prometheus_fastapi_instrumentator is kept as a MagicMock stub — the real
    # package is not needed for integration tests and removing the stub breaks
    # serving/app/main.py imports.
    "psycopg2",
    "redis",
    "xgboost",
}
for _mod in list(sys.modules):
    root = _mod.split(".")[0]
    if root in _STUB_ROOTS:
        del sys.modules[_mod]

from pathlib import Path

import pytest
from testcontainers.kafka import KafkaContainer
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

TIMESCALEDB_IMAGE = "timescale/timescaledb:latest-pg15"
TIMESCALEDB_MIGRATION = Path("database/timescaledb/migrations/001_initial_schema.sql")


@pytest.fixture(scope="session")
def kafka_container():
    with KafkaContainer(image="confluentinc/cp-kafka:7.6.0") as kafka:
        yield kafka


@pytest.fixture(scope="session")
def timescaledb_container():
    with PostgresContainer(
        image=TIMESCALEDB_IMAGE,
        username="test_user",
        password="test_pass",
        dbname="test_tsdb",
    ) as tsdb:
        import psycopg2

        conn = psycopg2.connect(
            host=tsdb.get_container_host_ip(),
            port=tsdb.get_exposed_port(5432),
            user="test_user",
            password="test_pass",
            dbname="test_tsdb",
        )
        # TimescaleDB DDL (including continuous aggregates) requires autocommit.
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            cur.execute(TIMESCALEDB_MIGRATION.read_text())
        conn.close()
        yield tsdb


@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer(image="redis:7.2-alpine") as redis:
        yield redis
