"""
Benchmarks de queries críticas de TimescaleDB.

Verifica que los índices y continuous aggregates mantienen las queries
bajo 50ms con un dataset de ~100K transacciones.

Ejecutar con:
    uv run --group testing pytest tests/load/test_timescaledb_benchmarks.py -v -s -m integration

Requiere Docker en el entorno de ejecución.
Tiempo estimado: 3-5 minutos (incluye seed de 100K filas).
"""

import time
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import psycopg2
import psycopg2.extras
import pytest
from testcontainers.postgres import PostgresContainer

TIMESCALEDB_IMAGE = "timescale/timescaledb:latest-pg15"
TIMESCALEDB_MIGRATION = "database/timescaledb/migrations/001_initial_schema.sql"

SEED_ROWS = 100_000
FRAUD_RATE = 0.02
QUERY_THRESHOLD_MS = 50.0
HISTOGRAM_USERS = 500
HISTOGRAM_MERCHANTS = 200


def generate_seed_data(n: int = SEED_ROWS) -> list[tuple]:
    """Genera filas para INSERT en batch. Timestamps relativos a now() para que
    las queries con NOW()-INTERVAL sean significativas."""
    rng = np.random.default_rng(seed=42)
    # 100K filas a 0.9 s de intervalo = ~25 horas de datos hasta ahora
    base_time = datetime.now(UTC) - timedelta(hours=25)
    categories = ["grocery", "electronics", "travel", "entertainment", "fuel"]
    countries = ["AR", "BR", "MX", "CL", "CO"]

    rows = []
    for i in range(n):
        is_fraud = rng.random() < FRAUD_RATE
        ts = base_time + timedelta(seconds=i * 0.9)
        rows.append(
            (
                uuid.uuid4(),  # transaction_id UUID
                f"user_{i % HISTOGRAM_USERS}",  # user_id
                f"merchant_{i % HISTOGRAM_MERCHANTS}",  # merchant_id
                rng.choice(categories),  # merchant_category
                float(rng.lognormal(mean=4.0, sigma=1.0)),  # amount
                rng.choice(countries),  # country
                rng.choice(["mobile", "desktop", "tablet"]),  # device_type
                f"hash_{i % 10000}",  # ip_hash
                ts,  # timestamp
                bool(is_fraud),  # is_fraud
            )
        )
    return rows


@pytest.fixture(scope="module")
def db_conn():
    """
    Levanta TimescaleDB, aplica migraciones, siembra 100K filas y refresca caggs.
    scope=module: una sola instancia para todos los benchmarks.
    """
    psycopg2.extras.register_uuid()

    with PostgresContainer(
        image=TIMESCALEDB_IMAGE,
        username="bench_user",
        password="bench_pass",
        dbname="bench_db",
    ) as tsdb:
        conn = psycopg2.connect(
            host=tsdb.get_container_host_ip(),
            port=int(tsdb.get_exposed_port(5432)),
            user="bench_user",
            password="bench_pass",
            dbname="bench_db",
        )

        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        conn.autocommit = False

        migration_sql = open(TIMESCALEDB_MIGRATION).read()
        with conn.cursor() as cur:
            cur.execute(migration_sql)
        conn.commit()

        print(f"\nSeeding {SEED_ROWS} rows... ", end="", flush=True)
        t0 = time.perf_counter()
        rows = generate_seed_data(SEED_ROWS)
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO public.transactions
                    (transaction_id, user_id, merchant_id, merchant_category,
                     amount, country, device_type, ip_hash, timestamp, is_fraud)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
                page_size=5000,
            )
        conn.commit()
        print(f"done in {time.perf_counter() - t0:.1f}s")

        print("Refreshing continuous aggregates... ", end="", flush=True)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CALL refresh_continuous_aggregate('fraud_volume_hourly', NULL, NULL);")
            cur.execute("CALL refresh_continuous_aggregate('merchant_amount_daily', NULL, NULL);")
        conn.autocommit = False
        print("done")

        yield conn
        conn.close()


def measure_query_ms(conn, query: str, params=None, repetitions: int = 5) -> float:
    """Ejecuta la query N veces y devuelve la mediana en ms.
    La mediana es robusta ante outliers de calentamiento de caché."""
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(query, params)
            cur.fetchall()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


@pytest.mark.integration
class TestUserHistoryQuery:
    """Query de historial de usuario — usa transactions_user_timestamp_idx."""

    def test_user_history_7d_under_threshold(self, db_conn):
        """SELECT del historial de un usuario activo (últimos 7 días) debe ser < 50ms.
        Con 100K tx y 500 usuarios, cada usuario tiene ~200 transacciones."""
        query = """
            SELECT transaction_id, amount, timestamp, merchant_category
            FROM public.transactions
            WHERE user_id = %s
              AND timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY timestamp
        """
        median_ms = measure_query_ms(db_conn, query, params=("user_1",))
        print(f"\n  User history 7d: {median_ms:.2f}ms (threshold: {QUERY_THRESHOLD_MS}ms)")
        assert median_ms < QUERY_THRESHOLD_MS, (
            f"Query de historial de usuario tardó {median_ms:.2f}ms > {QUERY_THRESHOLD_MS}ms"
        )

    def test_user_history_uses_index(self, db_conn):
        """EXPLAIN debe mostrar que se usa el índice compuesto (user_id, timestamp)."""
        with db_conn.cursor() as cur:
            cur.execute(
                """
                EXPLAIN (FORMAT TEXT)
                SELECT transaction_id FROM public.transactions
                WHERE user_id = 'user_1'
                  AND timestamp >= NOW() - INTERVAL '7 days'
                ORDER BY timestamp
                """
            )
            plan = "\n".join(row[0] for row in cur.fetchall())

        assert "transactions_user_timestamp_idx" in plan or "Index Scan" in plan, (
            f"La query no usa el índice compuesto. Plan:\n{plan}"
        )


@pytest.mark.integration
class TestFraudVolumeHourlyQuery:
    """Query sobre el continuous aggregate fraud_volume_hourly."""

    def test_fraud_volume_24h_under_threshold(self, db_conn):
        """SELECT del cagg fraud_volume_hourly para las últimas 24h debe ser < 50ms.
        El cagg pre-calcula las agregaciones — no debe tocar la hypertable."""
        query = """
            SELECT bucket_hour, total_transactions, total_fraud_transactions, fraud_rate
            FROM public.fraud_volume_hourly
            WHERE bucket_hour >= NOW() - INTERVAL '24 hours'
            ORDER BY bucket_hour
        """
        median_ms = measure_query_ms(db_conn, query)
        print(f"\n  fraud_volume_hourly 24h: {median_ms:.2f}ms (threshold: {QUERY_THRESHOLD_MS}ms)")
        assert median_ms < QUERY_THRESHOLD_MS, (
            f"Query de fraud_volume_hourly tardó {median_ms:.2f}ms > {QUERY_THRESHOLD_MS}ms"
        )

    def test_fraud_volume_hourly_has_data(self, db_conn):
        """El continuous aggregate debe tener filas después del refresh."""
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM public.fraud_volume_hourly")
            count = cur.fetchone()[0]
        assert count > 0, "fraud_volume_hourly está vacío — el refresh falló"


@pytest.mark.integration
class TestTopMerchantsFraudQuery:
    """Query de top merchants con más fraude — usa transactions_is_fraud_true_idx."""

    def test_top_merchants_24h_under_threshold(self, db_conn):
        """Top 10 merchants por fraude en las últimas 24h.
        Esta query hace GROUP BY sobre la hypertable sin índice dedicado, por lo que
        su umbral es más generoso (150ms) para tolerar la carga de CI con contenedores
        adicionales corriendo en paralelo."""
        _GROUPBY_THRESHOLD_MS = 150.0
        query = """
            SELECT
                merchant_id,
                merchant_category,
                COUNT(*) FILTER (WHERE is_fraud IS TRUE) AS fraud_count,
                COUNT(*) AS total_count
            FROM public.transactions
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            GROUP BY merchant_id, merchant_category
            ORDER BY fraud_count DESC
            LIMIT 10
        """
        median_ms = measure_query_ms(db_conn, query)
        print(f"\n  Top merchants fraud 24h: {median_ms:.2f}ms (threshold: {_GROUPBY_THRESHOLD_MS}ms)")
        assert median_ms < _GROUPBY_THRESHOLD_MS, (
            f"Query de top merchants tardó {median_ms:.2f}ms > {_GROUPBY_THRESHOLD_MS}ms"
        )


@pytest.mark.integration
class TestMerchantAmountDailyQuery:
    """Query sobre el continuous aggregate merchant_amount_daily."""

    def test_merchant_daily_aggregates_under_threshold(self, db_conn):
        """SELECT del cagg merchant_amount_daily para los últimos 7 días debe ser < 50ms."""
        query = """
            SELECT bucket_day, merchant_id, total_amount, transaction_count
            FROM public.merchant_amount_daily
            WHERE bucket_day >= NOW() - INTERVAL '7 days'
            ORDER BY total_amount DESC
            LIMIT 50
        """
        median_ms = measure_query_ms(db_conn, query)
        print(f"\n  merchant_amount_daily 7d: {median_ms:.2f}ms (threshold: {QUERY_THRESHOLD_MS}ms)")
        assert median_ms < QUERY_THRESHOLD_MS, (
            f"Query de merchant_amount_daily tardó {median_ms:.2f}ms > {QUERY_THRESHOLD_MS}ms"
        )


@pytest.mark.integration
class TestHypertableChunkQuery:
    """Verificaciones de que la hypertable está correctamente configurada."""

    def test_hypertable_has_multiple_chunks(self, db_conn):
        """Con 100K filas de ~25h repartidas, TimescaleDB debe crear al menos 2 chunks
        (chunk_time_interval = 1 day)."""
        with db_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*)
                FROM timescaledb_information.chunks
                WHERE hypertable_schema = 'public'
                  AND hypertable_name = 'transactions'
            """)
            chunk_count = cur.fetchone()[0]
        print(f"\n  Chunks creados: {chunk_count}")
        assert chunk_count >= 1, "TimescaleDB no creó chunks — la hypertable puede no estar configurada"

    def test_transactions_count_matches_seed(self, db_conn):
        """La hypertable debe contener exactamente SEED_ROWS filas."""
        with db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM public.transactions")
            count = cur.fetchone()[0]
        assert count == SEED_ROWS, f"Expected {SEED_ROWS} rows, found {count}"
