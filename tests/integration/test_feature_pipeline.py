import io
import time
import uuid
from datetime import UTC, datetime

import fastavro
import pytest

from streaming.features.feature_publisher import FeaturePublisher
from streaming.features.feature_types import HistoricalFeatures, WindowFeatures
from streaming.features.historical_profile_store import HistoricalProfileStore
from streaming.features.sliding_window_store import SlidingWindowStore
from streaming.features.transaction_store import TransactionStore
from streaming.features.user_store import UserStore
from streaming.models import Transaction

FEATURES_SCHEMA_PATH = "streaming/schemas/transaction_features.avsc"


def make_test_transaction(user_id: str = "user_int_test") -> Transaction:
    return Transaction(
        transaction_id=str(uuid.uuid4()),
        user_id=user_id,
        merchant_id="merchant_001",
        merchant_category="grocery",
        amount=99.99,
        country="AR",
        timestamp=datetime(2025, 1, 15, 14, 30, 0, tzinfo=UTC),
        device_type="mobile",
        ip_hash="hash_abc",
    )


@pytest.mark.integration
class TestTransactionStoreIntegration:
    def test_write_transaction_to_timescaledb(self, timescaledb_container):
        """TransactionStore inserta correctamente una Transaction en la hypertable."""
        tx = make_test_transaction()
        writer = TransactionStore(
            host=timescaledb_container.get_container_host_ip(),
            port=int(timescaledb_container.get_exposed_port(5432)),
            user="test_user",
            password="test_pass",
            db="test_tsdb",
        )
        assert writer.is_available, "TransactionStore debe estar disponible"

        writer.write(tx)

        import psycopg2

        conn = psycopg2.connect(
            host=timescaledb_container.get_container_host_ip(),
            port=int(timescaledb_container.get_exposed_port(5432)),
            user="test_user",
            password="test_pass",
            dbname="test_tsdb",
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM public.transactions WHERE transaction_id = %s",
                (uuid.UUID(tx.transaction_id),),
            )
            count = cur.fetchone()[0]
        conn.close()
        writer.close()

        assert count == 1

    def test_write_is_idempotent(self, timescaledb_container):
        """Escribir la misma transacción dos veces no duplica filas (ON CONFLICT DO NOTHING)."""
        tx = make_test_transaction(user_id="user_idempotent")
        writer = TransactionStore(
            host=timescaledb_container.get_container_host_ip(),
            port=int(timescaledb_container.get_exposed_port(5432)),
            user="test_user",
            password="test_pass",
            db="test_tsdb",
        )
        writer.write(tx)
        writer.write(tx)

        import psycopg2

        conn = psycopg2.connect(
            host=timescaledb_container.get_container_host_ip(),
            port=int(timescaledb_container.get_exposed_port(5432)),
            user="test_user",
            password="test_pass",
            dbname="test_tsdb",
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM public.transactions WHERE transaction_id = %s",
                (uuid.UUID(tx.transaction_id),),
            )
            count = cur.fetchone()[0]
        conn.close()
        writer.close()

        assert count == 1


@pytest.mark.integration
class TestFeaturePipelineKafkaIntegration:
    def test_feature_publisher_produces_to_kafka(self, kafka_container):
        """FeaturePublisher publica un mensaje Avro válido al topic transactions.features."""
        from confluent_kafka import Consumer
        from confluent_kafka.admin import AdminClient, NewTopic

        broker = kafka_container.get_bootstrap_server()

        admin = AdminClient({"bootstrap.servers": broker})
        admin.create_topics(
            [
                NewTopic("transactions.features", num_partitions=1, replication_factor=1),
            ]
        )
        time.sleep(1)

        publisher = FeaturePublisher(
            broker_url=broker,
            topic="transactions.features",
            schema_path=FEATURES_SCHEMA_PATH,
        )
        tx = make_test_transaction()
        window_f = WindowFeatures(
            tx_count_1h=2,
            tx_count_24h=5,
            tx_count_7d=20,
            amount_sum_1h=200.0,
            amount_sum_24h=500.0,
            tx_velocity_1h=2 / 60,
            seconds_since_last_tx=300.0,
        )
        historical_f = HistoricalFeatures(
            amount_ratio_vs_user_avg=1.0,
            is_country_new=0.0,
            distinct_countries_seen=2,
            is_merchant_new=1.0,
            distinct_merchants_seen=5,
        )
        publisher.publish(tx, window_f, historical_f)
        publisher.flush()
        publisher.close()

        consumer = Consumer(
            {
                "bootstrap.servers": broker,
                "group.id": "test-verifier",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe(["transactions.features"])

        received = None
        deadline = time.time() + 15
        while time.time() < deadline:
            msg = consumer.poll(timeout=1.0)
            if msg is not None and not msg.error():
                received = msg.value()
                break
        consumer.close()

        assert received is not None, "Debe haber recibido un mensaje en transactions.features"

        schema = fastavro.schema.load_schema(FEATURES_SCHEMA_PATH)
        decoded = fastavro.schemaless_reader(io.BytesIO(received), schema)
        assert decoded["transaction_id"] == tx.transaction_id
        assert "features" in decoded
        assert "tx_count_1h" in decoded["features"]


@pytest.mark.integration
class TestUserStoreIntegration:
    def test_save_and_load_user_window(self, redis_container):
        """UserStore guarda y recupera el estado del usuario correctamente."""
        store = UserStore(
            host=redis_container.get_container_host_ip(),
            port=int(redis_container.get_exposed_port(6379)),
        )
        assert store.is_available, "UserStore debe estar disponible"

        tx = make_test_transaction(user_id="user_redis_test")
        window_store = SlidingWindowStore()
        window_store.add(tx)
        historical_store = HistoricalProfileStore()
        historical_store.compute_features(tx)
        historical_store.update(tx)

        store.save_user_state(
            tx.user_id,
            window_store.get_user_window(tx.user_id),
            historical_store.to_snapshot(tx.user_id),
        )

        loaded_window = store.load_user_window(tx.user_id)
        loaded_profile = store.load_user_historical(tx.user_id)

        assert loaded_window is not None
        assert len(loaded_window) == 1
        assert loaded_profile is not None
        assert loaded_profile["amount_count"] == 1
        store.close()
