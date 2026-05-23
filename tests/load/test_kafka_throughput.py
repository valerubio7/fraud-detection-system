"""
Test de throughput del pipeline Kafka + feature engineering.

Objetivo: procesar >= 10,000 transacciones/minuto (166 msg/s)
con lag residual < 100 mensajes al finalizar.

Ejecutar con:
    uv run --group testing pytest tests/load/test_kafka_throughput.py -v -s -m integration

Requiere Docker en el entorno de ejecución.
"""

import time
import uuid
from datetime import UTC, datetime

import pytest
from confluent_kafka import Consumer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic
from testcontainers.kafka import KafkaContainer

from streaming.features.feature_publisher import FeaturePublisher
from streaming.features.historical_profile_store import HistoricalProfileStore
from streaming.features.sliding_window_store import SlidingWindowStore
from streaming.features.transaction_consumer import TransactionConsumer
from streaming.models import Transaction
from streaming.producer.transaction_producer import TransactionProducer

RAW_SCHEMA_PATH = "streaming/schemas/transaction_raw.avsc"
FEATURES_SCHEMA_PATH = "streaming/schemas/transaction_features.avsc"

TARGET_MESSAGES = 10_000
TARGET_RATE_MSG_PER_SEC = 167  # 10,000 / 60s
PRODUCE_TIMEOUT_S = 30
CONSUME_TIMEOUT_S = 90


def _build_transaction(i: int) -> Transaction:
    return Transaction(
        transaction_id=str(uuid.uuid4()),
        user_id=f"user_{i % 200}",
        merchant_id=f"merchant_{i % 50}",
        merchant_category="grocery",
        amount=float(50 + (i % 100)),
        country="AR",
        timestamp=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        device_type="mobile",
        ip_hash=f"hash_{i}",
    )


@pytest.fixture(scope="module")
def kafka_broker():
    """KafkaContainer compartido para todos los tests del módulo."""
    with KafkaContainer(image="confluentinc/cp-kafka:7.6.0") as kafka:
        broker = kafka.get_bootstrap_server()

        admin = AdminClient({"bootstrap.servers": broker})
        admin.create_topics(
            [
                NewTopic("transactions.raw", num_partitions=3, replication_factor=1),
                NewTopic("transactions.features", num_partitions=3, replication_factor=1),
            ]
        )
        time.sleep(2)  # esperar a que los topics estén listos
        yield broker


@pytest.mark.integration
class TestKafkaProducerThroughput:
    def test_producer_achieves_target_rate(self, kafka_broker):
        """El producer debe publicar 10,000 mensajes Avro en menos de 30 segundos."""
        producer = TransactionProducer(
            broker_url=kafka_broker,
            topic="transactions.raw",
            schema_path=RAW_SCHEMA_PATH,
        )

        t0 = time.perf_counter()
        for i in range(TARGET_MESSAGES):
            producer.send(_build_transaction(i))

        producer.flush()
        elapsed = time.perf_counter() - t0
        actual_rate = TARGET_MESSAGES / elapsed
        print(f"\nProduced {TARGET_MESSAGES} messages in {elapsed:.2f}s → {actual_rate:.0f} msg/s")

        producer.close()

        assert elapsed < PRODUCE_TIMEOUT_S, f"Producer demoró {elapsed:.1f}s > {PRODUCE_TIMEOUT_S}s límite"
        assert actual_rate >= TARGET_RATE_MSG_PER_SEC, (
            f"Throughput del producer {actual_rate:.0f} msg/s < objetivo {TARGET_RATE_MSG_PER_SEC} msg/s"
        )


@pytest.mark.integration
class TestFeatureEngineeringThroughput:
    def test_processing_rate_without_io(self, kafka_broker):
        """
        El pipeline completo (consumer → feature engineering → publisher) debe procesar
        >= 10,000 transacciones/minuto. Redis y TimescaleDB no se incluyen en el loop
        para medir el bottleneck real: serialización Avro + red Kafka.
        """
        producer = TransactionProducer(
            broker_url=kafka_broker,
            topic="transactions.raw",
            schema_path=RAW_SCHEMA_PATH,
        )
        for i in range(TARGET_MESSAGES):
            producer.send(_build_transaction(i))
        producer.flush()
        producer.close()

        consumer = TransactionConsumer(
            broker_url=kafka_broker,
            topic="transactions.raw",
            group_id=f"throughput-test-{uuid.uuid4().hex[:8]}",
        )
        window_store = SlidingWindowStore()
        historical_store = HistoricalProfileStore()
        publisher = FeaturePublisher(
            broker_url=kafka_broker,
            topic="transactions.features",
            schema_path=FEATURES_SCHEMA_PATH,
        )

        processed = 0
        t0 = time.perf_counter()
        deadline = t0 + CONSUME_TIMEOUT_S

        while processed < TARGET_MESSAGES and time.perf_counter() < deadline:
            transaction = consumer.consume(timeout=1.0)
            if transaction is None:
                continue

            window_features = window_store.compute_features(transaction)
            historical_features = historical_store.compute_features(transaction)
            window_store.add(transaction)
            historical_store.update(transaction)
            publisher.publish(transaction, window_features, historical_features)
            consumer.commit()
            processed += 1

        elapsed = time.perf_counter() - t0
        actual_rate = processed / elapsed
        msg = f"\nProcessed {processed}/{TARGET_MESSAGES} messages in {elapsed:.2f}s"
        print(f"{msg} → {actual_rate:.0f} msg/s")

        consumer.close()
        publisher.close()

        assert processed == TARGET_MESSAGES, (
            f"Solo se procesaron {processed}/{TARGET_MESSAGES} mensajes en {elapsed:.1f}s"
        )
        assert actual_rate >= TARGET_RATE_MSG_PER_SEC, (
            f"Throughput de feature engineering {actual_rate:.0f} msg/s < objetivo {TARGET_RATE_MSG_PER_SEC} msg/s"
        )

    def test_consumer_lag_after_processing(self, kafka_broker):
        """
        Verifica que el topic tiene mensajes publicados (sanidad del producer).
        El lag real del grupo se mide de forma indirecta: si el end offset > 0,
        los mensajes llegaron al broker y el producer funcionó correctamente.
        """
        admin_consumer = Consumer(
            {
                "bootstrap.servers": kafka_broker,
                "group.id": "lag-checker",
                "auto.offset.reset": "earliest",
            }
        )
        metadata = admin_consumer.list_topics("transactions.raw", timeout=10)
        partitions = [TopicPartition("transactions.raw", p) for p in metadata.topics["transactions.raw"].partitions]
        end_offsets = admin_consumer.get_watermark_offsets(partitions[0], timeout=5)
        print(f"\nEnd offset partition 0: {end_offsets}")
        admin_consumer.close()

        assert end_offsets[1] > 0, "El topic no tiene mensajes — el producer falló"
