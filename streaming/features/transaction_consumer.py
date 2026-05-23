import io
import json
import logging
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, KafkaException, Message
from fastavro import parse_schema, schemaless_reader

from streaming.models import Transaction

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "transaction_raw.avsc"
DEFAULT_GROUP_ID = "fraud-feature-engineering"


class TransactionConsumer:
    def __init__(
        self,
        broker_url: str = os.getenv("KAFKA_BROKER_URL", "kafka:29092"),
        topic: str = os.getenv("KAFKA_TOPICS_RAW", "transactions.raw"),
        group_id: str = DEFAULT_GROUP_ID,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,
        client_id: str = "fraud-feature-engineering-consumer",
    ) -> None:
        self._broker_url = broker_url
        self._topic = topic
        self._schema_path = Path(schema_path)
        self._avro_schema = self._load_schema(self._schema_path)

        config = {
            "bootstrap.servers": broker_url,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": enable_auto_commit,
            "client.id": client_id,
        }
        self._consumer = Consumer(config)
        self._consumer.subscribe([self._topic])

        self._last_message: Message | None = None
        self._deserialization_failures: dict[tuple[int, int], int] = {}
        self._retry_queue: deque[Message] = deque()

        logger.info("Initialized Kafka consumer for topic %s with broker %s", self._topic, self._broker_url)

    @staticmethod
    def _load_schema(schema_path: Path) -> dict[str, Any]:
        try:
            with schema_path.open("r", encoding="utf-8") as file:
                raw_schema = json.load(file)
        except FileNotFoundError:
            logger.error("Avro schema not found at %s", schema_path)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Invalid Avro schema JSON at %s", schema_path)
            raise ValueError("Invalid Avro schema JSON") from exc

        return parse_schema(raw_schema)

    def _poll_message(self, timeout: float) -> Message | None:
        try:
            return self._consumer.poll(timeout)
        except KafkaException as exc:
            logger.error("Kafka poll failed: %s", exc)
            return None

    def _deserialize(self, payload: bytes) -> dict[str, Any]:
        buffer = io.BytesIO(payload)
        return schemaless_reader(buffer, self._avro_schema)

    @staticmethod
    def _parse_timestamp(raw_value: Any) -> datetime:
        # fastavro deserializes timestamp-millis logical type directly to datetime
        if isinstance(raw_value, datetime):
            return raw_value if raw_value.tzinfo is not None else raw_value.replace(tzinfo=UTC)
        try:
            timestamp_millis = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid timestamp value") from exc
        return datetime.fromtimestamp(timestamp_millis / 1000, tz=UTC)

    def _handle_deserialization_failure(self, message: Message, exc: Exception) -> None:
        key = (message.partition(), message.offset())
        failures = self._deserialization_failures.get(key, 0) + 1
        self._deserialization_failures[key] = failures

        if failures > 1:
            logger.error(
                "Dead-lettered message at %s [%s] offset %s: %s",
                message.topic(),
                message.partition(),
                message.offset(),
                exc,
            )
            self._deserialization_failures.pop(key, None)
            return

        logger.warning(
            "Failed to deserialize message at %s [%s] offset %s: %s. Retrying once.",
            message.topic(),
            message.partition(),
            message.offset(),
            exc,
        )
        self._retry_queue.append(message)

    def consume(self, timeout: float) -> Transaction | None:
        message = self._retry_queue.popleft() if self._retry_queue else self._poll_message(timeout)
        if message is None:
            return None

        if message.error() is not None:
            logger.warning("Kafka message error: %s", message.error())
            return None

        payload = message.value()
        if payload is None:
            self._handle_deserialization_failure(message, ValueError("Message payload is empty"))
            return None

        try:
            data = self._deserialize(payload)
            transaction = Transaction.from_dict(data, self._parse_timestamp(data["timestamp"]))
        except Exception as exc:
            self._handle_deserialization_failure(message, exc)
            return None

        self._last_message = message
        return transaction

    def commit(self) -> None:
        if self._last_message is None:
            return

        try:
            self._consumer.commit(message=self._last_message, asynchronous=False)
        except KafkaException as exc:
            logger.error(
                "Failed to commit offset for %s [%s] at %s: %s",
                self._topic,
                self._last_message.partition(),
                self._last_message.offset(),
                exc,
            )
        else:
            self._last_message = None

    def close(self) -> None:
        try:
            self._consumer.close()
        except KafkaException as exc:
            logger.error("Failed to close consumer: %s", exc)


__all__ = ["TransactionConsumer"]
