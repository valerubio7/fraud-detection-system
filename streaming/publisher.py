import io
import json
import logging
from pathlib import Path
from typing import Any

from confluent_kafka import KafkaException, Producer
from fastavro import parse_schema, schemaless_writer

logger = logging.getLogger(__name__)


class AvroPublisher:
    def __init__(
        self,
        broker_url: str,
        topic: str,
        schema_path: str,
        client_id: str,
        acks: str = "all",
        retries: int = 3,
        max_in_flight: int = 5,
    ) -> None:
        self._broker_url = broker_url
        self._topic = topic
        self._avro_schema = self._load_schema(Path(schema_path))

        self._producer = Producer(
            {
                "bootstrap.servers": broker_url,
                "acks": acks,
                "retries": retries,
                "max.in.flight.requests.per.connection": max_in_flight,
                "compression.type": "snappy",
                "enable.idempotence": True,
                "client.id": client_id,
            }
        )
        logger.info("Initialized Kafka publisher for topic %s with broker %s", topic, broker_url)

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

    def _serialize_avro(self, data: dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        schemaless_writer(buffer, self._avro_schema, data)
        return buffer.getvalue()

    def _publish(self, key: str, payload: bytes) -> None:
        try:
            self._producer.produce(
                self._topic,
                key=key,
                value=payload,
                on_delivery=lambda err, msg: self._delivery_callback(err, msg, key),
            )
            self._producer.poll(0.0)
        except KafkaException as exc:
            logger.error("Failed to publish message %s: %s", key, exc)
            raise
        except BufferError as exc:
            logger.error("Publisher queue full for message %s: %s", key, exc)
            raise

    def flush(self) -> None:
        try:
            remaining = self._producer.flush()
        except KafkaException as exc:
            logger.error("Failed to flush producer: %s", exc)
            raise
        if remaining:
            logger.warning("Flush completed with %s messages remaining", remaining)

    def close(self) -> None:
        logger.info("Closing Kafka publisher for topic %s", self._topic)
        self.flush()

    def _delivery_callback(self, error: Exception | None, message: Any, key: str) -> None:
        if error is not None:
            logger.error("Delivery failed for %s: %s", key, error)
            return
        logger.debug(
            "Delivered %s to %s [%s] at offset %s",
            key,
            message.topic(),
            message.partition(),
            message.offset(),
        )


__all__ = ["AvroPublisher"]
