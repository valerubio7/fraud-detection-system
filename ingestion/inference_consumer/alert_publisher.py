from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ingestion.avro_producer import AvroKafkaProducer

_SCHEMA_PATH = str(Path(__file__).resolve().parents[1] / "schemas" / "fraud_alert.avsc")


class AlertPublisher(AvroKafkaProducer):
    """Publish fraud alerts to Kafka using Avro serialization."""

    def __init__(self, broker_url: str, topic: str) -> None:
        super().__init__(
            broker_url=broker_url,
            topic=topic,
            schema_path=_SCHEMA_PATH,
            client_id="fraud-alert-publisher",
        )

    def publish(self, transaction_id: str, prediction_score: float, severity: str) -> None:
        payload = {
            "transaction_id": transaction_id,
            "alert_type": "FRAUD_DETECTED",
            "severity": severity,
            "prediction_score": prediction_score,
            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
        }
        self._produce(transaction_id, self._serialize_avro(payload))


__all__ = ["AlertPublisher"]
