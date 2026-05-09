"""Kafka producer for transaction events."""

from __future__ import annotations

from typing import Any

from ingestion.kafka_base import AvroKafkaProducer

from .models import Transaction


class TransactionProducer(AvroKafkaProducer):
    """Produce transaction events to Kafka using Avro serialization."""

    def __init__(
        self,
        broker_url: str,
        topic: str,
        schema_path: str,
        acks: str = "all",
        retries: int = 3,
        max_in_flight: int = 5,
    ) -> None:
        super().__init__(
            broker_url=broker_url,
            topic=topic,
            schema_path=schema_path,
            client_id="fraud-detection-producer",
            acks=acks,
            retries=retries,
            max_in_flight=max_in_flight,
        )

    def send(self, transaction: Transaction) -> None:
        """Send a transaction to the Kafka topic."""
        self._produce(
            str(transaction.transaction_id),
            self._serialize_avro(self._transaction_to_dict(transaction)),
        )

    def _transaction_to_dict(self, transaction: Transaction) -> dict[str, Any]:
        timestamp = transaction.timestamp
        if timestamp.tzinfo is None:
            raise ValueError("Transaction timestamp must be timezone-aware")
        return {
            "transaction_id": str(transaction.transaction_id),
            "user_id": transaction.user_id,
            "merchant_id": transaction.merchant_id,
            "merchant_category": transaction.merchant_category,
            "amount": float(transaction.amount),
            "country": transaction.country,
            "timestamp": int(timestamp.timestamp() * 1000),
            "device_type": transaction.device_type,
            "ip_hash": transaction.ip_hash,
        }


__all__ = ["TransactionProducer"]
