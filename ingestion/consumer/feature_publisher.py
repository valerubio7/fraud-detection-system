"""Kafka publisher for transaction feature messages."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC
from typing import Any

from ingestion.kafka_base import AvroKafkaProducer

from .feature_models import HistoricalFeatures, WindowFeatures
from .models import TransactionRaw


class FeaturePublisher(AvroKafkaProducer):
    """Publish enriched transaction features to Kafka using Avro serialization."""

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
            client_id="fraud-feature-engineering-publisher",
            acks=acks,
            retries=retries,
            max_in_flight=max_in_flight,
        )

    def publish(
        self,
        transaction: TransactionRaw,
        window_features: WindowFeatures,
        historical_features: HistoricalFeatures,
    ) -> None:
        """Publish a transaction enriched with calculated features."""
        payload = self._build_payload(transaction, window_features, historical_features)
        self._produce(transaction.transaction_id, self._serialize_avro(payload))

    @staticmethod
    def _flatten_features(
        window_features: WindowFeatures,
        historical_features: HistoricalFeatures,
    ) -> dict[str, float]:
        return {
            **{k: float(v) for k, v in asdict(window_features).items()},
            **{k: float(v) for k, v in asdict(historical_features).items()},
        }

    def _build_payload(
        self,
        transaction: TransactionRaw,
        window_features: WindowFeatures,
        historical_features: HistoricalFeatures,
    ) -> dict[str, Any]:
        timestamp = transaction.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)
        return {
            "transaction_id": transaction.transaction_id,
            "user_id": transaction.user_id,
            "merchant_id": transaction.merchant_id,
            "merchant_category": transaction.merchant_category,
            "amount": float(transaction.amount),
            "country": transaction.country,
            "timestamp": int(timestamp.timestamp() * 1000),
            "device_type": transaction.device_type,
            "ip_hash": transaction.ip_hash,
            "features": self._flatten_features(window_features, historical_features),
        }


__all__ = ["FeaturePublisher"]
