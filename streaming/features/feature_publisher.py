from dataclasses import asdict
from datetime import UTC
from typing import Any

from streaming.features.feature_types import HistoricalFeatures, WindowFeatures
from streaming.models import Transaction
from streaming.publisher import AvroPublisher


class FeaturePublisher(AvroPublisher):
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
        transaction: Transaction,
        window_features: WindowFeatures,
        historical_features: HistoricalFeatures,
    ) -> None:
        payload = self._build_payload(transaction, window_features, historical_features)
        self._publish(transaction.transaction_id, self._serialize_avro(payload))

    @staticmethod
    def _flatten_features(window_features: WindowFeatures, historical_features: HistoricalFeatures) -> dict[str, float]:
        return {
            **{k: float(v) for k, v in asdict(window_features).items()},
            **{k: float(v) for k, v in asdict(historical_features).items()},
        }

    def _build_payload(
        self,
        transaction: Transaction,
        window_features: WindowFeatures,
        historical_features: HistoricalFeatures,
    ) -> dict[str, Any]:
        timestamp = transaction.timestamp.astimezone(UTC)
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
