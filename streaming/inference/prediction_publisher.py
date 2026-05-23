from datetime import UTC, datetime
from pathlib import Path

from streaming.publisher import AvroPublisher

_SCHEMA_PATH = str(Path(__file__).resolve().parents[1] / "schemas" / "transaction_prediction.avsc")


class PredictionPublisher(AvroPublisher):
    def __init__(self, broker_url: str, topic: str) -> None:
        super().__init__(
            broker_url=broker_url, topic=topic, schema_path=_SCHEMA_PATH, client_id="fraud-inference-publisher"
        )

    def publish(
        self,
        transaction_id: str,
        prediction_score: float,
        prediction_label: bool,
        model_version_id: int,
        latency_ms: float,
    ) -> None:
        payload = {
            "transaction_id": transaction_id,
            "prediction_score": prediction_score,
            "prediction_label": prediction_label,
            "model_version_id": model_version_id,
            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
            "latency_ms": latency_ms,
        }
        self._publish(transaction_id, self._serialize_avro(payload))


__all__ = ["PredictionPublisher"]
