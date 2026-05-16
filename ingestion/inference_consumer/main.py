"""Entry point for the inference consumer."""

from __future__ import annotations

import logging
import os
import signal
import threading

from config import kafka_settings

from .alert_publisher import AlertPublisher
from .api_client import InferenceApiClient
from .features_consumer import FeaturesConsumer
from .prediction_publisher import PredictionPublisher

logger = logging.getLogger(__name__)

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://fastapi:8000")


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def install_signal_handlers(stop_event: threading.Event) -> None:
    def _handle_signal(signum, _frame) -> None:
        logger.info("Received signal %s, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


def _classify_severity(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    if score >= 0.75:
        return "HIGH"
    return "WARNING"


def main() -> None:
    configure_logging()

    consumer = FeaturesConsumer(
        broker_url=kafka_settings.broker_url,
        topic=kafka_settings.topics_features,
    )
    api_client = InferenceApiClient(base_url=FASTAPI_BASE_URL)
    publisher = PredictionPublisher(
        broker_url=kafka_settings.broker_url,
        topic=kafka_settings.topics_predictions,
    )
    alert_publisher = AlertPublisher(
        broker_url=kafka_settings.broker_url,
        topic=kafka_settings.topics_alerts,
    )

    model_info = api_client.fetch_model_info()
    deployment_id: int = model_info["deployment_id"]
    logger.info(
        "Active model: %s v%s (deployment_id=%s, threshold=%s)",
        model_info.get("model_name"),
        model_info.get("model_version"),
        deployment_id,
        model_info.get("fraud_score_threshold"),
    )

    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    try:
        while not stop_event.is_set():
            message = consumer.consume(timeout=1.0)
            if message is None:
                continue

            try:
                prediction = api_client.predict(message)
            except Exception as exc:
                logger.error(
                    "Prediction failed for transaction %s: %s",
                    message.get("transaction_id"),
                    exc,
                )
                continue

            publisher.publish(
                transaction_id=prediction["transaction_id"],
                prediction_score=prediction["prediction_score"],
                prediction_label=prediction["prediction_label"],
                model_version_id=deployment_id,
                latency_ms=prediction["latency_ms"],
            )

            if prediction["prediction_label"]:
                severity = _classify_severity(prediction["prediction_score"])
                try:
                    alert_publisher.publish(
                        transaction_id=message["transaction_id"],
                        prediction_score=prediction["prediction_score"],
                        severity=severity,
                    )
                    logger.info(
                        "Fraud alert published: transaction_id=%s score=%.4f severity=%s",
                        message["transaction_id"],
                        prediction["prediction_score"],
                        severity,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to publish fraud alert for transaction %s: %s",
                        message["transaction_id"],
                        exc,
                    )

            consumer.commit()
    finally:
        alert_publisher.close()
        publisher.close()
        api_client.close()
        consumer.close()


if __name__ == "__main__":
    main()
