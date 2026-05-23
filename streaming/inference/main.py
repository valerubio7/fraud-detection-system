import logging
import os
import signal
import threading
import time

from streaming.inference.alert_publisher import AlertPublisher
from streaming.inference.api_client import InferenceApiClient
from streaming.inference.circuit_breaker import CircuitBreaker
from streaming.inference.feature_consumer import FeatureConsumer
from streaming.inference.prediction_publisher import PredictionPublisher

logger = logging.getLogger(__name__)

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://serving:8000")
INFERENCE_FAILURE_THRESHOLD = int(os.getenv("INFERENCE_FAILURE_THRESHOLD", "5"))
INFERENCE_COOLDOWN_SECONDS = float(os.getenv("INFERENCE_COOLDOWN_SECONDS", "30.0"))
INFERENCE_RATE_LIMIT_MS = int(os.getenv("INFERENCE_RATE_LIMIT_MS", "0"))


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


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

    consumer = FeatureConsumer(
        broker_url=os.getenv("KAFKA_BROKER_URL", "kafka:29092"),
        topic=os.getenv("KAFKA_TOPICS_FEATURES", "transactions.features"),
    )
    api_client = InferenceApiClient(base_url=FASTAPI_BASE_URL)
    publisher = PredictionPublisher(
        broker_url=os.getenv("KAFKA_BROKER_URL", "kafka:29092"),
        topic=os.getenv("KAFKA_TOPICS_PREDICTIONS", "transactions.predictions"),
    )
    alert_publisher = AlertPublisher(
        broker_url=os.getenv("KAFKA_BROKER_URL", "kafka:29092"),
        topic=os.getenv("KAFKA_TOPICS_ALERTS", "transactions.fraud.alerts"),
    )
    circuit_breaker = CircuitBreaker(
        failure_threshold=INFERENCE_FAILURE_THRESHOLD, cooldown_seconds=INFERENCE_COOLDOWN_SECONDS
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

            if circuit_breaker.is_open():
                logger.warning("Circuit breaker OPEN — skipping transaction %s", message["transaction_id"])
                continue

            try:
                prediction = api_client.predict(message)

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
                            "Failed to publish fraud alert for transaction %s: %s", message["transaction_id"], exc
                        )

                circuit_breaker.record_success()
                consumer.commit()

                if INFERENCE_RATE_LIMIT_MS > 0:
                    time.sleep(INFERENCE_RATE_LIMIT_MS / 1000)

            except Exception as exc:
                circuit_breaker.record_failure()
                logger.error(
                    "Inference failed for %s (circuit=%s): %s", message["transaction_id"], circuit_breaker.state(), exc
                )
    finally:
        alert_publisher.close()
        publisher.close()
        api_client.close()
        consumer.close()


if __name__ == "__main__":
    main()
