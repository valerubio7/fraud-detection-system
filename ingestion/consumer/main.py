"""Entry point for the transaction consumer."""

from __future__ import annotations

import logging
import os
import signal
import threading
from datetime import datetime
from pathlib import Path

from config import kafka_settings, redis_settings, timescaledb_settings

from .feature_publisher import FeaturePublisher
from .historical_store import HistoricalProfileStore
from .kafka_consumer import TransactionConsumer
from .redis_store import RedisFeatureStore
from .timescale_writer import TimescaleWriter
from .window_store import SEVEN_DAYS_SECONDS, SlidingWindowStore

logger = logging.getLogger(__name__)


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


def hydrate_user_state(
    user_id: str,
    reference_time: datetime,
    window_store: SlidingWindowStore,
    historical_store: HistoricalProfileStore,
    redis_store: RedisFeatureStore,
    max_window_seconds: int,
) -> None:
    """Load cached state from Redis into in-memory stores."""
    window_transactions = redis_store.load_user_window(user_id)
    if window_transactions:
        window_store.hydrate(window_transactions, reference_time, max_window_seconds)

    historical_profile = redis_store.load_user_historical(user_id)
    if historical_profile:
        historical_store.hydrate(user_id, historical_profile)


def main() -> None:
    configure_logging()

    window_max_seconds = SEVEN_DAYS_SECONDS
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "transaction_features.avsc"
    consumer = TransactionConsumer(
        broker_url=kafka_settings.broker_url,
        topic=kafka_settings.topics_raw,
        group_id="fraud-feature-engineering",
    )
    window_store = SlidingWindowStore(max_window_seconds=window_max_seconds)
    historical_store = HistoricalProfileStore()
    redis_store = RedisFeatureStore(
        host=redis_settings.host,
        port=redis_settings.port,
    )
    feature_publisher = FeaturePublisher(
        broker_url=kafka_settings.broker_url,
        topic=kafka_settings.topics_features,
        schema_path=str(schema_path),
    )
    timescale_writer = TimescaleWriter(
        host=timescaledb_settings.host,
        port=timescaledb_settings.port,
        user=timescaledb_settings.user,
        password=timescaledb_settings.password,
        db=timescaledb_settings.db,
    )
    initialized_users: set[str] = set()

    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    try:
        while not stop_event.is_set():
            transaction = consumer.consume(timeout=1.0)
            if transaction is None:
                continue
            if transaction.user_id not in initialized_users:
                if redis_store.is_available:
                    hydrate_user_state(
                        transaction.user_id,
                        transaction.timestamp,
                        window_store,
                        historical_store,
                        redis_store,
                        window_max_seconds,
                    )
                initialized_users.add(transaction.user_id)
            logger.debug(
                "Consumed transaction %s for user %s",
                transaction.transaction_id,
                transaction.user_id,
            )
            window_features = window_store.compute_features(transaction)
            historical_features = historical_store.compute_features(transaction)
            window_store.add(transaction)
            historical_store.update(transaction)
            if redis_store.is_available:
                redis_store.save_user_state(
                    transaction.user_id,
                    window_store.get_user_window(transaction.user_id),
                    historical_store.to_snapshot(transaction.user_id),
                )
            if timescale_writer.is_available:
                timescale_writer.write(transaction)
            try:
                feature_publisher.publish(transaction, window_features, historical_features)
            except Exception as exc:
                logger.error(
                    "Failed to publish features for %s: %s",
                    transaction.transaction_id,
                    exc,
                )
            logger.debug(
                "Computed window features for %s: %s", transaction.transaction_id, window_features
            )
            logger.debug(
                "Computed historical features for %s: %s",
                transaction.transaction_id,
                historical_features,
            )
            consumer.commit()
    finally:
        feature_publisher.close()
        timescale_writer.close()
        redis_store.close()
        consumer.close()


if __name__ == "__main__":
    main()
