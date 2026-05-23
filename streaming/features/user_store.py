import json
import logging
from datetime import UTC, datetime
from typing import Any

import redis

from streaming.models import Transaction

logger = logging.getLogger(__name__)

WINDOW_KEY_PREFIX = "features:window"
HISTORICAL_KEY_PREFIX = "features:historical"
MAX_WINDOW_TRANSACTIONS = 500


class UserStore:
    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        key_ttl_seconds: int = 7 * 24 * 3600,
        socket_timeout: float = 0.5,
    ) -> None:
        self._key_ttl_seconds = key_ttl_seconds
        self._client: redis.Redis | None = None
        self._is_available = False

        try:
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                socket_timeout=socket_timeout,
                decode_responses=True,
            )
            client.ping()
        except redis.RedisError as exc:
            logger.warning("Redis unavailable at %s:%s: %s", host, port, exc)
        else:
            self._client = client
            self._is_available = True
            logger.info("Connected to Redis at %s:%s", host, port)

    @property
    def is_available(self) -> bool:
        return self._is_available

    def save_user_state(
        self,
        user_id: str,
        transactions: list[Transaction],
        historical_profile: dict[str, Any],
    ) -> None:
        if not self._is_available or self._client is None:
            logger.debug("Redis unavailable; skipping save for user %s", user_id)
            return

        window_key = self._window_key(user_id)
        historical_key = self._historical_key(user_id)

        trimmed = transactions[-MAX_WINDOW_TRANSACTIONS:]
        window_payload = [self._serialize_transaction(t) for t in trimmed]

        try:
            self._client.set(window_key, json.dumps(window_payload), ex=self._key_ttl_seconds)
            self._client.set(historical_key, json.dumps(historical_profile), ex=self._key_ttl_seconds)
        except redis.RedisError as exc:
            self._handle_redis_error("save", user_id, exc)

    def load_user_window(self, user_id: str) -> list[Transaction]:
        raw = self._get_json(self._window_key(user_id), "load window", user_id)
        if raw is None:
            return []
        if not isinstance(raw, list):
            logger.error("Invalid window payload for user %s", user_id)
            return []
        transactions: list[Transaction] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                transactions.append(self._deserialize_transaction(item))
            except (KeyError, ValueError) as exc:
                logger.error("Failed to deserialize transaction for user %s: %s", user_id, exc)
        return transactions

    def load_user_historical(self, user_id: str) -> dict[str, Any] | None:
        raw = self._get_json(self._historical_key(user_id), "load historical", user_id)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            logger.error("Invalid historical payload for user %s", user_id)
            return None
        return raw

    def _get_json(self, key: str, action: str, user_id: str) -> Any:
        if not self._is_available or self._client is None:
            logger.debug("Redis unavailable; skipping %s for user %s", action, user_id)
            return None
        try:
            payload = self._client.get(key)
        except redis.RedisError as exc:
            self._handle_redis_error(action, user_id, exc)
            return None
        if payload is None:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode %s payload for user %s: %s", action, user_id, exc)
            return None

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.close()
        except redis.RedisError as exc:
            logger.error("Failed to close Redis client: %s", exc)

    @staticmethod
    def _window_key(user_id: str) -> str:
        return f"{WINDOW_KEY_PREFIX}:{user_id}"

    @staticmethod
    def _historical_key(user_id: str) -> str:
        return f"{HISTORICAL_KEY_PREFIX}:{user_id}"

    @staticmethod
    def _serialize_transaction(transaction: Transaction) -> dict[str, Any]:
        return {
            "transaction_id": transaction.transaction_id,
            "user_id": transaction.user_id,
            "merchant_id": transaction.merchant_id,
            "merchant_category": transaction.merchant_category,
            "amount": float(transaction.amount),
            "country": transaction.country,
            "timestamp": transaction.timestamp.astimezone(UTC).isoformat(),
            "device_type": transaction.device_type,
            "ip_hash": transaction.ip_hash,
        }

    @staticmethod
    def _deserialize_transaction(payload: dict[str, Any]) -> Transaction:
        return Transaction.from_dict(payload, UserStore._parse_timestamp(payload["timestamp"]))

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if not isinstance(value, str):
            raise ValueError("Timestamp must be a string")
        text = value if not value.endswith("Z") else f"{value[:-1]}+00:00"
        timestamp = datetime.fromisoformat(text)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC)

    def _handle_redis_error(self, action: str, user_id: str, exc: Exception) -> None:
        logger.error("Redis %s failed for user %s: %s", action, user_id, exc)
        self._is_available = False


__all__ = ["UserStore"]
