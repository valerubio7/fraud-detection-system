import json
import logging

from redis import Redis
from redis.exceptions import RedisError

_log = logging.getLogger(__name__)


class PredictionCache:
    def __init__(self, host: str, port: int) -> None:
        self._client = Redis(host=host, port=port, socket_timeout=0.1, decode_responses=True)
        try:
            self._client.ping()
            self.is_available = True
        except Exception:
            self.is_available = False
            _log.warning("Redis not available at %s:%d — cache disabled", host, port)

    def get(self, transaction_id: str) -> dict | None:
        try:
            raw = self._client.get(f"prediction:{transaction_id}")
            return json.loads(raw) if raw is not None else None
        except RedisError as exc:
            _log.warning("Redis GET failed for %s: %s", transaction_id, exc)
            return None

    def set(self, transaction_id: str, data: dict, ttl_seconds: int = 60) -> None:
        try:
            self._client.set(f"prediction:{transaction_id}", json.dumps(data), ex=ttl_seconds)
        except RedisError as exc:
            _log.warning("Redis SET failed for %s: %s", transaction_id, exc)
