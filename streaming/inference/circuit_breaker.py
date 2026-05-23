import logging
import time

logger = logging.getLogger(__name__)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._state = "CLOSED"
        self._failure_count = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self._state == "OPEN":
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._state = "HALF_OPEN"
                logger.info("Circuit breaker transitioned to HALF_OPEN")
                return False
            return True
        return False

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._state == "HALF_OPEN":
            self._state = "OPEN"
            self._opened_at = time.monotonic()
            return
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            self._opened_at = time.monotonic()
            logger.warning("Circuit breaker OPEN after %d consecutive failures", self._failure_count)

    def state(self) -> str:
        return self._state


__all__ = ["CircuitBreaker"]
