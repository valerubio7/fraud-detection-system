import time

from streaming.inference.circuit_breaker import CircuitBreaker


class TestCircuitBreakerInitialState:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state() == "CLOSED"

    def test_initial_is_not_open(self):
        cb = CircuitBreaker()
        assert cb.is_open() is False


class TestCircuitBreakerOpening:
    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state() == "OPEN"
        assert cb.is_open() is True

    def test_does_not_open_below_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(2):
            cb.record_failure()
        assert cb.state() == "CLOSED"

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.state() == "CLOSED"


class TestCircuitBreakerHalfOpen:
    def test_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state() == "OPEN"
        time.sleep(0.02)
        is_open = cb.is_open()
        assert is_open is False
        assert cb.state() == "HALF_OPEN"

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.is_open()  # triggers HALF_OPEN transition
        cb.record_failure()
        assert cb.state() == "OPEN"

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.is_open()  # triggers HALF_OPEN transition
        cb.record_success()
        assert cb.state() == "CLOSED"
        assert cb.is_open() is False

    def test_still_open_within_cooldown(self):
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=3600.0)
        cb.record_failure()
        assert cb.is_open() is True
        assert cb.state() == "OPEN"
