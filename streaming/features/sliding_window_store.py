from collections import deque
from datetime import datetime

from streaming.features.feature_types import WindowFeatures
from streaming.models import Transaction

ONE_HOUR_SECONDS = 3600
TWENTY_FOUR_HOURS_SECONDS = 24 * ONE_HOUR_SECONDS
SEVEN_DAYS_SECONDS = 7 * TWENTY_FOUR_HOURS_SECONDS


class SlidingWindowStore:
    def __init__(self, max_window_seconds: int = SEVEN_DAYS_SECONDS) -> None:
        if max_window_seconds <= 0:
            raise ValueError("max_window_seconds must be positive")
        self._max_window_seconds = max_window_seconds
        self._history: dict[str, deque[Transaction]] = {}

    def add(self, transaction: Transaction) -> None:
        history = self._history.setdefault(transaction.user_id, deque())
        history.append(transaction)
        self._evict_old(history, transaction.timestamp)

    def compute_features(self, transaction: Transaction) -> WindowFeatures:
        history = self._history.get(transaction.user_id)
        if not history:
            return WindowFeatures(
                tx_count_1h=0,
                tx_count_24h=0,
                tx_count_7d=0,
                amount_sum_1h=0.0,
                amount_sum_24h=0.0,
                tx_velocity_1h=0.0,
                seconds_since_last_tx=-1.0,
            )

        reference_time = transaction.timestamp
        self._evict_old(history, reference_time)

        tx_count_1h = 0
        tx_count_24h = 0
        tx_count_7d = 0
        amount_sum_1h = 0.0
        amount_sum_24h = 0.0
        seconds_since_last_tx = -1.0

        for item in reversed(history):
            delta_seconds = self._seconds_between(reference_time, item.timestamp)
            if delta_seconds < 0:
                continue
            if delta_seconds <= SEVEN_DAYS_SECONDS:
                tx_count_7d += 1
            if delta_seconds <= TWENTY_FOUR_HOURS_SECONDS:
                tx_count_24h += 1
                amount_sum_24h += float(item.amount)
            if delta_seconds <= ONE_HOUR_SECONDS:
                tx_count_1h += 1
                amount_sum_1h += float(item.amount)
            if seconds_since_last_tx < 0:
                seconds_since_last_tx = delta_seconds

        tx_velocity_1h = float(tx_count_1h) / 60.0

        return WindowFeatures(
            tx_count_1h=tx_count_1h,
            tx_count_24h=tx_count_24h,
            tx_count_7d=tx_count_7d,
            amount_sum_1h=amount_sum_1h,
            amount_sum_24h=amount_sum_24h,
            tx_velocity_1h=tx_velocity_1h,
            seconds_since_last_tx=seconds_since_last_tx,
        )

    def get_user_window(self, user_id: str) -> list[Transaction]:
        history = self._history.get(user_id)
        return list(history) if history else []

    def hydrate(self, transactions: list[Transaction], reference_time: datetime, max_window_seconds: int) -> None:
        cutoff = reference_time.timestamp() - max_window_seconds
        valid = sorted(
            (t for t in transactions if cutoff <= t.timestamp.timestamp() <= reference_time.timestamp()),
            key=lambda t: t.timestamp,
        )
        for transaction in valid:
            self.add(transaction)

    def _evict_old(self, history: deque[Transaction], reference_time: datetime) -> None:
        cutoff = reference_time.timestamp() - self._max_window_seconds
        while history and history[0].timestamp.timestamp() < cutoff:
            history.popleft()

    @staticmethod
    def _seconds_between(reference_time: datetime, item_time: datetime) -> float:
        return (reference_time - item_time).total_seconds()


__all__ = ["SlidingWindowStore"]
