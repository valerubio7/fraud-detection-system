from datetime import UTC, datetime, timedelta

import pytest

from streaming.features.sliding_window_store import SlidingWindowStore
from tests.unit.conftest import make_transaction


class TestSlidingWindowStoreNewUser:
    def test_new_user_returns_zero_features(self):
        store = SlidingWindowStore()
        tx = make_transaction()
        features = store.compute_features(tx)
        assert features.tx_count_1h == 0
        assert features.tx_count_24h == 0
        assert features.tx_count_7d == 0
        assert features.amount_sum_1h == 0.0
        assert features.seconds_since_last_tx == -1.0


class TestSlidingWindowStoreWindowCounts:
    def test_tx_within_1h_is_counted(self):
        base = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        store = SlidingWindowStore()
        for i in range(2):
            tx = make_transaction(timestamp=base + timedelta(minutes=30 * i))
            store.add(tx)
        query_tx = make_transaction(timestamp=base + timedelta(minutes=60))
        features = store.compute_features(query_tx)
        assert features.tx_count_1h == 2
        assert features.tx_count_24h == 2
        assert features.tx_count_7d == 2

    def test_tx_outside_1h_not_counted_in_1h(self):
        base = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        store = SlidingWindowStore()
        store.add(make_transaction(timestamp=base))
        query_tx = make_transaction(timestamp=base + timedelta(minutes=90))
        features = store.compute_features(query_tx)
        assert features.tx_count_1h == 0
        assert features.tx_count_24h == 1

    def test_amount_sum_1h_is_correct(self):
        base = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        store = SlidingWindowStore()
        store.add(make_transaction(amount=50.0, timestamp=base))
        store.add(make_transaction(amount=75.0, timestamp=base + timedelta(minutes=20)))
        query_tx = make_transaction(amount=100.0, timestamp=base + timedelta(minutes=40))
        features = store.compute_features(query_tx)
        assert features.amount_sum_1h == pytest.approx(125.0)

    def test_seconds_since_last_tx(self):
        base = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        store = SlidingWindowStore()
        store.add(make_transaction(timestamp=base))
        query_tx = make_transaction(timestamp=base + timedelta(seconds=300))
        features = store.compute_features(query_tx)
        assert features.seconds_since_last_tx == pytest.approx(300.0)

    def test_eviction_beyond_7_days(self):
        base = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        store = SlidingWindowStore()
        store.add(make_transaction(timestamp=base))
        query_tx = make_transaction(timestamp=base + timedelta(days=8))
        features = store.compute_features(query_tx)
        assert features.tx_count_7d == 0


class TestSlidingWindowStoreUserIsolation:
    def test_different_users_are_isolated(self):
        base = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        store = SlidingWindowStore()
        store.add(make_transaction(user_id="user_A", amount=100.0, timestamp=base))
        features_b = store.compute_features(make_transaction(user_id="user_B", timestamp=base))
        assert features_b.tx_count_1h == 0
        assert features_b.tx_count_24h == 0
        assert features_b.seconds_since_last_tx == -1.0


class TestSlidingWindowStoreHydrate:
    def test_hydrate_restores_window(self):
        base = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        transactions = [make_transaction(timestamp=base + timedelta(hours=i)) for i in range(3)]
        store = SlidingWindowStore()
        store.hydrate(
            transactions,
            reference_time=base + timedelta(hours=3),
            max_window_seconds=3600 * 24 * 7,
        )
        query_tx = make_transaction(timestamp=base + timedelta(hours=3))
        features = store.compute_features(query_tx)
        assert features.tx_count_24h == 3
