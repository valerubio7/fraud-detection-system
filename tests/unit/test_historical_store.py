import pytest

from streaming.features.historical_profile_store import HistoricalProfileStore
from tests.unit.conftest import make_transaction


class TestHistoricalProfileStoreNewUser:
    def test_new_user_ratio_is_one(self):
        store = HistoricalProfileStore()
        tx = make_transaction(amount=500.0)
        features = store.compute_features(tx)
        assert features.amount_ratio_vs_user_avg == pytest.approx(1.0)

    def test_new_user_all_new_flags(self):
        store = HistoricalProfileStore()
        tx = make_transaction(country="AR", merchant_id="m1")
        features = store.compute_features(tx)
        assert features.is_country_new == 1.0
        assert features.is_merchant_new == 1.0
        assert features.distinct_countries_seen == 0
        assert features.distinct_merchants_seen == 0


class TestHistoricalProfileStoreReturningUser:
    def test_known_country_is_not_new(self):
        store = HistoricalProfileStore()
        tx1 = make_transaction(country="AR", merchant_id="m1")
        store.compute_features(tx1)
        store.update(tx1)
        tx2 = make_transaction(country="AR", merchant_id="m2")
        features = store.compute_features(tx2)
        assert features.is_country_new == 0.0
        assert features.distinct_countries_seen == 1

    def test_new_country_detected(self):
        store = HistoricalProfileStore()
        tx1 = make_transaction(country="AR")
        store.compute_features(tx1)
        store.update(tx1)
        tx2 = make_transaction(country="BR")
        features = store.compute_features(tx2)
        assert features.is_country_new == 1.0

    def test_amount_ratio_vs_known_avg(self):
        store = HistoricalProfileStore()
        for _ in range(2):
            tx = make_transaction(amount=100.0)
            store.compute_features(tx)
            store.update(tx)
        tx_big = make_transaction(amount=200.0)
        features = store.compute_features(tx_big)
        assert features.amount_ratio_vs_user_avg == pytest.approx(2.0)


class TestHistoricalProfileStoreSnapshot:
    def test_snapshot_of_unknown_user_returns_empty(self):
        store = HistoricalProfileStore()
        snap = store.to_snapshot("user_never_seen")
        assert snap["amount_count"] == 0
        assert snap["amount_total"] == 0.0
        assert snap["countries_seen"] == []
        assert snap["merchants_seen"] == []

    def test_snapshot_roundtrip(self):
        store = HistoricalProfileStore()
        tx = make_transaction(amount=100.0, country="AR", merchant_id="m1")
        store.compute_features(tx)
        store.update(tx)
        snapshot = store.to_snapshot("user_1")
        new_store = HistoricalProfileStore()
        new_store.hydrate("user_1", snapshot)
        tx2 = make_transaction(amount=200.0, country="AR")
        features = new_store.compute_features(tx2)
        assert features.is_country_new == 0.0
        assert features.amount_ratio_vs_user_avg == pytest.approx(2.0)
