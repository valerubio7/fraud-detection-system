# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

for _stub in ["pandas", "joblib"]:
    _mod = sys.modules.get(_stub)
    if isinstance(_mod, MagicMock):
        sys.modules.pop(_stub, None)
        for _key in list(sys.modules):
            if _key.startswith(_stub + "."):
                sys.modules.pop(_key, None)

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from offline_features.featurizer import (
    ALL_FEATURES,
    TransactionFeaturizer,
    _historical_features_for_user,
    _window_features_for_user,
)


def _ts(*args) -> np.ndarray:
    """Helper: build a sorted int64 nanosecond timestamp array."""
    return np.array([int(t.timestamp() * 1e9) for t in args], dtype=np.int64)


def _make_df(n: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2025, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "transaction_id": [f"tx_{i}" for i in range(n)],
            "user_id": ["user_1"] * n,
            "merchant_id": [f"m_{i % 3}" for i in range(n)],
            "merchant_category": rng.choice(["grocery", "electronics"], size=n).tolist(),
            "amount": rng.uniform(10.0, 200.0, size=n).tolist(),
            "country": rng.choice(["AR", "BR"], size=n).tolist(),
            "device_type": rng.choice(["mobile", "desktop"], size=n).tolist(),
            "timestamp": [base + timedelta(hours=i) for i in range(n)],
            "is_fraud": [0] * (n - 1) + [1],
        }
    )


# ---------------------------------------------------------------------------
# _window_features_for_user
# ---------------------------------------------------------------------------


class TestWindowFeaturesForUser:
    def test_first_tx_has_no_history(self):
        base = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
        times = _ts(base)
        amounts = np.array([100.0])
        (c1, c24, c7, s1, s24, v1, secs) = _window_features_for_user(times, amounts)
        assert c1[0] == 0
        assert c24[0] == 0
        assert c7[0] == 0
        assert s1[0] == 0.0
        assert secs[0] == pytest.approx(-1.0)

    def test_two_txs_within_1h(self):
        base = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
        times = _ts(base, base + timedelta(minutes=30))
        amounts = np.array([50.0, 75.0])
        (c1, c24, c7, s1, _, _, secs) = _window_features_for_user(times, amounts)
        assert c1[1] == 1
        assert c24[1] == 1
        assert s1[1] == pytest.approx(50.0)
        assert secs[1] == pytest.approx(1800.0)

    def test_tx_outside_1h_not_counted_in_1h(self):
        base = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
        times = _ts(base, base + timedelta(minutes=90))
        amounts = np.array([50.0, 75.0])
        (c1, c24, _, _, _, _, _) = _window_features_for_user(times, amounts)
        assert c1[1] == 0
        assert c24[1] == 1

    def test_tx_outside_7d_not_counted(self):
        base = datetime(2025, 1, 1, tzinfo=UTC)
        times = _ts(base, base + timedelta(days=8))
        amounts = np.array([50.0, 75.0])
        (_, _, c7, _, _, _, _) = _window_features_for_user(times, amounts)
        assert c7[1] == 0

    def test_amount_sum_1h_sums_within_window(self):
        base = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
        times = _ts(base, base + timedelta(minutes=20), base + timedelta(minutes=40))
        amounts = np.array([50.0, 30.0, 20.0])
        (_, _, _, s1, _, _, _) = _window_features_for_user(times, amounts)
        assert s1[2] == pytest.approx(80.0)

    def test_velocity_equals_count(self):
        base = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
        times = _ts(base, base + timedelta(minutes=30))
        amounts = np.array([50.0, 75.0])
        (c1, _, _, _, _, v1, _) = _window_features_for_user(times, amounts)
        assert v1[1] == pytest.approx(float(c1[1]))


# ---------------------------------------------------------------------------
# _historical_features_for_user
# ---------------------------------------------------------------------------


class TestHistoricalFeaturesForUser:
    def test_first_tx_all_new(self):
        base = datetime(2025, 1, 1, tzinfo=UTC)
        times = _ts(base)
        amounts = np.array([100.0])
        countries = np.array(["AR"])
        merchants = np.array(["m1"])
        ratio, cn_new, cn_dist, mer_new, mer_dist = _historical_features_for_user(times, amounts, countries, merchants)
        assert ratio[0] == pytest.approx(1.0)
        assert cn_new[0] == 1.0
        assert cn_dist[0] == 0
        assert mer_new[0] == 1.0
        assert mer_dist[0] == 0

    def test_second_tx_known_country(self):
        base = datetime(2025, 1, 1, tzinfo=UTC)
        times = _ts(base, base + timedelta(hours=1))
        amounts = np.array([100.0, 100.0])
        countries = np.array(["AR", "AR"])
        merchants = np.array(["m1", "m2"])
        _, cn_new, cn_dist, _, _ = _historical_features_for_user(times, amounts, countries, merchants)
        assert cn_new[1] == 0.0
        assert cn_dist[1] == 1

    def test_amount_ratio_vs_avg(self):
        base = datetime(2025, 1, 1, tzinfo=UTC)
        times = _ts(base, base + timedelta(hours=1), base + timedelta(hours=2))
        amounts = np.array([100.0, 100.0, 300.0])
        countries = np.array(["AR"] * 3)
        merchants = np.array(["m1"] * 3)
        ratio, _, _, _, _ = _historical_features_for_user(times, amounts, countries, merchants)
        # Third tx: avg of previous = (100+100)/2 = 100.0, current = 300.0
        assert ratio[2] == pytest.approx(3.0)

    def test_new_country_detected(self):
        base = datetime(2025, 1, 1, tzinfo=UTC)
        times = _ts(base, base + timedelta(hours=1))
        amounts = np.array([100.0, 100.0])
        countries = np.array(["AR", "BR"])
        merchants = np.array(["m1", "m1"])
        _, cn_new, _, _, _ = _historical_features_for_user(times, amounts, countries, merchants)
        assert cn_new[1] == 1.0


# ---------------------------------------------------------------------------
# TransactionFeaturizer
# ---------------------------------------------------------------------------


class TestTransactionFeaturizer:
    @pytest.fixture
    def featurizer_and_data(self):
        df = _make_df(n=20, seed=42)
        y = df["is_fraud"].astype(int)
        feat = TransactionFeaturizer()
        return feat, df, y

    def test_fit_transform_returns_all_features(self, featurizer_and_data):
        feat, df, y = featurizer_and_data
        result = feat.fit_transform(df, y)
        assert set(ALL_FEATURES).issubset(set(result.columns))

    def test_output_has_same_row_count(self, featurizer_and_data):
        feat, df, y = featurizer_and_data
        result = feat.fit_transform(df, y)
        assert len(result) == len(df)

    def test_transform_preserves_original_index(self, featurizer_and_data):
        feat, df, y = featurizer_and_data
        df_shuffled = df.sample(frac=1, random_state=42)
        feat.fit(df_shuffled, y[df_shuffled.index])
        result = feat.transform(df_shuffled)
        assert list(result.index) == list(df_shuffled.index)

    def test_transform_before_fit_raises(self):
        feat = TransactionFeaturizer()
        with pytest.raises(RuntimeError, match="fit"):
            feat.transform(_make_df())

    def test_log_amount_column_is_log1p(self, featurizer_and_data):
        feat, df, y = featurizer_and_data
        result = feat.fit_transform(df, y)
        for i, row in df.iterrows():
            expected = np.log1p(row["amount"])
            assert result.loc[i, "log_amount"] == pytest.approx(expected, rel=1e-5)

    def test_validate_columns_raises_on_missing(self):
        feat = TransactionFeaturizer()
        bad_df = pd.DataFrame({"user_id": ["a"], "amount": [10.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            feat._validate_columns(bad_df)

    def test_apply_selection_filters_columns(self, featurizer_and_data):
        from offline_features.feature_selection import FeatureSelectionReport

        feat, df, y = featurizer_and_data
        feat.fit(df, y)
        selected = ["log_amount", "hour_of_day", "tx_count_1h"]
        dropped = [f for f in ALL_FEATURES if f not in selected]
        report = FeatureSelectionReport(
            all_features=ALL_FEATURES,
            selected_features=selected,
            dropped_features=dropped,
            drop_reason={f: "redundant" for f in dropped},
            importance_df=pd.DataFrame({"feature": ALL_FEATURES, "importance": [0.05] * len(ALL_FEATURES)}),
            redundant_pairs=[],
        )
        feat.apply_selection(report)
        result = feat.transform(df)
        assert list(result.columns) == selected

    def test_get_feature_names_default(self, featurizer_and_data):
        feat, df, y = featurizer_and_data
        feat.fit(df, y)
        assert feat.get_feature_names() == ALL_FEATURES

    def test_fit_saves_encoder_when_dir_provided(self, featurizer_and_data, tmp_path):
        feat = TransactionFeaturizer(encoders_dir=tmp_path)
        df, y = featurizer_and_data[1], featurizer_and_data[2]
        feat.fit(df, y)
        assert (tmp_path / "categorical_encoder.joblib").exists()

    def test_load_existing_encoder_from_dir(self, featurizer_and_data, tmp_path):
        df, y = featurizer_and_data[1], featurizer_and_data[2]
        TransactionFeaturizer(encoders_dir=tmp_path).fit(df, y)
        loaded_feat = TransactionFeaturizer(encoders_dir=tmp_path)
        assert loaded_feat._is_fitted is True

    def test_multi_user_features_are_isolated(self):
        base = datetime(2025, 1, 1, tzinfo=UTC)
        df = pd.DataFrame(
            {
                "transaction_id": ["tx_a", "tx_b"],
                "user_id": ["user_A", "user_B"],
                "merchant_id": ["m1", "m1"],
                "merchant_category": ["grocery", "grocery"],
                "amount": [100.0, 100.0],
                "country": ["AR", "AR"],
                "device_type": ["mobile", "mobile"],
                "timestamp": [base, base + timedelta(minutes=1)],
                "is_fraud": [0, 0],
            }
        )
        y = df["is_fraud"].astype(int)
        feat = TransactionFeaturizer()
        result = feat.fit_transform(df, y)
        # Neither user has prior transactions, so tx_count_1h should be 0 for both
        assert result["tx_count_1h"].tolist() == [0, 0]
