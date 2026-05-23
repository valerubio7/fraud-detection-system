from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from offline_features.encoders import CategoricalEncoderPipeline

if TYPE_CHECKING:
    from offline_features.feature_selection import FeatureSelectionReport

_ONE_HOUR_NS = np.int64(3_600 * 1_000_000_000)
_TWENTY_FOUR_HOURS_NS = np.int64(86_400 * 1_000_000_000)
_SEVEN_DAYS_NS = np.int64(604_800 * 1_000_000_000)

CATEGORICAL_COLUMNS: list[str] = ["merchant_category", "country", "device_type"]
ENCODER_FILENAME = "categorical_encoder.joblib"

DIRECT_FEATURES: list[str] = [
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "merchant_category_encoded",
    "country_encoded",
    "device_type_encoded",
]
WINDOW_FEATURES: list[str] = [
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "amount_sum_1h",
    "amount_sum_24h",
    "tx_velocity_1h",
    "seconds_since_last_tx",
]
HISTORICAL_FEATURES: list[str] = [
    "amount_ratio_vs_user_avg",
    "is_country_new",
    "distinct_countries_seen",
    "is_merchant_new",
    "distinct_merchants_seen",
]
ALL_FEATURES: list[str] = DIRECT_FEATURES + WINDOW_FEATURES + HISTORICAL_FEATURES

_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "transaction_id",
        "user_id",
        "merchant_id",
        "merchant_category",
        "amount",
        "country",
        "device_type",
        "timestamp",
        "is_fraud",
    }
)


def _window_features_for_user(times_ns: np.ndarray, amounts: np.ndarray) -> tuple[np.ndarray, ...]:
    n = len(times_ns)
    tx_count_1h = np.zeros(n, dtype=np.int64)
    tx_count_24h = np.zeros(n, dtype=np.int64)
    tx_count_7d = np.zeros(n, dtype=np.int64)
    amount_sum_1h = np.zeros(n, dtype=np.float64)
    amount_sum_24h = np.zeros(n, dtype=np.float64)
    tx_velocity_1h = np.zeros(n, dtype=np.float64)
    seconds_since_last_tx = np.full(n, -1.0, dtype=np.float64)

    # prefix[b] - prefix[a] == sum(amounts[a:b])
    prefix = np.zeros(n + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(amounts)

    for i in range(n):
        t_i = times_ns[i]
        # searchsorted 'left' → first index with value >= t_i;
        # everything before is strictly prior to t_i.
        prior_end = int(np.searchsorted(times_ns, t_i, side="left"))
        if prior_end == 0:
            continue

        seconds_since_last_tx[i] = float(t_i - times_ns[prior_end - 1]) / 1e9

        s7 = int(np.searchsorted(times_ns, t_i - _SEVEN_DAYS_NS, side="left"))
        s24 = int(np.searchsorted(times_ns, t_i - _TWENTY_FOUR_HOURS_NS, side="left"))
        s1 = int(np.searchsorted(times_ns, t_i - _ONE_HOUR_NS, side="left"))

        # Clamp so we never exceed the strictly-prior boundary
        s7 = min(s7, prior_end)
        s24 = min(s24, prior_end)
        s1 = min(s1, prior_end)

        tx_count_7d[i] = prior_end - s7
        tx_count_24h[i] = prior_end - s24
        amount_sum_24h[i] = prefix[prior_end] - prefix[s24]
        tx_count_1h[i] = prior_end - s1
        amount_sum_1h[i] = prefix[prior_end] - prefix[s1]
        tx_velocity_1h[i] = float(tx_count_1h[i])

    return (
        tx_count_1h,
        tx_count_24h,
        tx_count_7d,
        amount_sum_1h,
        amount_sum_24h,
        tx_velocity_1h,
        seconds_since_last_tx,
    )


def _historical_features_for_user(
    times_ns: np.ndarray,
    amounts: np.ndarray,
    countries: np.ndarray,
    merchants: np.ndarray,
) -> tuple[np.ndarray, ...]:
    n = len(times_ns)
    amount_ratio = np.ones(n, dtype=np.float64)
    is_country_new = np.zeros(n, dtype=np.float64)
    distinct_countries = np.zeros(n, dtype=np.int64)
    is_merchant_new = np.zeros(n, dtype=np.float64)
    distinct_merchants = np.zeros(n, dtype=np.int64)

    seen_countries: set[str] = set()
    seen_merchants: set[str] = set()
    amount_total = 0.0
    amount_count = 0
    cursor = 0

    for i in range(n):
        t_i = times_ns[i]
        prior_end = int(np.searchsorted(times_ns, t_i, side="left"))

        # Advance running state to cover all strictly-prior transactions
        while cursor < prior_end:
            seen_countries.add(str(countries[cursor]))
            seen_merchants.add(str(merchants[cursor]))
            amount_total += float(amounts[cursor])
            amount_count += 1
            cursor += 1

        if amount_count == 0:
            amount_ratio[i] = 1.0
        else:
            avg = amount_total / amount_count
            amount_ratio[i] = float(amounts[i]) / avg if avg > 0.0 else 1.0

        c_i = str(countries[i])
        m_i = str(merchants[i])
        is_country_new[i] = 1.0 if c_i not in seen_countries else 0.0
        distinct_countries[i] = len(seen_countries)
        is_merchant_new[i] = 1.0 if m_i not in seen_merchants else 0.0
        distinct_merchants[i] = len(seen_merchants)

    return (amount_ratio, is_country_new, distinct_countries, is_merchant_new, distinct_merchants)


class TransactionFeaturizer:
    def __init__(self, encoders_dir: str | Path | None = None) -> None:
        self._encoders_dir = Path(encoders_dir) if encoders_dir is not None else None
        self._cat_pipeline: CategoricalEncoderPipeline | None = None
        self._is_fitted = False
        self.selected_features_: list[str] | None = None

        if self._encoders_dir is not None:
            encoder_path = self._encoders_dir / ENCODER_FILENAME
            if encoder_path.exists():
                self._cat_pipeline = CategoricalEncoderPipeline.load(encoder_path)
                self._is_fitted = True

    def fit(self, df: pd.DataFrame, y: pd.Series) -> TransactionFeaturizer:
        self._validate_columns(df)
        self._cat_pipeline = CategoricalEncoderPipeline()
        self._cat_pipeline.fit(df, y)

        if self._encoders_dir is not None:
            self._encoders_dir.mkdir(parents=True, exist_ok=True)
            self._cat_pipeline.save(self._encoders_dir / ENCODER_FILENAME)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted or self._cat_pipeline is None:
            raise RuntimeError("Call fit() before transform().")
        self._validate_columns(df)

        ts_ns = pd.to_datetime(df["timestamp"]).values.astype(np.int64)
        user_arr = df["user_id"].values.astype(str)

        # Stable sort by (user_id primary, timestamp secondary)
        sort_idx = np.lexsort([ts_ns, user_arr])
        inv_sort_idx = np.argsort(sort_idx, kind="stable")

        times_ns = ts_ns[sort_idx]
        amounts = df["amount"].values.astype(np.float64)[sort_idx]
        countries = df["country"].values.astype(str)[sort_idx]
        merchants = df["merchant_id"].values.astype(str)[sort_idx]
        user_ids = user_arr[sort_idx]

        n = len(df)
        dti = pd.DatetimeIndex(times_ns)

        # --- Direct features (fully vectorised) ---
        cat_df = pd.DataFrame(
            {
                "merchant_category": df["merchant_category"].values.astype(str)[sort_idx],
                "country": countries,
                "device_type": df["device_type"].values.astype(str)[sort_idx],
            }
        )
        cat_result = self._cat_pipeline.transform(cat_df)

        log_amount = np.log1p(amounts)
        hour_of_day = dti.hour.to_numpy(dtype=np.int64)
        day_of_week = dti.dayofweek.to_numpy(dtype=np.int64)

        # --- Per-user temporal features (initialised to no-history defaults) ---
        w_tx_1h = np.zeros(n, dtype=np.int64)
        w_tx_24h = np.zeros(n, dtype=np.int64)
        w_tx_7d = np.zeros(n, dtype=np.int64)
        w_sum_1h = np.zeros(n, dtype=np.float64)
        w_sum_24h = np.zeros(n, dtype=np.float64)
        w_vel_1h = np.zeros(n, dtype=np.float64)
        w_secs = np.full(n, -1.0, dtype=np.float64)
        h_ratio = np.ones(n, dtype=np.float64)
        h_cn_new = np.zeros(n, dtype=np.float64)
        h_cn_dist = np.zeros(n, dtype=np.int64)
        h_mer_new = np.zeros(n, dtype=np.float64)
        h_mer_dist = np.zeros(n, dtype=np.int64)

        # user_ids is sorted → np.unique returns contiguous group boundaries
        unique_users, first_idx = np.unique(user_ids, return_index=True)
        boundaries = np.concatenate([first_idx, [n]])

        for k in range(len(unique_users)):
            lo, hi = int(boundaries[k]), int(boundaries[k + 1])

            (
                w_tx_1h[lo:hi],
                w_tx_24h[lo:hi],
                w_tx_7d[lo:hi],
                w_sum_1h[lo:hi],
                w_sum_24h[lo:hi],
                w_vel_1h[lo:hi],
                w_secs[lo:hi],
            ) = _window_features_for_user(times_ns[lo:hi], amounts[lo:hi])

            (
                h_ratio[lo:hi],
                h_cn_new[lo:hi],
                h_cn_dist[lo:hi],
                h_mer_new[lo:hi],
                h_mer_dist[lo:hi],
            ) = _historical_features_for_user(times_ns[lo:hi], amounts[lo:hi], countries[lo:hi], merchants[lo:hi])

        # Build result in sorted order, then restore original row order
        out_sorted = pd.DataFrame(
            {
                "log_amount": log_amount,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "merchant_category_encoded": cat_result["merchant_category_encoded"].to_numpy(dtype=np.float64),
                "country_encoded": cat_result["country_encoded"].to_numpy(dtype=np.float64),
                "device_type_encoded": cat_result["device_type_encoded"].to_numpy(dtype=np.float64),
                "tx_count_1h": w_tx_1h,
                "tx_count_24h": w_tx_24h,
                "tx_count_7d": w_tx_7d,
                "amount_sum_1h": w_sum_1h,
                "amount_sum_24h": w_sum_24h,
                "tx_velocity_1h": w_vel_1h,
                "seconds_since_last_tx": w_secs,
                "amount_ratio_vs_user_avg": h_ratio,
                "is_country_new": h_cn_new,
                "distinct_countries_seen": h_cn_dist,
                "is_merchant_new": h_mer_new,
                "distinct_merchants_seen": h_mer_dist,
            }
        )
        out = out_sorted.iloc[inv_sort_idx].copy()
        out.index = df.index
        if self.selected_features_ is not None:
            out = out[self.selected_features_]
        return out

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def get_feature_names(self) -> list[str]:
        if self.selected_features_ is not None:
            return list(self.selected_features_)
        return list(ALL_FEATURES)

    def apply_selection(self, report: FeatureSelectionReport) -> None:
        self.selected_features_ = list(report.selected_features)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")


__all__ = [
    "TransactionFeaturizer",
    "ALL_FEATURES",
    "DIRECT_FEATURES",
    "WINDOW_FEATURES",
    "HISTORICAL_FEATURES",
]
