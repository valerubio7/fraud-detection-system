"""Data models for window-based features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowFeatures:
    """Feature values computed from sliding transaction windows."""

    tx_count_1h: int
    tx_count_24h: int
    tx_count_7d: int
    amount_sum_1h: float
    amount_sum_24h: float
    tx_velocity_1h: float
    seconds_since_last_tx: float


@dataclass(frozen=True)
class HistoricalFeatures:
    """Feature values derived from historical user behavior."""

    amount_ratio_vs_user_avg: float
    is_country_new: float
    distinct_countries_seen: int
    is_merchant_new: float
    distinct_merchants_seen: int


__all__ = ["HistoricalFeatures", "WindowFeatures"]
