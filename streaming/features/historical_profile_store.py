from dataclasses import dataclass
from typing import Any

from streaming.features.feature_types import HistoricalFeatures
from streaming.models import Transaction


@dataclass
class _UserProfile:
    amount_total: float
    amount_count: int
    countries_seen: set[str]
    merchants_seen: set[str]


class HistoricalProfileStore:
    def __init__(self) -> None:
        self._profiles: dict[str, _UserProfile] = {}

    def compute_features(self, transaction: Transaction) -> HistoricalFeatures:
        profile = self._profiles.get(transaction.user_id)
        if profile is None or profile.amount_count == 0:
            ratio = 1.0
            countries_seen = set()
            merchants_seen = set()
        else:
            avg_amount = profile.amount_total / profile.amount_count
            ratio = float(transaction.amount) / avg_amount if avg_amount > 0 else 1.0
            countries_seen = profile.countries_seen
            merchants_seen = profile.merchants_seen

        is_country_new = 1.0 if transaction.country not in countries_seen else 0.0
        is_merchant_new = 1.0 if transaction.merchant_id not in merchants_seen else 0.0

        return HistoricalFeatures(
            amount_ratio_vs_user_avg=ratio,
            is_country_new=is_country_new,
            distinct_countries_seen=len(countries_seen),
            is_merchant_new=is_merchant_new,
            distinct_merchants_seen=len(merchants_seen),
        )

    def update(self, transaction: Transaction) -> None:
        profile = self._profiles.get(transaction.user_id)
        if profile is None:
            profile = _UserProfile(
                amount_total=0.0,
                amount_count=0,
                countries_seen=set(),
                merchants_seen=set(),
            )
            self._profiles[transaction.user_id] = profile

        profile.amount_total += float(transaction.amount)
        profile.amount_count += 1
        profile.countries_seen.add(transaction.country)
        profile.merchants_seen.add(transaction.merchant_id)

    def to_snapshot(self, user_id: str) -> dict[str, Any]:
        profile = self._profiles.get(user_id)
        if profile is None:
            return {
                "amount_total": 0.0,
                "amount_count": 0,
                "countries_seen": [],
                "merchants_seen": [],
            }
        return {
            "amount_total": float(profile.amount_total),
            "amount_count": int(profile.amount_count),
            "countries_seen": sorted(profile.countries_seen),
            "merchants_seen": sorted(profile.merchants_seen),
        }

    def hydrate(self, user_id: str, raw_profile: dict[str, Any]) -> None:
        countries_raw = raw_profile.get("countries_seen", [])
        merchants_raw = raw_profile.get("merchants_seen", [])
        self._profiles[user_id] = _UserProfile(
            amount_total=float(raw_profile.get("amount_total", 0.0)),
            amount_count=int(raw_profile.get("amount_count", 0)),
            countries_seen={str(c) for c in countries_raw} if isinstance(countries_raw, list) else set(),
            merchants_seen={str(m) for m in merchants_raw} if isinstance(merchants_raw, list) else set(),
        )


__all__ = ["HistoricalProfileStore"]
