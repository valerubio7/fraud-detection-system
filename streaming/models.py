from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    user_id: str
    merchant_id: str
    merchant_category: str
    amount: float
    country: str
    timestamp: datetime
    device_type: str
    ip_hash: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any], timestamp: datetime) -> Transaction:
        return cls(
            transaction_id=str(payload["transaction_id"]),
            user_id=str(payload["user_id"]),
            merchant_id=str(payload["merchant_id"]),
            merchant_category=str(payload["merchant_category"]),
            amount=float(payload["amount"]),
            country=str(payload["country"]),
            timestamp=timestamp,
            device_type=str(payload["device_type"]),
            ip_hash=str(payload["ip_hash"]),
        )


__all__ = ["Transaction"]
