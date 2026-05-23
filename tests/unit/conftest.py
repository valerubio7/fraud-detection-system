from datetime import UTC, datetime

from streaming.models import Transaction


def make_transaction(
    user_id: str = "user_1",
    amount: float = 100.0,
    country: str = "AR",
    merchant_id: str = "merchant_1",
    merchant_category: str = "grocery",
    timestamp: datetime | None = None,
) -> Transaction:
    return Transaction(
        transaction_id="tx_test",
        user_id=user_id,
        merchant_id=merchant_id,
        merchant_category=merchant_category,
        amount=amount,
        country=country,
        timestamp=timestamp or datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
        device_type="mobile",
        ip_hash="abc123",
    )
