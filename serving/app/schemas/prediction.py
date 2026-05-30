from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user_AR_001",
                "merchant_id": "merchant_supermaxi",
                "merchant_category": "grocery",
                "amount": 150.75,
                "country": "AR",
                "timestamp": "2025-01-15T14:30:00Z",
                "device_type": "mobile",
                "ip_hash": "a3f8b2c1d4e5",
                "features": {
                    "tx_count_1h": 3.0,
                    "tx_count_24h": 10.0,
                    "tx_count_7d": 52.0,
                    "amount_sum_1h": 320.50,
                    "amount_sum_24h": 1080.00,
                    "seconds_since_last_tx": 1800.0,
                    "amount_ratio_vs_user_avg": 1.4,
                    "is_country_new": 0.0,
                    "is_merchant_new": 0.0,
                    "distinct_merchants_seen": 8.0,
                },
            }
        }
    )

    transaction_id: str = Field(description="Unique transaction identifier.")
    user_id: str = Field(description="Identifier of the user making the transaction.")
    merchant_id: str = Field(description="Merchant identifier.")
    merchant_category: str = Field(description="Merchant category (e.g.: grocery, electronics, travel).")
    amount: float = Field(gt=0, description="Transaction amount in local currency. Must be greater than 0.")
    country: str = Field(description="ISO 3166-1 alpha-2 country code for the transaction (e.g.: AR, BR, MX).")
    timestamp: datetime = Field(description="Transaction timestamp in ISO 8601 format.")
    device_type: str = Field(description="Device type: mobile, desktop, or tablet.")
    ip_hash: str = Field(description="Hash of the source IP (the real IP is not stored).")
    features: dict[str, float] = Field(
        description=(
            "Features pre-computed by the online feature engineering pipeline. "
            "Must contain exactly 10 keys: "
            "`tx_count_1h`, `tx_count_24h`, `tx_count_7d`, "
            "`amount_sum_1h`, `amount_sum_24h`, `seconds_since_last_tx`, "
            "`amount_ratio_vs_user_avg`, `is_country_new`, "
            "`is_merchant_new`, `distinct_merchants_seen`."
        )
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
                "prediction_score": 0.847,
                "prediction_label": True,
                "model_version": "3",
                "latency_ms": 4.2,
            }
        }
    )

    transaction_id: str = Field(description="Identifier of the evaluated transaction.")
    prediction_score: float = Field(description="Fraud probability between 0.0 and 1.0.")
    prediction_label: bool = Field(description="True if `prediction_score` >= `fraud_score_threshold` (default 0.5).")
    model_version: str = Field(description="XGBoost model version used for this prediction.")
    latency_ms: float = Field(description="Inference latency in milliseconds (feature prep + XGBoost).")


class BatchPredictionRequest(BaseModel):
    items: list[TransactionRequest] = Field(
        min_length=1, max_length=500, description="List of transactions to evaluate. Minimum 1, maximum 500."
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(description="List of predictions in the same order as `items`.")
    total: int = Field(description="Total number of predictions returned.")
    latency_ms: float = Field(description="Total batch latency in milliseconds.")
