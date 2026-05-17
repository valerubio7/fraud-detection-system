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
                    "distinct_countries_seen": 2.0,
                    "is_merchant_new": 0.0,
                    "distinct_merchants_seen": 8.0,
                },
            }
        }
    )

    transaction_id: str = Field(description="Identificador único de la transacción.")
    user_id: str = Field(description="Identificador del usuario que realiza la transacción.")
    merchant_id: str = Field(description="Identificador del comercio.")
    merchant_category: str = Field(
        description="Categoría del comercio (ej: grocery, electronics, travel)."
    )
    amount: float = Field(
        gt=0, description="Monto de la transacción en moneda local. Debe ser mayor que 0."
    )
    country: str = Field(
        description="Código ISO 3166-1 alpha-2 del país de la transacción (ej: AR, BR, MX)."
    )
    timestamp: datetime = Field(description="Timestamp de la transacción en formato ISO 8601.")
    device_type: str = Field(description="Tipo de dispositivo: mobile, desktop, o tablet.")
    ip_hash: str = Field(description="Hash del IP de origen (no se almacena el IP real).")
    features: dict[str, float] = Field(
        description=(
            "Features pre-calculadas por el pipeline de feature engineering online. "
            "Debe contener exactamente las 11 keys: "
            "`tx_count_1h`, `tx_count_24h`, `tx_count_7d`, "
            "`amount_sum_1h`, `amount_sum_24h`, `seconds_since_last_tx`, "
            "`amount_ratio_vs_user_avg`, `is_country_new`, `distinct_countries_seen`, "
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

    transaction_id: str = Field(description="Identificador de la transacción evaluada.")
    prediction_score: float = Field(description="Probabilidad de fraude entre 0.0 y 1.0.")
    prediction_label: bool = Field(
        description="True si `prediction_score` >= `fraud_score_threshold` (por defecto 0.5)."
    )
    model_version: str = Field(description="Versión del modelo XGBoost usado para esta predicción.")
    latency_ms: float = Field(
        description="Latencia de inferencia en milisegundos (feature prep + XGBoost)."
    )


class BatchPredictionRequest(BaseModel):
    items: list[TransactionRequest] = Field(
        min_length=1,
        max_length=500,
        description="Lista de transacciones a evaluar. Mínimo 1, máximo 500.",
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(
        description="Lista de predicciones en el mismo orden que `items`."
    )
    total: int = Field(description="Cantidad total de predicciones devueltas.")
    latency_ms: float = Field(description="Latencia total del batch en milisegundos.")
