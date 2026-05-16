from datetime import datetime

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    merchant_category: str
    amount: float = Field(gt=0)
    country: str
    timestamp: datetime
    device_type: str
    ip_hash: str
    features: dict[str, float]


class PredictionResponse(BaseModel):
    transaction_id: str
    prediction_score: float
    prediction_label: bool
    model_version: str
    latency_ms: float
