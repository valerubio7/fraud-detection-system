import time

from fastapi import APIRouter, Request

import config

from ..schemas.prediction import PredictionResponse, TransactionRequest

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest, request: Request) -> PredictionResponse:
    model_loader = request.app.state.model_loader

    raw = {
        "amount": req.amount,
        "timestamp": req.timestamp,
        "merchant_category": req.merchant_category,
        "country": req.country,
        "device_type": req.device_type,
    }

    t0 = time.perf_counter()
    features_array = model_loader.prepare_features(raw, req.features)
    prediction_score = float(model_loader._model.predict_proba(features_array)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000

    prediction_label = prediction_score >= config.model_settings.fraud_score_threshold

    return PredictionResponse(
        transaction_id=req.transaction_id,
        prediction_score=prediction_score,
        prediction_label=prediction_label,
        model_version=model_loader.model_version,
        latency_ms=latency_ms,
    )
