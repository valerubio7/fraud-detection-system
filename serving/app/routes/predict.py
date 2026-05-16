import time

import numpy as np
from fastapi import APIRouter, Request

import config

from ..schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TransactionRequest,
)

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


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    model_loader = request.app.state.model_loader
    threshold = config.model_settings.fraud_score_threshold

    t0 = time.perf_counter()

    arrays = []
    for item in req.items:
        raw = {
            "amount": item.amount,
            "timestamp": item.timestamp,
            "merchant_category": item.merchant_category,
            "country": item.country,
            "device_type": item.device_type,
        }
        arrays.append(model_loader.prepare_features(raw, item.features))

    batch_array = np.vstack(arrays)
    scores = model_loader._model.predict_proba(batch_array)[:, 1]

    latency_ms = (time.perf_counter() - t0) * 1000
    per_item_ms = latency_ms / len(req.items)

    predictions = [
        PredictionResponse(
            transaction_id=item.transaction_id,
            prediction_score=float(score),
            prediction_label=float(score) >= threshold,
            model_version=model_loader.model_version,
            latency_ms=per_item_ms,
        )
        for item, score in zip(req.items, scores, strict=False)
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        latency_ms=latency_ms,
    )
