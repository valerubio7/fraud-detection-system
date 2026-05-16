import json
import logging
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
_log = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest, request: Request) -> PredictionResponse:
    model_loader = request.app.state.model_loader
    prediction_store = request.app.state.prediction_store

    raw = {
        "amount": req.amount,
        "timestamp": req.timestamp,
        "merchant_category": req.merchant_category,
        "country": req.country,
        "device_type": req.device_type,
    }

    t0 = time.perf_counter()
    features_array = model_loader.prepare_features(raw, req.features)
    feature_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    prediction_score = float(model_loader._model.predict_proba(features_array)[0, 1])
    inference_ms = (time.perf_counter() - t1) * 1000

    prediction_label = prediction_score >= config.model_settings.fraud_score_threshold

    t2 = time.perf_counter()
    prediction_store.save(
        req.transaction_id, prediction_score, prediction_label, feature_ms + inference_ms
    )
    db_ms = (time.perf_counter() - t2) * 1000

    total_ms = feature_ms + inference_ms + db_ms
    threshold = config.model_settings.slow_request_threshold_ms
    log_payload = {
        "event": "predict",
        "transaction_id": req.transaction_id,
        "feature_ms": round(feature_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "db_ms": round(db_ms, 3),
        "total_ms": round(total_ms, 3),
    }
    if total_ms > threshold:
        log_payload["slow_request"] = True
        _log.warning(json.dumps(log_payload))
    else:
        _log.info(json.dumps(log_payload))

    return PredictionResponse(
        transaction_id=req.transaction_id,
        prediction_score=prediction_score,
        prediction_label=prediction_label,
        model_version=model_loader.model_version,
        latency_ms=feature_ms + inference_ms,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    model_loader = request.app.state.model_loader
    prediction_store = request.app.state.prediction_store
    threshold_score = config.model_settings.fraud_score_threshold
    n = len(req.items)

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
    feature_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    batch_array = np.vstack(arrays)
    scores = model_loader._model.predict_proba(batch_array)[:, 1]
    inference_ms = (time.perf_counter() - t1) * 1000

    per_item_latency_ms = (feature_ms + inference_ms) / n

    t2 = time.perf_counter()
    predictions = []
    for item, score in zip(req.items, scores, strict=False):
        label = float(score) >= threshold_score
        prediction_store.save(item.transaction_id, float(score), label, per_item_latency_ms)
        predictions.append(
            PredictionResponse(
                transaction_id=item.transaction_id,
                prediction_score=float(score),
                prediction_label=label,
                model_version=model_loader.model_version,
                latency_ms=per_item_latency_ms,
            )
        )
    db_ms = (time.perf_counter() - t2) * 1000

    total_ms = feature_ms + inference_ms + db_ms
    threshold_slow = config.model_settings.slow_request_threshold_ms
    log_payload = {
        "event": "predict_batch",
        "batch_size": n,
        "feature_ms": round(feature_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "avg_inference_ms": round(inference_ms / n, 3),
        "db_ms": round(db_ms, 3),
        "total_ms": round(total_ms, 3),
    }
    if total_ms > threshold_slow:
        log_payload["slow_request"] = True
        _log.warning(json.dumps(log_payload))
    else:
        _log.info(json.dumps(log_payload))

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        latency_ms=feature_ms + inference_ms,
    )
