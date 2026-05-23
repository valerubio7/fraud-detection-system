import json
import logging
import os
import time

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from serving.app.schemas import BatchPredictionRequest, BatchPredictionResponse, PredictionResponse, TransactionRequest

router = APIRouter(tags=["predictions"])
_log = logging.getLogger(__name__)

_FRAUD_SCORE_THRESHOLD = float(os.getenv("FRAUD_SCORE_THRESHOLD", "0.5"))
_SLOW_REQUEST_THRESHOLD_MS = float(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "50.0"))


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict fraud on a single transaction",
    description=(
        "Evaluates a bank transaction and returns the fraud probability. "
        "Features must be pre-computed by the online feature engineering pipeline "
        "(`SlidingWindowStore` + `HistoricalProfileStore`). "
        "The prediction is persisted to PostgreSQL asynchronously (does not block the response). "
        "Requests with the same `transaction_id` return the cached result."
    ),
    responses={
        200: {"description": "Successful prediction."},
        422: {"description": "Invalid request: `amount` <= 0 or required fields missing."},
        503: {"description": "Model is not available (degraded mode)."},
    },
)
async def predict(req: TransactionRequest, request: Request, background_tasks: BackgroundTasks) -> PredictionResponse:
    model_loader = request.app.state.model_loader
    if model_loader._model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prediction_store = request.app.state.prediction_store
    cache = request.app.state.prediction_cache

    cached = cache.get(req.transaction_id)
    if cached is not None:
        return PredictionResponse(**cached)

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

    prediction_label = prediction_score >= _FRAUD_SCORE_THRESHOLD
    latency_ms = feature_ms + inference_ms

    background_tasks.add_task(prediction_store.save, req.transaction_id, prediction_score, prediction_label, latency_ms)

    total_ms = latency_ms
    threshold = _SLOW_REQUEST_THRESHOLD_MS
    log_payload = {
        "event": "predict",
        "transaction_id": req.transaction_id,
        "feature_ms": round(feature_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "total_ms": round(total_ms, 3),
    }
    if total_ms > threshold:
        log_payload["slow_request"] = True
        _log.warning(json.dumps(log_payload))
    else:
        _log.info(json.dumps(log_payload))

    response = PredictionResponse(
        transaction_id=req.transaction_id,
        prediction_score=prediction_score,
        prediction_label=prediction_label,
        model_version=model_loader.model_version,
        latency_ms=latency_ms,
    )
    cache.set(req.transaction_id, response.model_dump())
    return response


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict fraud on a batch of transactions",
    description=(
        "Evaluates between 1 and 500 transactions in a single call. "
        "The batch is processed by vectorizing features with `numpy.vstack` and a single "
        "`predict_proba` call, improving throughput over individual requests. "
        "The reported latency covers the full batch; per-transaction latency is `latency_ms / total`."
    ),
    responses={
        200: {"description": "Batch predictions."},
        422: {"description": "Empty list or more than 500 items."},
        503: {"description": "Model is not available."},
    },
)
async def predict_batch(
    req: BatchPredictionRequest, request: Request, background_tasks: BackgroundTasks
) -> BatchPredictionResponse:
    model_loader = request.app.state.model_loader
    if model_loader._model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prediction_store = request.app.state.prediction_store
    threshold_score = _FRAUD_SCORE_THRESHOLD
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

    predictions = []
    for item, score in zip(req.items, scores, strict=False):
        label = float(score) >= threshold_score
        background_tasks.add_task(prediction_store.save, item.transaction_id, float(score), label, per_item_latency_ms)
        predictions.append(
            PredictionResponse(
                transaction_id=item.transaction_id,
                prediction_score=float(score),
                prediction_label=label,
                model_version=model_loader.model_version,
                latency_ms=per_item_latency_ms,
            )
        )

    total_ms = feature_ms + inference_ms
    threshold_slow = _SLOW_REQUEST_THRESHOLD_MS
    log_payload = {
        "event": "predict_batch",
        "batch_size": n,
        "feature_ms": round(feature_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "avg_inference_ms": round(inference_ms / n, 3),
        "total_ms": round(total_ms, 3),
    }
    if total_ms > threshold_slow:
        log_payload["slow_request"] = True
        _log.warning(json.dumps(log_payload))
    else:
        _log.info(json.dumps(log_payload))

    return BatchPredictionResponse(predictions=predictions, total=len(predictions), latency_ms=total_ms)
