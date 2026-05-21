import json
import logging
import os
import time

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Request

from ..schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TransactionRequest,
)

router = APIRouter(tags=["predictions"])
_log = logging.getLogger(__name__)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predecir fraude en una transacción",
    description=(
        "Evalúa una transacción bancaria y devuelve la probabilidad de fraude. "
        "Las features deben ser pre-calculadas por el pipeline de feature engineering "
        "(`SlidingWindowStore` + `HistoricalProfileStore`). "
        "La predicción se guarda en PostgreSQL de forma asíncrona (no bloquea la respuesta). "
        "Requests con el mismo `transaction_id` devuelven el resultado cacheado."
    ),
    responses={
        200: {"description": "Predicción exitosa."},
        422: {"description": "Request inválido: `amount` <= 0 o campos requeridos faltantes."},
        503: {"description": "El modelo no está disponible (modo degradado)."},
    },
)
async def predict(
    req: TransactionRequest, request: Request, background_tasks: BackgroundTasks
) -> PredictionResponse:
    model_loader = request.app.state.model_loader
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

    prediction_label = prediction_score >= float(os.getenv("FRAUD_SCORE_THRESHOLD", "0.5"))
    latency_ms = feature_ms + inference_ms

    background_tasks.add_task(
        prediction_store.save, req.transaction_id, prediction_score, prediction_label, latency_ms
    )

    total_ms = latency_ms
    threshold = float(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "50.0"))
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
    summary="Predecir fraude en un batch de transacciones",
    description=(
        "Evalúa entre 1 y 500 transacciones en una sola llamada. "
        "El batch se procesa vectorizando las features con `numpy.vstack` y una sola "
        "llamada a `predict_proba`, lo que mejora el throughput respecto a llamadas individuales. "
        "La latencia reportada es la del batch completo; la latencia por transacción "
        "es `latency_ms / total`."
    ),
    responses={
        200: {"description": "Predicciones del batch."},
        422: {"description": "Lista vacía o con más de 500 items."},
        503: {"description": "El modelo no está disponible."},
    },
)
async def predict_batch(
    req: BatchPredictionRequest, request: Request, background_tasks: BackgroundTasks
) -> BatchPredictionResponse:
    model_loader = request.app.state.model_loader
    prediction_store = request.app.state.prediction_store
    threshold_score = float(os.getenv("FRAUD_SCORE_THRESHOLD", "0.5"))
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
        background_tasks.add_task(
            prediction_store.save, item.transaction_id, float(score), label, per_item_latency_ms
        )
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
    threshold_slow = float(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "50.0"))
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

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        latency_ms=total_ms,
    )
