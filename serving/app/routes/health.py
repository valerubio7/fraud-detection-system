from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

import config

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> JSONResponse:
    loader = getattr(request.app.state, "model_loader", None)
    model_loaded = loader is not None and loader._model is not None
    if model_loaded:
        return JSONResponse({"status": "ok", "model_loaded": True}, status_code=200)
    return JSONResponse({"status": "degraded", "model_loaded": False}, status_code=503)


@router.get("/model/info")
def model_info(request: Request) -> dict:
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None or loader._model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": loader.model_name,
        "model_version": loader.model_version,
        "model_stage": loader.model_stage,
        "loaded_at": loader.loaded_at.isoformat(),
        "fraud_score_threshold": config.model_settings.fraud_score_threshold,
        "deployment_id": loader.deployment_id,
    }
