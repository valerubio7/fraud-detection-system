import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Service health",
    description=(
        "Returns the current service status. "
        "`status: ok` means the model is loaded and ready for inference. "
        "`status: degraded` means the model failed to load at startup — "
        "predictions are unavailable but the service is still responding."
    ),
    responses={200: {"description": "Service is operational (ok or degraded)."}},
)
def health(request: Request) -> JSONResponse:
    loader = getattr(request.app.state, "model_loader", None)
    model_loaded = loader is not None and loader._model is not None
    status = "ok" if model_loaded else "degraded"
    status_code = 200 if model_loaded else 503
    return JSONResponse({"status": status, "model_loaded": model_loaded}, status_code=status_code)


@router.get(
    "/model/info",
    summary="Loaded model information",
    description=(
        "Returns metadata for the XGBoost model currently in memory: "
        "version, MLflow stage, load timestamp, classification threshold, "
        "and PostgreSQL deployment ID."
    ),
    responses={
        200: {"description": "Active model information."},
        503: {"description": "Model is not loaded — service is in degraded mode."},
    },
)
def model_info(request: Request) -> dict:
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None or loader._model is None or loader.loaded_at is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": loader.model_name,
        "model_version": loader.model_version,
        "model_stage": loader.model_stage,
        "loaded_at": loader.loaded_at.isoformat(),
        "fraud_score_threshold": float(os.getenv("FRAUD_SCORE_THRESHOLD", "0.5")),
        "deployment_id": loader.deployment_id,
    }
