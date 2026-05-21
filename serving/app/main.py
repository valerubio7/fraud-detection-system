import logging
import os
import time
from contextlib import asynccontextmanager
from urllib.parse import quote_plus

import asyncpg
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator

from .routes.health import router as health_router
from .routes.predict import router as predict_router
from .services.cache import PredictionCache
from .services.model_loader import ModelLoader
from .services.prediction_store import PredictionStore

_log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loader = ModelLoader()
    try:
        loader.load()
    except Exception as exc:
        _log.error("Model load failed, starting in degraded mode: %s", exc)

    app.state.model_loader = loader

    _pg_user = quote_plus(os.getenv("POSTGRES_USER", "fraud_metadata_user"))
    _pg_password = quote_plus(os.getenv("POSTGRES_PASSWORD"))
    _pg_host = os.getenv("POSTGRES_HOST", "postgresql")
    _pg_port = os.getenv("POSTGRES_PORT", "5432")
    _pg_db = os.getenv("POSTGRES_DB", "fraud_metadata")
    dsn = f"postgresql://{_pg_user}:{_pg_password}@{_pg_host}:{_pg_port}/{_pg_db}"
    app.state.pg_pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=int(os.getenv("PG_POOL_MIN_SIZE", "2")),
        max_size=int(os.getenv("PG_POOL_MAX_SIZE", "10")),
    )
    app.state.prediction_store = PredictionStore(app.state.pg_pool, loader.deployment_id)

    app.state.prediction_cache = PredictionCache(
        os.getenv("REDIS_HOST", "redis"),
        int(os.getenv("REDIS_PORT", "6379")),
    )
    yield

    await app.state.pg_pool.close()


app = FastAPI(
    title="Fraud Detection Serving API",
    version="0.1.0",
    description=(
        "API de inferencia de fraude en tiempo real. "
        "Clasifica transacciones bancarias como fraudulentas o legítimas "
        "usando un modelo XGBoost con latencia P99 < 100ms bajo carga.\n\n"
        "**Modelo**: cargado desde MLflow Registry (stage: Production) al startup. "
        "Si no hay modelo en Production, la API inicia en modo degradado "
        "y `/health` devuelve `status: degraded`.\n\n"
        "**Idempotencia**: el mismo `transaction_id` devuelve siempre el mismo resultado "
        "(cacheado en Redis por 5 minutos)."
    ),
    openapi_tags=[
        {
            "name": "health",
            "description": "Estado del servicio y del modelo cargado en memoria.",
        },
        {
            "name": "predictions",
            "description": "Inferencia de fraude para transacciones individuales y en batch.",
        },
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(predict_router)

Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response
