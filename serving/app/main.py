from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes.health import router as health_router
from .routes.predict import router as predict_router
from .services.model_loader import ModelLoader


@asynccontextmanager
async def lifespan(app: FastAPI):
    loader = ModelLoader()
    loader.load()
    app.state.model_loader = loader
    yield


app = FastAPI(
    title="Fraud Detection Serving API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(predict_router)
