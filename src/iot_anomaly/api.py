"""window-aware: FastAPI application. Run with:
    uvicorn iot_anomaly.api:app --app-dir src --host 127.0.0.1 --port 8000

The API is stateless, so it serves single-reading models only. Sliding-window
models need per-device history and are served by scripts/consume_score.py.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .inference import AnomalyDetector
from .schemas import HealthResponse, PredictResponse, SensorReading
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.detector = (
        AnomalyDetector(settings) if settings.onnx_path.exists() else None
    )
    yield


app = FastAPI(title="IoT Sensor Anomaly Detection", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = getattr(app.state, "detector", None) is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(reading: SensorReading) -> PredictResponse:
    detector: AnomalyDetector | None = getattr(app.state, "detector", None)
    if detector is None:
        raise HTTPException(503, "Model not loaded. Run: python scripts/train_export.py")
    if detector.window is not None:
        raise HTTPException(
            501,
            "This model needs per-device history (sliding window). Score it with "
            "scripts/consume_score.py, or unset IOT_MODEL_PATH to serve a single-reading model.",
        )
    settings = get_settings()
    features = [getattr(reading, name) for name in settings.feature_names]
    score = detector.score(features)
    return PredictResponse(
        anomaly_score=score, is_anomaly=score >= detector.threshold,
        threshold=detector.threshold,
    )
