"""Pydantic request/response models (keep in sync with configs/model.json)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    humidity: float
    current: float
    rpm: float


class PredictResponse(BaseModel):
    anomaly_score: float = Field(ge=0.0, le=1.0)
    is_anomaly: bool
    threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
