import pytest
from fastapi.testclient import TestClient

from iot_anomaly.api import app
from iot_anomaly.settings import get_settings

PAYLOAD = {n: 0.0 for n in get_settings().feature_names}


def test_health_endpoint_responds():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert "model_loaded" in r.json()


def test_predict_validation_error():
    with TestClient(app) as client:
        assert client.post("/predict", json={"temperature": 1.0}).status_code == 422


def test_predict_roundtrip():
    if not get_settings().onnx_path.exists():
        pytest.skip("model not built; run scripts/train_export.py")
    with TestClient(app) as client:
        r = client.post("/predict", json=PAYLOAD)
        assert r.status_code == 200
        assert 0.0 <= r.json()["anomaly_score"] <= 1.0
