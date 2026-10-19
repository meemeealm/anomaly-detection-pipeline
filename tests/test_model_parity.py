import numpy as np
import pytest

from iot_anomaly.inference import AnomalyDetector
from iot_anomaly.settings import get_settings

s = get_settings()


@pytest.mark.skipif(
    not (s.onnx_path.exists() and s.lightgbm_path.exists()),
    reason="model not built; run scripts/train_export.py",
)
def test_onnx_matches_lightgbm():
    lgb = pytest.importorskip("lightgbm")
    x = np.random.default_rng(3).normal(0, 2, size=(500, len(s.feature_names))).astype(np.float32)
    expected = lgb.Booster(model_file=str(s.lightgbm_path)).predict(x)
    actual = AnomalyDetector(s).score_batch(x)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
