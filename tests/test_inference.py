import numpy as np
import pytest

from iot_anomaly.inference import AnomalyDetector
from iot_anomaly.settings import get_settings

s = get_settings()
pytestmark = pytest.mark.skipif(
    not s.onnx_path.exists(), reason="model not built; run scripts/train_export.py"
)


def test_scores_in_unit_interval():
    det = AnomalyDetector(s)
    x = np.random.default_rng(1).normal(size=(32, det.n_features)).astype(np.float32)
    scores = det.score_batch(x)
    assert scores.shape == (32,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_wrong_shape_rejected():
    det = AnomalyDetector(s)
    with pytest.raises(ValueError):
        det.score_batch(np.zeros((1, det.n_features + 1), dtype=np.float32))
