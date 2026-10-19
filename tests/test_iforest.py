import dataclasses

import numpy as np
import pytest

from iot_anomaly.inference import AnomalyDetector, anomaly_from_scores, meta_path_for
from iot_anomaly.settings import get_settings

S = get_settings()
IF_PATH = S.onnx_path.with_name("iforest.onnx")


def test_decision_function_mapping():
    meta = {"scores_kind": "decision_function", "offset": -0.5}
    np.testing.assert_allclose(anomaly_from_scores([0.1, -0.2], meta), [0.4, 0.7])


def test_score_samples_mapping():
    np.testing.assert_allclose(anomaly_from_scores([-0.3], {"scores_kind": "score_samples"}), [0.3])


@pytest.mark.skipif(
    not (IF_PATH.exists() and meta_path_for(IF_PATH).exists() and IF_PATH.with_suffix(".joblib").exists()),
    reason="Isolation Forest not built; run scripts/train_iforest.py",
)
def test_onnx_matches_sklearn():
    joblib = pytest.importorskip("joblib")
    model = joblib.load(IF_PATH.with_suffix(".joblib"))
    det = AnomalyDetector(dataclasses.replace(S, onnx_path=IF_PATH))
    assert det.model_type == "isolation_forest"
    x = np.random.default_rng(0).normal(0, 1.5, size=(500, det.n_features)).astype(np.float32)
    diff = np.abs(det.score_batch(x) - np.clip(-model.score_samples(x), 0, 1))
    assert np.quantile(diff, 0.999) <= 1e-4
