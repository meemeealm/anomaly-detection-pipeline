"""ONNX Runtime (CPU) inference wrapper (iforest-aware, window-aware).

score_batch() always returns a score in [0, 1]; higher means more anomalous.

* LightGBM classifier (default): output 1 holds class probabilities; P(anomaly) = column 1.
* Isolation Forest: detected by a sidecar "<model stem>_meta.json" next to the .onnx.
  If that meta has a "window" entry the model expects sliding-window features
  (see features.py) and needs per-device history, so `.window` is set.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .settings import Settings


def meta_path_for(onnx_path: Path) -> Path:
    return onnx_path.with_name(onnx_path.stem + "_meta.json")


def anomaly_from_scores(scores, meta: dict) -> np.ndarray:
    """Map raw Isolation Forest ONNX scores to an anomaly score in [0, 1].

    sklearn: score_samples = -(anomaly score) in [-1, 0);
             decision_function = score_samples - offset_.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if meta["scores_kind"] == "decision_function":
        s = s + float(meta["offset"])
    return np.clip(-s, 0.0, 1.0)


class AnomalyDetector:
    def __init__(self, settings: Settings) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.onnx_intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(settings.onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        # Use the model's real input name: skl2onnx names it "X", the LightGBM export "input".
        self._input = self._session.get_inputs()[0].name
        self.model_id = settings.onnx_path.stem
        self.n_features = len(settings.feature_names)
        self.threshold = settings.threshold
        self.window: dict | None = None
        self._meta: dict | None = None
        self._scores_idx = 1

        meta_file = meta_path_for(settings.onnx_path)
        if meta_file.exists():
            self._meta = json.loads(meta_file.read_text())
            base = self._meta.get("base_features", self._meta.get("feature_names"))
            if base is not None and list(base) != list(settings.feature_names):
                raise ValueError(
                    f"Model was trained on sensors {list(base)}, "
                    f"but configs/model.json has {list(settings.feature_names)}"
                )
            if "input_features" in self._meta:
                self.n_features = len(self._meta["input_features"])
            self.window = self._meta.get("window")
            names = [o.name for o in self._session.get_outputs()]
            wanted = self._meta.get("scores_output", "scores")
            self._scores_idx = names.index(wanted) if wanted in names else 1
            if "IOT_THRESHOLD" not in os.environ:
                self.threshold = float(self._meta["threshold"])

    @property
    def model_type(self) -> str:
        return self._meta["model_type"] if self._meta else "lightgbm"

    def score_batch(self, x: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1] for each row of a (n, n_features) array."""
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected shape (n, {self.n_features}), got {x.shape}")
        x = np.ascontiguousarray(x, dtype=np.float32)
        outputs = self._session.run(None, {self._input: x})
        if self._meta is None:
            return np.asarray(outputs[1])[:, 1]
        return anomaly_from_scores(outputs[self._scores_idx], self._meta)

    def score(self, features: list[float]) -> float:
        return float(self.score_batch(np.asarray([features], dtype=np.float32))[0])
