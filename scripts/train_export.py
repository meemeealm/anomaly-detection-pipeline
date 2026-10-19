"""Train a LightGBM classifier on SYNTHETIC data and export it to ONNX.

Replace make_data() with a real dataset for anything beyond a demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iot_anomaly.settings import get_settings  # noqa: E402


def make_data(n: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, n_features))
    y = (rng.random(n) < 0.05).astype(int)
    signs = rng.choice([-1.0, 1.0], size=(int(y.sum()), n_features))
    x[y == 1] += signs * rng.normal(4.0, 1.0, size=(int(y.sum()), n_features))
    return x.astype(np.float32), y


def main() -> int:
    import lightgbm as lgb
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    s = get_settings()
    n_features = len(s.feature_names)
    x, y = make_data(20_000, n_features, seed=42)

    model = lgb.LGBMClassifier(
        n_estimators=100, num_leaves=15, learning_rate=0.1, n_jobs=1, verbose=-1
    )
    model.fit(x, y)

    s.onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(s.lightgbm_path))
    onnx_model = convert_lightgbm(
        model,
        initial_types=[(s.input_name, FloatTensorType([None, n_features]))],
        zipmap=False,
    )
    s.onnx_path.write_bytes(onnx_model.SerializeToString())
    print(f"Wrote {s.onnx_path}\nWrote {s.lightgbm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
