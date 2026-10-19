"""Check ONNX Runtime output matches native LightGBM predictions."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iot_anomaly.inference import AnomalyDetector  # noqa: E402
from iot_anomaly.settings import get_settings  # noqa: E402

TOLERANCE = 1e-4


def main() -> int:
    import lightgbm as lgb

    s = get_settings()
    for p in (s.onnx_path, s.lightgbm_path):
        if not p.exists():
            print(f"Missing {p}. Run: python scripts/train_export.py", file=sys.stderr)
            return 2
    rng = np.random.default_rng(7)
    x = rng.normal(0, 2, size=(2000, len(s.feature_names))).astype(np.float32)
    expected = lgb.Booster(model_file=str(s.lightgbm_path)).predict(x)
    actual = AnomalyDetector(s).score_batch(x)
    max_diff = float(np.max(np.abs(expected - actual)))
    print(f"max |lightgbm - onnx| = {max_diff:.3e} (tolerance {TOLERANCE:.0e})")
    return 0 if max_diff <= TOLERANCE else 1


if __name__ == "__main__":
    raise SystemExit(main())
