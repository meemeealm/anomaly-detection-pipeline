"""In-process single-row latency + RSS measurement. Run manually; it reports
what it measured and makes no claim about any target."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iot_anomaly.inference import AnomalyDetector  # noqa: E402
from iot_anomaly.settings import get_settings  # noqa: E402


def main() -> int:
    import psutil

    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=500)
    args = ap.parse_args()

    s = get_settings()
    if not s.onnx_path.exists():
        print("Missing model. Run: python scripts/train_export.py", file=sys.stderr)
        return 2
    det = AnomalyDetector(s)
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(args.iterations + args.warmup, det.n_features)).astype(np.float32)

    for r in rows[: args.warmup]:
        det.score_batch(r[None, :])
    lat_ms = []
    for r in rows[args.warmup:]:
        t0 = time.perf_counter()
        det.score_batch(r[None, :])
        lat_ms.append((time.perf_counter() - t0) * 1000)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": args.iterations,
        "p50_ms": float(np.percentile(lat_ms, 50)),
        "p95_ms": float(np.percentile(lat_ms, 95)),
        "p99_ms": float(np.percentile(lat_ms, 99)),
        "rss_mb": psutil.Process().memory_info().rss / 1024**2,
    }
    out = ROOT / "results" / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
