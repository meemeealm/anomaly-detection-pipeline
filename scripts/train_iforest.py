"""window-aware: train an unsupervised Isolation Forest on simulator data and export to ONNX.

Two variants:
    python scripts/train_iforest.py                 # single-reading model -> artifacts/iforest.onnx
    python scripts/train_iforest.py --window 10     # sliding-window model -> artifacts/iforest_window.onnx

Training data is UNLABELED and contains a small share of hidden spikes
(--train-contamination). The threshold is calibrated on a separate clean
"known-good" period to hit --target-fpr. Labels are used only for evaluation.
Windowed models need per-device history, so they are served by the consumer:
    IOT_MODEL_PATH=artifacts/iforest_window.onnx python scripts/consume_score.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iot_anomaly.features import DeviceWindow, feature_names  # noqa: E402
from iot_anomaly.inference import anomaly_from_scores, meta_path_for  # noqa: E402
from iot_anomaly.settings import get_settings  # noqa: E402
from iot_anomaly.simulation import Fleet, to_z  # noqa: E402

PARITY_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=0, help="sliding-window size W (0 = single-reading model)")
    ap.add_argument("--max-gap-s", type=float, default=30.0, help="silence that resets a device window (stored in meta)")
    ap.add_argument("--n-train", type=int, default=30_000)
    ap.add_argument("--n-val", type=int, default=10_000)
    ap.add_argument("--n-eval", type=int, default=10_000)
    ap.add_argument("--train-contamination", type=float, default=0.01)
    ap.add_argument("--train-stuck-rate", type=float, default=0.0)
    ap.add_argument("--eval-anomaly-rate", type=float, default=0.03)
    ap.add_argument("--eval-stuck-rate", type=float, default=0.001)
    ap.add_argument("--target-fpr", type=float, default=0.01, help="false-positive rate used to set the threshold")
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--n-devices", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset; lower it if onnxruntime rejects the model")
    ap.add_argument("--ml-opset", type=int, default=3, help="ai.onnx.ml opset (skl2onnx currently supports up to 3)")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def generate(names, n: int, anomaly_rate: float, stuck_rate: float, seed: int, n_devices: int, window: int):
    """Features + labels from the fleet simulator.

    Returns (x, y, y_any). y = fault in the CURRENT reading; y_any = a fault anywhere in the
    current window (a window keeps "remembering" a spike for W readings).
    """
    fleet = Fleet(names, n_devices=n_devices, seed=seed, anomaly_rate=anomaly_rate, stuck_rate=stuck_rate)
    if window <= 0:
        x = np.empty((n, len(names)), dtype=np.float32)
        y = np.empty(n, dtype=np.int8)
        for i in range(n):
            readings, label = fleet.sample(fleet.devices[i % len(fleet.devices)])
            x[i] = [to_z(name, readings[name]) for name in names]
            y[i] = label
        return x, y, y.copy()

    wins = {d.device_id: DeviceWindow(window) for d in fleet.devices}
    recent = {d.device_id: deque(maxlen=window) for d in fleet.devices}
    xs, ys, yany = [], [], []
    while len(xs) < n:
        for dev in fleet.devices:
            readings, label = fleet.sample(dev)
            z = np.array([to_z(nm, readings[nm]) for nm in names])
            recent[dev.device_id].append(label)
            f = wins[dev.device_id].push(z)
            if f is not None:
                xs.append(f)
                ys.append(label)
                yany.append(int(any(recent[dev.device_id])))
    return (np.vstack(xs[:n]).astype(np.float32),
            np.array(ys[:n], dtype=np.int8), np.array(yany[:n], dtype=np.int8))


def _prf(flagged: np.ndarray, truth: np.ndarray):
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    return (tp / (tp + fp) if tp + fp else float("nan"),
            tp / (tp + fn) if tp + fn else float("nan"))


def main() -> int:
    import joblib
    import onnxruntime as ort
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        from skl2onnx import to_onnx
    except ImportError:
        print("skl2onnx missing. Run: pip install 'skl2onnx>=1.16,<2.0'", file=sys.stderr)
        return 2

    args = parse_args()
    s = get_settings()
    names = s.feature_names
    input_names = feature_names(names) if args.window > 0 else list(names)
    default_out = "artifacts/iforest_window.onnx" if args.window > 0 else "artifacts/iforest.onnx"
    out = Path(args.out or default_out)
    if not out.is_absolute():
        out = ROOT / out

    print(f"Generating data (window={args.window or 'none'}, {len(input_names)} input features) ...")
    x_train, _, _ = generate(names, args.n_train, args.train_contamination, args.train_stuck_rate,
                             args.seed, args.n_devices, args.window)
    x_val, _, _ = generate(names, args.n_val, 0.0, 0.0, args.seed + 1, args.n_devices, args.window)
    x_eval, y_eval, y_any = generate(names, args.n_eval, args.eval_anomaly_rate, args.eval_stuck_rate,
                                     args.seed + 2, args.n_devices, args.window)

    print(f"Training IsolationForest(n_estimators={args.n_estimators}, max_samples={args.max_samples}) ...")
    model = IsolationForest(
        n_estimators=args.n_estimators, max_samples=args.max_samples,
        contamination="auto", random_state=args.seed, n_jobs=1,
    ).fit(x_train)

    print("Exporting to ONNX ...")
    onx = to_onnx(model, x_train[:1], target_opset={"": args.opset, "ai.onnx.ml": args.ml_opset})
    try:
        sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        print(f"onnxruntime rejected the exported model: {exc}\nTry a lower --opset (e.g. 15).", file=sys.stderr)
        return 1
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    scores_name = "scores" if "scores" in out_names else out_names[-1]
    scores_idx = out_names.index(scores_name)

    # Parity gate: find which sklearn function the ONNX "scores" output equals.
    raw = np.asarray(sess.run(None, {in_name: x_val})[scores_idx], dtype=np.float64).reshape(-1)
    diffs = {
        "decision_function": np.abs(raw - model.decision_function(x_val)),
        "score_samples": np.abs(raw - model.score_samples(x_val)),
    }
    kind = min(diffs, key=lambda k: float(np.quantile(diffs[k], 0.999)))
    err = float(np.quantile(diffs[kind], 0.999))
    print(f"ONNX scores match sklearn {kind}: 99.9th-pct abs diff = {err:.2e} (tolerance {PARITY_TOL:.0e})")
    if err > PARITY_TOL:
        print("Parity check FAILED; nothing was written.", file=sys.stderr)
        return 1

    meta = {
        "model_type": "isolation_forest",
        "scores_output": scores_name,
        "scores_kind": kind,
        "offset": float(model.offset_),
        "base_features": list(names),
        "input_features": input_names,
    }
    if args.window > 0:
        meta["window"] = {"size": args.window, "max_gap_s": args.max_gap_s}
    val_scores = anomaly_from_scores(raw, meta)
    threshold = float(np.quantile(val_scores, 1.0 - args.target_fpr))
    meta.update({
        "threshold": threshold,
        "target_fpr": args.target_fpr,
        "n_estimators": args.n_estimators,
        "max_samples": args.max_samples,
        "train_contamination": args.train_contamination,
        "created": datetime.now(timezone.utc).isoformat(),
    })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())
    meta_path_for(out).write_text(json.dumps(meta, indent=2))
    joblib.dump(model, out.with_suffix(".joblib"))

    # Evaluation on held-out labeled data, via the same code path as serving.
    ev = anomaly_from_scores(sess.run(None, {in_name: x_eval})[scores_idx], meta)
    flagged = ev >= threshold
    truth = y_eval == 1
    precision, recall = _prf(flagged, truth)
    print(f"\nWrote {out}\n      {meta_path_for(out)}\n      {out.with_suffix('.joblib')}")
    print(f"threshold (anomaly score >= t, FPR target {args.target_fpr:.1%}) = {threshold:.4f}")
    print(f"eval (spikes {args.eval_anomaly_rate:.1%}/reading, stuck-sensor starts {args.eval_stuck_rate:.2%}/reading):")
    print(f"  ROC-AUC={roc_auc_score(truth, ev):.3f} PR-AUC={average_precision_score(truth, ev):.3f} "
          f"precision={precision:.3f} recall={recall:.3f}")
    if args.window > 0:
        p_any, _ = _prf(flagged, y_any == 1)
        print(f"  window-aware precision={p_any:.3f}  (an alert is correct if ANY reading in its window was faulty;")
        print("  after a spike, alerts keep firing until it leaves the window)")
        print("  note: stuck sensors are largely invisible to an Isolation Forest, even with windows;")
        print("  a flatline rule (rolling std ~ 0) is the usual fix. See docs/observability.md.")
    print("\nUse it:  export IOT_MODEL_PATH=" + str(out.relative_to(ROOT)).replace("\\", "/")
          + "   (then restart the consumer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
