"""window-aware, flatline-aware, config-aware: MQTT consumer.

validate -> (sliding-window features) -> ONNX score -> SQLite

* Every process has a run_id (logs in results/logs/, row in the `runs` table).
* Nothing fails silently: rejected messages are written to `failed_events`
  (malformed, non_finite_value, out_of_order, gap_reset, scoring_error); queue
  overflow and warm-up skips are counted in the run summary.
* Ctrl+C drains the queue before exiting.
* Settings resolve in this order: command-line flag > environment (IOT_<NAME>) >
  configs/stream.json > built-in default. The effective value and its source are logged at
  startup and stored in runs.config. Model-produced values (threshold, window size) come from
  the model's meta file, never from flags or the config.
* Windowed models also get a flatline rule: any sensor whose rolling std is below --flat-eps
  (z-units) flags the reading, because an Isolation Forest barely notices stuck sensors.

    python scripts/consume_score.py
    IOT_MODEL_PATH=artifacts/iforest_window.onnx python scripts/consume_score.py
"""
from __future__ import annotations

import argparse
import json
import queue
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iot_anomaly import storage  # noqa: E402
from iot_anomaly.features import StreamFeaturizer  # noqa: E402
from iot_anomaly.inference import AnomalyDetector  # noqa: E402
from iot_anomaly.logging_setup import new_run_id, setup_logging  # noqa: E402
from iot_anomaly.rules import alert_reason, flatline_sensors  # noqa: E402
from iot_anomaly.runtime_config import (  # noqa: E402
    CONSUMER_SETTINGS, format_settings, model_sources, resolve_all)
from iot_anomaly.settings import get_settings  # noqa: E402
from iot_anomaly.simulation import to_z  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults are None on purpose: the real default and its source are resolved in runtime_config.
    ap.add_argument("--host", default=None, help="MQTT broker host")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--topic", default=None)
    ap.add_argument("--qos", type=int, choices=(0, 1), default=None)
    ap.add_argument("--db", default=None, help="SQLite path (relative paths are inside the repo)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-wait-ms", type=float, default=None)
    ap.add_argument("--queue-size", type=int, default=None)
    ap.add_argument("--flat-eps", type=float, default=None,
                    help="flatline rule: flag a sensor whose rolling std (z-units) is below this; 0 disables")
    ap.add_argument("--max-gap-s", type=float, default=None,
                    help="silence (seconds) after which a device's window restarts")
    ap.add_argument("--log-level", default=None)
    ap.add_argument("--config", default=None, help="stream config JSON (default configs/stream.json)")
    ap.add_argument("--run-id", default=None)
    return ap.parse_args()


def main() -> int:
    import paho.mqtt.client as mqtt

    args = parse_args()
    try:
        resolved = resolve_all(args, CONSUMER_SETTINGS)   # fills args.<name>
    except ValueError as exc:
        print(f"Invalid settings: {exc}", file=sys.stderr)
        return 2
    run_id = args.run_id or new_run_id()
    log = setup_logging("consumer", run_id, args.log_level)
    s = get_settings()
    if not s.onnx_path.exists():
        log.error("model missing; run scripts/train_export.py or scripts/train_iforest.py",
                  extra={"path": str(s.onnx_path)})
        return 2
    det = AnomalyDetector(s)
    conn = storage.connect(args.db)
    stale = storage.close_stale_runs(conn)
    if stale:
        log.warning("marked earlier runs as unclean_shutdown (no heartbeat)", extra={"count": stale})

    window = det.window
    featurizer = (StreamFeaturizer(int(window["size"]), float(args.max_gap_s))
                  if window else None)
    cfg = {"model": det.model_id, "model_type": det.model_type, "threshold": det.threshold,
           "window": window, "batch_size": args.batch_size, "max_wait_ms": args.max_wait_ms,
           "topic": args.topic, "qos": args.qos, "db": str(args.db), "flat_eps": args.flat_eps,
           "max_gap_s": args.max_gap_s}
    all_settings = {**resolved, **model_sources(s, det)}
    cfg["settings"] = all_settings
    storage.start_run(conn, run_id, "consumer", cfg)
    log.info("consumer starting", extra={"cfg": cfg})
    log.info("effective settings, value [source]", extra={"resolved": format_settings(all_settings)})

    q: queue.Queue = queue.Queue(maxsize=args.queue_size)
    counters = {k: 0 for k in ("processed", "dropped", "malformed", "non_finite", "out_of_order",
                               "gap_reset", "warmup", "scoring_error", "db_error", "flatline_flags")}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            log.error("MQTT connect failed", extra={"reason": str(reason_code)})
            return
        client.subscribe([(args.topic, args.qos), ("fleet/+/meta", 1)])
        log.info("connected and subscribed", extra={"topic": args.topic})

    def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
        log.warning("MQTT disconnected (client will retry)", extra={"reason": str(reason_code)})

    def on_message(client, userdata, msg):
        try:
            q.put_nowait((msg.topic, msg.payload, time.time()))
        except queue.Full:
            counters["dropped"] += 1   # backpressure: shed load, never block the network thread

    def process(batch: list) -> None:
        failures: list[dict] = []
        valid: list[tuple] = []      # (device_id, event_ts, message, recv_ts, source_run_id, flat_sensors)
        feats: list[np.ndarray] = []

        def fail(reason: str, detail: str, payload: bytes, m=None, dev=None) -> None:
            src = m.get("run_id") if isinstance(m, dict) else None
            if dev is None and isinstance(m, dict):
                dev = m.get("device_id")
            failures.append({
                "run_id": run_id, "ts": time.time(), "source_run_id": src,
                "device_id": None if dev is None else str(dev),
                "reason": reason, "detail": detail,
                "payload": payload.decode("utf-8", "replace") if isinstance(payload, bytes) else str(payload),
            })

        for topic, payload, recv_ts in batch:
            m = None
            try:
                m = json.loads(payload)
                if topic.startswith("fleet/"):
                    storage.upsert_device(conn, m)
                    continue
                dev = str(m["device_id"])
                ts = float(m["ts"])
                z = [to_z(n, float(m["readings"][n])) for n in s.feature_names]
                if not np.all(np.isfinite(z)):
                    counters["non_finite"] += 1
                    fail("non_finite_value", "NaN or inf in readings", payload, m)
                    continue
                src_run, seq = m.get("run_id"), m.get("seq")
                if featurizer is not None:
                    vec, status = featurizer.update(dev, z, run_id=src_run, seq=seq, ts=ts)
                    if status == "warmup":
                        counters["warmup"] += 1
                        continue
                    if status == "gap_reset":
                        counters["gap_reset"] += 1
                        fail("gap_reset", f"no data for > {featurizer.max_gap_s}s; window restarted", payload, m)
                        continue
                    if status == "out_of_order":
                        counters["out_of_order"] += 1
                        fail("out_of_order", f"seq={seq} not newer than last seen", payload, m)
                        continue
                    flat = flatline_sensors(vec, s.feature_names, args.flat_eps)
                else:
                    vec = np.asarray(z, dtype=np.float32)
                    flat = []
                if flat:
                    counters["flatline_flags"] += 1
                feats.append(vec)
                valid.append((dev, ts, m, recv_ts, src_run, flat))
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                counters["malformed"] += 1
                fail("malformed", f"{type(exc).__name__}: {exc}", payload, m)

        rows: list[dict] = []
        if valid:
            try:
                x = np.vstack(feats).astype(np.float32)
                t0 = time.perf_counter()
                scores = det.score_batch(x)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                done = time.time()
                rows = [{
                    "device_id": dev, "seq": m.get("seq"), "event_ts": ts, "recv_ts": recv,
                    "readings": json.dumps(m["readings"]), "score": float(sc),
                    "is_anomaly": int(sc >= det.threshold or bool(flat)),
                    "alert_reason": alert_reason(bool(sc >= det.threshold), flat), "label": m.get("label"),
                    "e2e_ms": (done - ts) * 1000.0, "infer_ms": infer_ms, "batch_size": len(valid),
                    "run_id": run_id, "source_run_id": src, "model": det.model_id,
                } for (dev, ts, m, recv, src, flat), sc in zip(valid, scores)]
            except Exception as exc:  # noqa: BLE001 - a bad batch must not kill the consumer
                counters["scoring_error"] += len(valid)
                log.exception("scoring failed for a batch", extra={"rows": len(valid)})
                for dev, ts, m, recv, src, _flat in valid:
                    fail("scoring_error", f"{type(exc).__name__}: {exc}", json.dumps(m), m, dev)

        try:
            if rows:
                storage.insert_readings(conn, rows)
                counters["processed"] += len(rows)
            storage.record_failures(conn, failures)
        except sqlite3.Error:
            counters["db_error"] += len(rows) + len(failures)
            log.exception("database write failed", extra={"rows": len(rows), "failures": len(failures)})

    def drain() -> None:
        batch: list = []
        while True:
            try:
                batch.append(q.get_nowait())
            except queue.Empty:
                break
            if len(batch) >= args.batch_size:
                process(batch)
                batch = []
        if batch:
            process(batch)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"consumer-{run_id}")
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    try:
        client.connect(args.host, args.port, keepalive=30)
    except OSError as exc:
        log.error("cannot reach MQTT broker; is mosquitto running?",
                  extra={"host": args.host, "port": args.port, "error": str(exc)})
        storage.end_run(conn, run_id, "crashed", {"error": str(exc)})
        conn.close()
        return 2
    client.loop_start()

    status = "stopped"
    max_wait = args.max_wait_ms / 1000.0
    last_report, last_count = time.monotonic(), 0
    try:
        while True:
            try:
                batch = [q.get(timeout=0.5)]
            except queue.Empty:
                batch = []
            deadline = time.monotonic() + max_wait
            while batch and len(batch) < args.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(q.get(timeout=remaining))
                except queue.Empty:
                    break
            if batch:
                process(batch)
            now = time.monotonic()
            if now - last_report >= 5.0:
                rate = (counters["processed"] - last_count) / (now - last_report)
                storage.heartbeat(conn, run_id)
                log.info("progress", extra={"rate_per_s": round(rate), "queue": q.qsize(), **counters})
                last_report, last_count = now, counters["processed"]
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        log.exception("consumer crashed")
        status = "crashed"
    finally:
        client.loop_stop()
        client.disconnect()
        try:
            drain()
        except Exception:  # noqa: BLE001
            log.exception("drain on shutdown failed")
        storage.end_run(conn, run_id, status, counters)
        conn.close()
        log.info("consumer stopped", extra={"status": status, **counters})
    return 0 if status == "stopped" else 1


if __name__ == "__main__":
    raise SystemExit(main())
