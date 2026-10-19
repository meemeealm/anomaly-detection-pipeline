"""MQTT consumer: micro-batch -> standardize -> ONNX Runtime scoring -> SQLite.

Example:
    python scripts/consume_score.py --batch-size 32 --max-wait-ms 20
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from iot_anomaly import storage  # noqa: E402
from iot_anomaly.inference import AnomalyDetector  # noqa: E402
from iot_anomaly.settings import get_settings  # noqa: E402
from iot_anomaly.simulation import to_z  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="sensors/+/readings")
    ap.add_argument("--qos", type=int, choices=(0, 1), default=0)
    ap.add_argument("--db", default=str(ROOT / "results" / "stream.db"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-wait-ms", type=float, default=20.0)
    ap.add_argument("--queue-size", type=int, default=10_000)
    return ap.parse_args()


def main() -> int:
    import paho.mqtt.client as mqtt

    args = parse_args()
    s = get_settings()
    if not s.onnx_path.exists():
        print("Model missing. Run: python scripts/train_export.py", file=sys.stderr)
        return 2
    det = AnomalyDetector(s)
    conn = storage.connect(args.db)
    q: queue.Queue = queue.Queue(maxsize=args.queue_size)
    counters = {"processed": 0, "dropped": 0, "malformed": 0}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code.is_failure:
            print(f"MQTT connect failed: {reason_code}", file=sys.stderr)
            return
        client.subscribe([(args.topic, args.qos), ("fleet/+/meta", 1)])
        print(f"Connected; subscribed to {args.topic}")

    def on_message(client, userdata, msg):
        try:
            q.put_nowait((msg.topic, msg.payload, time.time()))
        except queue.Full:
            counters["dropped"] += 1   # backpressure: shed load, never block the network thread

    def process(batch: list) -> None:
        valid = []   # (device_id, event_ts, message, recv_ts)
        feats = []
        for topic, payload, recv_ts in batch:
            try:
                m = json.loads(payload)
                if topic.startswith("fleet/"):
                    storage.upsert_device(conn, m)
                    continue
                feats.append([to_z(n, float(m["readings"][n])) for n in s.feature_names])
                valid.append((str(m["device_id"]), float(m["ts"]), m, recv_ts))
            except (ValueError, KeyError, TypeError):
                counters["malformed"] += 1
                if feats and len(feats) > len(valid):
                    feats.pop()   # keep feats aligned with valid
        if not valid:
            return
        x = np.asarray(feats, dtype=np.float32)
        t0 = time.perf_counter()
        scores = det.score_batch(x)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        done = time.time()
        rows = [
            (dev, m.get("seq"), ts, recv, json.dumps(m["readings"]), float(sc),
             int(sc >= det.threshold), m.get("label"), (done - ts) * 1000.0, infer_ms, len(valid))
            for (dev, ts, m, recv), sc in zip(valid, scores)
        ]
        storage.insert_scores(conn, rows)
        counters["processed"] += len(rows)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="anomaly-consumer")
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    try:
        client.connect(args.host, args.port, keepalive=30)
    except OSError as exc:
        print(f"Cannot reach MQTT broker at {args.host}:{args.port} ({exc}). Is mosquitto running?",
              file=sys.stderr)
        return 2
    client.loop_start()

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
                print(f"processed={counters['processed']} ({rate:.0f}/s) "
                      f"queue={q.qsize()} dropped={counters['dropped']} malformed={counters['malformed']}")
                last_report, last_count = now, counters["processed"]
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        conn.close()
        print(f"Stopped. {counters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
