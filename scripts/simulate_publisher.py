"""Fake IoT fleet publishing readings to an MQTT broker.

Topics:  sensors/<device_id>/readings   (QoS configurable)
         fleet/<device_id>/meta         (retained, QoS 1)

Example:
    python scripts/simulate_publisher.py --devices 50 --rate 1 --anomaly-rate 0.03
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iot_anomaly.settings import get_settings  # noqa: E402
from iot_anomaly.simulation import Fleet  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--devices", type=int, default=20)
    ap.add_argument("--rate", type=float, default=1.0, help="readings per second PER device")
    ap.add_argument("--anomaly-rate", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qos", type=int, choices=(0, 1), default=0)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = run until Ctrl+C")
    ap.add_argument("--drift", type=float, default=0.0, help="z-units of drift per minute")
    ap.add_argument("--drift-feature", default="temperature")
    return ap.parse_args()


def main() -> int:
    import paho.mqtt.client as mqtt

    args = parse_args()
    s = get_settings()
    fleet = Fleet(
        s.feature_names,
        n_devices=args.devices,
        seed=args.seed,
        anomaly_rate=args.anomaly_rate,
        drift_per_min=args.drift,
        drift_feature=args.drift_feature if args.drift else None,
    )

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="sim-publisher")
    try:
        client.connect(args.host, args.port, keepalive=30)
    except OSError as exc:
        print(f"Cannot reach MQTT broker at {args.host}:{args.port} ({exc}). Is mosquitto running?",
              file=sys.stderr)
        return 2
    client.loop_start()

    for dev in fleet.devices:
        client.publish(f"fleet/{dev.device_id}/meta", json.dumps(dev.meta()), qos=1, retain=True)
    print(f"Publishing {args.devices} devices x {args.rate}/s = "
          f"{args.devices * args.rate:.0f} msg/s (anomaly rate {args.anomaly_rate:.1%}). Ctrl+C to stop.")

    interval = 1.0 / args.rate
    start = next_t = last_report = time.monotonic()
    seq = sent = 0
    warned = False
    try:
        while True:
            now = time.monotonic()
            if args.duration and now - start >= args.duration:
                break
            elapsed_min = (now - start) / 60.0
            for dev in fleet.devices:
                readings, label = fleet.sample(dev, elapsed_min)
                payload = json.dumps({
                    "device_id": dev.device_id, "seq": seq, "ts": time.time(),
                    "readings": readings, "label": label,
                })
                client.publish(f"sensors/{dev.device_id}/readings", payload, qos=args.qos)
                sent += 1
            seq += 1
            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0 and not warned:
                print("WARN: publisher cannot keep up with the requested rate.", file=sys.stderr)
                warned = True
            if time.monotonic() - last_report >= 5.0:
                last_report = time.monotonic()
                print(f"sent={sent}")
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(0.3)
        client.loop_stop()
        client.disconnect()
        print(f"Stopped. Total sent: {sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
