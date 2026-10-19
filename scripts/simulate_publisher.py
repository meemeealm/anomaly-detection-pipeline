"""window-aware, config-aware: fake IoT fleet publishing readings to an MQTT broker.

Topics:  sensors/<device_id>/readings   (QoS configurable)
         fleet/<device_id>/meta         (retained, QoS 1)
Every message carries this process's run_id; logs go to results/logs/.

Example:
    python scripts/simulate_publisher.py --devices 50 --rate 1 --anomaly-rate 0.03 --stuck-rate 0.001
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iot_anomaly.logging_setup import new_run_id, setup_logging  # noqa: E402
from iot_anomaly.runtime_config import PUBLISHER_SETTINGS, format_settings, resolve_all  # noqa: E402
from iot_anomaly.settings import get_settings  # noqa: E402
from iot_anomaly.simulation import Fleet  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # None = resolved from env / configs/stream.json / built-in default (see runtime_config.py)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--devices", type=int, default=20)
    ap.add_argument("--rate", type=float, default=1.0, help="readings per second PER device")
    ap.add_argument("--anomaly-rate", type=float, default=0.02, help="spike probability per reading")
    ap.add_argument("--stuck-rate", type=float, default=0.0, help="stuck-sensor fault start probability per reading")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qos", type=int, choices=(0, 1), default=None)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = run until Ctrl+C")
    ap.add_argument("--drift", type=float, default=0.0, help="z-units of drift per minute")
    ap.add_argument("--drift-feature", default="temperature")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--log-level", default=None)
    ap.add_argument("--config", default=None, help="stream config JSON (default configs/stream.json)")
    return ap.parse_args()


def main() -> int:
    import paho.mqtt.client as mqtt

    args = parse_args()
    try:
        resolved = resolve_all(args, PUBLISHER_SETTINGS)   # fills args.<name>
    except ValueError as exc:
        print(f"Invalid settings: {exc}", file=sys.stderr)
        return 2
    run_id = args.run_id or new_run_id()
    log = setup_logging("publisher", run_id, args.log_level)
    log.info("effective settings, value [source]", extra={"resolved": format_settings(resolved)})
    s = get_settings()
    fleet = Fleet(
        s.feature_names,
        n_devices=args.devices,
        seed=args.seed,
        anomaly_rate=args.anomaly_rate,
        drift_per_min=args.drift,
        drift_feature=args.drift_feature if args.drift else None,
        stuck_rate=args.stuck_rate,
    )

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"sim-{run_id}")
    try:
        client.connect(args.host, args.port, keepalive=30)
    except OSError as exc:
        log.error("cannot reach MQTT broker; is mosquitto running?",
                  extra={"host": args.host, "port": args.port, "error": str(exc)})
        return 2
    client.loop_start()

    for dev in fleet.devices:
        client.publish(f"fleet/{dev.device_id}/meta", json.dumps(dev.meta()), qos=1, retain=True)
    log.info("publisher starting", extra={
        "devices": args.devices, "rate_per_device": args.rate,
        "msg_per_s": args.devices * args.rate, "anomaly_rate": args.anomaly_rate,
        "stuck_rate": args.stuck_rate, "drift": args.drift, "seed": args.seed, "qos": args.qos})

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
                    "run_id": run_id, "device_id": dev.device_id, "seq": seq, "ts": time.time(),
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
                log.warning("publisher cannot keep up with the requested rate", extra={"lag_s": round(-delay, 2)})
                warned = True
            if time.monotonic() - last_report >= 5.0:
                last_report = time.monotonic()
                log.info("progress", extra={"sent": sent})
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep(0.3)
        client.loop_stop()
        client.disconnect()
        log.info("publisher stopped", extra={"sent": sent})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
