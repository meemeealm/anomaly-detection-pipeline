"""metrics-aware: Prometheus metrics for the consumer.

Deliberately low-cardinality: labels are outcome / alert class only. Never label by device_id
(50 devices x every metric would multiply series; use the SQLite tables for per-device questions).

The consumer keeps plain counters; sync() copies their deltas into Prometheus counters each loop,
and observe_batch() records latency / batch histograms for rows that were stored.
prometheus-client is imported lazily so the rest of the pipeline works without it.
"""
from __future__ import annotations

import time

OUTCOMES = ("processed", "dropped", "malformed", "non_finite", "out_of_order",
            "gap_reset", "warmup", "scoring_error", "db_error")
ALERT_CLASSES = ("model", "flatline", "model+flatline")
E2E_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
INFER_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)

# Every series name the endpoint exposes. Dashboard and alert rules may only use these
# (checked by tests/test_observability_files.py).
EXPOSED_SERIES = frozenset({
    "iot_messages_total", "iot_flatline_flags_total", "iot_alerts_total",
    "iot_e2e_latency_seconds_bucket", "iot_e2e_latency_seconds_sum", "iot_e2e_latency_seconds_count",
    "iot_inference_seconds_bucket", "iot_inference_seconds_sum", "iot_inference_seconds_count",
    "iot_batch_size_bucket", "iot_batch_size_sum", "iot_batch_size_count",
    "iot_queue_depth", "iot_last_scored_timestamp_seconds", "iot_run_info",
})


def reason_class(reason: str | None) -> str | None:
    """'model+flatline:temperature,rpm' -> 'model+flatline' (drops sensor names: keeps cardinality at 3)."""
    if not reason:
        return None
    return "+".join(part.split(":", 1)[0] for part in reason.split("+"))


class Metrics:
    def __init__(self, registry=None) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self._messages = Counter("iot_messages", "Messages by outcome (processed = scored and stored)",
                                 ["outcome"], registry=r)
        self._flat = Counter("iot_flatline_flags", "Readings flagged by the flatline rule", registry=r)
        self._alerts = Counter("iot_alerts", "Flagged readings by alert class", ["reason"], registry=r)
        self._e2e = Histogram("iot_e2e_latency_seconds", "Device timestamp to scored and stored",
                              buckets=E2E_BUCKETS, registry=r)
        self._infer = Histogram("iot_inference_seconds", "Model time per micro-batch",
                                buckets=INFER_BUCKETS, registry=r)
        self._batch = Histogram("iot_batch_size", "Rows per scored micro-batch",
                                buckets=BATCH_BUCKETS, registry=r)
        self._queue = Gauge("iot_queue_depth", "Messages waiting in the consumer queue", registry=r)
        self._last = Gauge("iot_last_scored_timestamp_seconds", "Unix time of the last stored batch", registry=r)
        self._info = Gauge("iot_run_info", "Constant 1; labels describe this consumer run",
                           ["run_id", "model", "model_type", "window"], registry=r)
        for outcome in OUTCOMES:          # create series at 0 so rate() and dashboards work from the start
            self._messages.labels(outcome)
        for cls in ALERT_CLASSES:
            self._alerts.labels(cls)
        self._synced: dict[str, int] = {}

    def start(self, host: str, port: int) -> None:
        from prometheus_client import start_http_server

        start_http_server(port, addr=host, registry=self.registry)

    def set_info(self, run_id: str, model: str, model_type: str, window_size) -> None:
        self._info.labels(run_id, model, model_type, str(window_size or 0)).set(1)

    def sync(self, counters: dict, queue_depth: int) -> None:
        for key in OUTCOMES:
            current = counters.get(key, 0)
            delta = current - self._synced.get(key, 0)
            if delta > 0:
                self._messages.labels(key).inc(delta)
                self._synced[key] = current
        current = counters.get("flatline_flags", 0)
        delta = current - self._synced.get("flatline_flags", 0)
        if delta > 0:
            self._flat.inc(delta)
            self._synced["flatline_flags"] = current
        self._queue.set(queue_depth)

    def observe_batch(self, rows: list[dict]) -> None:
        if not rows:
            return
        self._infer.observe(rows[0]["infer_ms"] / 1000.0)
        self._batch.observe(len(rows))
        for row in rows:
            self._e2e.observe(max(0.0, row["e2e_ms"] / 1000.0))
            cls = reason_class(row.get("alert_reason"))
            if cls:
                self._alerts.labels(cls).inc()
        self._last.set(time.time())


def make_metrics(registry=None) -> Metrics | None:
    """Return Metrics, or None when prometheus-client is not installed."""
    try:
        return Metrics(registry)
    except ImportError:
        return None
