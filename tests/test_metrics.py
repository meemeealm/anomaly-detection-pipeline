import time

import pytest

pytest.importorskip("prometheus_client")

from iot_anomaly.metrics import EXPOSED_SERIES, Metrics, reason_class


def _row(reason=None, e2e_ms=40.0, infer_ms=2.0):
    return {"e2e_ms": e2e_ms, "infer_ms": infer_ms, "alert_reason": reason}


def test_reason_class_drops_sensor_names():
    assert reason_class(None) is None
    assert reason_class("model") == "model"
    assert reason_class("flatline:temperature") == "flatline"
    assert reason_class("model+flatline:temperature,rpm") == "model+flatline"


def test_sync_copies_counter_deltas_once():
    m = Metrics()
    counters = {"processed": 5, "malformed": 2, "flatline_flags": 3}
    m.sync(counters, queue_depth=7)
    m.sync(counters, queue_depth=7)                      # no change: must not double count
    counters["processed"] = 9
    m.sync(counters, queue_depth=1)
    get = m.registry.get_sample_value
    assert get("iot_messages_total", {"outcome": "processed"}) == 9
    assert get("iot_messages_total", {"outcome": "malformed"}) == 2
    assert get("iot_messages_total", {"outcome": "dropped"}) == 0        # pre-created at 0
    assert get("iot_flatline_flags_total") == 3
    assert get("iot_queue_depth") == 1


def test_observe_batch_records_histograms_alerts_and_freshness():
    m = Metrics()
    before = time.time()
    m.observe_batch([_row(), _row("model"), _row("flatline:rpm"), _row("model+flatline:a,b")])
    get = m.registry.get_sample_value
    assert get("iot_e2e_latency_seconds_count") == 4
    assert get("iot_inference_seconds_count") == 1
    assert get("iot_batch_size_sum") == 4
    assert get("iot_alerts_total", {"reason": "model"}) == 1
    assert get("iot_alerts_total", {"reason": "flatline"}) == 1
    assert get("iot_alerts_total", {"reason": "model+flatline"}) == 1
    assert get("iot_last_scored_timestamp_seconds") >= before


def test_negative_latency_from_clock_skew_is_clamped():
    m = Metrics()
    m.observe_batch([_row(e2e_ms=-500.0)])
    assert m.registry.get_sample_value("iot_e2e_latency_seconds_sum") == 0.0


def test_exposed_series_list_matches_the_real_exposition():
    from prometheus_client import generate_latest

    m = Metrics()
    m.set_info("RUN", "iforest_window", "isolation_forest", 10)
    m.observe_batch([_row()])
    text = generate_latest(m.registry).decode()
    sample_names = {line.split("{")[0].split(" ")[0] for line in text.splitlines() if line and not line.startswith("#")}
    missing = EXPOSED_SERIES - sample_names
    assert not missing, f"EXPOSED_SERIES lists series the endpoint does not expose: {sorted(missing)}"
