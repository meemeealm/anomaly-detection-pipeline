import numpy as np

from iot_anomaly.features import DeviceWindow, feature_names
from iot_anomaly.rules import alert_reason, flatline_sensors

BASE = ["a", "b", "c"]


def _features(columns):
    w = DeviceWindow(len(columns[0]))
    out = None
    for t in range(len(columns[0])):
        out = w.push([col[t] for col in columns])
    return out


def test_layout_matches_features_module():
    names = feature_names(BASE)
    for i, n in enumerate(BASE):
        assert names[2 * len(BASE) + i] == f"{n}_std"


def test_flatline_detected_on_the_right_sensor():
    rng = np.random.default_rng(0)
    f = _features([rng.normal(size=10), np.full(10, 1.7), rng.normal(size=10)])
    assert flatline_sensors(f, BASE, 0.02) == ["b"]


def test_normal_noise_not_flagged():
    rng = np.random.default_rng(1)
    flags = sum(bool(flatline_sensors(_features([rng.normal(size=10) * 0.9 for _ in BASE]), BASE, 0.02))
                for _ in range(500))
    assert flags == 0


def test_eps_zero_disables_rule():
    f = _features([np.zeros(10)] * 3)
    assert flatline_sensors(f, BASE, 0.0) == []


def test_alert_reason_formatting():
    assert alert_reason(True, []) == "model"
    assert alert_reason(False, ["a"]) == "flatline:a"
    assert alert_reason(True, ["a", "b"]) == "model+flatline:a,b"
    assert alert_reason(False, []) is None
