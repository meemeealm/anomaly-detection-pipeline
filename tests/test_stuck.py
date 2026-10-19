import pytest

pytest.importorskip("faker")

from iot_anomaly.settings import get_settings
from iot_anomaly.simulation import Fleet

NAMES = get_settings().feature_names


def test_stuck_sensor_flatlines_and_is_labeled():
    f = Fleet(NAMES, n_devices=1, seed=3, anomaly_rate=0.0, stuck_rate=1.0, stuck_len=(5, 5))
    dev = f.devices[0]
    samples = [f.sample(dev) for _ in range(6)]       # fault start + 5 frozen readings
    idx = NAMES[dev.stuck_idx]
    assert len({r[idx] for r, _ in samples}) == 1
    assert all(label == 1 for _, label in samples)


def test_no_stuck_faults_by_default():
    f = Fleet(NAMES, n_devices=1, seed=3, anomaly_rate=0.0)
    assert all(f.sample(f.devices[0])[1] == 0 for _ in range(500))
