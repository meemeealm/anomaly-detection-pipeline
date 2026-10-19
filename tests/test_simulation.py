import numpy as np
import pytest

pytest.importorskip("faker")

from iot_anomaly.settings import get_settings
from iot_anomaly.simulation import PROFILES, Fleet, to_raw, to_z

NAMES = get_settings().feature_names


def test_z_roundtrip():
    for name in PROFILES:
        assert to_z(name, to_raw(name, 1.7)) == pytest.approx(1.7)


def test_fleet_is_deterministic_for_a_seed():
    a, b = Fleet(NAMES, n_devices=3, seed=1), Fleet(NAMES, n_devices=3, seed=1)
    assert [d.site for d in a.devices] == [d.site for d in b.devices]
    assert a.sample(a.devices[0]) == b.sample(b.devices[0])


def test_readings_cover_all_features():
    f = Fleet(NAMES, n_devices=1, anomaly_rate=0.0)
    readings, label = f.sample(f.devices[0])
    assert set(readings) == set(NAMES) and label == 0


def test_anomaly_rate_is_roughly_respected():
    f = Fleet(NAMES, n_devices=1, anomaly_rate=0.1, seed=5)
    labels = [f.sample(f.devices[0])[1] for _ in range(3000)]
    assert 0.07 < np.mean(labels) < 0.13


def test_bad_drift_feature_rejected():
    with pytest.raises(ValueError):
        Fleet(NAMES, drift_feature="nope", drift_per_min=0.1)
