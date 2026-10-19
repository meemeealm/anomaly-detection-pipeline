import numpy as np

from iot_anomaly.features import DeviceWindow, StreamFeaturizer, feature_names


def test_warmup_then_features():
    w = DeviceWindow(3)
    z = np.array([1.0, 2.0])
    assert w.push(z) is None
    assert w.push(z) is None
    assert w.push(z).shape == (8,)


def test_feature_values():
    w = DeviceWindow(3)
    w.push([1.0, 10.0])
    w.push([2.0, 20.0])
    f = w.push([4.0, 30.0])
    np.testing.assert_allclose(f[0:2], [4.0, 30.0])                              # current
    np.testing.assert_allclose(f[2:4], [7 / 3, 20.0], rtol=1e-6)                 # mean
    np.testing.assert_allclose(f[4:6], [np.std([1, 2, 4]), np.std([10, 20, 30])], rtol=1e-6)
    np.testing.assert_allclose(f[6:8], [2.0, 10.0])                              # delta


def test_flatline_has_zero_std():
    w = DeviceWindow(5)
    for _ in range(5):
        f = w.push([3.0, 3.0])
    np.testing.assert_allclose(f[4:6], [0.0, 0.0], atol=1e-7)


def test_names_match_width():
    assert len(feature_names(["a", "b", "c"])) == 12
    assert feature_names(["a"]) == ["a", "a_mean", "a_std", "a_delta"]


def test_out_of_order_and_duplicate_ignored():
    sf = StreamFeaturizer(2)
    assert sf.update("d", [0.0], run_id="r", seq=0, ts=0.0)[1] == "warmup"
    assert sf.update("d", [0.0], run_id="r", seq=1, ts=1.0)[1] == "ok"
    assert sf.update("d", [0.0], run_id="r", seq=1, ts=1.0)[1] == "out_of_order"
    assert sf.update("d", [0.0], run_id="r", seq=0, ts=0.5)[1] == "out_of_order"
    assert sf.update("d", [0.0], run_id="r", seq=2, ts=2.0)[1] == "ok"


def test_gap_resets_window():
    sf = StreamFeaturizer(2, max_gap_s=10)
    sf.update("d", [0.0], run_id="r", seq=0, ts=0.0)
    assert sf.update("d", [0.0], run_id="r", seq=1, ts=1.0)[1] == "ok"
    assert sf.update("d", [0.0], run_id="r", seq=2, ts=100.0)[1] == "gap_reset"
    assert sf.update("d", [0.0], run_id="r", seq=3, ts=101.0)[1] == "ok"


def test_new_publisher_run_resets_sequence():
    sf = StreamFeaturizer(2)
    sf.update("d", [0.0], run_id="r1", seq=50, ts=0.0)
    sf.update("d", [0.0], run_id="r1", seq=51, ts=1.0)
    # publisher restarted: seq starts again at 0 but must NOT count as out_of_order
    assert sf.update("d", [0.0], run_id="r2", seq=0, ts=2.0)[1] == "warmup"
    assert sf.update("d", [0.0], run_id="r2", seq=1, ts=3.0)[1] == "ok"


def test_devices_are_independent():
    sf = StreamFeaturizer(2)
    sf.update("a", [0.0], run_id="r", seq=0, ts=0.0)
    assert sf.update("b", [0.0], run_id="r", seq=0, ts=0.0)[1] == "warmup"
    assert sf.update("a", [0.0], run_id="r", seq=1, ts=1.0)[1] == "ok"
