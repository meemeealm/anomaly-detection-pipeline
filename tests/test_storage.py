from iot_anomaly import storage


def test_roundtrip(tmp_path):
    conn = storage.connect(tmp_path / "t.db")
    storage.upsert_device(conn, {"device_id": "dev-000", "site": "Manila", "firmware": "v1.0.0", "serial": "SN-1"})
    storage.upsert_device(conn, {"device_id": "dev-000", "site": "Cebu", "firmware": "v1.0.1", "serial": "SN-1"})
    storage.insert_scores(conn, [("dev-000", 0, 1.0, 1.1, "{}", 0.9, 1, 1, 100.0, 1.0, 1)])
    assert conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 1
    assert conn.execute("SELECT site FROM devices").fetchone()[0] == "Cebu"
    assert conn.execute("SELECT is_anomaly, label FROM readings").fetchone() == (1, 1)
