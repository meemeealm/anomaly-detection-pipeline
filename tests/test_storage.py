import pytest
import sqlite3

from iot_anomaly import storage


def test_roundtrip(tmp_path):
    conn = storage.connect(tmp_path / "t.db")
    storage.upsert_device(conn, {"device_id": "dev-000", "site": "Manila", "firmware": "v1.0.0", "serial": "SN-1"})
    storage.upsert_device(conn, {"device_id": "dev-000", "site": "Cebu", "firmware": "v1.0.1", "serial": "SN-1"})
    storage.insert_scores(conn, [("dev-000", 0, 1.0, 1.1, "{}", 0.9, 1, 1, 100.0, 1.0, 1)])
    assert conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 1
    assert conn.execute("SELECT site FROM devices").fetchone()[0] == "Cebu"
    assert conn.execute("SELECT is_anomaly, label FROM readings").fetchone() == (1, 1)


def test_insert_scores_rolls_back_failed_batch(tmp_path):
    conn = storage.connect(tmp_path / "rollback.db")

    valid_row = (
        "dev-001", 1, 100.0, 100.1, "{}",
        0.2, 0, 0, 100.0, 1.0, 1
    )

    invalid_row = (
        "dev-002", 2, None, 100.2, "{}",
        0.3, 0, 0, 100.0, 1.0, 1
    )

    with pytest.raises(sqlite3.IntegrityError):
        storage.insert_scores(conn, [valid_row, invalid_row])

    count = conn.execute(
        "SELECT COUNT(*) FROM readings"
    ).fetchone()[0]

    assert count == 0
    conn.close()
