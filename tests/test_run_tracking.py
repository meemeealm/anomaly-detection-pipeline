import sqlite3
import time

from iot_anomaly import storage


def test_run_lifecycle(tmp_path):
    conn = storage.connect(tmp_path / "t.db")
    storage.start_run(conn, "R1", "consumer", {"model": "m"})
    assert conn.execute("SELECT status FROM runs WHERE run_id='R1'").fetchone()[0] == "running"
    storage.end_run(conn, "R1", "stopped", {"processed": 3})
    status, summary = conn.execute("SELECT status, summary FROM runs WHERE run_id='R1'").fetchone()
    assert status == "stopped" and '"processed": 3' in summary


def test_stale_running_run_is_marked_unclean(tmp_path):
    conn = storage.connect(tmp_path / "t.db")
    storage.start_run(conn, "R2", "consumer", {})
    conn.execute("UPDATE runs SET heartbeat_at=? WHERE run_id='R2'", (time.time() - 600,))
    conn.commit()
    assert storage.close_stale_runs(conn, older_than_s=60) == 1
    assert conn.execute("SELECT status FROM runs WHERE run_id='R2'").fetchone()[0] == "unclean_shutdown"


def test_fresh_running_run_is_left_alone(tmp_path):
    conn = storage.connect(tmp_path / "t.db")
    storage.start_run(conn, "R3", "consumer", {})
    assert storage.close_stale_runs(conn, older_than_s=60) == 0


def test_failed_events_and_payload_truncation(tmp_path):
    conn = storage.connect(tmp_path / "t.db")
    storage.record_failures(conn, [
        {"run_id": "R1", "ts": 1.0, "device_id": "d", "reason": "malformed", "detail": "x", "payload": "p" * 5000},
        {"run_id": "R1", "ts": 2.0, "reason": "out_of_order"},
    ])
    assert conn.execute("SELECT COUNT(*) FROM failed_events").fetchone()[0] == 2
    assert conn.execute("SELECT LENGTH(payload) FROM failed_events WHERE reason='malformed'").fetchone()[0] == 1000


def test_insert_readings_with_run_columns(tmp_path):
    conn = storage.connect(tmp_path / "t.db")
    storage.insert_readings(conn, [{
        "device_id": "d", "seq": 1, "event_ts": 1.0, "recv_ts": 1.1, "readings": "{}", "score": 0.4,
        "is_anomaly": 0, "label": 0, "run_id": "R1", "source_run_id": "P1", "model": "iforest_window",
    }])
    assert conn.execute("SELECT run_id, source_run_id, model FROM readings").fetchone() == ("R1", "P1", "iforest_window")


def test_old_database_is_migrated_not_dropped(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE readings (id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL, seq INTEGER,
            event_ts REAL NOT NULL, recv_ts REAL NOT NULL, readings TEXT NOT NULL, score REAL NOT NULL,
            is_anomaly INTEGER NOT NULL, label INTEGER, e2e_ms REAL, infer_ms REAL, batch_size INTEGER);
        INSERT INTO readings (device_id, event_ts, recv_ts, readings, score, is_anomaly)
            VALUES ('d', 1.0, 1.1, '{}', 0.5, 0);
    """)
    old.commit()
    old.close()
    conn = storage.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(readings)")}
    assert {"run_id", "source_run_id", "model"} <= cols
    assert conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 1
