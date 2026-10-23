
import sqlite3

import pytest

from iot_anomaly import storage
from iot_anomaly.db_retry import retry_locked as real_retry_locked


@pytest.fixture
def db(tmp_path):
    """Create an isolated SQLite database for each test."""
    conn = storage.connect(tmp_path / "test_stream.db")
    yield conn
    conn.close()


def install_one_time_lock(monkeypatch):
    """Inject one transient lock, then use the real retry helper."""
    attempts = 0

    def retry_with_injected_lock(operation):
        def flaky_operation():
            nonlocal attempts
            attempts += 1

            if attempts == 1:
                raise sqlite3.OperationalError("database is locked")

            return operation()

        return real_retry_locked(
            flaky_operation,
            max_attempts=3,
            delay_s=0,
        )

    monkeypatch.setattr(storage, "retry_locked", retry_with_injected_lock)
    return lambda: attempts


def test_insert_readings_retries_and_saves_once(db, monkeypatch):
    get_attempts = install_one_time_lock(monkeypatch)

    rows = [
        {
            "device_id": "sensor-001",
            "seq": 1,
            "event_ts": 1000.0,
            "recv_ts": 1001.0,
            "readings": '{"temperature": 25.5}',
            "score": 0.12,
            "is_anomaly": 0,
        }
    ]

    storage.insert_readings(db, rows)

    assert get_attempts() == 2

    saved = db.execute(
        "SELECT device_id, seq FROM readings"
    ).fetchall()

    assert saved == [("sensor-001", 1)]


def test_record_failures_retries_and_saves_once(db, monkeypatch):
    get_attempts = install_one_time_lock(monkeypatch)

    failures = [
        {
            "run_id": "run-001",
            "ts": 1001.0,
            "device_id": "sensor-001",
            "reason": "malformed",
            "detail": "Invalid JSON",
            "payload": "{invalid",
        }
    ]

    storage.record_failures(db, failures)

    assert get_attempts() == 2

    saved = db.execute(
        "SELECT device_id, reason FROM failed_events"
    ).fetchall()

    assert saved == [("sensor-001", "malformed")]


def test_insert_readings_does_not_retry_unrelated_error(
    db, monkeypatch
):
    attempts = 0

    def retry_with_tracking(operation):
        nonlocal attempts
        attempts += 1
        return real_retry_locked(operation, max_attempts=3, delay_s=0)

    monkeypatch.setattr(storage, "retry_locked", retry_with_tracking)

    # A missing required column value causes an integrity error.
    rows = [
        {
            "device_id": "sensor-001",
            "event_ts": None,
            "recv_ts": 1001.0,
            "readings": "{}",
            "score": 0.1,
            "is_anomaly": 0,
        }
    ]

    with pytest.raises(sqlite3.IntegrityError):
        storage.insert_readings(db, rows)

    # The storage function invokes the retry helper once.
    # The helper does not retry IntegrityError.
    assert attempts == 1

    count = db.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    assert count == 0
