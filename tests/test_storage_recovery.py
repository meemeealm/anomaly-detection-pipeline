
import sqlite3

import pytest

from iot_anomaly import storage


def test_database_lock_raises_after_timeout(tmp_path):
    db_path = tmp_path / "locked.db"

    # Open two connections to the same database.
    conn1 = storage.connect(db_path)
    conn2 = storage.connect(db_path)

    try:
        # Make the first connection hold a write lock.
        conn1.execute("BEGIN IMMEDIATE")

        # Prevent the second connection from waiting for 10 seconds.
        conn2.execute("PRAGMA busy_timeout = 100")

        # The second connection cannot acquire the write lock.
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with conn2:
                conn2.execute(
                    "INSERT INTO devices "
                    "(device_id, updated_at) VALUES (?, ?)",
                    ("dev-lock-test", 1.0),
                )

    finally:
        conn1.rollback()
        conn1.close()
        conn2.close()
