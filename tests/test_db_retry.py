
import sqlite3

import pytest

from iot_anomaly.db_retry import retry_locked


def test_retries_until_operation_succeeds():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise sqlite3.OperationalError("database is locked")

        return "saved"

    result = retry_locked(operation, max_attempts=3, delay_s=0)

    assert result == "saved"
    assert attempts == 3


def test_does_not_retry_unrelated_database_errors():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError("no such table: readings")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        retry_locked(operation, max_attempts=3, delay_s=0)

    assert attempts == 1
