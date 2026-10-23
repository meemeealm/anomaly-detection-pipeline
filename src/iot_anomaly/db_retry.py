
import sqlite3
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_locked(
    operation: Callable[[], T],
    max_attempts: int = 3,
    delay_s: float = 0.1,
) -> T:
    """Retry an operation after transient SQLite lock errors."""
    for attempt in range(max_attempts):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay_s * (2 ** attempt))

    raise RuntimeError("unreachable")
