"""window-aware, flatline-aware: SQLite persistence for scored readings, runs and failed events.

WAL mode: one writer (the consumer), many readers (the dashboard).
Older databases are migrated in place (new columns are added, nothing is dropped).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from iot_anomaly.db_retry import retry_locked

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id  TEXT PRIMARY KEY,
    site       TEXT,
    firmware   TEXT,
    serial     TEXT,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS readings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT NOT NULL,
    seq           INTEGER,
    event_ts      REAL NOT NULL,   -- device clock (epoch seconds)
    recv_ts       REAL NOT NULL,   -- consumer receive time
    readings      TEXT NOT NULL,   -- raw physical values as JSON
    score         REAL NOT NULL,
    is_anomaly    INTEGER NOT NULL,
    label         INTEGER,         -- simulator ground truth (evaluation only)
    e2e_ms        REAL,            -- event_ts -> scored
    infer_ms      REAL,            -- model time for the whole micro-batch
    batch_size    INTEGER,
    run_id        TEXT,            -- consumer run that scored this row
    source_run_id TEXT,            -- publisher run that produced it
    model         TEXT,            -- model file stem, e.g. iforest_window
    alert_reason  TEXT             -- NULL | model | flatline:<sensors> | model+flatline:<sensors>
);
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    component    TEXT NOT NULL,
    started_at   REAL NOT NULL,
    ended_at     REAL,
    heartbeat_at REAL,
    status       TEXT NOT NULL,    -- running | stopped | crashed | unclean_shutdown
    config       TEXT,
    summary      TEXT
);
CREATE TABLE IF NOT EXISTS failed_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    ts            REAL NOT NULL,
    source_run_id TEXT,
    device_id     TEXT,
    reason        TEXT NOT NULL,
    detail        TEXT,
    payload       TEXT
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(event_ts);
CREATE INDEX IF NOT EXISTS idx_readings_dev_ts ON readings(device_id, event_ts);
CREATE INDEX IF NOT EXISTS idx_readings_run ON readings(run_id);
CREATE INDEX IF NOT EXISTS idx_failed_ts ON failed_events(ts);
CREATE INDEX IF NOT EXISTS idx_failed_reason ON failed_events(reason);
"""

_NEW_READING_COLUMNS = {"run_id": "TEXT", "source_run_id": "TEXT", "model": "TEXT", "alert_reason": "TEXT"}

_READING_COLS = (
    "device_id", "seq", "event_ts", "recv_ts", "readings", "score", "is_anomaly", "label",
    "e2e_ms", "infer_ms", "batch_size", "run_id", "source_run_id", "model", "alert_reason",
)
_FAILURE_COLS = ("run_id", "ts", "source_run_id", "device_id", "reason", "detail", "payload")

_LEGACY_INSERT = (
    "INSERT INTO readings (device_id, seq, event_ts, recv_ts, readings, score,"
    " is_anomaly, label, e2e_ms, infer_ms, batch_size) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
)


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(readings)")}
    for name, sql_type in _NEW_READING_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE readings ADD COLUMN {name} {sql_type}")
    conn.commit()


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.executescript(INDEXES)
    return conn


def upsert_device(conn: sqlite3.Connection, meta: dict) -> None:
    device_id = str(meta["device_id"])
    with conn:
        conn.execute(
            "INSERT INTO devices (device_id, site, firmware, serial, updated_at) VALUES (?,?,?,?,?)"
            " ON CONFLICT(device_id) DO UPDATE SET site=excluded.site,"
            " firmware=excluded.firmware, serial=excluded.serial, updated_at=excluded.updated_at",
            (device_id, meta.get("site"), meta.get("firmware"), meta.get("serial"), time.time()),
        )


def insert_scores(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """Legacy 11-column tuple insert (kept for backward compatibility)."""
    with conn:
        conn.executemany(_LEGACY_INSERT, rows)



def insert_readings(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """Insert scored readings, retrying transient SQLite lock errors."""
    if not rows:
        return

    sql = (
        f"INSERT INTO readings ({','.join(_READING_COLS)}) "
        f"VALUES ({','.join(':' + c for c in _READING_COLS)})"
    )
    cleaned = [{c: r.get(c) for c in _READING_COLS} for r in rows]

    def operation() -> None:
        with conn:
            conn.executemany(sql, cleaned)

    retry_locked(operation)


def record_failures(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """Insert failed events, retrying transient SQLite lock errors."""
    if not rows:
        return

    sql = (
        f"INSERT INTO failed_events ({','.join(_FAILURE_COLS)}) "
        f"VALUES ({','.join(':' + c for c in _FAILURE_COLS)})"
    )
    cleaned = []
    for r in rows:
        item = {c: r.get(c) for c in _FAILURE_COLS}
        if item["payload"] is not None:
            item["payload"] = str(item["payload"])[:1000]
        cleaned.append(item)

    def operation() -> None:
        with conn:
            conn.executemany(sql, cleaned)

    retry_locked(operation)


# ------------------------------------------------------------------- runs ----
def start_run(conn: sqlite3.Connection, run_id: str, component: str, config: dict) -> None:
    now = time.time()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, component, started_at, heartbeat_at, status, config)"
            " VALUES (?,?,?,?,?,?)",
            (run_id, component, now, now, "running", json.dumps(config, default=str)),
        )


def heartbeat(conn: sqlite3.Connection, run_id: str) -> None:
    with conn:
        conn.execute("UPDATE runs SET heartbeat_at=? WHERE run_id=?", (time.time(), run_id))


def end_run(conn: sqlite3.Connection, run_id: str, status: str, summary: dict) -> None:
    now = time.time()
    with conn:
        conn.execute(
            "UPDATE runs SET ended_at=?, heartbeat_at=?, status=?, summary=? WHERE run_id=?",
            (now, now, status, json.dumps(summary, default=str), run_id),
        )


def close_stale_runs(conn: sqlite3.Connection, older_than_s: float = 60.0) -> int:
    """Mark runs that stopped heart-beating (kill -9, power loss) as unclean_shutdown."""
    cutoff = time.time() - older_than_s
    with conn:
        cur = conn.execute(
            "UPDATE runs SET status='unclean_shutdown', ended_at=COALESCE(heartbeat_at, started_at)"
            " WHERE status='running' AND COALESCE(heartbeat_at, started_at) < ?",
            (cutoff,),
        )
    return cur.rowcount
