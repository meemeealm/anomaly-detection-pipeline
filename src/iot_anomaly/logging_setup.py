"""window-aware: structured logging with a per-process run_id.

Console: short human-readable lines. File: JSON lines in results/logs/<component>_<run_id>.jsonl
(every line carries run_id, so a run can be reconstructed with grep/jq).
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


def new_run_id() -> str:
    """Sortable, filename-safe id, e.g. 20261019T051400Z-a1b2c3."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in vars(record).items() if k not in _RESERVED and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def __init__(self, component: str, run_id: str) -> None:
        super().__init__()
        self.component, self.run_id = component, run_id

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "component": self.component,
            "run_id": self.run_id,
            "msg": record.getMessage(),
            **_extras(record),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, component: str, run_id: str) -> None:
        super().__init__()
        self.component, self.run_id = component, run_id

    def format(self, record: logging.LogRecord) -> str:
        t = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        extras = " ".join(f"{k}={json.dumps(v, default=str)}" for k, v in _extras(record).items())
        line = f"{t} {record.levelname:<7} {self.component}[{self.run_id}] {record.getMessage()} {extras}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(component: str, run_id: str, level: str = "INFO",
                  log_dir: str | Path | None = None, to_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(f"iot.{component}")
    logger.setLevel(level.upper())
    for h in list(logger.handlers):
        h.close()
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter(component, run_id))
    logger.addHandler(console)

    if to_file:
        directory = Path(log_dir) if log_dir else ROOT / "results" / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(directory / f"{component}_{run_id}.jsonl", encoding="utf-8")
        fh.setFormatter(JsonFormatter(component, run_id))
        logger.addHandler(fh)
    return logger
