"""config-aware: layered runtime settings.

Precedence, highest first:
    command-line flag  >  environment variable IOT_<NAME>  >  configs/stream.json  >  built-in default

What lives where:
  * configs/model.json   - sensors, model paths, LightGBM threshold  (read by settings.py)
  * configs/stream.json  - operational choices: broker, batching, flatline rule, gap tolerance, log level
  * artifacts/<model>_meta.json - values TRAINING produced: threshold, window size, offset, feature list.
    These are never overridden by flags or the config (except IOT_THRESHOLD for experiments).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH_ENV = "IOT_STREAM_CONFIG"

DEFAULTS: dict = {
    "host": "127.0.0.1",
    "port": 1883,
    "topic": "sensors/+/readings",
    "qos": 0,
    "db": "results/stream.db",
    "batch_size": 32,
    "max_wait_ms": 20.0,
    "queue_size": 10_000,
    "flat_eps": 0.02,
    "max_gap_s": 30.0,
    "log_level": "INFO",
}
CONSUMER_SETTINGS = tuple(DEFAULTS)
PUBLISHER_SETTINGS = ("host", "port", "qos", "log_level")

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def load_stream_config(path: str | os.PathLike | None = None):
    """Return (settings dict, path or None). Unknown keys are an error (catches typos)."""
    import json

    p = Path(path or os.environ.get(CONFIG_PATH_ENV) or ROOT / "configs" / "stream.json")
    if not p.exists():
        if path or os.environ.get(CONFIG_PATH_ENV):
            raise ValueError(f"stream config not found: {p}")
        return {}, None
    data = json.loads(p.read_text(encoding="utf-8"))
    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown setting(s) in {p.name}: {unknown}; allowed: {sorted(DEFAULTS)}")
    return data, p


def _cast(name: str, raw):
    try:
        return type(DEFAULTS[name])(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}={raw!r} is not a valid {type(DEFAULTS[name]).__name__}") from exc


def resolve(name: str, cli_value, file_cfg: dict, file_label: str):
    """Return (value, source) for one setting."""
    if cli_value is not None:
        return cli_value, "command line"
    env_name = "IOT_" + name.upper()
    if env_name in os.environ:
        return _cast(name, os.environ[env_name]), f"env {env_name}"
    if name in file_cfg:
        return _cast(name, file_cfg[name]), file_label
    return DEFAULTS[name], "default"


_CHECKS = {
    "port": lambda v: 1 <= v <= 65535,
    "qos": lambda v: v in (0, 1),
    "batch_size": lambda v: v >= 1,
    "max_wait_ms": lambda v: v >= 0,
    "queue_size": lambda v: v >= 1,
    "flat_eps": lambda v: v >= 0,
    "max_gap_s": lambda v: v > 0,
    "log_level": lambda v: str(v).upper() in _LOG_LEVELS,
}


def _validate(name: str, value) -> None:
    check = _CHECKS.get(name)
    if check is not None and not check(value):
        raise ValueError(f"{name}={value!r} is out of range")


def resolve_all(args, names, config_path=None) -> dict:
    """Resolve each setting, write it back onto `args`, return {name: {"value", "source"}}."""
    file_cfg, path = load_stream_config(getattr(args, "config", None) or config_path)
    label = f"configs/{path.name}" if path else "config file"
    out = {}
    for name in names:
        value, source = resolve(name, getattr(args, name, None), file_cfg, label)
        _validate(name, value)
        if name == "db":
            p = Path(value)
            value = str(p if p.is_absolute() else ROOT / p)
        setattr(args, name, value)
        out[name] = {"value": value, "source": source}
    return out


def resolve_one(name: str, config_path=None):
    file_cfg, path = load_stream_config(config_path)
    value, source = resolve(name, None, file_cfg, f"configs/{path.name}" if path else "config file")
    if name == "db":
        p = Path(value)
        value = str(p if p.is_absolute() else ROOT / p)
    return value, source


def model_sources(settings, detector) -> dict:
    """Where the model-related values came from (for the startup log and runs.config)."""
    is_if = detector.model_type != "lightgbm"
    if "IOT_THRESHOLD" in os.environ:
        th_src = "env IOT_THRESHOLD"
    else:
        th_src = "model meta (set by training)" if is_if else "configs/model.json"
    return {
        "model_path": {"value": str(settings.onnx_path),
                       "source": "env IOT_MODEL_PATH" if "IOT_MODEL_PATH" in os.environ else "configs/model.json"},
        "threshold": {"value": detector.threshold, "source": th_src},
        "sensors": {"value": list(settings.feature_names), "source": "configs/model.json"},
        "window_size": {"value": (detector.window or {}).get("size"),
                        "source": "model meta (set by training)" if detector.window else "n/a (single-reading model)"},
    }


def format_settings(d: dict) -> str:
    return ", ".join(f"{k}={v['value']} [{v['source']}]" for k, v in d.items())
