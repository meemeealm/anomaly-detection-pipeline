import argparse
import json
import os

from iot_anomaly import runtime_config as rc

NAMES = ("host", "port", "batch_size", "flat_eps", "db")


def _ns(**kw):
    base = {n: None for n in rc.DEFAULTS}
    base["config"] = None
    base.update(kw)
    return argparse.Namespace(**base)


class _env:
    """Set env vars for the duration of a block (no pytest fixtures needed)."""
    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _cfg(tmp_path, **data):
    p = tmp_path / "stream.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_precedence_cli_env_file_default(tmp_path):
    path = _cfg(tmp_path, batch_size=64, port=1999)
    with _env(IOT_PORT="1884", IOT_BATCH_SIZE="128"):
        args = _ns(port=1111, config=path)            # cli wins over env and file
        out = rc.resolve_all(args, NAMES)
        assert (out["port"]["value"], out["port"]["source"]) == (1111, "command line")
        assert (out["batch_size"]["value"], out["batch_size"]["source"]) == (128, "env IOT_BATCH_SIZE")
    args = _ns(config=path)                            # file wins over default
    out = rc.resolve_all(args, NAMES)
    assert out["batch_size"] == {"value": 64, "source": "configs/stream.json"}
    assert out["host"] == {"value": "127.0.0.1", "source": "default"}


def test_values_are_written_back_onto_args(tmp_path):
    args = _ns(config=_cfg(tmp_path, flat_eps=0.1))
    rc.resolve_all(args, NAMES)
    assert args.flat_eps == 0.1 and args.port == 1883


def test_env_values_are_cast_to_the_right_type(tmp_path):
    with _env(IOT_PORT="1884", IOT_FLAT_EPS="0.5"):
        out = rc.resolve_all(_ns(config=_cfg(tmp_path)), NAMES)
    assert out["port"]["value"] == 1884 and isinstance(out["port"]["value"], int)
    assert out["flat_eps"]["value"] == 0.5


def test_unknown_key_is_rejected(tmp_path):
    try:
        rc.resolve_all(_ns(config=_cfg(tmp_path, flat_epsilon=0.1)), NAMES)
    except ValueError as exc:
        assert "flat_epsilon" in str(exc)
    else:
        raise AssertionError("typo in config was accepted")


def test_out_of_range_and_bad_type_are_rejected(tmp_path):
    for bad in ({"qos": 2}, {"batch_size": 0}, {"port": "abc"}):
        try:
            rc.resolve_all(_ns(config=_cfg(tmp_path, **bad)), tuple(rc.DEFAULTS))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_missing_explicit_config_is_an_error(tmp_path):
    try:
        rc.resolve_all(_ns(config=str(tmp_path / "nope.json")), NAMES)
    except ValueError:
        return
    raise AssertionError("missing config file was ignored")


def test_relative_db_is_placed_inside_the_repo(tmp_path):
    out = rc.resolve_all(_ns(config=_cfg(tmp_path, db="results/x.db")), NAMES)
    assert out["db"]["value"] == str(rc.ROOT / "results" / "x.db")


def test_format_settings_shows_sources(tmp_path):
    out = rc.resolve_all(_ns(config=_cfg(tmp_path), port=1234), ("host", "port"))
    text = rc.format_settings(out)
    assert "port=1234 [command line]" in text and "host=127.0.0.1 [default]" in text
