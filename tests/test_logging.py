import json
import logging

from iot_anomaly.logging_setup import new_run_id, setup_logging


def test_run_id_format_is_unique_and_filename_safe():
    a, b = new_run_id(), new_run_id()
    assert a != b
    assert ":" not in a and "/" not in a


def test_json_log_lines_carry_run_id_and_extras(tmp_path):
    log = setup_logging("unit", "RUN123", log_dir=tmp_path)
    log.info("hello", extra={"device": "dev-001"})
    for h in log.handlers:
        h.flush()
    lines = (tmp_path / "unit_RUN123.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    assert rec["run_id"] == "RUN123" and rec["component"] == "unit"
    assert rec["msg"] == "hello" and rec["device"] == "dev-001" and rec["level"] == "INFO"
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)
    logging.shutdown()
