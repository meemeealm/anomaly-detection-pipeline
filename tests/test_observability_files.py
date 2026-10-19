import json
import re
from pathlib import Path

import pytest

from iot_anomaly.metrics import EXPOSED_SERIES
from iot_anomaly.runtime_config import DEFAULTS

OBS = Path(__file__).resolve().parents[1] / "observability"


def _promql_series(text: str) -> set[str]:
    return set(re.findall(r"\biot_[a-z0-9_]+", text))


def test_dashboard_json_is_valid_with_unique_panel_ids():
    d = json.loads((OBS / "grafana" / "dashboards" / "iot-anomaly.json").read_text())
    ids = [p["id"] for p in d["panels"]]
    assert len(ids) == len(set(ids)) and d["uid"] == "iot-anomaly"
    assert all(t["datasource"]["uid"] == "prometheus" for p in d["panels"] for t in p["targets"])


def test_dashboard_only_queries_series_the_consumer_exposes():
    d = json.loads((OBS / "grafana" / "dashboards" / "iot-anomaly.json").read_text())
    used = set()
    for p in d["panels"]:
        for t in p["targets"]:
            used |= _promql_series(t["expr"])
    assert used and used <= EXPOSED_SERIES, sorted(used - EXPOSED_SERIES)


def test_alert_rules_only_use_exposed_series():
    used = _promql_series((OBS / "alerts.yml").read_text())
    assert used and used <= EXPOSED_SERIES, sorted(used - EXPOSED_SERIES)


def test_prometheus_scrape_port_matches_the_default_metrics_port():
    ports = re.findall(r"127\.0\.0\.1:(\d+)", (OBS / "prometheus.yml").read_text())
    assert ports == [str(DEFAULTS["metrics_port"])]


def test_yaml_files_parse():
    yaml = pytest.importorskip("yaml")
    for path in (OBS / "prometheus.yml", OBS / "alerts.yml",
                 OBS / "grafana" / "provisioning" / "datasources" / "prometheus.yml"):
        assert yaml.safe_load(path.read_text())
    alerts = yaml.safe_load((OBS / "alerts.yml").read_text())
    assert all("expr" in r for g in alerts["groups"] for r in g["rules"])
