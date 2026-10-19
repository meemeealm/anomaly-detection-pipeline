# Metrics: Prometheus + Grafana

```
consumer :8001/metrics  <--scrape 5s--  Prometheus :9090  <--query--  Grafana :3000
```
No Docker needed. Run Prometheus and Grafana natively (download from prometheus.io and grafana.com, or your
package manager, e.g. `brew install prometheus grafana`). Check each project's docs for current install steps.

## 1. Start the consumer (metrics on by default)
```bash
python scripts/consume_score.py          # logs "metrics endpoint up url=http://127.0.0.1:8001/metrics"
curl -s localhost:8001/metrics | grep -E "^iot_" | head
```
`metrics_port` / `metrics_host` can go in `configs/stream.json` (optional keys), env `IOT_METRICS_PORT`, or flags.
`0` turns metrics off. If the port is busy the consumer keeps scoring and logs an error. Without `prometheus-client`
installed it logs a warning and runs without metrics.

## 2. Prometheus
```bash
prometheus --config.file=observability/prometheus.yml       # UI: http://localhost:9090
```
Check Status > Targets: `iot-consumer` must be UP. Alert rules (`observability/alerts.yml`) show under Alerts; there is
no Alertmanager, so nothing is sent anywhere. If you change `metrics_port`, change the target in `prometheus.yml` too.

## 3. Grafana
Provisioning files are in `observability/grafana/`. `add_metrics.sh` generated the machine-specific
`provisioning/dashboards/dashboards.yml` (absolute dashboard path, git-ignored) from the `.template`.
Point Grafana at the provisioning folder, then open http://localhost:3000 (default login admin/admin):
```bash
export GF_PATHS_PROVISIONING="$(pwd)/observability/grafana/provisioning"    # Windows: set it as an environment variable
grafana server       # or grafana-server, depending on how you installed it
```
Dashboard: IoT > "IoT anomaly pipeline". If provisioning is awkward, add a Prometheus data source
(URL `http://localhost:9090`, uid `prometheus`) and import `observability/grafana/dashboards/iot-anomaly.json`
through Dashboards > New > Import.

## Series
| series | meaning |
|---|---|
| `iot_messages_total{outcome}` | processed, dropped, malformed, non_finite, out_of_order, gap_reset, warmup, scoring_error, db_error |
| `iot_alerts_total{reason}` | flagged readings: `model`, `flatline`, `model+flatline` |
| `iot_flatline_flags_total` | readings flagged by the flatline rule |
| `iot_e2e_latency_seconds` (histogram) | device timestamp to stored |
| `iot_inference_seconds`, `iot_batch_size` (histograms) | model time and rows per micro-batch |
| `iot_queue_depth` | messages waiting in the consumer queue |
| `iot_last_scored_timestamp_seconds` | for "no data" detection |
| `iot_run_info{run_id,model,model_type,window}` | constant 1; ties a graph to a run in the `runs` table |

No `device_id` label on purpose (cardinality). Per-device questions belong in SQLite / the Streamlit dashboard.
End-to-end latency compares the device clock with the consumer clock, so it is only meaningful when they agree
(true locally; with real devices, watch for clock skew).
Metrics are copied from the consumer's counters about every 0.5 s, so they lag the SQLite tables slightly.
