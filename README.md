# Anomaly Detection Pipeline 

This repo is demonstration of a real-time IoT monitoring system that simulates telemetry data using Faker and NumPy, streams it via MQTT, and processes it through a sliding-window consumer. It applies both rule-based checks (flatline detection) and machine learning models (ONNX Isolation Forest) to detect anomalies, storing results and exposing metrics for live monitoring via Prometheus and Grafana.

![Anomaly Detection Pipeline Diagram](https://github.com/meemeealm/anomaly-detection-pipeline/raw/main/docs/imgs/mermaid_diagram.png)

## Setup 

from root -

### 1. Install uv 
```bash
uv --version
uv python install 3.11
```
### 2. Create the virtual environment
```bash
uv venv --python 3.11 .venv
```

### 3. Run setup using the environment's Python
```bash
PYTHON_BIN="$PWD/.venv/bin/python" ./setup.sh
```

### 4. Run dependencies
```bash
uv run pip install -r requirements.txt
uv run pip install -r requirements-dev.txt
```

### 5. Run scripts 

```bash
uv run python scripts/train_export.py 		# export lgbm  
python scripts/train_iforest.py   			  # export isolation-forest
uv run python scripts/validate_onnx.py
uv run pytest -q
```

open fastapi:  
```bash
uv run uvicorn iot_anomaly.api:app --app-dir src --host 127.0.0.1 --port 8000
```

in each terminals:  
```bash
uv run python scripts/consume_score.py
uv run python scripts/simulate_publisher.py --devices 20 --rate 1 --anomaly-rate 0.03
uv run streamlit run scripts/dashboard.py
```

to test and run MQQT:  
```bash
mosquitto_sub -h localhost -t "test/topic" -v
```

```bash
mosquitto_pub -h 127.0.0.1 \
  -t 'sensors/test' \
  -m '{"temperature":25,"vibration":0.5,"pressure":101.3,"humidity":60,"current":2.4,"rpm":1500}'
```

run the stream with the windowed model in streamlit:  
```bash
  export IOT_MODEL_PATH=artifacts/iforest_window.onnx
  python scripts/consume_score.py
  python scripts/simulate_publisher.py --devices 20 --anomaly-rate 0.02 --stuck-rate 0.001
  streamlit run scripts/dashboard.py
```

run in Grafana:  
```bash
& "C:\Program Files\GrafanaLabs\grafana\bin\grafana.exe" server
```
To unset it (unset IOT_MODEL_PATH) to go back to LightGBM, in `inference.py`.




