# Anomaly Detection Pipeline 

This is demonstration of IoT sensor data anomaly detection pipeline.


## Setup with uv

from root -

### 1. Install uv 
`bash
uv --version
`

### 2. Install Python 3.11 managed by uv
`bash
uv python install 3.11
`

### 3. Create the virtual environment
`bash
uv venv --python 3.11 .venv
`

### 4. Run setup using the environment's Python
`bash
PYTHON_BIN="$PWD/.venv/bin/python" ./setup.sh
`

### 5. Run dependencies
`bash
uv run pip install -r requirements.txt
uv run pip install -r requirements-dev.txt
`

## Run scripts step-by-step

uv run python scripts/train_export.py 		# export lgbm  
python scripts/train_iforest.py   			  # export isolation-forest
uv run python scripts/validate_onnx.py
uv run pytest -q

open fastapi:  
`uv run uvicorn iot_anomaly.api:app --app-dir src --host 127.0.0.1 --port 8000`

in each terminals:  
`bash
uv run python scripts/consume_score.py
uv run python scripts/simulate_publisher.py --devices 20 --rate 1 --anomaly-rate 0.03
uv run streamlit run scripts/dashboard.py
`

to test and run MQQT:  
`mosquitto_sub -h localhost -t "test/topic" -v`

`bash
mosquitto_pub -h 127.0.0.1 \
  -t 'sensors/test' \
  -m '{"temperature":25,"vibration":0.5,"pressure":101.3,"humidity":60,"current":2.4,"rpm":1500}'`


Run the stream with the windowed model in streamlit:  
` bash
  export IOT_MODEL_PATH=artifacts/iforest_window.onnx
  python scripts/consume_score.py
  python scripts/simulate_publisher.py --devices 20 --anomaly-rate 0.02 --stuck-rate 0.001
  streamlit run scripts/dashboard.py
`

Run in Grafana:  
`& "C:\Program Files\GrafanaLabs\grafana\bin\grafana.exe" server`


To unset it (unset IOT_MODEL_PATH) to go back to LightGBM, in `inference.py`.




