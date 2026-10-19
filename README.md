
# Anomaly Detection Pipeline 


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
python scripts/train_iforest.py   			# export isolation-forest

uv run python scripts/validate_onnx.py
uv run pytest -q
uv run uvicorn iot_anomaly.api:app --app-dir src --host 127.0.0.1 --port 8000

uv run python scripts/consume_score.py
uv run python scripts/simulate_publisher.py --devices 20 --rate 1 --anomaly-rate 0.03
uv run streamlit run scripts/dashboard.py

mosquitto_sub -h localhost -t "test/topic" -v

mosquitto_pub -h 127.0.0.1 \
  -t 'sensors/test' \
  -m '{"temperature":25,"vibration":0.5,"pressure":101.3,"humidity":60,"current":2.4,"rpm":1500}'




Next:
  python scripts/train_iforest.py --window 10         # -> artifacts/iforest_window.onnx
  python -m pytest -q

Run the stream with the windowed model:
  export IOT_MODEL_PATH=artifacts/iforest_window.onnx
  python scripts/consume_score.py
  python scripts/simulate_publisher.py --devices 20 --anomaly-rate 0.02 --stuck-rate 0.001
  streamlit run scripts/dashboard.py





Unset it (unset IOT_MODEL_PATH) to go back to LightGBM, in `inference.py`.





Assumptions and limitations
Dependency versions are bounded ranges, not exact pins. I couldn’t check them against the package index, so I didn’t want to invent pins. After your first successful install, run make lock to freeze exact versions into requirements.lock.
Check the ONNX conversion first. I wrote train_export.py from memory of the onnxmltools API, which I couldn’t install here. Run train_export.py and validate_onnx.py before relying on it. The conversion needs zipmap=False so the output is a plain probability array.
The stubs are working demos, not just empty files. They include a synthetic-data trainer, an ONNX inference wrapper, /health and /predict endpoints, and tests that skip until a model exists. Swap in real data before drawing any conclusions.
The Makefile gets real tabs. The script converts a marker into tabs, so it doesn’t break when copied.
Platform: Linux and macOS with Bash 3.2 or newer. It is not meant for Windows outside WSL.
No performance claims. The docs and benchmark.py only report what was measured. The LICENSE has a <YOUR NAME> placeholder to fill in.
Executable bit. The file I saved doesn’t have the executable bit set, so you’ll need the chmod +x.