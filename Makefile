PY := .venv/bin/python

.PHONY: setup train validate test serve benchmark loadtest lock

setup:
	./setup.sh

train:
	$(PY) scripts/train_export.py

validate:
	$(PY) scripts/validate_onnx.py

test:
	$(PY) -m pytest -q

serve:
	.venv/bin/uvicorn iot_anomaly.api:app --app-dir src --host 127.0.0.1 --port 8000

benchmark:
	$(PY) scripts/benchmark.py

loadtest:
	.venv/bin/locust -f loadtest/locustfile.py --host http://127.0.0.1:8000

lock:
	$(PY) -m pip freeze > requirements.lock

.PHONY: publish consume dashboard

publish:
	$(PY) scripts/simulate_publisher.py

consume:
	$(PY) scripts/consume_score.py

dashboard:
	.venv/bin/streamlit run scripts/dashboard.py

.PHONY: train-iforest

train-iforest:
	$(PY) scripts/train_iforest.py
