# Benchmarking

Benchmarks are run manually and separately from setup, training and tests.

1. `python scripts/train_export.py`
2. `python scripts/validate_onnx.py` (correctness first)
3. `python scripts/benchmark.py` (in-process latency + RSS -> `results/`)
4. Start the API, then run Locust (see `loadtest/locustfile.py`) for end-to-end numbers.

Record hardware, OS, Python/ORT versions and thread settings next to every result.
Report only what was measured; do not state targets as achieved without data.
