# Architecture

```
offline:  data -> LightGBM (train_export.py) -> ONNX (onnxmltools) -> artifacts/model.onnx
online:   HTTP -> FastAPI (api.py) -> ONNX Runtime CPU (inference.py) -> score + decision
```

- **Training/export** needs the dev dependencies; **serving** needs only `requirements.txt`.
- The model is loaded once at app startup; `/health` reports `model_loaded`.
- Feature order and threshold live in `configs/model.json`.


