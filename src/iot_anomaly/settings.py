"""Runtime settings, loaded from configs/model.json with env-var overrides."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_name: str
    input_name: str
    feature_names: tuple[str, ...]
    threshold: float
    onnx_path: Path
    lightgbm_path: Path
    onnx_intra_op_threads: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_path = Path(os.environ.get("IOT_CONFIG", ROOT / "configs" / "model.json"))
    cfg = json.loads(config_path.read_text())
    onnx_path = Path(os.environ.get("IOT_MODEL_PATH", cfg["onnx_path"]))
    if not onnx_path.is_absolute():
        onnx_path = ROOT / onnx_path
    lgbm_path = Path(cfg["lightgbm_path"])
    if not lgbm_path.is_absolute():
        lgbm_path = ROOT / lgbm_path
    return Settings(
        model_name=cfg["model_name"],
        input_name=cfg["input_name"],
        feature_names=tuple(cfg["feature_names"]),
        threshold=float(os.environ.get("IOT_THRESHOLD", cfg["threshold"])),
        onnx_path=onnx_path,
        lightgbm_path=lgbm_path,
        onnx_intra_op_threads=int(cfg.get("onnx_intra_op_threads", 1)),
    )
