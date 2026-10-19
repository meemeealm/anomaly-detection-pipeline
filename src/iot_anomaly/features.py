"""window-aware: per-device sliding-window features, shared by training and streaming.

Both the trainer and the consumer call the SAME DeviceWindow.push(), so features
cannot drift apart between training and serving.

For k base sensors (z-scores) and window size W the feature vector has 4*k values:
    [ z (k) | mean over last W (k) | std over last W (k) | delta vs previous reading (k) ]
A device needs W readings before it produces features (warm-up).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def feature_names(base: list[str] | tuple[str, ...]) -> list[str]:
    base = list(base)
    return (base + [f"{n}_mean" for n in base] + [f"{n}_std" for n in base]
            + [f"{n}_delta" for n in base])


class DeviceWindow:
    def __init__(self, size: int) -> None:
        if size < 2:
            raise ValueError("window size must be >= 2")
        self.size = size
        self._buf: deque[np.ndarray] = deque(maxlen=size)

    def __len__(self) -> int:
        return len(self._buf)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, z) -> np.ndarray | None:
        """Add one reading; return the feature vector, or None while warming up."""
        z = np.asarray(z, dtype=np.float64)
        prev = self._buf[-1] if self._buf else None
        self._buf.append(z)
        if len(self._buf) < self.size:
            return None
        arr = np.vstack(self._buf)
        return np.concatenate([z, arr.mean(axis=0), arr.std(axis=0), z - prev]).astype(np.float32)


@dataclass
class _DeviceState:
    window: DeviceWindow
    run_id: str | None = None
    last_seq: int | None = None
    last_ts: float | None = None


class StreamFeaturizer:
    """Per-device windows plus the stream hygiene a real feed needs.

    update() returns (features | None, status):
      ok            features ready
      warmup        window not full yet (also right after a publisher restart)
      gap_reset     device was silent > max_gap_s, window restarted
      out_of_order  seq <= last seq seen for this device (duplicate / late message); ignored
    """

    def __init__(self, size: int, max_gap_s: float = 30.0) -> None:
        self.size = size
        self.max_gap_s = max_gap_s
        self._states: dict[str, _DeviceState] = {}

    def update(self, device_id: str, z, *, run_id: str | None = None,
               seq: int | None = None, ts: float | None = None):
        st = self._states.get(device_id)
        if st is None:
            st = self._states[device_id] = _DeviceState(DeviceWindow(self.size))
        if run_id != st.run_id:                      # new publisher run: sequence restarts
            st.window.reset()
            st.last_seq = st.last_ts = None
            st.run_id = run_id
        if seq is not None and st.last_seq is not None and seq <= st.last_seq:
            return None, "out_of_order"
        gap = ts is not None and st.last_ts is not None and (ts - st.last_ts) > self.max_gap_s
        if seq is not None:
            st.last_seq = seq
        if ts is not None:
            st.last_ts = ts
        if gap:
            st.window.reset()
            st.window.push(z)
            return None, "gap_reset"
        feats = st.window.push(z)
        return (feats, "ok") if feats is not None else (None, "warmup")
