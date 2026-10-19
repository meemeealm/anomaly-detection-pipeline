"""flatline-aware: rule-based detectors that complement the model.

An Isolation Forest barely notices a stuck sensor (1 of 24 window features), but the
signal is trivial to state as a rule: the rolling std of that sensor collapses to ~0.
Layout of a window feature vector (see features.py): [z | mean | std | delta], k each.
Caveat: a sensor that is legitimately constant (e.g. rpm while idle) will trip this rule.
"""
from __future__ import annotations

import numpy as np


def flatline_sensors(features, base_names, eps: float) -> list[str]:
    """Names of sensors whose rolling std (z-units) is below eps. eps <= 0 disables the rule."""
    if eps <= 0:
        return []
    k = len(base_names)
    feats = np.asarray(features)
    if feats.shape[-1] < 3 * k:
        raise ValueError(f"expected window features (>= {3 * k} values), got {feats.shape[-1]}")
    std = feats[2 * k:3 * k]
    return [base_names[i] for i in np.flatnonzero(std < eps)]


def alert_reason(model_flag: bool, flat_sensors: list[str]) -> str | None:
    """NULL | model | flatline:<sensors> | model+flatline:<sensors>"""
    parts = []
    if model_flag:
        parts.append("model")
    if flat_sensors:
        parts.append("flatline:" + ",".join(flat_sensors))
    return "+".join(parts) if parts else None
