"""window-aware: fake IoT fleet + sensor signal generator.

NumPy generates the signals; Faker only generates device metadata.
Readings are emitted in physical units. The consumer standardizes them with
to_z(), because models are trained on standardized (z-score) features.

Fault types: point anomalies (spikes on 1-3 sensors) and stuck sensors
(one sensor freezes at its current value for N readings). Both get label=1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# name -> (mean, std) in physical units
PROFILES: dict[str, tuple[float, float]] = {
    "temperature": (60.0, 8.0),     # deg C
    "vibration": (2.5, 0.6),        # mm/s
    "pressure": (101.0, 4.0),       # kPa
    "humidity": (45.0, 9.0),        # %
    "current": (12.0, 2.0),         # A
    "rpm": (1800.0, 150.0),
}


def to_z(name: str, value: float) -> float:
    mean, std = PROFILES[name]
    return (value - mean) / std


def to_raw(name: str, z: float) -> float:
    mean, std = PROFILES[name]
    return mean + z * std


@dataclass
class Device:
    device_id: str
    site: str
    firmware: str
    serial: str
    offset: np.ndarray   # fixed per-device bias (in z units)
    state: np.ndarray    # AR(1) noise state (in z units)
    stuck_left: int = 0  # readings remaining in a stuck-sensor fault
    stuck_idx: int = -1
    stuck_z: float = 0.0

    def meta(self) -> dict:
        return {
            "device_id": self.device_id,
            "site": self.site,
            "firmware": self.firmware,
            "serial": self.serial,
        }


class Fleet:
    PHI = 0.5            # autocorrelation of the noise
    NOISE_STD = 0.9      # stationary noise std (z units)

    def __init__(
        self,
        names: tuple[str, ...] | list[str],
        n_devices: int = 20,
        seed: int = 42,
        anomaly_rate: float = 0.02,
        drift_per_min: float = 0.0,
        drift_feature: str | None = None,
        stuck_rate: float = 0.0,
        stuck_len: tuple[int, int] = (20, 60),
    ) -> None:
        from faker import Faker  # imported lazily: only needed for simulation

        unknown = [n for n in names if n not in PROFILES]
        if unknown:
            raise ValueError(f"No simulation profile for features: {unknown}")
        if drift_feature is not None and drift_feature not in names:
            raise ValueError(f"drift_feature {drift_feature!r} not in {list(names)}")

        Faker.seed(seed)
        fake = Faker()
        self.names = tuple(names)
        self.rng = np.random.default_rng(seed)
        self.anomaly_rate = anomaly_rate
        self.drift_per_min = drift_per_min
        self.drift_feature = drift_feature
        self.stuck_rate = stuck_rate
        self.stuck_len = stuck_len
        n = len(self.names)
        self.devices = [
            Device(
                device_id=f"dev-{i:03d}",
                site=fake.city(),
                firmware=fake.numerify("v#.#.#"),
                serial=fake.bothify("SN-????-#####"),
                offset=self.rng.normal(0.0, 0.3, n),
                state=np.zeros(n),
            )
            for i in range(n_devices)
        ]

    def sample(self, device: Device, elapsed_min: float = 0.0) -> tuple[dict[str, float], int]:
        """One reading for one device. Returns (readings in physical units, label)."""
        n = len(self.names)
        eps = self.rng.normal(0.0, 1.0, n)
        device.state = self.PHI * device.state + np.sqrt(1 - self.PHI**2) * self.NOISE_STD * eps
        z = device.offset + device.state
        if self.drift_feature is not None:
            z[self.names.index(self.drift_feature)] += self.drift_per_min * elapsed_min
        label = 0

        # stuck-sensor fault: one sensor flatlines (invisible to a single-reading model)
        if device.stuck_left > 0:
            z[device.stuck_idx] = device.stuck_z
            device.stuck_left -= 1
            label = 1
        elif self.stuck_rate > 0 and self.rng.random() < self.stuck_rate:
            device.stuck_idx = int(self.rng.integers(0, n))
            device.stuck_z = float(z[device.stuck_idx])
            device.stuck_left = int(self.rng.integers(self.stuck_len[0], self.stuck_len[1] + 1))
            label = 1

        # point anomaly: spike on 1-3 sensors
        if self.rng.random() < self.anomaly_rate:
            k = int(self.rng.integers(1, min(3, n) + 1))
            idx = self.rng.choice(n, size=k, replace=False)
            z[idx] += self.rng.choice([-1.0, 1.0], size=k) * self.rng.uniform(3.5, 5.5, size=k)
            label = 1
        readings = {name: round(float(to_raw(name, z[i])), 3) for i, name in enumerate(self.names)}
        return readings, label
