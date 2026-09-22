"""
Controlled fault laboratory for PHI-SPIKE (campaign-grade).

Upgrades:
  - Randomised fault onset in [onset_low, onset_high]
  - Full severity grid including 1.0
  - Clean metadata for robustness matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class FaultType(str, Enum):
    HEALTHY = "healthy"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_NOISE = "sensor_noise"
    ACTUATOR_DEGRADATION = "actuator_degradation"
    ENGINE_PERFORMANCE_LOSS = "engine_performance_loss"
    CONTROL_SURFACE_LAG = "control_surface_lag"
    FUEL_FLOW_ANOMALY = "fuel_flow_anomaly"
    VIBRATION_ANOMALY = "vibration_anomaly"


FAULT_CLASSES = [f.value for f in FaultType]


@dataclass
class FaultSpec:
    fault_type: FaultType
    severity: float
    start_idx: int
    channel: Optional[str] = None
    target_channel_idx: Optional[int] = None
    onset_fraction: float = 0.3


CHANNEL_NAMES = [
    "airspeed_kt",
    "altitude_ft",
    "pitch_deg",
    "roll_deg",
    "yaw_rate_dps",
    "engine_rpm",
    "throttle",
    "fuel_flow",
    "oil_pressure",
    "oil_temp",
    "egt",
    "vibration_proxy",
]


class FaultInjector:
    # Base bias/noise scales, keyed by the *physical* channel type — looked
    # up after stripping any _1/_2 engine suffix, so scale stays correct
    # regardless of single- or twin-engine schema.
    _BASE_SCALE = {
        "airspeed_kt": 15.0, "altitude_ft": 200.0, "pitch_deg": 5.0, "roll_deg": 8.0,
        "yaw_rate_dps": 10.0, "engine_rpm": 300.0, "throttle": 0.3, "fuel_flow": 0.02,
        "oil_pressure": 25.0, "oil_temp": 20.0, "egt": 80.0, "vibration_proxy": 1.5,
    }
    _ENGINE_FAMILY_BASES = {"engine_rpm", "throttle", "fuel_flow", "oil_pressure", "oil_temp", "egt"}

    def __init__(self, seed: int = 42, channel_names: Optional[List[str]] = None):
        self.rng = np.random.default_rng(seed)
        # Backward-compatible default: unspecified channel_names means the
        # original single-engine 12-channel schema, so c172x behavior is
        # byte-identical to before this change.
        self.channel_names = list(channel_names) if channel_names is not None else list(CHANNEL_NAMES)
        self.name_to_idx = {n: i for i, n in enumerate(self.channel_names)}
        self.is_twin = "engine_rpm_1" in self.name_to_idx and "engine_rpm_2" in self.name_to_idx

    def _base_name(self, channel: str) -> str:
        for suf in ("_1", "_2"):
            if channel.endswith(suf):
                return channel[: -len(suf)]
        return channel

    def _resolve(self, base_name: str, engine_suffix: str) -> str:
        """Map a physical base channel to its concrete name for this schema.
        On twin-engine, engine-family channels get an explicit engine suffix
        (using engine_suffix if given, else a fresh random pick); airframe-
        level channels (airspeed, altitude, attitude, vibration) are shared
        and returned unchanged."""
        if self.is_twin and base_name in self._ENGINE_FAMILY_BASES:
            suffix = engine_suffix or f"_{self.rng.integers(1, 3)}"
            return f"{base_name}{suffix}"
        return base_name

    def inject(
        self,
        data: np.ndarray,
        fault_type: FaultType,
        severity: float,
        start_fraction: Optional[float] = None,
        onset_range: Tuple[float, float] = (0.15, 0.80),
        channel: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, FaultSpec]:
        T, C = data.shape
        assert C == len(self.channel_names), (
            f"data has {C} channels but this FaultInjector was built for "
            f"{len(self.channel_names)} channels ({self.channel_names}). "
            f"Pass channel_names= matching the aircraft that generated this data."
        )
        out = data.copy()
        if start_fraction is None:
            start_fraction = float(self.rng.uniform(onset_range[0], onset_range[1]))
        start_idx = int(start_fraction * T)
        labels = np.zeros(T, dtype=np.int64)

        if fault_type == FaultType.HEALTHY or severity <= 0.0:
            return out, labels, FaultSpec(FaultType.HEALTHY, 0.0, 0, onset_fraction=0.0)

        class_idx = FAULT_CLASSES.index(fault_type.value)
        labels[start_idx:] = class_idx

        # For a single fault EVENT that touches multiple engine-family
        # channels at once (a real engine degrading affects its own rpm,
        # fuel flow, and EGT together), pick ONE engine and use it
        # consistently across all channels this fault type touches — not
        # independently per channel, which would produce an unphysical
        # "half of engine 1, half of engine 2" signature.
        engine_suffix = ""
        if self.is_twin and fault_type in (
            FaultType.ACTUATOR_DEGRADATION,
            FaultType.ENGINE_PERFORMANCE_LOSS,
            FaultType.FUEL_FLOW_ANOMALY,
        ):
            engine_suffix = f"_{self.rng.integers(1, 3)}"

        if channel is None:
            channel = self._default_channel(fault_type, engine_suffix)
        ch_idx = self.name_to_idx.get(channel, 0)

        if fault_type == FaultType.SENSOR_BIAS:
            bias = severity * self._bias_scale(channel)
            out[start_idx:, ch_idx] += bias

        elif fault_type == FaultType.SENSOR_DRIFT:
            t = np.arange(T - start_idx)
            drift = severity * self._bias_scale(channel) * (t / max(len(t), 1))
            out[start_idx:, ch_idx] += drift

        elif fault_type == FaultType.SENSOR_STUCK:
            stuck_val = out[start_idx, ch_idx]
            out[start_idx:, ch_idx] = stuck_val

        elif fault_type == FaultType.SENSOR_NOISE:
            noise = self.rng.normal(
                0, severity * self._noise_scale(channel), size=T - start_idx
            )
            out[start_idx:, ch_idx] += noise

        elif fault_type == FaultType.ACTUATOR_DEGRADATION:
            rpm_ch = self._resolve("engine_rpm", engine_suffix)
            out[start_idx:, self.name_to_idx[rpm_ch]] *= 1.0 - 0.4 * severity
            out[start_idx:, self.name_to_idx["airspeed_kt"]] *= 1.0 - 0.25 * severity
            channel, ch_idx = rpm_ch, self.name_to_idx[rpm_ch]

        elif fault_type == FaultType.ENGINE_PERFORMANCE_LOSS:
            rpm_ch = self._resolve("engine_rpm", engine_suffix)
            fuel_ch = self._resolve("fuel_flow", engine_suffix)
            egt_ch = self._resolve("egt", engine_suffix)
            out[start_idx:, self.name_to_idx[rpm_ch]] *= 1.0 - 0.5 * severity
            out[start_idx:, self.name_to_idx[fuel_ch]] *= 1.0 + 0.3 * severity
            out[start_idx:, self.name_to_idx[egt_ch]] += 80.0 * severity
            out[start_idx:, self.name_to_idx["vibration_proxy"]] += 0.8 * severity
            channel, ch_idx = rpm_ch, self.name_to_idx[rpm_ch]

        elif fault_type == FaultType.CONTROL_SURFACE_LAG:
            alpha = 0.3 + 0.5 * severity
            for ch_name in ("pitch_deg", "roll_deg"):
                ci = self.name_to_idx[ch_name]
                for t in range(start_idx + 1, T):
                    out[t, ci] = alpha * out[t - 1, ci] + (1 - alpha) * out[t, ci]

        elif fault_type == FaultType.FUEL_FLOW_ANOMALY:
            fuel_ch = self._resolve("fuel_flow", engine_suffix)
            fi = self.name_to_idx[fuel_ch]
            out[start_idx:, fi] *= 1.0 + 0.6 * severity * np.sin(
                np.linspace(0, 6 * np.pi, T - start_idx)
            )

        elif fault_type == FaultType.VIBRATION_ANOMALY:
            vi = self.name_to_idx["vibration_proxy"]
            out[start_idx:, vi] += severity * (
                1.5 + self.rng.normal(0, 0.3, size=T - start_idx)
            )

        spec = FaultSpec(
            fault_type=fault_type,
            severity=severity,
            start_idx=start_idx,
            channel=channel,
            target_channel_idx=ch_idx,
            onset_fraction=start_fraction,
        )
        return out, labels, spec

    def _default_channel(self, fault_type: FaultType, engine_suffix: str = "") -> str:
        mapping = {
            FaultType.SENSOR_BIAS: "oil_pressure",
            FaultType.SENSOR_DRIFT: "oil_temp",
            FaultType.SENSOR_STUCK: "egt",
            FaultType.SENSOR_NOISE: "vibration_proxy",
            FaultType.FUEL_FLOW_ANOMALY: "fuel_flow",
            FaultType.VIBRATION_ANOMALY: "vibration_proxy",
        }
        base = mapping.get(fault_type, "oil_pressure")
        return self._resolve(base, engine_suffix)

    def _bias_scale(self, channel: str) -> float:
        return self._BASE_SCALE[self._base_name(channel)]

    def _noise_scale(self, channel: str) -> float:
        return self._bias_scale(channel) * 0.15

    def generate_dataset(
        self,
        clean_trajectories: List[np.ndarray],
        n_per_class: int = 3,
        severities: Optional[List[float]] = None,
        onset_range: Tuple[float, float] = (0.15, 0.80),
        start_fraction: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[FaultSpec]]:
        if severities is None:
            severities = [0.10, 0.25, 0.50, 0.75, 1.00]
        X, Y, specs = [], [], []
        for traj in clean_trajectories:
            X.append(traj)
            Y.append(np.zeros(len(traj), dtype=np.int64))
            specs.append(FaultSpec(FaultType.HEALTHY, 0.0, 0, onset_fraction=0.0))

            for ft in FaultType:
                if ft == FaultType.HEALTHY:
                    continue
                for sev in severities:
                    for _ in range(n_per_class):
                        corrupted, labels, spec = self.inject(
                            traj,
                            ft,
                            sev,
                            start_fraction=start_fraction,
                            onset_range=onset_range,
                        )
                        X.append(corrupted)
                        Y.append(labels)
                        specs.append(spec)
        return X, Y, specs


def smoke_test() -> None:
    from simulation.aircraft_sim import AircraftSimulator, FlightConfig

    sim = AircraftSimulator(FlightConfig(duration_s=20.0))
    data, _, _ = sim.generate_trajectory(duration_s=20.0, seed=7)
    inj = FaultInjector(seed=7)
    corrupted, labels, spec = inj.inject(
        data, FaultType.ENGINE_PERFORMANCE_LOSS, 0.5, onset_range=(0.2, 0.6)
    )
    print(f"Fault: {spec.fault_type.value}, severity={spec.severity}, onset={spec.onset_fraction:.2f}")
    print(f"Label unique: {np.unique(labels)}")


if __name__ == "__main__":
    smoke_test()
