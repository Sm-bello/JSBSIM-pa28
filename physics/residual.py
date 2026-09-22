"""
Physics residual engine for PHI-SPIKE (campaign-grade).

Improvements over prototype:
  - Channel-wise robust normalisation (different physical units)
  - Explicit residual scaling for membrane conditioning
  - Temporal residual consistency helpers
  - Cleaner interface for learnable gating downstream
  - Physical bound and jump-consistency diagnostics retained

Residual definition:
    r_t = normalize(x_{t+1} - A x_t)
where A is a domain-informed discrete linear transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PhysicsConfig:
    airspeed_min_kt: float = 40.0
    airspeed_max_kt: float = 180.0
    rpm_min: float = 500.0
    rpm_max: float = 2800.0
    oil_pressure_min: float = 20.0
    oil_pressure_max: float = 100.0
    egt_min: float = 400.0
    egt_max: float = 900.0
    max_pitch_rate_dps: float = 30.0
    max_roll_rate_dps: float = 60.0
    channel_scales: List[float] = field(
        default_factory=lambda: [
            20.0, 500.0, 8.0, 12.0, 15.0, 400.0, 0.25, 0.03, 15.0, 15.0, 50.0, 0.8,
        ]
    )


class PhysicsResidualEngine:
    """
    Multi-channel residual between measured telemetry and a first-order
    expected-dynamics model, with channel-wise normalisation.
    """

    CHANNEL_IDX = {
        "airspeed_kt": 0,
        "altitude_ft": 1,
        "pitch_deg": 2,
        "roll_deg": 3,
        "yaw_rate_dps": 4,
        "engine_rpm": 5,
        "throttle": 6,
        "fuel_flow": 7,
        "oil_pressure": 8,
        "oil_temp": 9,
        "egt": 10,
        "vibration_proxy": 11,
    }

    # Twin-engine variant: airframe-level channels keep the same indices,
    # every engine-specific channel gets a mirrored pair. New, additive —
    # does not alter CHANNEL_IDX above, so c172x's physics is untouched.
    CHANNEL_IDX_TWIN = {
        "airspeed_kt": 0,
        "altitude_ft": 1,
        "pitch_deg": 2,
        "roll_deg": 3,
        "yaw_rate_dps": 4,
        "engine_rpm_1": 5,
        "engine_rpm_2": 6,
        "throttle_1": 7,
        "throttle_2": 8,
        "fuel_flow_1": 9,
        "fuel_flow_2": 10,
        "oil_pressure_1": 11,
        "oil_pressure_2": 12,
        "oil_temp_1": 13,
        "oil_temp_2": 14,
        "egt_1": 15,
        "egt_2": 16,
        "vibration_proxy": 17,
    }

    def __init__(
        self,
        dt: float = 0.02,
        cfg: Optional[PhysicsConfig] = None,
        normalize: bool = True,
        twin_engine: bool = False,
    ):
        self.dt = dt
        self.cfg = cfg or PhysicsConfig()
        self.normalize = normalize
        self.twin_engine = twin_engine

        if twin_engine:
            self.CHANNEL_IDX = self.CHANNEL_IDX_TWIN  # instance override, class default untouched
            self.n = len(self.CHANNEL_IDX_TWIN)
            self.A = np.eye(self.n)
            self.A[0, 0] = 0.995
            self.A[1, 1] = 1.0
            self.A[2, 2] = 0.98
            self.A[3, 3] = 0.97
            self.A[4, 4] = 0.90
            # Each engine's coupling mirrors the single-engine model exactly
            # (rpm <- throttle, fuel/oil-pressure/oil-temp/egt <- rpm),
            # applied independently per engine so a single-engine fault
            # (e.g. one engine's actuator degrading) produces a residual
            # localized to that engine's channels, not a shared/blended one.
            self.A[5, 5] = 0.99
            self.A[5, 7] = 50.0 * dt   # rpm1 <- throttle1
            self.A[6, 6] = 0.99
            self.A[6, 8] = 50.0 * dt   # rpm2 <- throttle2
            self.A[9, 7] = 0.03 * dt   # fuel_flow1 <- throttle1
            self.A[10, 8] = 0.03 * dt  # fuel_flow2 <- throttle2
            self.A[11, 5] = 0.01 * dt  # oil_pressure1 <- rpm1
            self.A[12, 6] = 0.01 * dt  # oil_pressure2 <- rpm2
            self.A[13, 5] = 0.005 * dt  # oil_temp1 <- rpm1
            self.A[14, 6] = 0.005 * dt  # oil_temp2 <- rpm2
            self.A[15, 5] = 0.02 * dt  # egt1 <- rpm1
            self.A[16, 6] = 0.02 * dt  # egt2 <- rpm2
            self.A[17, 0] = 0.001 * dt  # vibration <- airspeed (shared airframe channel)
            self.B = np.zeros((self.n, 1))
            twin_scales = self.cfg.channel_scales
            if len(twin_scales) != self.n:
                # Auto-mirror the single-engine 12-value default into the
                # 18-value twin layout unless the caller explicitly passed
                # a PhysicsConfig already sized for twin_engine=True.
                s = self.cfg.channel_scales
                twin_scales = [
                    s[0], s[1], s[2], s[3], s[4],   # airspeed, altitude, pitch, roll, yaw
                    s[5], s[5],                      # rpm1, rpm2
                    s[6], s[6],                      # throttle1, throttle2
                    s[7], s[7],                      # fuel_flow1, fuel_flow2
                    s[8], s[8],                      # oil_pressure1, oil_pressure2
                    s[9], s[9],                      # oil_temp1, oil_temp2
                    s[10], s[10],                     # egt1, egt2
                    s[11],                            # vibration_proxy
                ]
            self._scale = np.array(twin_scales, dtype=np.float64)
            self._fitted = False
            return

        self.n = len(self.CHANNEL_IDX)
        self.A = np.eye(self.n)
        self.A[0, 0] = 0.995
        self.A[1, 1] = 1.0
        self.A[2, 2] = 0.98
        self.A[3, 3] = 0.97
        self.A[4, 4] = 0.90
        self.A[5, 5] = 0.99
        self.A[5, 6] = 50.0 * dt
        self.A[7, 6] = 0.03 * dt
        self.A[8, 5] = 0.01 * dt
        self.A[9, 5] = 0.005 * dt
        self.A[10, 5] = 0.02 * dt
        self.A[11, 0] = 0.001 * dt
        self.B = np.zeros((self.n, 1))
        self._scale = np.array(self.cfg.channel_scales, dtype=np.float64)
        self._fitted = False

    def fit_scales(self, residuals) -> None:
        """
        Fit per-channel normalisation scale from the *training* residuals.

        BUGFIX (was causing NaN loss on small/smoke datasets):
        The old floor `max(median_abs_residual, 1e-6)` lets any channel that
        happens to look near-constant in a small training split (e.g. a
        stuck sensor, or simply too few trajectories) collapse to a scale of
        1e-6. Any ordinary residual value on val/test data for that channel
        (~0.01-0.1 in physical units) then gets divided by 1e-6, producing
        normalised residuals in the tens-of-thousands, whose *square* (fed
        straight into physics_informed_loss's MSE term) overflows/propagates
        as NaN within the first few batches.

        Fix: never let a channel's fitted scale drop below a fixed fraction
        of its known physical `channel_scales` (from PhysicsConfig). This
        keeps normalisation sane even when a channel is quiet/degenerate in
        a tiny training split, while still allowing genuine per-channel
        adaptation when there's enough data to estimate it reliably.
        """
        if isinstance(residuals, list):
            if len(residuals) == 0:
                return
            stacked = np.concatenate(residuals, axis=0)
        else:
            stacked = residuals.reshape(-1, residuals.shape[-1])
        med = np.median(np.abs(stacked), axis=0)
        static_floor = 0.05 * np.asarray(self.cfg.channel_scales, dtype=np.float64)
        med = np.maximum(med, static_floor)
        med = np.maximum(med, 1e-3)  # absolute safety floor, was 1e-6
        self._scale = med
        self._fitted = True

    def predict_next(self, x: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.A).all() or not np.isfinite(x).all():
            # Handle or mask NaNs to prevent silent blocking or numerical hangs
            self.A = np.nan_to_num(self.A, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.A @ x

    def residual_sequence(self, data: np.ndarray, apply_norm: Optional[bool] = None) -> np.ndarray:
        apply_norm = self.normalize if apply_norm is None else apply_norm
        T, C = data.shape
        assert C == self.n, f"Expected {self.n} channels, got {C}"
        r = np.zeros((T - 1, C), dtype=np.float64)
        for t in range(T - 1):
            x_pred = self.predict_next(data[t])
            r[t] = data[t + 1] - x_pred
        if apply_norm:
            r = r / (self._scale + 1e-8)
        return r

    def residual_norm(self, data: np.ndarray) -> np.ndarray:
        r = self.residual_sequence(data, apply_norm=True)
        return np.linalg.norm(r, axis=1)

    def physical_bound_violations(self, data: np.ndarray) -> Dict[str, float]:
        cfg = self.cfg
        n = len(data)
        if n == 0:
            return {}
        viol = {
            "airspeed": float(
                np.mean((data[:, 0] < cfg.airspeed_min_kt) | (data[:, 0] > cfg.airspeed_max_kt))
            ),
            "rpm": float(np.mean((data[:, 5] < cfg.rpm_min) | (data[:, 5] > cfg.rpm_max))),
            "oil_pressure": float(
                np.mean(
                    (data[:, 8] < cfg.oil_pressure_min) | (data[:, 8] > cfg.oil_pressure_max)
                )
            ),
            "egt": float(np.mean((data[:, 10] < cfg.egt_min) | (data[:, 10] > cfg.egt_max))),
        }
        viol["overall"] = float(np.mean(list(viol.values())))
        return viol

    def temporal_consistency_score(self, data: np.ndarray) -> float:
        if len(data) < 2:
            return 0.0
        d = np.diff(data, axis=0)
        limits = np.array(
            [5.0, 50.0, 5.0, 8.0, 20.0, 100.0, 0.2, 0.05, 5.0, 3.0, 20.0, 0.5]
        )
        jumps = np.abs(d) > limits
        return float(np.mean(jumps))

    def temporal_residual_consistency(self, residual: np.ndarray, window: int = 5) -> np.ndarray:
        norms = np.linalg.norm(residual, axis=1)
        if len(norms) < window:
            return norms
        kernel = np.ones(window) / window
        return np.convolve(norms, kernel, mode="same")


def smoke_test() -> None:
    from simulation.aircraft_sim import AircraftSimulator, FlightConfig

    sim = AircraftSimulator(FlightConfig(duration_s=10.0))
    data, _, _ = sim.generate_trajectory(duration_s=10.0, seed=1)
    eng = PhysicsResidualEngine(dt=0.02, normalize=True)
    r = eng.residual_sequence(data)
    eng.fit_scales([r])
    r2 = eng.residual_sequence(data)
    r_norm = eng.residual_norm(data)
    viol = eng.physical_bound_violations(data)
    print(f"Residual shape: {r2.shape}")
    print(f"Residual norm mean (normed): {r_norm.mean():.4f}")
    print(f"Bound violations: {viol}")
    print(f"Temporal consistency violations: {eng.temporal_consistency_score(data):.4f}")
    print(f"Scales fitted: {eng._fitted}, scale sample: {eng._scale[:3]}")


if __name__ == "__main__":
    smoke_test()