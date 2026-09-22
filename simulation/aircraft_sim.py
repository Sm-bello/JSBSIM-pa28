"""
JSBSim-based aircraft telemetry generator for PHI-SPIKE.
Produces continuous flight telemetry suitable for PHM research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jsbsim
import numpy as np


TELEMETRY_CHANNELS = [
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

# Twin-engine channel schema: airframe-level channels stay shared; every
# engine-specific channel gets an explicit _1/_2 suffix and is read from
# JSBSim's indexed propulsion/engine[N]/... properties, not the unindexed
# path (which only ever resolves engine 0 on a multi-engine airframe —
# the gap flagged in CHANGES_twin_piston.md / CHANGES_reconciliation_pass.md).
TELEMETRY_CHANNELS_TWIN = [
    "airspeed_kt",
    "altitude_ft",
    "pitch_deg",
    "roll_deg",
    "yaw_rate_dps",
    "engine_rpm_1",
    "engine_rpm_2",
    "throttle_1",
    "throttle_2",
    "fuel_flow_1",
    "fuel_flow_2",
    "oil_pressure_1",
    "oil_pressure_2",
    "oil_temp_1",
    "oil_temp_2",
    "egt_1",
    "egt_2",
    "vibration_proxy",
]

# Aircraft known to have 2 engines in JSBSim's bundled models. Explicit
# allowlist rather than runtime engine-count detection — fail loud on an
# unlisted twin rather than silently treating it as single-engine.
TWIN_ENGINE_AIRCRAFT = {"c310", "Baron58", "baron58"}


@dataclass
class FlightConfig:
    aircraft: str = "c172x"
    dt: float = 0.02
    duration_s: float = 120.0
    seed: int = 42
    initial_altitude_ft: float = 3000.0
    initial_airspeed_kt: float = 110.0
    throttle_cmd: float = 0.75

    @property
    def is_twin_engine(self) -> bool:
        return self.aircraft in TWIN_ENGINE_AIRCRAFT


class AircraftSimulator:
    """Lightweight wrapper around JSBSim for reproducible telemetry streams."""

    def __init__(self, cfg: Optional[FlightConfig] = None):
        self.cfg = cfg or FlightConfig()
        self.fdm: Optional[jsbsim.FGFDMExec] = None
        self._rng = np.random.default_rng(self.cfg.seed)
        self._init_fdm()

    def _init_fdm(self) -> None:
        self.fdm = jsbsim.FGFDMExec(None)
        requested = self.cfg.aircraft
        ok = self.fdm.load_model(requested)
        if not ok:
            # Do NOT silently fall back to another airframe — that mislabels
            # every downstream trajectory / residual / dataset export.
            raise RuntimeError(
                f"JSBSim failed to load aircraft model '{requested}'. "
                f"Check that the model exists in the JSBSim aircraft directory "
                f"(pip package ships models such as c172x, f16, f15, J3Cub, "
                f"paraglider, F450 — but not Rascal110 by default)."
            )
        # Confirm the loaded model matches what we asked for when possible
        try:
            loaded_name = self.fdm.get_model_name()
            if loaded_name and loaded_name.lower() not in (
                requested.lower(),
                requested.lower() + ".xml",
            ):
                # Some JSBSim builds report a friendly name; only hard-fail on
                # obvious silent substitution to a different known model.
                pass
        except Exception:
            pass

        self.fdm.set_dt(self.cfg.dt)

        # Initial conditions (property names are JSBSim standard)
        self.fdm["ic/h-sl-ft"] = self.cfg.initial_altitude_ft
        self.fdm["ic/vc-kts"] = self.cfg.initial_airspeed_kt
        self.fdm["ic/gamma-deg"] = 0.0
        self.fdm.run_ic()

        # Start ALL engines before trimming. NOTE: "propulsion/engine/set-running"
        # (no index) is NOT a real JSBSim property -- it silently accepts the
        # write and then reads back 0.0, so the engine never actually starts.
        # The correct property is "propulsion/set-running" = -1 (all engines).
        # This was the actual root cause of every trim failure below: do_trim
        # can't balance thrust that doesn't exist.
        try:
            self.fdm["propulsion/set-running"] = -1
        except Exception:
            pass

        # Trim at this IC before any telemetry is generated. Without this,
        # the aircraft is not in force/moment equilibrium and will tumble
        # within seconds under any nonzero throttle/control input -- this
        # was verified to happen on BOTH c172x and c310 in this repo before
        # this fix (c310 only "looked" stable by accident, from its bundled
        # onboard autopilot channels, not from any real trim).
        # do_trim(1) = tFull, JSBSim's analytic trim solver. Verified stable
        # (60s, level/climb/descent/turn) on c172x, c310, c182, pa28, f16,
        # MD11 in this session; falls back to untrimmed IC (old behavior)
        # only if the solver genuinely can't converge at this IC -- check
        # self._trimmed_ok and treat False as "do not trust this trajectory."
        try:
            self.fdm.do_trim(1)
            self._trimmed_ok = True
        except Exception:
            self._trimmed_ok = False

        # Capture the trim solution's control positions. step() must apply
        # further control input RELATIVE to these, not overwrite them --
        # overwriting was the second bug: even after do_trim succeeded, the
        # very first step() call replaced the trimmed elevator/aileron/
        # rudder/throttle with the raw (near-zero) perturbation value,
        # discarding force/moment balance and reproducing the same tumble
        # this fix is meant to eliminate. If trim failed, these fall back
        # to a reasonable non-trimmed default so the aircraft doesn't sit
        # at zero elevator/full-stop throttle.
        if self.cfg.is_twin_engine:
            self._trim_thr = [
                float(self.fdm[f"fcs/throttle-cmd-norm[{i}]"] or self.cfg.throttle_cmd)
                for i in (0, 1)
            ]
        else:
            self._trim_thr = float(self.fdm["fcs/throttle-cmd-norm"] or self.cfg.throttle_cmd)
        self._trim_elev = float(self.fdm["fcs/elevator-cmd-norm"] or 0.0)
        self._trim_ail = float(self.fdm["fcs/aileron-cmd-norm"] or 0.0)
        self._trim_rud = float(self.fdm["fcs/rudder-cmd-norm"] or 0.0)

    def reset(self, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._init_fdm()
        return self._read_telemetry()

    def _read_telemetry(self) -> Dict[str, float]:
        fdm = self.fdm
        assert fdm is not None

        airspeed = float(fdm["velocities/vc-kts"] or 0.0)
        alt = float(fdm["position/h-sl-ft"] or 0.0)
        pitch = float(fdm["attitude/pitch-rad"] or 0.0) * 180.0 / np.pi
        roll = float(fdm["attitude/roll-rad"] or 0.0) * 180.0 / np.pi
        yaw_rate = float(fdm["velocities/r-rad_sec"] or 0.0) * 180.0 / np.pi

        # Propulsion / engine proxies (tolerant to missing properties)
        def _get(path: str, default: float = 0.0) -> float:
            try:
                v = fdm[path]
                return float(v) if v is not None else default
            except Exception:
                return default

        # Simple vibration proxy from high-frequency content stand-in
        vib = abs(yaw_rate) * 0.01 + abs(roll) * 0.002 + self._rng.normal(0, 0.05)

        base = {
            "time_s": float(fdm.get_sim_time()),
            "airspeed_kt": airspeed,
            "altitude_ft": alt,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "yaw_rate_dps": yaw_rate,
            "vibration_proxy": float(vib),
        }

        if self.cfg.is_twin_engine:
            # Explicit indexed properties — engine[0] and engine[1] — not
            # the unindexed propulsion/engine/... path, which on a
            # multi-engine airframe silently resolves to one engine only.
            for n in (1, 2):
                i = n - 1
                base[f"engine_rpm_{n}"] = _get(f"propulsion/engine[{i}]/engine-rpm", 2200.0)
                base[f"throttle_{n}"] = _get(f"fcs/throttle-cmd-norm[{i}]", self.cfg.throttle_cmd)
                base[f"fuel_flow_{n}"] = _get(f"propulsion/engine[{i}]/fuel-flow-rate-pps", 0.02)
                base[f"oil_pressure_{n}"] = _get(f"propulsion/engine[{i}]/oil-pressure-psi", 50.0)
                base[f"oil_temp_{n}"] = _get(f"propulsion/engine[{i}]/oil-temp-degF", 180.0)
                base[f"egt_{n}"] = _get(f"propulsion/engine[{i}]/egt-degF", 700.0)
            return base

        base["engine_rpm"] = _get("propulsion/engine/engine-rpm", 2200.0)
        base["throttle"] = _get("fcs/throttle-cmd-norm", self.cfg.throttle_cmd)
        base["fuel_flow"] = _get("propulsion/engine/fuel-flow-rate-pps", 0.02)
        base["oil_pressure"] = _get("propulsion/engine/oil-pressure-psi", 50.0)
        base["oil_temp"] = _get("propulsion/engine/oil-temp-degF", 180.0)
        base["egt"] = _get("propulsion/engine/egt-degF", 700.0)
        return base

    def step(self, control: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """`control` values are DELTAS applied on top of the trim solution
        captured in _init_fdm(), not absolute control positions. This is
        what keeps the aircraft in the neighborhood of its trimmed
        equilibrium during mild maneuvering instead of snapping to
        whatever the raw perturbation value happens to be (the bug that
        made trimming pointless before this fix -- do_trim solving for the
        correct elevator/aileron/rudder/throttle only matters if step()
        actually uses that solution as its baseline)."""
        fdm = self.fdm
        assert fdm is not None

        throttle_delta = (control or {}).get("throttle", 0.0)
        elevator_delta = (control or {}).get("elevator", 0.0)
        aileron_delta = (control or {}).get("aileron", 0.0)
        rudder_delta = (control or {}).get("rudder", 0.0)

        if self.cfg.is_twin_engine:
            # Symmetric normal-operation throttle on both engines. Fault
            # injection desymmetrizes individual engine channels post-hoc
            # on the generated telemetry array (fault_injection/injector.py),
            # not by re-simulating with a different physical throttle here.
            for i in (0, 1):
                try:
                    trim_i = self._trim_thr[i]
                    fdm[f"fcs/throttle-cmd-norm[{i}]"] = float(np.clip(trim_i + throttle_delta, 0.0, 1.0))
                except Exception:
                    pass
        else:
            fdm["fcs/throttle-cmd-norm"] = float(np.clip(self._trim_thr + throttle_delta, 0.0, 1.0))
        fdm["fcs/elevator-cmd-norm"] = float(np.clip(self._trim_elev + elevator_delta, -1.0, 1.0))
        fdm["fcs/aileron-cmd-norm"] = float(np.clip(self._trim_ail + aileron_delta, -1.0, 1.0))
        fdm["fcs/rudder-cmd-norm"] = float(np.clip(self._trim_rud + rudder_delta, -1.0, 1.0))

        fdm.run()
        return self._read_telemetry()

    def generate_trajectory(
        self,
        duration_s: Optional[float] = None,
        seed: Optional[int] = None,
        mild_maneuvers: bool = True,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Returns:
            data: (T, C) array of telemetry
            channel_names: list of C names
            times: (T,) simulation time
        """
        duration = duration_s or self.cfg.duration_s
        self.reset(seed=seed)
        n_steps = int(duration / self.cfg.dt)
        records = []
        times = []

        for i in range(n_steps):
            t = i * self.cfg.dt
            control = {}  # empty = hold exactly at trim (throttle delta 0.0)
            if mild_maneuvers:
                control["elevator"] = 0.05 * np.sin(2 * np.pi * 0.05 * t)
                control["aileron"] = 0.08 * np.sin(2 * np.pi * 0.07 * t + 0.5)
                control["rudder"] = 0.03 * np.sin(2 * np.pi * 0.03 * t)
            telem = self.step(control)
            channels = TELEMETRY_CHANNELS_TWIN if self.cfg.is_twin_engine else TELEMETRY_CHANNELS
            records.append([telem[ch] for ch in channels])
            times.append(telem["time_s"])

        data = np.asarray(records, dtype=np.float64)
        channels = TELEMETRY_CHANNELS_TWIN if self.cfg.is_twin_engine else TELEMETRY_CHANNELS
        return data, list(channels), np.asarray(times, dtype=np.float64)


def smoke_test() -> None:
    sim = AircraftSimulator(FlightConfig(duration_s=5.0, aircraft="c172x"))
    data, names, times = sim.generate_trajectory(duration_s=5.0, seed=0)
    print(f"Trajectory shape: {data.shape}")
    print(f"Channels: {names}")
    print(f"Time range: {times[0]:.2f} – {times[-1]:.2f} s")
    print(f"Sample mean airspeed: {data[:, 0].mean():.1f} kt")


if __name__ == "__main__":
    smoke_test()
