"""
Per-aircraft PASS GATE for PHI-SPIKE.

Stage 1 of the master build plan uses three airframes sequentially
(C172P -> F-16 -> UAV-class). Before any unified run (training,
baselines, edge demo, dashboard, or Stage 14 cross-aircraft generalisation),
each airframe must independently pass this gate:

  1. JSBSim loads the model
  2. A short trajectory generates with no NaN/Inf telemetry
  3. The physics residual engine runs cleanly on that trajectory
  4. Fault injection + spike encoding run cleanly on that trajectory

Usage:
    python -m scripts.aircraft_gate
    python -m scripts.aircraft_gate --aircraft c172p_general_aviation f16_fighter
    python -m scripts.aircraft_gate --duration 10

Exits 0 only if every requested aircraft passes. Writes:
  - verification/aircraft_gate_report.json
  - console_log.txt  (full tee of stdout/stderr for this run)
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# JSBSim model identifiers to try for each conceptual airframe.
#
# IMPORTANT: the official pip `jsbsim` package does NOT ship a Rascal110 model.
# Verified aircraft folder names include (among others):
#   c172x, c172p, c172r, f16, f15, f22, J3Cub, paraglider, F450, p51d, ...
# Rascal110 must be installed manually (FlightGear/FGAddon / ArduPilot SITL)
# or we fall through to a shipped UAV / light-aircraft proxy (F450, paraglider, J3Cub).
AIRCRAFT_CANDIDATES = {
    "c172p_general_aviation": ["c172x", "c172p", "c172r"],
    "f16_fighter": ["f16"],
    # Twin-piston generalization target. Baron58 is not bundled in the
    # official pip jsbsim package (verified: load fails, no aircraft/
    # Baron58/ directory ships with jsbsim==1.3.1) and was not found in
    # the JSBSim-Team GitHub repos either -- it exists only as
    # third-party FGAddon content that was not fetched here (unverified
    # source/licensing). c310 (Cessna 310) is used instead: officially
    # bundled, twin-piston, same asymmetric-thrust generalization value.
    # If a real Baron58/Baron58.xml is later placed in the JSBSim
    # aircraft directory, it is tried first automatically.
    "twin_piston_generalization": ["Baron58", "c310"],
    # Prefer true Rascal if the user installed it; otherwise use a shipped
    # light / UAV-class stand-in so the gate can still clear.
    "rascal110_uav": [
        "Rascal110-JSBSim",
        "Rascal",
        "rascal110",
        "Rascal110",
        "F450",          # quad-style UAV model shipped with recent JSBSim
        "paraglider",    # light airframe proxy
        "J3Cub",         # light GA stand-in
    ],
}


class _Tee:
    """Write to both a file and the original stream."""

    def __init__(self, stream, path: Path):
        self._stream = stream
        self._file = open(path, "w", encoding="utf-8", errors="replace")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


def _try_load(candidates):
    """Return (working_name, sim, error_messages) for the first candidate that loads.

    aircraft_sim no longer silently falls back to f16. We still treat a
    resolved name that is outside the candidate list as a failure so that a
    future silent-fallback regression cannot mislabel data.
    """
    from simulation.aircraft_sim import AircraftSimulator, FlightConfig

    errors = []
    for name in candidates:
        try:
            sim = AircraftSimulator(FlightConfig(duration_s=3.0, aircraft=name))
            resolved = sim.cfg.aircraft
            if resolved not in candidates:
                errors.append(
                    f"{name}: resolved to '{resolved}' which is not in the "
                    f"candidate list {candidates} — treating as failure"
                )
                continue
            return resolved, sim, errors
        except Exception as e:
            errors.append(f"{name}: {e}")
    return None, None, errors


def run_gate(label: str, candidates, duration_s: float) -> dict:
    from physics.residual import PhysicsResidualEngine
    from fault_injection.injector import FaultInjector, FaultType
    from encoding.spike_encoder import SpikeEncoder

    result = {
        "aircraft": label,
        "candidates_tried": candidates,
        "status": "FAIL",
        "checks": {},
    }

    resolved, sim, load_errors = _try_load(candidates)
    if sim is None:
        result["error"] = "no candidate model could be loaded: " + "; ".join(load_errors)
        result["load_errors"] = load_errors
        return result
    result["resolved_model"] = resolved
    result["checks"]["load_model"] = "PASS"
    if resolved.lower() not in {c.lower() for c in ("Rascal110-JSBSim", "Rascal", "rascal110", "Rascal110")}:
        if label == "rascal110_uav":
            result["note"] = (
                f"Rascal110 not present in this JSBSim install; using shipped "
                f"proxy model '{resolved}'. Install a Rascal110 aircraft folder "
                f"into jsbsim/aircraft/ if you need the real airframe."
            )
            print(f"  [NOTE ] {result['note']}")

    try:
        data, names, times = sim.generate_trajectory(duration_s=duration_s, seed=0)
        assert data.shape[0] > 10, "trajectory too short"
        assert np.isfinite(data).all(), "non-finite telemetry values"
        result["checks"]["trajectory_generation"] = "PASS"
        result["samples"] = int(data.shape[0])
        result["channels"] = list(names)
    except Exception as e:
        result["checks"]["trajectory_generation"] = f"FAIL: {e}"
        return result

    try:
        eng = PhysicsResidualEngine(dt=0.02)
        r = eng.residual_norm(data)
        assert len(r) > 0 and np.isfinite(r).all(), "invalid physics residual"
        result["checks"]["physics_residual"] = "PASS"
        result["residual_norm_mean"] = float(np.mean(r))
    except Exception as e:
        result["checks"]["physics_residual"] = f"FAIL: {e}"
        return result

    try:
        inj = FaultInjector(seed=0)
        corrupted, labels, spec = inj.inject(data, FaultType.SENSOR_BIAS, 0.5)
        assert corrupted.shape == data.shape
        enc = SpikeEncoder(method="hybrid")
        enc.fit_normalisation(corrupted)
        spikes = enc.encode(corrupted)
        assert spikes.shape[0] == corrupted.shape[0]
        result["checks"]["fault_injection_and_encoding"] = "PASS"
    except Exception as e:
        result["checks"]["fault_injection_and_encoding"] = f"FAIL: {e}"
        return result

    result["status"] = "PASS"
    return result


def main():
    log_path = ROOT / "console_log.txt"
    tee_out = _Tee(sys.stdout, log_path)
    tee_err = _Tee(sys.stderr, log_path)
    sys.stdout = tee_out
    sys.stderr = tee_err

    try:
        parser = argparse.ArgumentParser(description="PHI-SPIKE per-aircraft pass gate")
        parser.add_argument(
            "--aircraft",
            nargs="*",
            default=list(AIRCRAFT_CANDIDATES.keys()),
            help="Which airframes to gate (default: all three in the plan)",
        )
        parser.add_argument(
            "--duration",
            type=float,
            default=6.0,
            help="Seconds of trajectory to smoke-test",
        )
        args = parser.parse_args()

        print("=== PHI-SPIKE Aircraft Pass Gate ===")
        print(f"(Stage 1 of the master plan: {', '.join(args.aircraft)})")
        print(f"Full console log -> {log_path}\n")

        results = {}
        for label in args.aircraft:
            candidates = AIRCRAFT_CANDIDATES.get(label)
            if candidates is None:
                print(f"[SKIP] Unknown aircraft key: {label}")
                continue
            print(f"--- Gating: {label} ---")
            print(f"  candidates: {candidates}")
            try:
                res = run_gate(label, candidates, args.duration)
            except Exception as e:
                res = {"aircraft": label, "status": "FAIL", "error": str(e)}
                traceback.print_exc()
            results[label] = res
            for check, outcome in res.get("checks", {}).items():
                tag = outcome.split(":")[0]
                print(f"  [{tag:5s}] {check}")
            if res["status"] == "PASS":
                print(f"  -> PASS  (resolved model: {res.get('resolved_model')})\n")
            else:
                print(f"  -> FAIL  ({res.get('error', 'see checks above')})\n")

        out_path = ROOT / "verification" / "aircraft_gate_report.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
        n_total = len(results)
        print(f"{n_pass}/{n_total} airframes passed. Report -> {out_path}")
        print(f"Console log also written to -> {log_path}")

        if n_pass < n_total:
            print("\nGATE NOT CLEARED. Do not proceed to training / baselines / dashboard")
            print("until every requested airframe shows PASS above.")
            return 1

        print("\nGATE CLEARED. Safe to proceed to the unified verification suite and training.")
        return 0
    finally:
        sys.stdout = tee_out._stream
        sys.stderr = tee_err._stream
        tee_out.close()
        tee_err.close()


if __name__ == "__main__":
    sys.exit(main())
