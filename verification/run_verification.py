"""
Minimal V&V gate script for PHI-SPIKE TRL-4 evidence.
Runs structural checks and records pass/fail for each component.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = {}


def check(name, fn):
    try:
        fn()
        RESULTS[name] = {"status": "PASS"}
        print(f"  [PASS] {name}")
    except Exception as e:
        RESULTS[name] = {"status": "FAIL", "error": str(e)}
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


def test_jsbsim():
    from simulation.aircraft_sim import AircraftSimulator, FlightConfig
    sim = AircraftSimulator(FlightConfig(duration_s=3.0, aircraft="c172x"))
    data, names, times = sim.generate_trajectory(duration_s=3.0, seed=0)
    assert data.shape[0] > 10
    assert len(names) == 12


def test_physics():
    from simulation.aircraft_sim import AircraftSimulator, FlightConfig
    from physics.residual import PhysicsResidualEngine
    data, _, _ = AircraftSimulator(FlightConfig(duration_s=5.0)).generate_trajectory(duration_s=5.0)
    eng = PhysicsResidualEngine()
    r = eng.residual_norm(data)
    assert len(r) > 0
    viol = eng.physical_bound_violations(data)
    assert "overall" in viol


def test_faults():
    from simulation.aircraft_sim import AircraftSimulator, FlightConfig
    from fault_injection.injector import FaultInjector, FaultType
    data, _, _ = AircraftSimulator(FlightConfig(duration_s=8.0)).generate_trajectory(duration_s=8.0)
    inj = FaultInjector()
    corrupted, labels, spec = inj.inject(data, FaultType.SENSOR_BIAS, 0.5)
    assert corrupted.shape == data.shape
    assert labels.max() > 0


def test_encoder():
    import numpy as np
    from encoding.spike_encoder import SpikeEncoder
    x = np.random.randn(50, 12).astype("float32")
    enc = SpikeEncoder(method="hybrid")
    enc.fit_normalisation(x)
    s = enc.encode(x)
    assert s.shape[0] == 50


def test_vanilla_snn():
    import torch
    from models.vanilla_snn.lif_snn import VanillaSNN
    m = VanillaSNN(n_inputs=24, n_hidden=32, n_outputs=10)
    x = (torch.rand(20, 2, 24) > 0.9).float()
    logits, spars = m(x)
    assert logits.shape == (2, 10)


def test_phi_spike():
    import torch
    from models.phi_spike.phi_snn import PHISPIKE, physics_informed_loss
    m = PHISPIKE(n_inputs=24, n_hidden=32, n_outputs=10, residual_dim=12)
    x = (torch.rand(20, 2, 24) > 0.9).float()
    r = torch.randn(20, 2, 12) * 0.1
    # BUGFIX: PHISPIKE.forward() always returns 4 values with the default
    # return_states=False: (logits, sparsity, residual_pred, temporal_pred).
    logits, spars, r_pred, temporal_pred = m(x, r)
    loss, _ = physics_informed_loss(logits, torch.randint(0, 10, (2,)), r_pred, r.mean(0), spars)
    assert loss.ndim == 0


def test_edge_pipeline():
    from edge_emulation.streaming_infer import run_demo
    # short run
    run_demo(duration_s=5.0, inject_fault=True)


def main():
    print("=== PHI-SPIKE Verification Suite ===")
    check("JSBSim aircraft telemetry", test_jsbsim)
    check("Physics residual engine", test_physics)
    check("Fault injection laboratory", test_faults)
    check("Spike encoder", test_encoder)
    check("Vanilla SNN forward", test_vanilla_snn)
    check("PHI-SPIKE forward + loss", test_phi_spike)
    check("Edge streaming pipeline", test_edge_pipeline)

    out = ROOT / "verification" / "verification_report.json"
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2)
    n_pass = sum(1 for v in RESULTS.values() if v["status"] == "PASS")
    print(f"\n{n_pass}/{len(RESULTS)} checks passed. Report → {out}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
