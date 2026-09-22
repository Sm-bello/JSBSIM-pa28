"""
Simulated edge deployment environment for PHI-SPIKE.

Constraints:
  - batch_size = 1
  - streaming inference (no future leakage)
  - CPU-only path
  - latency measurement
  - optional artificial CPU/memory pressure reporting
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.spike_encoder import SpikeEncoder
from fault_injection.injector import FaultInjector, FaultType
from models.phi_spike.phi_snn import PHISPIKE
from physics.residual import PhysicsResidualEngine
from simulation.aircraft_sim import AircraftSimulator, FlightConfig


class EdgeEmulator:
    def __init__(
        self,
        model: torch.nn.Module,
        encoder: SpikeEncoder,
        residual_engine: PhysicsResidualEngine,
        window: int = 40,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.encoder = encoder
        self.residual_engine = residual_engine
        self.window = window
        self.device = device
        self.buffer: List[np.ndarray] = []
        self.latencies_ms: List[float] = []

    @torch.no_grad()
    def step(self, telemetry_row: np.ndarray) -> Dict:
        """Ingest one telemetry sample; when buffer full, run inference."""
        self.buffer.append(telemetry_row)
        if len(self.buffer) < self.window:
            return {"status": "warming", "ready": False}

        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window :]

        window_data = np.stack(self.buffer, axis=0)  # (T, C)
        spikes = self.encoder.encode(window_data)  # (T, F)
        residual = self.residual_engine.residual_sequence(window_data)
        # pad residual to T
        if len(residual) < self.window:
            pad = np.zeros((self.window - len(residual), residual.shape[-1]))
            residual = np.concatenate([pad, residual], axis=0)

        x = torch.from_numpy(spikes).float().unsqueeze(1).to(self.device)  # (T, 1, F)
        r = torch.from_numpy(residual).float().unsqueeze(1).to(self.device)

        t0 = time.perf_counter()
        if isinstance(self.model, PHISPIKE):
            # PHISPIKE.forward always returns 4 values with return_states=False:
            # (logits, sparsity, residual_pred, temporal_pred).
            logits, spars, _residual_pred, _temporal_pred = self.model(x, r)
        else:
            logits, spars = self.model(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(latency_ms)

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])
        return {
            "status": "ok",
            "ready": True,
            "fault_class": pred,
            "confidence": conf,
            "sparsity": float(spars) if torch.is_tensor(spars) else float(spars),
            "latency_ms": latency_ms,
            "avg_latency_ms": float(np.mean(self.latencies_ms[-50:])),
        }


def run_demo(
    duration_s: float = 30.0,
    inject_fault: bool = True,
    ckpt_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    aircraft: str = "c172x",
) -> Dict:
    print(f"[edge] Building simulator ({aircraft}) + PHI-SPIKE edge pipeline …")
    sim = AircraftSimulator(FlightConfig(duration_s=duration_s, aircraft=aircraft))
    data, names, times = sim.generate_trajectory(duration_s=duration_s, seed=11)

    if inject_fault:
        inj = FaultInjector(seed=11)
        data, labels, spec = inj.inject(data, FaultType.ENGINE_PERFORMANCE_LOSS, 0.6, start_fraction=0.4)
        print(f"[edge] Injected {spec.fault_type.value} @ severity {spec.severity} (idx {spec.start_idx})")
    else:
        labels = np.zeros(len(data), dtype=np.int64)

    encoder = SpikeEncoder(method="hybrid")
    encoder.fit_normalisation(data)
    residual_eng = PhysicsResidualEngine(dt=0.02)

    n_inputs = encoder.encode(data[:10]).shape[-1]
    # BUGFIX: training always builds full_phi_spike with n_hidden=128
    # (train_campaign.py); this was hardcoded to 64 here, causing every
    # load_state_dict() to fail with a silent shape-mismatch, caught by the
    # bare except and leaving this demo running on random weights.
    model = PHISPIKE(n_inputs=n_inputs, n_hidden=128, n_outputs=10, residual_dim=12)

    # BUGFIX: previously hardcoded to ROOT/"experiments"/"phi_spike_final.pt",
    # which never existed for ANY aircraft. Auto-selects the best
    # full_phi_spike checkpoint by test F1 under
    # experiments/full_campaign/{aircraft}/, matching train_campaign.py's
    # --aircraft-aware save_dir convention, unless one is passed explicitly.
    weights_loaded = False
    if ckpt_path is None:
        campaign_dir = ROOT / "experiments" / "full_campaign" / aircraft
        if not campaign_dir.exists():
            # legacy flat layout fallback, c172x only (pre---aircraft campaigns)
            campaign_dir = ROOT / "experiments" / "full_campaign"
        best_path, best_f1 = None, -1.0
        for p in sorted(campaign_dir.glob("full_phi_spike_seed*_metrics.json")):
            try:
                m = json.loads(p.read_text())
                f1 = m.get("test", {}).get("f1", -1.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_path = campaign_dir / f"{p.stem.replace('_metrics', '')}_final.pt"
            except Exception:
                continue
        if best_path is not None and best_path.exists():
            ckpt_path = best_path
            print(f"[edge] Auto-selected best {aircraft} checkpoint by test F1={best_f1:.3f}: {ckpt_path.name}")

    if ckpt_path is not None and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        try:
            model.load_state_dict(ckpt["state_dict"])
            weights_loaded = True
            print(f"[edge] Loaded trained PHI-SPIKE weights from {ckpt_path}")
        except Exception as e:
            print(f"[edge] Could not load weights ({e}); using random init for pipeline demo")
    else:
        print(f"[edge] WARNING: no checkpoint found for aircraft={aircraft} — running with "
              f"RANDOM/UNTRAINED weights. Predictions below are meaningless for accuracy, "
              f"only useful for latency/pipeline structure.")

    # BUGFIX: training used sequence_len=80; window=40 fed the model half the
    # temporal context it was trained on, independently hurting accuracy.
    edge = EdgeEmulator(model, encoder, residual_eng, window=80)
    detections = []
    for i, row in enumerate(data):
        out = edge.step(row)
        if out.get("ready"):
            out["true_label"] = int(labels[i])
            out["t"] = float(times[i])
            if i % 25 == 0:
                print(
                    f"  t={times[i]:6.1f}s  pred={out['fault_class']:2d}  "
                    f"conf={out['confidence']:.2f}  lat={out['latency_ms']:.2f} ms  "
                    f"true={labels[i]}"
                )
            detections.append(out)

    result = {"aircraft": aircraft, "weights_loaded": weights_loaded,
              "ckpt_path": str(ckpt_path) if ckpt_path else None}
    if edge.latencies_ms:
        lat_mean = float(np.mean(edge.latencies_ms))
        lat_p95 = float(np.percentile(edge.latencies_ms, 95))
        lat_max = float(np.max(edge.latencies_ms))
        print(f"[edge] Latency: mean={lat_mean:.2f} ms  p95={lat_p95:.2f} ms  max={lat_max:.2f} ms")
        result.update({"latency_ms_mean": lat_mean, "latency_ms_p95": lat_p95, "latency_ms_max": lat_max})

    if weights_loaded and detections:
        correct = sum(1 for d in detections if d["fault_class"] == d["true_label"])
        acc = correct / len(detections)
        print(f"[edge] Streaming accuracy (trained weights): {acc:.3f} ({correct}/{len(detections)})")
        result["streaming_accuracy"] = acc

    print("[edge] Streaming pipeline OK — batch_size=1, CPU path.")
    result["detections"] = detections

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "edge_emulation_report.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[edge] Report written to {out_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--ckpt", type=str, default=None,
                         help="Explicit checkpoint path; auto-selects best checkpoint for --aircraft if omitted")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--aircraft", type=str, default="c172x")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_demo(
        duration_s=args.duration_s,
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        aircraft=args.aircraft,
    )
