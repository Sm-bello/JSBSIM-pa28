"""
PHI-SPIKE live dashboard server.

Runs the same pipeline as edge_emulation/streaming_infer.py (simulator ->
physics residual -> spike encoder -> PHI-SPIKE SNN) in a background thread,
and serves the rolling state over HTTP so dashboard/templates/index.html can
poll it and render live telemetry, spikes, physics residuals and diagnosis.

Run:
    python -m dashboard.server

Then open http://127.0.0.1:8765 in a browser.
"""

from __future__ import annotations

import sys
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np
import torch
from flask import Flask, jsonify, render_template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from encoding.spike_encoder import SpikeEncoder  # noqa: E402
from fault_injection.injector import FAULT_CLASSES, FaultInjector, FaultType  # noqa: E402
from models.phi_spike.phi_snn import PHISPIKE  # noqa: E402
from physics.residual import PhysicsResidualEngine  # noqa: E402
from simulation.aircraft_sim import AircraftSimulator, FlightConfig, TELEMETRY_CHANNELS  # noqa: E402

HISTORY_LEN = 150  # points kept per channel for the live charts
WINDOW = 80  # matches training sequence_len=80

app = Flask(__name__)

STATE_LOCK = threading.Lock()
STATE: Dict = {
    "connected": False,
    "aircraft": "C172P",
    "mission_time_s": 0.0,
    "telemetry_channels": TELEMETRY_CHANNELS,
    "telemetry_history": {ch: [] for ch in TELEMETRY_CHANNELS},
    "residual_history": {ch: [] for ch in TELEMETRY_CHANNELS},
    "spike_raster": [],  # list of {t, channel_idx} recent spike events
    "fault_class": "healthy",
    "fault_confidence": 0.0,
    "true_fault": "healthy",
    "health_score": 100,
    "latency_ms": 0.0,
    "avg_latency_ms": 0.0,
    "spike_rate_hz": 0.0,
    "sparsity_pct": 0.0,
    "status": "starting",
}


def _health_score(fault_idx: int, confidence: float) -> int:
    if fault_idx == 0:  # healthy
        return int(round(100 - confidence * 5))
    return int(round(max(5, 100 - confidence * 90)))


def _pipeline_loop(duration_s: float = 600.0, aircraft: str = "c172x") -> None:
    """Continuously generates telemetry, injects a fault partway through,
    runs the PHI-SPIKE streaming inference, and writes results into STATE."""
    while True:
        try:
            with STATE_LOCK:
                STATE["status"] = "loading simulator"

            sim = AircraftSimulator(FlightConfig(duration_s=duration_s, aircraft=aircraft))
            data, names, times = sim.generate_trajectory(duration_s=duration_s, seed=int(time.time()) % 10_000)

            inj = FaultInjector(seed=int(time.time()) % 10_000)
            fault_choice = FaultType.ENGINE_PERFORMANCE_LOSS
            data, labels, spec = inj.inject(data, fault_choice, severity=0.6, start_fraction=0.5)

            encoder = SpikeEncoder(method="hybrid")
            encoder.fit_normalisation(data)
            residual_eng = PhysicsResidualEngine(dt=0.02)

            n_inputs = encoder.encode(data[:10]).shape[-1]
            # BUGFIX: training always builds full_phi_spike with n_hidden=128
            # (train_campaign.py); this was hardcoded to 64 here, which made
            # every load_state_dict() below fail with a shape mismatch,
            # silently swallowed by the bare `except: pass`, leaving this
            # dashboard running on random/untrained weights with no visible
            # indication in the UI.
            model = PHISPIKE(n_inputs=n_inputs, n_hidden=128, n_outputs=len(FAULT_CLASSES), residual_dim=12)
            # BUGFIX: this path never existed for any aircraft (real
            # checkpoints live under experiments/full_campaign/{aircraft}/
            # <variant>_seed<N>_final.pt since train_campaign.py's --aircraft
            # support). Auto-select the best full_phi_spike checkpoint for
            # this aircraft by test F1, same logic as edge_emulation.
            ckpt_path = None
            campaign_dir = ROOT / "experiments" / "full_campaign" / aircraft
            if not campaign_dir.exists():
                campaign_dir = ROOT / "experiments" / "full_campaign"  # legacy flat c172x layout
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
            weights_loaded = False
            if ckpt_path is not None and ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(ckpt["state_dict"])
                    weights_loaded = True
                except Exception as e:
                    with STATE_LOCK:
                        STATE["status"] = f"WARNING: checkpoint load failed ({e}) - random weights"
            model.eval()

            with STATE_LOCK:
                STATE["weights_loaded"] = weights_loaded
                STATE["ckpt_path"] = str(ckpt_path) if ckpt_path else None
                STATE["aircraft"] = aircraft

            buffer: Deque[np.ndarray] = deque(maxlen=WINDOW)
            latencies: Deque[float] = deque(maxlen=100)

            with STATE_LOCK:
                STATE["status"] = "streaming"
                STATE["connected"] = True

            for i, row in enumerate(data):
                buffer.append(row)
                t = float(times[i])

                with STATE_LOCK:
                    for c, ch in enumerate(TELEMETRY_CHANNELS):
                        hist = STATE["telemetry_history"][ch]
                        hist.append({"t": t, "v": float(row[c])})
                        if len(hist) > HISTORY_LEN:
                            del hist[0]
                    STATE["mission_time_s"] = t
                    STATE["true_fault"] = FAULT_CLASSES[int(labels[i])] if int(labels[i]) < len(FAULT_CLASSES) else "unknown"

                if len(buffer) == WINDOW:
                    window_data = np.stack(buffer, axis=0)
                    spikes = encoder.encode(window_data)
                    residual = residual_eng.residual_sequence(window_data)
                    if len(residual) < WINDOW:
                        pad = np.zeros((WINDOW - len(residual), residual.shape[-1]))
                        residual = np.concatenate([pad, residual], axis=0)

                    x = torch.from_numpy(spikes).float().unsqueeze(1)
                    r = torch.from_numpy(residual).float().unsqueeze(1)

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        # BUGFIX: PHISPIKE.forward() always returns 4 values
                        # with return_states=False: (logits, sparsity,
                        # residual_pred, temporal_pred), never 3.
                        logits, spars, _residual_pred, _temporal_pred = model(x, r)
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(latency_ms)

                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    pred = int(probs.argmax())
                    conf = float(probs[pred])
                    fault_name = FAULT_CLASSES[pred] if pred < len(FAULT_CLASSES) else f"class_{pred}"
                    sparsity_val = float(spars) if torch.is_tensor(spars) else float(spars)

                    # Build a small spike-raster snapshot from the latest encoded window
                    last_spikes = spikes[-1]  # (F,)
                    raster = [
                        {"t": t, "ch": int(k)}
                        for k in np.nonzero(last_spikes)[0][:40]
                    ]

                    with STATE_LOCK:
                        for c, ch in enumerate(TELEMETRY_CHANNELS):
                            rhist = STATE["residual_history"][ch]
                            rval = float(residual[-1][c]) if c < residual.shape[-1] else 0.0
                            rhist.append({"t": t, "v": rval})
                            if len(rhist) > HISTORY_LEN:
                                del rhist[0]
                        STATE["spike_raster"] = raster
                        STATE["fault_class"] = fault_name
                        STATE["fault_confidence"] = conf
                        STATE["health_score"] = _health_score(pred, conf)
                        STATE["latency_ms"] = latency_ms
                        STATE["avg_latency_ms"] = float(np.mean(latencies)) if latencies else 0.0
                        STATE["sparsity_pct"] = sparsity_val * 100.0 if sparsity_val <= 1.0 else sparsity_val
                        STATE["spike_rate_hz"] = float(last_spikes.sum() / max(1e-6, encoder.dt))

                time.sleep(0.02)  # ~50 Hz, matches telemetry dt

            with STATE_LOCK:
                STATE["status"] = "mission complete — restarting"

        except Exception as e:  # keep the server alive even if a run errors out
            with STATE_LOCK:
                STATE["status"] = f"error: {e}"
                STATE["connected"] = False
            time.sleep(2.0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    with STATE_LOCK:
        return jsonify(STATE)


def main():
    thread = threading.Thread(target=_pipeline_loop, daemon=True)
    thread.start()
    app.run(host="127.0.0.1", port=8765, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
