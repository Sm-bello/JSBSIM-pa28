"""
Dataset factory (campaign-grade).

Upgrades:
  - Randomised fault onset [15%, 80%]
  - Multiple severities + proper per-class sampling
  - Larger clean trajectory pool (30–50+)
  - Sequence length 50–100
  - Trajectory-level train/val/test split (zero leakage)
  - Residual scale fitting on train only
  - Support for paper_campaign.yaml settings
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from encoding.spike_encoder import SpikeEncoder
from fault_injection.injector import FAULT_CLASSES, FaultInjector, FaultType
from physics.residual import PhysicsResidualEngine
from simulation.aircraft_sim import AircraftSimulator, FlightConfig, TELEMETRY_CHANNELS, TELEMETRY_CHANNELS_TWIN


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectories: List[np.ndarray],
        labels: List[np.ndarray],
        residuals: List[np.ndarray],
        encoder: SpikeEncoder,
        sequence_len: int = 80,
        stride: int = 40,
    ):
        self.windows = []
        self.window_labels = []
        self.window_residuals = []
        self.encoder = encoder

        for traj, lab, res in zip(trajectories, labels, residuals):
            T = len(traj)
            for start in range(0, max(1, T - sequence_len), stride):
                end = start + sequence_len
                if end > T:
                    break
                self.windows.append(traj[start:end])
                self.window_labels.append(int(np.bincount(lab[start:end]).argmax()))
                r_slice = res[max(0, start - 1) : end - 1] if len(res) == T - 1 else res[start:end]
                if len(r_slice) < sequence_len:
                    pad = np.zeros((sequence_len - len(r_slice), res.shape[-1]))
                    r_slice = np.concatenate([pad, r_slice], axis=0)
                self.window_residuals.append(r_slice[:sequence_len])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x = self.windows[idx]
        spikes = self.encoder.encode(x)
        y = self.window_labels[idx]
        r = self.window_residuals[idx].astype(np.float32)
        return (
            torch.from_numpy(spikes).float(),
            torch.tensor(y, dtype=torch.long),
            torch.from_numpy(r).float(),
        )


def generate_clean_trajectories(
    n_traj: int = 40,
    duration_s: float = 80.0,
    seed: int = 42,
    aircraft: str = "c172x",
) -> List[np.ndarray]:
    # Verified-trimmable envelopes per aircraft (see CHANGES_trim_fix_pass.md).
    # The original hardcoded range (2500-3700ft / 95-119kt) was tuned for
    # c172x only and silently reused for every aircraft regardless of
    # --aircraft -- for pa28 that range is entirely outside its verified
    # working envelope (75-90kt), so every trajectory would fail trim.
    # Unlisted aircraft fall back to the original c172x-tuned range, same
    # as before this patch.
    ENVELOPES = {
        "c172x":  {"alt": (2500.0, 3700.0), "kt": (95.0, 119.0)},
        "c182":   {"alt": (2500.0, 5500.0), "kt": (88.0, 112.0)},
        "pa28":   {"alt": (1600.0, 2900.0), "kt": (77.0, 89.0)},
    }
    env = ENVELOPES.get(aircraft, ENVELOPES["c172x"])
    alt_lo, alt_hi = env["alt"]
    kt_lo, kt_hi = env["kt"]

    trajs = []
    attempts = 0
    i = 0
    while len(trajs) < n_traj and attempts < n_traj * 4:
        attempts += 1
        # Vary initial conditions across the aircraft's verified envelope
        cfg = FlightConfig(
            aircraft=aircraft,
            duration_s=duration_s,
            seed=seed + attempts,
            initial_altitude_ft=alt_lo + (attempts % 5) * (alt_hi - alt_lo) / 4.0,
            initial_airspeed_kt=kt_lo + (attempts % 4) * (kt_hi - kt_lo) / 3.0,
            throttle_cmd=0.65 + (attempts % 3) * 0.08,
        )
        sim = AircraftSimulator(cfg)
        data, _, _ = sim.generate_trajectory(duration_s=duration_s, seed=seed + attempts)

        # Validity gate: reject trajectories where trim failed, or where the
        # aircraft still drifted into an implausible attitude despite trim
        # succeeding. Previously nothing checked this -- an untrimmed or
        # diverged run would silently enter the "clean" dataset.
        if not getattr(sim, "_trimmed_ok", True):
            continue
        pitch_idx = 2  # TELEMETRY_CHANNELS / TELEMETRY_CHANNELS_TWIN both start
        roll_idx = 3   # [airspeed, altitude, pitch, roll, ...]
        if np.abs(data[:, pitch_idx]).max() > 30.0 or np.abs(data[:, roll_idx]).max() > 45.0:
            continue

        trajs.append(data)
        i += 1

    if len(trajs) < n_traj:
        print(f"[dataset] WARNING: only {len(trajs)}/{n_traj} clean trajectories passed "
              f"the trim/stability gate for aircraft={aircraft} after {attempts} attempts. "
              f"Consider widening/rechecking its verified envelope.")
    return trajs


def build_datasets(
    n_clean: int = 40,
    n_per_class: int = 3,
    severities: Optional[List[float]] = None,
    duration_s: float = 80.0,
    seed: int = 42,
    sequence_len: int = 80,
    stride: int = 40,
    onset_range: Tuple[float, float] = (0.15, 0.80),
    out_dir: str = "datasets/full",
    aircraft: str = "c172x",
    quick: bool = False,
) -> Dict:
    if severities is None:
        severities = [0.10, 0.25, 0.50, 0.75, 1.00]

    if quick:
        n_clean = min(n_clean, 6)
        n_per_class = 1
        duration_s = min(duration_s, 25.0)
        sequence_len = min(sequence_len, 40)
        severities = [0.25, 0.5, 0.75]

    print(f"[dataset] Generating {n_clean} clean trajectories ({aircraft}, {duration_s}s) …")
    clean = generate_clean_trajectories(
        n_traj=n_clean, duration_s=duration_s, seed=seed, aircraft=aircraft
    )

    n_train = max(1, int(0.6 * n_clean))
    n_val = max(1, int(0.2 * n_clean))
    train_clean = clean[:n_train]
    val_clean = clean[n_train : n_train + n_val]
    test_clean = clean[n_train + n_val :]

    # BUGFIX: FaultInjector and PhysicsResidualEngine both defaulted to the
    # single-engine 12-channel schema regardless of `aircraft`. On a twin
    # (aircraft in TWIN_ENGINE_AIRCRAFT), generate_clean_trajectories above
    # already produces 18-channel telemetry (TELEMETRY_CHANNELS_TWIN), which
    # would immediately fail FaultInjector's channel-count assertion and
    # silently mismatch PhysicsResidualEngine's 12x12 transition matrix.
    is_twin = FlightConfig(aircraft=aircraft).is_twin_engine
    channel_names = TELEMETRY_CHANNELS_TWIN if is_twin else TELEMETRY_CHANNELS
    inj = FaultInjector(seed=seed, channel_names=channel_names)
    eng = PhysicsResidualEngine(dt=0.02, normalize=True, twin_engine=is_twin)

    def expand(c_list, fit_scales: bool = False):
        X, Y, specs = inj.generate_dataset(
            c_list,
            n_per_class=n_per_class,
            severities=severities,
            onset_range=onset_range,
        )
        residuals = [eng.residual_sequence(x, apply_norm=False) for x in X]
        if fit_scales:
            eng.fit_scales(residuals)
            residuals = [eng.residual_sequence(x, apply_norm=True) for x in X]
        else:
            residuals = [r / (eng._scale + 1e-8) for r in residuals]
        return X, Y, residuals, specs

    print("[dataset] Injecting faults with randomised onset …")
    Xtr, Ytr, Rtr, _ = expand(train_clean, fit_scales=True)
    Xva, Yva, Rva, _ = expand(val_clean, fit_scales=False)
    Xte, Yte, Rte, _ = expand(test_clean, fit_scales=False)

    encoder = SpikeEncoder(method="hybrid", rate_max_hz=200.0)
    all_train = np.concatenate(Xtr, axis=0)
    encoder.fit_normalisation(all_train)

    train_ds = TrajectoryDataset(Xtr, Ytr, Rtr, encoder, sequence_len=sequence_len, stride=stride)
    val_ds = TrajectoryDataset(Xva, Yva, Rva, encoder, sequence_len=sequence_len, stride=stride)
    test_ds = TrajectoryDataset(Xte, Yte, Rte, encoder, sequence_len=sequence_len, stride=stride)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta = {
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "n_test_windows": len(test_ds),
        "n_classes": len(FAULT_CLASSES),
        "channels": TELEMETRY_CHANNELS,
        "fault_classes": FAULT_CLASSES,
        "n_clean": n_clean,
        "severities": severities,
        "onset_range": onset_range,
        "sequence_len": sequence_len,
        "aircraft": aircraft,
        "residual_scales": eng._scale.tolist(),
    }
    np.save(os.path.join(out_dir, "meta.npy"), meta, allow_pickle=True)
    print(
        f"[dataset] Windows — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )
    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "encoder": encoder,
        "meta": meta,
        "engine": eng,
    }


def get_loaders(datasets: Dict, batch_size: int = 32, num_workers: int = 0):
    return {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }


if __name__ == "__main__":
    ds = build_datasets(n_clean=4, n_per_class=1, duration_s=20.0, sequence_len=40, quick=True)
    print("Done.", ds["meta"])
