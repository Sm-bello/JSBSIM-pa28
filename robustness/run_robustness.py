"""
Robustness matrix for PHI-SPIKE.

Evaluates trained models under:
  - Gaussian noise
  - Sensor bias / drift (already in training; re-evaluate severity)
  - Packet loss / dropout
  - Missing telemetry

Outputs JSON + optional radar/bar plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.factory import build_datasets, get_loaders
from models.phi_spike.phi_snn import build_ablation_model
from training.train_campaign import evaluate, macro_f1


def apply_noise(spikes: torch.Tensor, level: float) -> torch.Tensor:
    if level <= 0:
        return spikes
    noise = torch.randn_like(spikes) * level
    return (spikes + noise).clamp(0, 1)


def apply_dropout(spikes: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0:
        return spikes
    mask = (torch.rand_like(spikes) > rate).float()
    return spikes * mask


def apply_packet_loss(spikes: torch.Tensor, rate: float) -> torch.Tensor:
    """Drop entire timesteps with given probability."""
    if rate <= 0:
        return spikes
    # spikes: (B, T, F)
    B, T, F = spikes.shape
    keep = (torch.rand(B, T, 1) > rate).float()
    return spikes * keep


@torch.no_grad()
def evaluate_corrupted(model, loader, device, flags, corrupt_fn):
    model.eval()
    all_preds, all_tgts = [], []
    sparsities = []
    physics = flags.get("physics_conditioning", False)
    for spikes, y, residual in loader:
        spikes = corrupt_fn(spikes)
        spikes_t = spikes.transpose(0, 1).to(device)
        y = y.to(device)
        residual = residual.transpose(0, 1).to(device)
        if physics or flags.get("use_physics_loss"):
            out = model(spikes_t, residual)
            logits, spars = out[0], out[1]
        else:
            out = model(spikes_t)
            if isinstance(out, tuple):
                logits, spars = out[0], out[1]
            else:
                logits, spars = out, torch.tensor(0.0)
        pred = logits.argmax(dim=-1)
        all_preds.append(pred.cpu().numpy())
        all_tgts.append(y.cpu().numpy())
        sparsities.append(spars.item() if torch.is_tensor(spars) else float(spars))
    preds = np.concatenate(all_preds)
    tgts = np.concatenate(all_tgts)
    f1, _ = macro_f1(tgts, preds)
    acc = float((preds == tgts).mean())
    return {"accuracy": acc, "f1": f1, "sparsity": float(np.mean(sparsities))}


def load_model(ckpt_path: str, device: str):
    # torch >=2.6 defaults weights_only=True, which rejects the plain
    # numpy/python scalars in our checkpoint's history list. These are our
    # own trusted checkpoints, so weights_only=False is safe here.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    variant = ckpt.get("variant", "full_phi_spike")
    n_inputs = ckpt["n_inputs"]
    residual_dim = ckpt["residual_dim"]
    model, flags = build_ablation_model(
        variant, n_inputs=n_inputs, residual_dim=residual_dim
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, flags, variant


def run_matrix(
    ckpt_paths: List[str],
    device: str = "cpu",
    quick: bool = False,
    out_dir: str = "experiments/robustness",
    aircraft: str = "c172x",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # BUGFIX: must match the exact dataset config the checkpoints were
    # trained on (n_clean=40, duration_s=80, sequence_len=80, seed=42,
    # quick=False), or the noise=0.00/dropout=0.00/packet=0.00 "clean"
    # baseline evaluates a different population than each checkpoint's
    # reported TEST metrics. Previously quick=True was hardcoded even in
    # the non---quick branch, and the aircraft was never threaded through
    # at all (always silently built c172x regardless of --out-dir).
    if quick:
        ds = build_datasets(
            n_clean=4, n_per_class=1, duration_s=20.0, sequence_len=30,
            seed=42, quick=True, aircraft=aircraft, out_dir=f"datasets/{aircraft}",
        )
    else:
        ds = build_datasets(
            n_clean=40, n_per_class=3, duration_s=80.0, sequence_len=80,
            seed=42, quick=False, aircraft=aircraft, out_dir=f"datasets/{aircraft}",
        )
    loaders = get_loaders(ds, batch_size=32 if not quick else 16)

    noise_levels = [0.0, 0.05, 0.10, 0.20] if quick else [0.0, 0.05, 0.10, 0.20, 0.30]
    dropout_rates = [0.0, 0.10, 0.20] if quick else [0.0, 0.05, 0.10, 0.20, 0.30]
    packet_rates = [0.0, 0.10, 0.20] if quick else [0.0, 0.05, 0.10, 0.20, 0.30]

    matrix = {"noise": {}, "dropout": {}, "packet_loss": {}}
    model_names = []

    for ckpt in ckpt_paths:
        model, flags, variant = load_model(ckpt, device)
        model_names.append(variant)
        print(f"Evaluating {variant} from {ckpt}")

        matrix["noise"][variant] = {}
        for lvl in noise_levels:
            m = evaluate_corrupted(
                model, loaders["test"], device, flags, lambda s, lv=lvl: apply_noise(s, lv)
            )
            matrix["noise"][variant][f"noise_{lvl}"] = m
            print(f"  noise={lvl:.2f}  F1={m['f1']:.3f}")

        matrix["dropout"][variant] = {}
        for r in dropout_rates:
            m = evaluate_corrupted(
                model, loaders["test"], device, flags, lambda s, rr=r: apply_dropout(s, rr)
            )
            matrix["dropout"][variant][f"dropout_{r}"] = m
            print(f"  dropout={r:.2f}  F1={m['f1']:.3f}")

        matrix["packet_loss"][variant] = {}
        for r in packet_rates:
            m = evaluate_corrupted(
                model,
                loaders["test"],
                device,
                flags,
                lambda s, rr=r: apply_packet_loss(s, rr),
            )
            matrix["packet_loss"][variant][f"packet_{r}"] = m
            print(f"  packet={r:.2f}  F1={m['f1']:.3f}")

    out_path = os.path.join(out_dir, "robustness_matrix.json")
    with open(out_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"Wrote {out_path}")

    # Compact F1-only matrix for plotting
    plot_matrix = {}
    for family in ("noise", "dropout", "packet_loss"):
        for variant in matrix[family]:
            for cond, mets in matrix[family][variant].items():
                plot_matrix.setdefault(cond, {})[variant] = mets["f1"]

    try:
        from evaluation.metrics_and_figures import plot_robustness_radar

        plot_robustness_radar(
            plot_matrix, os.path.join(out_dir, "robustness_bars.png")
        )
    except Exception as e:
        print(f"Plot skipped: {e}")

    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts",
        type=str,
        default="",
        help="Comma-separated checkpoint paths. If empty, looks in experiments/full_campaign/{aircraft}/",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", default=None,
                         help="Default: experiments/robustness/{aircraft}")
    parser.add_argument("--aircraft", type=str, default="c172x",
                         help="Must match the aircraft the checkpoints were trained on — "
                              "checkpoint discovery and dataset regeneration both key off this.")
    args = parser.parse_args()

    out_dir = args.out_dir or f"experiments/robustness/{args.aircraft}"

    if args.ckpts:
        ckpts = [p.strip() for p in args.ckpts.split(",")]
    else:
        # BUGFIX: previously always globbed experiments/full_campaign/ flat —
        # since train_campaign.py's --aircraft support writes each aircraft's
        # checkpoints to experiments/full_campaign/{aircraft}/, this silently
        # picked up c172x checkpoints (which still live in the old flat path)
        # for ANY aircraft you asked for, including c310.
        camp = Path("experiments/full_campaign") / args.aircraft
        ckpts = sorted(str(p) for p in camp.glob("*_final.pt"))
        if not ckpts and args.aircraft == "c172x":
            # legacy flat layout fallback, c172x only (pre---aircraft campaigns)
            camp = Path("experiments/full_campaign")
            ckpts = sorted(str(p) for p in camp.glob("*_final.pt"))
        if not ckpts:
            proto = Path("experiments/prototype_baseline")
            ckpts = sorted(str(p) for p in proto.glob("*.pt"))
    if not ckpts:
        print(f"No checkpoints found for aircraft={args.aircraft}. Train first.")
        return
    print(f"[robustness] aircraft={args.aircraft}  found {len(ckpts)} checkpoints  out_dir={out_dir}")
    run_matrix(ckpts, device=args.device, quick=args.quick, out_dir=out_dir, aircraft=args.aircraft)


if __name__ == "__main__":
    main()
