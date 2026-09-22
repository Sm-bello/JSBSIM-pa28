"""
Full campaign trainer for PHI-SPIKE.

- Multi-seed training with confidence intervals
- Ablation variants
- Preserves prototype baseline under experiments/prototype_baseline/
- Writes JSON metrics + checkpoints under experiments/full_campaign/ and experiments/ablations/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Tee:
    """Duplicate writes to both the real stream and a log file, so every
    tmux/console print (including tqdm progress, which writes to stderr)
    is preserved on disk even if the terminal session is lost."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.factory import build_datasets, get_loaders
from models.phi_spike.phi_snn import (
    PHISPIKE,
    physics_informed_loss,
    build_ablation_model,
)
from models.vanilla_snn.lif_snn import VanillaSNN


def macro_f1(y_true, y_pred, n_classes=10):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s)), f1s


def per_class_metrics(y_true, y_pred, n_classes=10, class_names=None):
    rows = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        support = np.sum(y_true == c)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        name = class_names[c] if class_names else str(c)
        rows.append(
            {
                "class": name,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": int(support),
            }
        )
    return rows


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def evaluate(model, loader, device, flags: Dict):
    model.eval()
    correct, total = 0, 0
    all_preds, all_tgts = [], []
    sparsities = []
    physics = flags.get("physics_conditioning", False)
    with torch.no_grad():
        for spikes, y, residual in loader:
            spikes = spikes.transpose(0, 1).to(device)
            y = y.to(device)
            residual = residual.transpose(0, 1).to(device)
            if physics:
                out = model(spikes, residual)
                logits, spars = out[0], out[1]
            else:
                # VanillaSNN or non-conditioning
                if hasattr(model, "physics_conditioning") and not model.physics_conditioning:
                    out = model(spikes, residual)
                    logits, spars = out[0], out[1]
                else:
                    out = model(spikes)
                    if isinstance(out, tuple):
                        logits, spars = out[0], out[1]
                    else:
                        logits, spars = out, torch.tensor(0.0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(y.cpu().numpy())
            sparsities.append(spars.item() if torch.is_tensor(spars) else float(spars))
    acc = correct / max(total, 1)
    preds = np.concatenate(all_preds)
    tgts = np.concatenate(all_tgts)
    f1, f1_per = macro_f1(tgts, preds)
    cm = confusion_matrix(tgts, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "f1_per_class": f1_per,
        "sparsity": float(np.mean(sparsities)),
        "confusion_matrix": cm.tolist(),
        "preds": preds,
        "targets": tgts,
    }


def train_one(
    variant: str,
    loaders,
    device: str,
    epochs: int = 40,
    lr: float = 1e-3,
    save_dir: str = "experiments/full_campaign",
    seed: int = 0,
):
    tag = f"{variant}_seed{seed}"
    run_dir = Path(save_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tee all console output (including tqdm's stderr progress bars) into
    # this run's own log.txt, so a lost tmux session never loses the run
    # history. Restored in the `finally` block below no matter how the
    # run ends (including a crash caught by _run_safely in main()).
    log_f = open(run_dir / "log.txt", "a")
    log_f.write(f"\n===== run started {datetime.now().isoformat()} =====\n")
    log_f.flush()
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_f)
    sys.stderr = Tee(orig_stderr, log_f)

    csv_path = run_dir / "history.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(
        ["epoch", "train_loss", "ce", "physics", "temporal", "sparsity_pen",
         "val_accuracy", "val_f1", "val_sparsity", "nan_batches_skipped"]
    )

    try:
        return _train_one_inner(
            variant, loaders, device, epochs, lr, save_dir, seed,
            tag, run_dir, csv_f, csv_writer,
        )
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_f.close()


def _train_one_inner(
    variant, loaders, device, epochs, lr, save_dir, seed,
    tag, run_dir, csv_f, csv_writer,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    sample_spikes, _, sample_r = next(iter(loaders["train"]))
    n_inputs = sample_spikes.shape[-1]
    residual_dim = sample_r.shape[-1]
    n_outputs = 10

    model, flags = build_ablation_model(
        variant, n_inputs=n_inputs, n_hidden=128, n_outputs=n_outputs, residual_dim=residual_dim
    )
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        parts_accum = {"ce": [], "physics": [], "temporal": [], "sparsity_pen": []}
        nan_skipped = 0
        for spikes, y, residual in tqdm(
            loaders["train"], desc=f"{variant} s{seed} ep{ep}", leave=False
        ):
            spikes = spikes.transpose(0, 1).to(device)
            y = y.to(device)
            residual = residual.transpose(0, 1).to(device)
            opt.zero_grad()

            if flags["physics_conditioning"] or flags["use_physics_loss"]:
                out = model(spikes, residual)
                logits, spars, r_pred = out[0], out[1], out[2]
                temporal_pred = out[3] if len(out) > 3 else None
                r_true = residual.mean(dim=0)
                loss, parts = physics_informed_loss(
                    logits,
                    y,
                    r_pred,
                    r_true,
                    spars,
                    temporal_pred=temporal_pred,
                    residual_seq=residual,
                    use_physics=flags["use_physics_loss"],
                    use_temporal=flags["use_temporal"],
                )
            else:
                # pure vanilla
                out = model(spikes)
                if isinstance(out, tuple):
                    logits, spars = out[0], out[1]
                else:
                    logits, spars = out, torch.tensor(0.0)
                loss = nn.functional.cross_entropy(logits, y)
                parts = {"ce": loss.item(), "total": loss.item()}

            # NaN/inf guard: skip the optimizer step for this batch rather
            # than letting a corrupted gradient permanently poison the
            # model weights for the rest of the run. physics_informed_loss
            # already falls back to CE-only internally, so this is a
            # second, independent safety net at the training-loop level.
            if not torch.isfinite(loss):
                nan_skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                nan_skipped += 1
                opt.zero_grad()
                continue
            opt.step()
            losses.append(parts.get("total", loss.item()))
            for k in parts_accum:
                if k in parts:
                    parts_accum[k].append(parts[k])

        val_metrics = evaluate(model, loaders["val"], device, flags)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        row = {
            "epoch": ep,
            "train_loss": mean_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_sparsity": val_metrics["sparsity"],
            "nan_batches_skipped": nan_skipped,
        }
        history.append(row)
        print(
            f"[{variant}|seed{seed}] ep {ep:02d}  loss={mean_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.3f}  val_f1={val_metrics['f1']:.3f}  "
            f"sparsity={val_metrics['sparsity']:.3f}"
            + (f"  [skipped {nan_skipped} non-finite batches]" if nan_skipped else "")
        )
        csv_writer.writerow(
            [
                ep,
                mean_loss,
                float(np.mean(parts_accum["ce"])) if parts_accum["ce"] else "",
                float(np.mean(parts_accum["physics"])) if parts_accum["physics"] else "",
                float(np.mean(parts_accum["temporal"])) if parts_accum["temporal"] else "",
                float(np.mean(parts_accum["sparsity_pen"])) if parts_accum["sparsity_pen"] else "",
                val_metrics["accuracy"],
                val_metrics["f1"],
                val_metrics["sparsity"],
                nan_skipped,
            ]
        )
        csv_f.flush()

    test_metrics = evaluate(model, loaders["test"], device, flags)
    print(
        f"[{variant}|seed{seed}] TEST  acc={test_metrics['accuracy']:.3f}  "
        f"f1={test_metrics['f1']:.3f}  sparsity={test_metrics['sparsity']:.3f}"
    )
    csv_f.close()

    ckpt = {
        "variant": variant,
        "seed": seed,
        "state_dict": model.state_dict(),
        "history": history,
        "test_metrics": {
            k: v
            for k, v in test_metrics.items()
            if k not in ("preds", "targets")
        },
        "flags": flags,
        "n_inputs": n_inputs,
        "residual_dim": residual_dim,
    }
    # Everything for this run lives in its own folder so nothing gets
    # overwritten or lost across variants/seeds: run_dir = save_dir/tag/
    path = run_dir / f"{tag}_final.pt"
    torch.save(ckpt, path)
    # Keep a flat-namespace copy too, for scripts/notebooks that expect the
    # old {save_dir}/{tag}_final.pt layout.
    torch.save(ckpt, os.path.join(save_dir, f"{tag}_final.pt"))
    metrics_out = {
        "history": history,
        "test": {k: v for k, v in test_metrics.items() if k not in ("preds", "targets")},
        "per_class": per_class_metrics(
            test_metrics["targets"],
            test_metrics["preds"],
            class_names=[
                "healthy",
                "sensor_bias",
                "sensor_drift",
                "sensor_stuck",
                "sensor_noise",
                "actuator_degradation",
                "engine_performance_loss",
                "control_surface_lag",
                "fuel_flow_anomaly",
                "vibration_anomaly",
            ],
        ),
        "confusion_matrix": test_metrics["confusion_matrix"],
    }
    with open(run_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    with open(os.path.join(save_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved {path}")
    print(f"Run folder: {run_dir}  (log.txt, history.csv, checkpoint, metrics.json)")
    return test_metrics, metrics_out


def aggregate_seeds(results: List[Dict]) -> Dict:
    accs = [r["accuracy"] for r in results]
    f1s = [r["f1"] for r in results]
    spars = [r["sparsity"] for r in results]
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "sparsity_mean": float(np.mean(spars)),
        "sparsity_std": float(np.std(spars)),
        "n_seeds": len(results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--variants",
        type=str,
        default="full_phi_spike",
        help="Comma-separated: vanilla_snn,snn_physics_loss,snn_membrane_conditioning,snn_temporal_physics,full_phi_spike",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--n-clean", type=int, default=40)
    parser.add_argument("--sequence-len", type=int, default=80)
    parser.add_argument("--save-dir", type=str, default=None,
                         help="Default: experiments/full_campaign/{aircraft}")
    parser.add_argument("--aircraft", type=str, default="c172x",
                         help="JSBSim aircraft model name, e.g. c172x, f16, c310. "
                              "aircraft_sim.py raises rather than silently substituting "
                              "if the model fails to load.")
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = f"experiments/full_campaign/{args.aircraft}"

    device = args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    campaign_log = open(os.path.join(args.save_dir, "campaign_log.txt"), "a")
    campaign_log.write(f"\n===== campaign started {datetime.now().isoformat()} =====\n")
    campaign_log.write(f"args: {vars(args)}\n")
    campaign_log.flush()
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, campaign_log)
    sys.stderr = Tee(orig_stderr, campaign_log)

    try:
        _main_body(args, seeds, variants, device, campaign_log, orig_stdout, orig_stderr)
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        campaign_log.close()


def _main_body(args, seeds, variants, device, campaign_log, orig_stdout, orig_stderr):
    print("[campaign] Building datasets …")
    ds = build_datasets(
        n_clean=args.n_clean,
        n_per_class=2 if args.quick else 3,
        duration_s=25.0 if args.quick else 80.0,
        sequence_len=40 if args.quick else args.sequence_len,
        seed=42,
        quick=args.quick,
        aircraft=args.aircraft,
        out_dir=f"datasets/{args.aircraft}",
    )
    loaders = get_loaders(ds, batch_size=args.batch_size)

    summary = {}
    failed_runs = []
    for variant in variants:
        seed_results = []
        for seed in seeds:
            # Crash isolation: one (variant, seed) run hitting an
            # unexpected exception (OOM, bad batch, etc.) must not take
            # down a multi-hour campaign. Log the traceback into that
            # run's own folder and move on to the next seed/variant.
            try:
                metrics, _ = train_one(
                    variant,
                    loaders,
                    device,
                    epochs=args.epochs if not args.quick else min(args.epochs, 8),
                    save_dir=args.save_dir,
                    seed=seed,
                )
                seed_results.append(metrics)
            except Exception:
                tb = traceback.format_exc()
                print(f"[campaign] !!! {variant} seed{seed} CRASHED, skipping. See crash.txt.")
                print(tb)
                crash_dir = Path(args.save_dir) / f"{variant}_seed{seed}"
                crash_dir.mkdir(parents=True, exist_ok=True)
                with open(crash_dir / "crash.txt", "a") as cf:
                    cf.write(f"\n===== crash {datetime.now().isoformat()} =====\n{tb}\n")
                failed_runs.append(f"{variant}_seed{seed}")
        if seed_results:
            summary[variant] = aggregate_seeds(seed_results)
            print(f"\n=== {variant} (n={len(seed_results)}/{len(seeds)} seeds succeeded) ===")
            print(
                f"  F1  {summary[variant]['f1_mean']:.3f} ± {summary[variant]['f1_std']:.3f}"
            )
            print(
                f"  Acc {summary[variant]['accuracy_mean']:.3f} ± {summary[variant]['accuracy_std']:.3f}"
            )
            print(
                f"  Sp  {summary[variant]['sparsity_mean']:.3f} ± {summary[variant]['sparsity_std']:.3f}"
            )
        else:
            print(f"\n=== {variant}: ALL seeds crashed, no results ===")

    with open(os.path.join(args.save_dir, "campaign_summary.json"), "w") as f:
        json.dump({"results": summary, "failed_runs": failed_runs}, f, indent=2)
    print(f"\nCampaign summary written to {args.save_dir}/campaign_summary.json")
    if failed_runs:
        print(f"Failed runs (see each run's crash.txt): {failed_runs}")
    print("Prototype baseline (original 30-epoch runs) is preserved under experiments/prototype_baseline/")


if __name__ == "__main__":
    main()
