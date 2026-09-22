"""
Train PHI-SPIKE and vanilla SNN baselines under identical conditions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.factory import build_datasets, get_loaders
from models.phi_spike.phi_snn import PHISPIKE, physics_informed_loss
from models.vanilla_snn.lif_snn import VanillaSNN


def evaluate(model, loader, device, physics: bool = True):
    model.eval()
    correct, total = 0, 0
    all_preds, all_tgts = [], []
    sparsities = []
    with torch.no_grad():
        for spikes, y, residual in loader:
            # spikes: (B, T, F) → (T, B, F)
            spikes = spikes.transpose(0, 1).to(device)
            y = y.to(device)
            residual = residual.transpose(0, 1).to(device)  # (T, B, D)
            if physics:
                # BUGFIX: PHISPIKE.forward() always returns 4 values with
                # return_states=False: (logits, sparsity, residual_pred,
                # temporal_pred), never 3.
                logits, spars, _residual_pred, _temporal_pred = model(spikes, residual)
            else:
                logits, spars = model(spikes)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(y.cpu().numpy())
            sparsities.append(spars.item() if torch.is_tensor(spars) else float(spars))
    acc = correct / max(total, 1)
    preds = np.concatenate(all_preds)
    tgts = np.concatenate(all_tgts)
    # Macro F1
    f1 = macro_f1(tgts, preds)
    return {"accuracy": acc, "f1": f1, "sparsity": float(np.mean(sparsities))}


def macro_f1(y_true, y_pred, n_classes=10):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s))


def train_one(
    model_name: str,
    loaders,
    device: str,
    epochs: int = 15,
    lr: float = 1e-3,
    save_dir: str = "experiments",
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Infer dims from first batch
    sample_spikes, _, sample_r = next(iter(loaders["train"]))
    n_inputs = sample_spikes.shape[-1]
    residual_dim = sample_r.shape[-1]
    n_outputs = 10

    if model_name == "phi_spike":
        model = PHISPIKE(
            n_inputs=n_inputs,
            n_hidden=128,
            n_outputs=n_outputs,
            residual_dim=residual_dim,
            alpha_physics=0.2,
            physics_conditioning=True,
        ).to(device)
        physics = True
    else:
        model = VanillaSNN(n_inputs=n_inputs, n_hidden=128, n_outputs=n_outputs).to(device)
        physics = False

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for spikes, y, residual in tqdm(loaders["train"], desc=f"{model_name} ep{ep}", leave=False):
            spikes = spikes.transpose(0, 1).to(device)
            y = y.to(device)
            residual = residual.transpose(0, 1).to(device)
            opt.zero_grad()
            if physics:
                # BUGFIX: PHISPIKE.forward() always returns 4 values with
                # return_states=False: (logits, sparsity, residual_pred,
                # temporal_pred), never 3.
                logits, spars, r_pred, _temporal_pred = model(spikes, residual)
                # mean residual target
                r_true = residual.mean(dim=0)
                loss, parts = physics_informed_loss(logits, y, r_pred, r_true, spars)
            else:
                logits, spars = model(spikes)
                loss = nn.functional.cross_entropy(logits, y)
                parts = {"ce": loss.item(), "total": loss.item()}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(parts["total"])

        val_metrics = evaluate(model, loaders["val"], device, physics=physics)
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[{model_name}] ep {ep:02d}  loss={row['train_loss']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.3f}  val_f1={val_metrics['f1']:.3f}  "
            f"sparsity={val_metrics['sparsity']:.3f}"
        )

    test_metrics = evaluate(model, loaders["test"], device, physics=physics)
    print(f"[{model_name}] TEST  acc={test_metrics['accuracy']:.3f}  f1={test_metrics['f1']:.3f}")

    ckpt = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "history": history,
        "test_metrics": test_metrics,
        "n_inputs": n_inputs,
        "residual_dim": residual_dim,
    }
    path = os.path.join(save_dir, f"{model_name}_final.pt")
    torch.save(ckpt, path)
    with open(os.path.join(save_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump({"history": history, "test": test_metrics}, f, indent=2)
    print(f"Saved {path}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--quick", action="store_true", help="Smaller data for smoke run")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if args.quick:
        ds = build_datasets(n_clean=4, n_per_class=1, duration_s=15.0, sequence_len=30, seed=0)
    else:
        ds = build_datasets(n_clean=10, n_per_class=2, duration_s=30.0, sequence_len=40, seed=42)

    loaders = get_loaders(ds, batch_size=args.batch_size)

    results = {}
    for name in ["vanilla_snn", "phi_spike"]:
        results[name] = train_one(name, loaders, device, epochs=args.epochs)

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"  {k}: F1={v['f1']:.3f}  Acc={v['accuracy']:.3f}  Sparsity={v['sparsity']:.3f}")


if __name__ == "__main__":
    main()
