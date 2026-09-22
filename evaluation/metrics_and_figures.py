"""
Publication-quality metrics and figures for PHI-SPIKE paper campaign.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


FAULT_NAMES = [
    "H",
    "SB",
    "SD",
    "SS",
    "SN",
    "AD",
    "EP",
    "CL",
    "FA",
    "VA",
]
FAULT_FULL = [
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
]


def load_metrics(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def plot_confusion_matrix(cm: List[List[int]], out_path: str, title: str = "Confusion Matrix"):
    if not HAS_MPL:
        print("matplotlib not available — skip CM plot")
        return
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(FAULT_NAMES)))
    ax.set_yticks(range(len(FAULT_NAMES)))
    ax.set_xticklabels(FAULT_NAMES)
    ax.set_yticklabels(FAULT_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_training_curves(history: List[Dict], out_path: str, title: str = "Training"):
    if not HAS_MPL:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], color="C0")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, [h.get("val_f1", h.get("val_f1", 0)) for h in history], color="C1")
    axes[1].set_title("Val Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[2].plot(epochs, [h.get("val_sparsity", 0) for h in history], color="C2")
    axes[2].set_title("Val Sparsity")
    axes[2].set_xlabel("Epoch")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_model_comparison(summary: Dict, out_path: str):
    if not HAS_MPL:
        return
    names = list(summary.keys())
    f1_m = [summary[n]["f1_mean"] for n in names]
    f1_s = [summary[n]["f1_std"] for n in names]
    acc_m = [summary[n]["accuracy_mean"] for n in names]
    sp_m = [summary[n]["sparsity_mean"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(names))
    axes[0].bar(x, f1_m, yerr=f1_s, capsize=4, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("F1 (± std across seeds)")
    axes[1].bar(x, acc_m, color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[2].bar(x, sp_m, color="darkorange")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("Sparsity")
    axes[2].set_title("Spike Sparsity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_per_class_f1(per_class: List[Dict], out_path: str, title: str = "Per-class F1"):
    if not HAS_MPL:
        return
    names = [r["class"] for r in per_class]
    f1s = [r["f1"] for r in per_class]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(names)), f1s, color="slateblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_robustness_radar(matrix: Dict, out_path: str):
    """
    matrix: {condition: {model: f1}}
    Simple grouped bar if radar is awkward.
    """
    if not HAS_MPL:
        return
    conditions = list(matrix.keys())
    models = list(next(iter(matrix.values())).keys()) if conditions else []
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [matrix[c].get(m, 0) for c in conditions]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Robustness matrix")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_trajectory_and_residual(
    traj: np.ndarray,
    residual: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    channel_names: Optional[List[str]] = None,
):
    if not HAS_MPL:
        return
    channel_names = channel_names or [f"ch{i}" for i in range(traj.shape[1])]
    T = traj.shape[0]
    t = np.arange(T)
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, height_ratios=[2, 2, 1])
    ax0 = fig.add_subplot(gs[0])
    for i in [0, 5, 10]:  # airspeed, rpm, egt
        ax0.plot(t, traj[:, i], label=channel_names[i], alpha=0.85)
    ax0.set_ylabel("Telemetry")
    ax0.legend(fontsize=7, ncol=3)
    ax0.set_title("State trajectories")

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    r_norm = np.linalg.norm(residual, axis=1)
    # residual is T-1
    ax1.plot(np.arange(len(r_norm)), r_norm, color="crimson", label="||r||")
    ax1.set_ylabel("Residual norm")
    ax1.legend()
    ax1.set_title("Physics residual evolution")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.plot(t, labels, color="black", drawstyle="steps-post")
    ax2.set_ylabel("Fault label")
    ax2.set_xlabel("Timestep")
    ax2.set_title("Ground-truth fault onset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_spike_raster(
    spikes: np.ndarray,
    voltages: Optional[np.ndarray],
    out_path: str,
    title: str = "Spike activity",
):
    """
    spikes: (T, N) binary
    voltages: (T, N) optional
    """
    if not HAS_MPL:
        return
    T, N = spikes.shape
    fig, axes = plt.subplots(2 if voltages is not None else 1, 1, figsize=(11, 5), sharex=True)
    if voltages is None:
        axes = [axes]
    # raster
    t_idx, n_idx = np.where(spikes > 0.5)
    axes[0].scatter(t_idx, n_idx, s=1, c="k", marker="|")
    axes[0].set_ylabel("Neuron")
    axes[0].set_title(title + " — raster")
    if voltages is not None:
        # mean membrane
        axes[1].plot(voltages.mean(axis=1), color="C0")
        axes[1].set_ylabel("Mean V")
        axes[1].set_xlabel("Timestep")
        axes[1].set_title("Mean membrane voltage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def generate_from_metrics_dir(metrics_dir: str, figure_dir: str):
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(metrics_dir)
    # Prototype
    proto = metrics_dir.parent / "prototype_baseline"
    if (proto / "phi_spike_metrics.json").exists():
        m = load_metrics(str(proto / "phi_spike_metrics.json"))
        if "history" in m:
            plot_training_curves(
                m["history"],
                str(Path(figure_dir) / "prototype_phi_spike_curves.png"),
                title="Prototype PHI-SPIKE (30 epochs)",
            )
    # Campaign summary
    summary_path = metrics_dir / "campaign_summary.json"
    if summary_path.exists():
        summary = load_metrics(str(summary_path))
        # train_campaign.py wraps per-variant stats under "results" (added
        # alongside a "failed_runs" list for crash-isolation reporting).
        # Support both shapes so this works regardless of which
        # train_campaign.py wrote the file.
        if "results" in summary:
            failed = summary.get("failed_runs", [])
            if failed:
                print(f"[figures] Note: campaign had failed runs, excluded from comparison: {failed}")
            summary = summary["results"]
        plot_model_comparison(summary, str(Path(figure_dir) / "model_comparison.png"))
    # Individual seed metrics
    for p in sorted(metrics_dir.glob("*_metrics.json")):
        m = load_metrics(str(p))
        stem = p.stem.replace("_metrics", "")
        if "history" in m:
            plot_training_curves(
                m["history"], str(Path(figure_dir) / f"{stem}_curves.png"), title=stem
            )
        if "confusion_matrix" in m:
            plot_confusion_matrix(
                m["confusion_matrix"],
                str(Path(figure_dir) / f"{stem}_cm.png"),
                title=f"CM — {stem}",
            )
        if "per_class" in m:
            plot_per_class_f1(
                m["per_class"],
                str(Path(figure_dir) / f"{stem}_per_class_f1.png"),
                title=f"Per-class F1 — {stem}",
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="experiments/full_campaign")
    parser.add_argument("--figure-dir", default="experiments/figures")
    args = parser.parse_args()
    generate_from_metrics_dir(args.metrics_dir, args.figure_dir)
    print("Figure generation complete.")


if __name__ == "__main__":
    main()
