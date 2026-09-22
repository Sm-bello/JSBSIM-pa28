"""
Classical and deep baselines under the same data splits as PHI-SPIKE.
XGBoost, MLP, LSTM, CNN-BiLSTM.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.factory import build_datasets


class LSTMClassifier(nn.Module):
    def __init__(self, n_feat, n_classes=10, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class CNNBiLSTM(nn.Module):
    def __init__(self, n_feat, n_classes=10, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        # x: (B, T, F) → conv expects (B, F, T)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1])


def windows_to_flat(dataset):
    """Convert TrajectoryDataset windows to (N, T*C) for XGBoost and (N, T, C) for nets."""
    Xs, ys = [], []
    for i in range(len(dataset)):
        spikes, y, _ = dataset[i]
        # Use the raw continuous window stored before encoding? 
        # For fairness we use the encoded spikes flattened / sequenced.
        Xs.append(spikes.numpy())
        ys.append(y.item())
    X = np.stack(Xs)  # (N, T, F)
    y = np.array(ys)
    return X, y


def train_xgboost(Xtr, ytr, Xte, yte):
    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed — skipping")
        return {"f1": 0.0, "accuracy": 0.0}
    Xtr_flat = Xtr.reshape(len(Xtr), -1)
    Xte_flat = Xte.reshape(len(Xte), -1)
    clf = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=10,
        n_jobs=2,
        verbosity=0,
    )
    clf.fit(Xtr_flat, ytr)
    pred = clf.predict(Xte_flat)
    return {
        "f1": float(f1_score(yte, pred, average="macro")),
        "accuracy": float(accuracy_score(yte, pred)),
    }


def train_torch_model(model, Xtr, ytr, Xte, yte, epochs=12, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte).float().to(device)).argmax(-1).cpu().numpy()
    return {
        "f1": float(f1_score(yte, pred, average="macro")),
        "accuracy": float(accuracy_score(yte, pred)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default=None,
                         help="Default: experiments/baselines/{aircraft}")
    parser.add_argument("--aircraft", type=str, default="c172x")
    parser.add_argument("--quick", action="store_true",
                         help="Use a small dataset for a fast sanity check, not for reporting")
    args = parser.parse_args()
    save_dir = args.save_dir or f"experiments/baselines/{args.aircraft}"

    print(f"[baselines] Building shared datasets (aircraft={args.aircraft}) …")
    # BUGFIX: defaults previously used a tiny dataset (n_clean=8, duration_s=25,
    # sequence_len=30) — a different population than the full campaign
    # (n_clean=40, duration_s=80, sequence_len=80), making these baselines
    # not a fair comparison against PHI-SPIKE's reported campaign numbers.
    # aircraft was also never threaded through at all (always silently
    # built c172x regardless of intent).
    if args.quick:
        ds = build_datasets(n_clean=8, n_per_class=1, duration_s=25.0, sequence_len=30,
                             seed=42, aircraft=args.aircraft)
    else:
        ds = build_datasets(n_clean=40, n_per_class=3, duration_s=80.0, sequence_len=80,
                             seed=42, aircraft=args.aircraft)
    Xtr, ytr = windows_to_flat(ds["train"])
    Xte, yte = windows_to_flat(ds["test"])
    n_feat = Xtr.shape[-1]
    print(f"  Train windows: {len(Xtr)}, Test: {len(Xte)}, Feat: {n_feat}")

    results = {}
    print("[baselines] XGBoost …")
    results["xgboost"] = train_xgboost(Xtr, ytr, Xte, yte)
    print(f"  → {results['xgboost']}")

    print("[baselines] LSTM …")
    results["lstm"] = train_torch_model(LSTMClassifier(n_feat), Xtr, ytr, Xte, yte)
    print(f"  → {results['lstm']}")

    print("[baselines] CNN-BiLSTM …")
    results["cnn_bilstm"] = train_torch_model(CNNBiLSTM(n_feat), Xtr, ytr, Xte, yte)
    print(f"  → {results['cnn_bilstm']}")

    # BUGFIX: --save-dir was previously accepted by nothing (no argparse
    # existed at all), silently ignored, output always hardcoded to
    # experiments/baselines_metrics.json regardless of what was passed.
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "baselines_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")
    return results


if __name__ == "__main__":
    main()
