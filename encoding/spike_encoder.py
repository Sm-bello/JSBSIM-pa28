"""
Spike encoding layer for continuous aircraft telemetry.

Supports rate, temporal (latency), delta, and hybrid population coding.
All encoders operate in a streaming-friendly fashion (batch size 1 ready).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class SpikeEncoder:
    def __init__(
        self,
        method: str = "hybrid",
        rate_max_hz: float = 200.0,
        dt: float = 0.02,
        n_population: int = 8,
        feature_min: Optional[np.ndarray] = None,
        feature_max: Optional[np.ndarray] = None,
    ):
        self.method = method
        self.rate_max_hz = rate_max_hz
        self.dt = dt
        self.n_population = n_population
        self.feature_min = feature_min
        self.feature_max = feature_max

    def fit_normalisation(self, data: np.ndarray) -> None:
        """data: (N, T, C) or (T, C)"""
        if data.ndim == 3:
            flat = data.reshape(-1, data.shape[-1])
        else:
            flat = data
        self.feature_min = flat.min(axis=0)
        self.feature_max = flat.max(axis=0)
        # Avoid zero range
        rng = self.feature_max - self.feature_min
        rng[rng < 1e-6] = 1.0
        self.feature_max = self.feature_min + rng

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        if self.feature_min is None:
            return x
        return (x - self.feature_min) / (self.feature_max - self.feature_min + 1e-8)

    def encode_rate(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, C) normalised [0,1]
        returns binary spikes (T, C) via Poisson rate coding
        """
        rates = np.clip(x, 0, 1) * self.rate_max_hz
        p = rates * self.dt
        return (np.random.rand(*x.shape) < p).astype(np.float32)

    def encode_temporal(self, x: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Latency coding: higher value → earlier spike inside a short window.
        Returns (T, C) binary.
        """
        T, C = x.shape
        spikes = np.zeros((T, C), dtype=np.float32)
        for t in range(0, T, window):
            end = min(t + window, T)
            seg = x[t:end]
            # rank within window → first spike time
            for c in range(C):
                vals = seg[:, c]
                if vals.size == 0:
                    continue
                order = np.argsort(-vals)  # high first
                spike_t = t + int(order[0] * (end - t - 1) / max(len(order), 1))
                if spike_t < T:
                    spikes[spike_t, c] = 1.0
        return spikes

    def encode_delta(self, x: np.ndarray, threshold: float = 0.05) -> np.ndarray:
        """
        Delta / change-based encoding: spike when |Δx| exceeds threshold.
        """
        dx = np.diff(x, axis=0, prepend=x[:1])
        return (np.abs(dx) > threshold).astype(np.float32)

    def encode_population(self, x: np.ndarray) -> np.ndarray:
        """
        Population / place coding: each channel expanded to n_population neurons
        with Gaussian receptive fields.
        Returns (T, C * n_population)
        """
        T, C = x.shape
        centers = np.linspace(0, 1, self.n_population)
        sigma = 0.15
        out = np.zeros((T, C * self.n_population), dtype=np.float32)
        for c in range(C):
            for k, mu in enumerate(centers):
                resp = np.exp(-0.5 * ((x[:, c] - mu) / sigma) ** 2)
                # Poisson spike from response
                p = resp * self.rate_max_hz * self.dt * 0.5
                out[:, c * self.n_population + k] = (np.random.rand(T) < p).astype(np.float32)
        return out

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, C) raw or already normalised
        returns spike tensor suitable for SNN input
        """
        xn = self._normalise(x)
        if self.method == "rate":
            return self.encode_rate(xn)
        if self.method == "temporal":
            return self.encode_temporal(xn)
        if self.method == "delta":
            return self.encode_delta(xn)
        if self.method == "population":
            return self.encode_population(xn)
        # hybrid: rate + delta concatenated
        r = self.encode_rate(xn)
        d = self.encode_delta(xn)
        return np.concatenate([r, d], axis=-1)

    def encode_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience for torch tensors (CPU numpy path)."""
        arr = x.detach().cpu().numpy()
        spikes = self.encode(arr)
        return torch.from_numpy(spikes).to(x.device)


def smoke_test() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 12)).astype(np.float32)
    enc = SpikeEncoder(method="hybrid")
    enc.fit_normalisation(x)
    s = enc.encode(x)
    print(f"Encoded shape: {s.shape}, sparsity: {1 - s.mean():.3f}")


if __name__ == "__main__":
    smoke_test()
