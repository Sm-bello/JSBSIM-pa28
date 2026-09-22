"""
Vanilla LIF Spiking Neural Network baseline (pure PyTorch).

Implements a simple multi-layer LIF network with surrogate gradient training.
No physics conditioning — used as the ablation baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_th):
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        return (v >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        # Fast sigmoid surrogate
        alpha = 10.0
        grad = grad_output * alpha * torch.sigmoid(alpha * (v - ctx.v_th)) * (
            1 - torch.sigmoid(alpha * (v - ctx.v_th))
        )
        return grad, None


def spike_fn(v: torch.Tensor, v_th: float = 1.0) -> torch.Tensor:
    return SurrogateSpike.apply(v, v_th)


class LIFLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        v_th: float = 1.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.v_th = v_th
        self.dt = dt
        self.alpha = torch.exp(torch.tensor(-dt / tau_mem))
        self.beta = torch.exp(torch.tensor(-dt / tau_syn))
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self, x_seq: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x_seq: (T, B, n_in). Caller (train_campaign.py) is responsible for
        transposing to this layout before calling -- this layer does NOT
        auto-detect/re-transpose. The previous version tried to guess the
        layout from tensor shape and would silently re-transpose an
        already-correct (T,B,F) input back to (B,T,F) whenever B != n_in,
        which is true for essentially every real batch. That scrambled
        batch and time together during BPTT. Fixed and verified on c182
        this session -- see CHANGES_c182_lif_fix.md / CHANGES_trim_fix_pass.md.
        returns spikes (T, B, n_out)
        """
        T, B, _ = x_seq.shape
        device = x_seq.device
        v = torch.zeros(B, self.fc.out_features, device=device)
        i_syn = torch.zeros(B, self.fc.out_features, device=device)
        spikes = []
        voltages = []

        alpha = self.alpha.to(device)
        beta = self.beta.to(device)

        for t in range(T):
            i_syn = beta * i_syn + self.fc(x_seq[t])
            # Clamp synaptic current and membrane potential every step --
            # verified necessary on c182: gradient-norm clipping alone does
            # not stop v/i_syn from compounding to overflow over many
            # thousands of batches at full training scale. See
            # CHANGES_c182_lif_fix.md for the full verification evidence.
            i_syn = torch.clamp(i_syn, -50.0, 50.0)
            v = alpha * v + i_syn
            v = torch.clamp(v, -50.0, 50.0)
            s = spike_fn(v, self.v_th)
            v = v * (1.0 - s)  # soft reset
            spikes.append(s)
            if return_states:
                voltages.append(v)

        spikes_t = torch.stack(spikes, dim=0)
        if return_states:
            return spikes_t, torch.stack(voltages, dim=0)
        return spikes_t, None


class VanillaSNN(nn.Module):
    def __init__(
        self,
        n_inputs: int = 24,  # hybrid encoding doubles channels
        n_hidden: int = 128,
        n_outputs: int = 10,
        tau_mem: float = 20.0,
        v_th: float = 1.0,
    ):
        super().__init__()
        self.layer1 = LIFLayer(n_inputs, n_hidden, tau_mem=tau_mem, v_th=v_th)
        self.layer2 = LIFLayer(n_hidden, n_hidden // 2, tau_mem=tau_mem, v_th=v_th)
        self.readout = nn.Linear(n_hidden // 2, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_seq: (T, B, F) spike trains
        returns:
            logits (B, n_outputs)  — from mean spike count
            spike_activity (scalar proxy for sparsity)
        """
        s1, _ = self.layer1(x_seq)
        s2, _ = self.layer2(s1)
        # Rate readout
        rate = s2.mean(dim=0)  # (B, H)
        logits = self.readout(rate)
        spike_count = s1.sum() + s2.sum()
        total_possible = s1.numel() + s2.numel()
        sparsity = 1.0 - (spike_count / (total_possible + 1e-8))
        return logits, sparsity

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke_test() -> None:
    model = VanillaSNN(n_inputs=24, n_hidden=64, n_outputs=10)
    T, B, F = 50, 4, 24
    x = (torch.rand(T, B, F) > 0.9).float()
    logits, sparsity = model(x)
    print(f"Logits: {logits.shape}, sparsity: {sparsity.item():.3f}")
    print(f"Parameters: {model.count_parameters()}")


if __name__ == "__main__":
    smoke_test()
