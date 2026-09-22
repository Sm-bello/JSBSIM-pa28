"""
PHI-SPIKE: Physics-Informed Spiking Neural Network (campaign-grade).

Key upgrades:
  - Learnable physics gate g_t = sigma(W_g [h_t, r_t])
  - Normalized residual pathway
  - Ablation-friendly flags: physics_loss, membrane_conditioning, temporal
  - Cleaner loss interface with temporal residual consistency option
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn

from models.vanilla_snn.lif_snn import LIFLayer, SurrogateSpike, spike_fn


class PhysicsConditionedLIF(nn.Module):
    """
    LIF layer with optional learnable physics gating on the residual pathway.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        v_th: float = 1.0,
        dt: float = 1.0,
        alpha_physics: float = 0.2,
        residual_dim: int = 12,
        learnable_gate: bool = True,
    ):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.residual_proj = nn.Linear(residual_dim, n_out)
        self.tau_mem = tau_mem
        self.v_th = v_th
        self.dt = dt
        self.alpha_physics = alpha_physics
        self.learnable_gate = learnable_gate
        self.alpha = torch.exp(torch.tensor(-dt / tau_mem))
        self.beta = torch.exp(torch.tensor(-dt / tau_syn))
        if learnable_gate:
            # gate from [current synaptic input proxy, residual]
            self.gate = nn.Sequential(
                nn.Linear(n_out + residual_dim, n_out),
                nn.Sigmoid(),
            )
            nn.init.xavier_uniform_(self.gate[0].weight, gain=0.1)
            nn.init.zeros_(self.gate[0].bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.residual_proj.weight, gain=0.1)
        nn.init.zeros_(self.residual_proj.bias)

    def forward(
        self,
        x_seq: torch.Tensor,
        residual_seq: Optional[torch.Tensor] = None,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x_seq.dim() == 3 and x_seq.shape[2] == self.fc.in_features:
            pass
        elif x_seq.dim() == 3 and x_seq.shape[1] == self.fc.in_features:
            x_seq = x_seq.transpose(0, 1)

        T, B, _ = x_seq.shape
        device = x_seq.device
        v = torch.zeros(B, self.fc.out_features, device=device)
        i_syn = torch.zeros(B, self.fc.out_features, device=device)
        spikes = []
        voltages = []

        alpha = self.alpha.to(device)
        beta = self.beta.to(device)

        if residual_seq is not None:
            if residual_seq.shape[0] == T - 1:
                pad = torch.zeros(1, B, residual_seq.shape[-1], device=device)
                residual_seq = torch.cat([pad, residual_seq], dim=0)
            residual_seq = residual_seq.to(device)

        for t in range(T):
            i_syn = beta * i_syn + self.fc(x_seq[t])
            i_syn = torch.clamp(i_syn, -50.0, 50.0)
            phys = 0.0
            if residual_seq is not None:
                r_t = torch.clamp(residual_seq[t], -50.0, 50.0)
                r_proj = self.residual_proj(r_t)
                if self.learnable_gate:
                    gate_in = torch.cat([i_syn.detach(), r_t], dim=-1)
                    g = self.gate(gate_in)
                    phys = g * r_proj
                else:
                    phys = self.alpha_physics * r_proj
            v = alpha * v + i_syn + phys
            v = torch.clamp(v, -50.0, 50.0)
            s = spike_fn(v, self.v_th)
            v = v * (1.0 - s)
            spikes.append(s)
            if return_states:
                voltages.append(v.clone())

        spikes_t = torch.stack(spikes, dim=0)
        if return_states:
            return spikes_t, torch.stack(voltages, dim=0)
        return spikes_t, None


class PHISPIKE(nn.Module):
    """
    Full PHI-SPIKE network with ablation flags.
    """

    def __init__(
        self,
        n_inputs: int = 24,
        n_hidden: int = 128,
        n_outputs: int = 10,
        residual_dim: int = 12,
        tau_mem: float = 20.0,
        v_th: float = 1.0,
        alpha_physics: float = 0.2,
        physics_conditioning: bool = True,
        learnable_gate: bool = True,
    ):
        super().__init__()
        self.physics_conditioning = physics_conditioning
        self.residual_dim = residual_dim

        if physics_conditioning:
            self.layer1 = PhysicsConditionedLIF(
                n_inputs,
                n_hidden,
                tau_mem=tau_mem,
                v_th=v_th,
                alpha_physics=alpha_physics,
                residual_dim=residual_dim,
                learnable_gate=learnable_gate,
            )
            self.layer2 = PhysicsConditionedLIF(
                n_hidden,
                n_hidden // 2,
                tau_mem=tau_mem,
                v_th=v_th,
                alpha_physics=alpha_physics * 0.5,
                residual_dim=residual_dim,
                learnable_gate=learnable_gate,
            )
        else:
            self.layer1 = LIFLayer(n_inputs, n_hidden, tau_mem=tau_mem, v_th=v_th)
            self.layer2 = LIFLayer(n_hidden, n_hidden // 2, tau_mem=tau_mem, v_th=v_th)

        self.readout = nn.Linear(n_hidden // 2, n_outputs)
        self.residual_head = nn.Linear(n_hidden // 2, residual_dim)
        # Temporal residual head (predict residual norm sequence mean)
        self.temporal_head = nn.Linear(n_hidden // 2, 1)
        self.n_outputs = n_outputs

    def forward(
        self,
        x_seq: torch.Tensor,
        residual_seq: Optional[torch.Tensor] = None,
        return_states: bool = False,
    ):
        if self.physics_conditioning and residual_seq is not None:
            s1, v1 = self.layer1(x_seq, residual_seq, return_states=return_states)
            s2, v2 = self.layer2(s1, residual_seq, return_states=return_states)
        else:
            s1, v1 = self.layer1(x_seq, return_states=return_states)
            s2, v2 = self.layer2(s1, return_states=return_states)

        rate = s2.mean(dim=0)
        logits = self.readout(rate)
        residual_pred = self.residual_head(rate)
        temporal_pred = self.temporal_head(rate).squeeze(-1)

        spike_count = s1.sum() + s2.sum()
        total = s1.numel() + s2.numel()
        sparsity = 1.0 - (spike_count / (total + 1e-8))

        if return_states:
            return logits, sparsity, residual_pred, temporal_pred, s1, s2, v1, v2
        return logits, sparsity, residual_pred, temporal_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def physics_informed_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    residual_pred: torch.Tensor,
    residual_true: torch.Tensor,
    sparsity: torch.Tensor,
    temporal_pred: Optional[torch.Tensor] = None,
    residual_seq: Optional[torch.Tensor] = None,
    lambda_p: float = 0.25,
    lambda_s: float = 0.05,
    lambda_t: float = 0.10,
    use_physics: bool = True,
    use_temporal: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    Composite loss with optional physics and temporal terms.
    """
    ce = nn.functional.cross_entropy(logits, targets)
    parts = {"ce": ce.item()}
    total = ce

    if use_physics and residual_true is not None:
        if residual_true.dim() == 3:
            residual_true = residual_true.mean(dim=0)
        # SAFETY: with tiny/smoke datasets, PhysicsResidualEngine.fit_scales
        # can still see rare per-channel outliers even after the residual.py
        # floor fix. Clamp targets and use a Huber (SmoothL1) loss instead of
        # raw MSE so a handful of extreme residual values can't square into
        # an overflow/NaN loss and blow up training. This does NOT change
        # behaviour on well-scaled data (Huber ~= MSE for small errors).
        residual_true = torch.nan_to_num(residual_true, nan=0.0, posinf=0.0, neginf=0.0)
        residual_true = torch.clamp(residual_true, -50.0, 50.0)
        phys = nn.functional.smooth_l1_loss(residual_pred, residual_true, beta=1.0)
        total = total + lambda_p * phys
        parts["physics"] = phys.item()
    else:
        parts["physics"] = 0.0

    if use_temporal and temporal_pred is not None and residual_seq is not None:
        # residual_seq: (T, B, D) → per-sample mean residual norm
        r_norm = residual_seq.norm(dim=-1).mean(dim=0)  # (B,)
        temp = nn.functional.mse_loss(temporal_pred, r_norm)
        total = total + lambda_t * temp
        parts["temporal"] = temp.item()
    else:
        parts["temporal"] = 0.0

    sparse_pen = 1.0 - sparsity
    total = total + lambda_s * sparse_pen
    parts["sparsity_pen"] = sparse_pen.item() if torch.is_tensor(sparse_pen) else float(sparse_pen)

    if not torch.isfinite(total):
        # Last-resort guard: fall back to CE alone rather than propagating
        # NaN/inf into backward() and permanently corrupting the model
        # weights for the rest of the run.
        total = ce
        parts["nan_guard_triggered"] = True
    parts["total"] = total.item()
    return total, parts


def build_ablation_model(
    variant: str,
    n_inputs: int,
    n_hidden: int = 128,
    n_outputs: int = 10,
    residual_dim: int = 12,
) -> Tuple[nn.Module, Dict]:
    """
    Factory for ablation variants.
    Returns model and flags dict used by training loop.
    """
    flags = {
        "physics_conditioning": False,
        "learnable_gate": False,
        "use_physics_loss": False,
        "use_temporal": False,
        "name": variant,
    }
    if variant == "vanilla_snn":
        from models.vanilla_snn.lif_snn import VanillaSNN
        model = VanillaSNN(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)
        return model, flags

    if variant == "snn_physics_loss":
        # conditioning off, but physics loss on
        model = PHISPIKE(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=n_outputs,
            residual_dim=residual_dim,
            physics_conditioning=False,
            learnable_gate=False,
        )
        flags["use_physics_loss"] = True
        return model, flags

    if variant == "snn_membrane_conditioning":
        model = PHISPIKE(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=n_outputs,
            residual_dim=residual_dim,
            physics_conditioning=True,
            learnable_gate=False,
        )
        flags["physics_conditioning"] = True
        flags["use_physics_loss"] = True
        return model, flags

    if variant == "snn_temporal_physics":
        model = PHISPIKE(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=n_outputs,
            residual_dim=residual_dim,
            physics_conditioning=True,
            learnable_gate=False,
        )
        flags["physics_conditioning"] = True
        flags["use_physics_loss"] = True
        flags["use_temporal"] = True
        return model, flags

    # full_phi_spike
    model = PHISPIKE(
        n_inputs=n_inputs,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        residual_dim=residual_dim,
        physics_conditioning=True,
        learnable_gate=True,
    )
    flags["physics_conditioning"] = True
    flags["learnable_gate"] = True
    flags["use_physics_loss"] = True
    flags["use_temporal"] = True
    return model, flags


def smoke_test() -> None:
    model = PHISPIKE(n_inputs=24, n_hidden=64, n_outputs=10, residual_dim=12, learnable_gate=True)
    T, B, F = 40, 4, 24
    x = (torch.rand(T, B, F) > 0.9).float()
    r = torch.randn(T, B, 12) * 0.1
    logits, spars, r_pred, t_pred = model(x, r)
    targets = torch.randint(0, 10, (B,))
    loss, parts = physics_informed_loss(
        logits, targets, r_pred, r.mean(0), spars, t_pred, r, use_physics=True, use_temporal=True
    )
    print(f"Logits {logits.shape}, sparsity {spars.item():.3f}, loss {loss.item():.4f}")
    print(parts)
    print(f"Params: {model.count_parameters()}")


if __name__ == "__main__":
    smoke_test()
