"""One-step trapezoidal Newton projection on PI-LSTM endpoint predictions."""
from __future__ import annotations

import torch

from .bateman_rhs import bateman_rhs_normalized, jacobian_norms, reaction_rates


def project_endpoint_trap(
    traj_norm: torch.Tensor,
    t_norm: torch.Tensor,
    phi_norm: torch.Tensor,
    energy_feature: torch.Tensor,
    *,
    t_ref_h: float = 500.0,
    damping: float = 0.5,
) -> torch.Tensor:
    """
    Reduce the final-interval trapezoidal residual with one damped Newton step.

    traj_norm: (batch, seq, 5) normalized species trajectory
    t_norm: (batch, seq) normalized time
    phi_norm: (batch,) or (batch, 1)
    energy_feature: (batch,) or (batch, 1)
  Returns: (batch, 5) projected endpoint in normalized units
    """
    if traj_norm.dim() != 3 or traj_norm.size(1) < 2:
        return traj_norm[:, -1, :]

    if phi_norm.dim() == 1:
        phi_norm = phi_norm.unsqueeze(-1)
    if energy_feature.dim() == 1:
        energy_feature = energy_feature.unsqueeze(-1)

    batch = traj_norm.size(0)
    n_prev = traj_norm[:, -2, :]
    n_end = traj_norm[:, -1, :]

    k_n2n, k_ng = reaction_rates(phi_norm, energy_feature)
    k_n2n_b = k_n2n.view(batch, 1)
    k_ng_b = k_ng.view(batch, 1)

    dt_h = (t_norm[:, -1] - t_norm[:, -2]).unsqueeze(-1).clamp(min=1e-8) * t_ref_h
    rhs_prev = bateman_rhs_normalized(n_prev.unsqueeze(1), k_n2n_b, k_ng_b).squeeze(1)
    rhs_end = bateman_rhs_normalized(n_end.unsqueeze(1), k_n2n_b, k_ng_b).squeeze(1)
    trap = 0.5 * dt_h * (rhs_prev + rhs_end)
    residual = (n_end - n_prev) - trap

    jn = jacobian_norms(k_n2n_b, k_ng_b).squeeze(1).clamp(min=1e-6)
    corrected = n_end - damping * (residual / jn)
    return corrected.clamp(min=1e-12)
