"""Self-adaptive per-species physics-loss weighting (PI_LSTM_ADAPTIVE_WEIGHTS=1).

Motivation: the integrated physics loss pools 5 species whose residual scales
differ by orders of magnitude. Fast species (Ra-227, T1/2 = 42 min) can end up
with vanishing gradient share — the loss then under-polices exactly the stiff
channel it exists for (cf. the ISEF readiness finding on Jacobian suppression).

Two standard remedies, implemented together:
  * McClenny & Braga-Neto 2023 (arXiv:2009.04544), "Self-Adaptive Physics-
    Informed Neural Networks using a Soft Attention Mechanism": trainable
    per-species multipliers rebalanced during optimization.
  * Wang, Yu & Perdikaris 2022 (arXiv:2007.14527), "When and why PINNs fail to
    train": NTK/gradient-pathology perspective — balance terms by their
    gradient-norm ratio so no term dominates or vanishes.

This module implements the gradient-norm-ratio variant (cheap, stable, no
extra optimizer state): every ``update_every`` epochs the per-species
grad-norms of the physics loss are measured on the first training batch and
the weights are refreshed as
    w_i <- EMA( clip( g_ref / g_i ) ),   normalized to mean 1,
where g_ref is the grad-norm of the (unweighted) pooled physics term. Species
with SMALL gradient share (suppressed, e.g. Ra-227) therefore receive LARGER
weights. Default OFF: PI_LSTM_ADAPTIVE_WEIGHTS unset/0 keeps the legacy
uniform pooling exactly.
"""
from __future__ import annotations

import numpy as np
import torch


class GradNormSpeciesWeighter:
    """EMA-smoothed grad-norm-ratio per-species weights for the physics loss."""

    def __init__(
        self,
        n_species: int = 5,
        *,
        update_every: int = 5,
        ema: float = 0.9,
        clip: tuple[float, float] = (0.1, 10.0),
    ):
        self.n_species = n_species
        self.update_every = max(1, int(update_every))
        self.ema = float(ema)
        self.clip = clip
        self.weights = np.ones(n_species, dtype=np.float64)
        self.last_update_epoch = -1
        self.history: list[dict] = []

    def should_update(self, epoch: int) -> bool:
        return epoch != self.last_update_epoch and (epoch == 1 or epoch % self.update_every == 0)

    @torch.no_grad()
    def tensor(self, device, dtype) -> torch.Tensor:
        return torch.as_tensor(self.weights, device=device, dtype=dtype)

    def update(
        self,
        model: torch.nn.Module,
        per_species_loss: torch.Tensor,
        pooled_loss: torch.Tensor,
        epoch: int,
    ) -> np.ndarray:
        """Refresh weights from per-species grad-norms on one batch.

        per_species_loss: (5,) graph tensor (per-species mean residual).
        pooled_loss: scalar physics loss used as the reference grad-norm.
        """
        params = [p for p in model.parameters() if p.requires_grad]

        def _gnorm(loss: torch.Tensor) -> float:
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            tot = 0.0
            for g in grads:
                if g is not None:
                    tot += float(g.detach().pow(2).sum())
            return tot ** 0.5

        g_ref = _gnorm(pooled_loss)
        g_i = np.array([_gnorm(per_species_loss[i]) for i in range(self.n_species)])
        ratio = g_ref / np.maximum(g_i, 1e-12)
        ratio = np.clip(ratio, self.clip[0], self.clip[1])
        ratio = ratio / ratio.mean()  # mean 1: keeps global loss scale stable
        self.weights = self.ema * self.weights + (1.0 - self.ema) * ratio
        self.weights = self.weights / self.weights.mean()
        self.last_update_epoch = epoch
        self.history.append(
            {
                "epoch": int(epoch),
                "g_ref": g_ref,
                "g_species": g_i.tolist(),
                "weights": self.weights.tolist(),
            }
        )
        return self.weights.copy()
