"""P1-7: v2 MLP-PINN as a frozen teacher for PI-LSTM distillation.

The frozen v2 model (held-out Ac-225 ~4.5% vs Radau5) predicts five-species
inventories pointwise from [t_norm, phi_norm, E_feat, IC(5)]. We evaluate it at
every timestep of a trajectory batch and use its normalized output as a soft
target. Distilling trajectory *shape* from v2 is the fastest route to v2-level
median error while the LSTM keeps its speed and integrated-physics consistency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinn_model import load_isotope_pinn_checkpoint  # noqa: E402

from .integrated_loss import SPECIES_WEIGHTS  # noqa: E402


class V2Teacher:
    """Frozen v2 PINN wrapped to emit normalized trajectory targets."""

    def __init__(self, weights_path: Path, device: torch.device, dtype: torch.dtype):
        self.model, _ = load_isotope_pinn_checkpoint(weights_path, map_location=device)
        self.device = device
        # v2 was trained/exported in float32; keep teacher in float32 for stability.
        self.model_dtype = torch.float32
        # Move the whole module (params AND registered buffers, e.g. Fourier freqs)
        # onto the training device so it matches GPU inputs.
        self.model.to(device=device, dtype=self.model_dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict_traj(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, seq, 8) -> normalized (batch, seq, 5) v2 prediction."""
        b, s, f = features.shape
        flat = features.reshape(b * s, f).to(device=self.device, dtype=self.model_dtype)
        out = self.model(flat)
        return out.reshape(b, s, 5).to(features.device)


def distillation_loss(
    student_traj: torch.Tensor,
    teacher_traj: torch.Tensor,
    *,
    species_weights: torch.Tensor | None = None,
    log_weight: float = 2.0,
) -> torch.Tensor:
    """Weighted linear + log-space Huber between student and v2 teacher."""
    w = species_weights if species_weights is not None else SPECIES_WEIGHTS
    w = w.to(device=student_traj.device, dtype=student_traj.dtype)
    teacher = teacher_traj.to(device=student_traj.device, dtype=student_traj.dtype)
    diff = student_traj - teacher
    lin = F.smooth_l1_loss(diff * w, torch.zeros_like(diff))
    eps = 1e-12
    log_diff = torch.log(student_traj.clamp(min=eps)) - torch.log(teacher.clamp(min=eps))
    log_w = w.clone()
    log_w[0] = 0.0  # Ra-226 linear only
    log_loss = F.smooth_l1_loss(log_diff * log_w, torch.zeros_like(log_diff))
    return lin + log_weight * log_loss
