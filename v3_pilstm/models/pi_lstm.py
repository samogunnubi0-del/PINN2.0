"""Physics-Informed LSTM with integrated trapezoidal loss (v3)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinn_model import (  # noqa: E402
    DEFAULT_T_REF_H,
    FourierEnergyEncoder,
    FourierTimeEncoder,
    N_OUTPUT_SPECIES,
)


class PhysicsInformedLSTM(nn.Module):
    """
    LSTM backbone predicting five-species trajectories.

    Inputs per timestep (8): [t_norm, phi_norm, energy_feature, ic_norm x5]
    Energy column is expanded with Fourier features before LSTM.
  Outputs: (batch, seq, 5) normalized positive inventories.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        n_energy_fourier: int = 8,
        n_time_fourier: int = 0,
        dropout: float = 0.0,
        hard_ic: bool = True,
        ic_time_scale: float = 0.02,
        t_ref_h: float = DEFAULT_T_REF_H,
    ):
        super().__init__()
        self.n_time_fourier = int(n_time_fourier)
        self.time_encoder = (
            FourierTimeEncoder(n_freqs=n_time_fourier, t_ref_h=t_ref_h)
            if n_time_fourier > 0
            else None
        )
        self.energy_encoder = FourierEnergyEncoder(n_freqs=n_energy_fourier)
        n_energy_out = self.energy_encoder.out_dim  # 2 * n_freqs (Tancik Fourier bands)
        n_time_out = self.time_encoder.out_dim if self.time_encoder else 1
        # phi, (raw t or encoded t), encoded E, 5 ICs
        lstm_in = 1 + n_time_out + n_energy_out + 5
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, N_OUTPUT_SPECIES)
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.constant_(self.head.bias, -2.0)
        self._n_energy_fourier = n_energy_fourier
        self.hard_ic = hard_ic
        if hard_ic:
            # learnable per-species blend time-scale (softplus keeps it positive)
            self.ic_time_scale = nn.Parameter(torch.full((N_OUTPUT_SPECIES,), float(ic_time_scale)))
        self.config = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "n_energy_fourier": n_energy_fourier,
            "n_time_fourier": n_time_fourier,
            "dropout": dropout,
            "hard_ic": hard_ic,
            "ic_time_scale": ic_time_scale,
            "t_ref_h": t_ref_h,
        }

    def _encode_features(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, seq, 8) -> LSTM input (batch, seq, lstm_in)."""
        t_norm = features[..., 0:1]
        phi_norm = features[..., 1:2]
        e_feat = features[..., 2:3]
        ic = features[..., 3:8]
        e_enc = self.energy_encoder(e_feat)
        if self.time_encoder is not None:
            t_enc = self.time_encoder(t_norm.reshape(-1, 1)).reshape(
                *t_norm.shape[:-1], self.time_encoder.out_dim
            )
            return torch.cat([phi_norm, t_enc, e_enc, ic], dim=-1)
        return torch.cat([phi_norm, t_norm, e_enc, ic], dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self._encode_features(features)
        out, _ = self.lstm(x)
        raw = self.head(out)
        net = F.softplus(raw) + 1e-12  # (batch, seq, 5) network prediction

        if not self.hard_ic:
            return net

        # P1-6 hard initial condition (Lagaris et al. 1998 trial-solution trick):
        #   N(t) = (1 - alpha(t)) * IC + alpha(t) * net(t),  alpha(0) = 0
        # Convex blend keeps outputs non-negative and forces N(t=0) == IC exactly.
        t_norm = features[..., 0:1]                 # (batch, seq, 1)
        ic = features[..., 3:8]                      # (batch, seq, 5) constant across seq
        tau = F.softplus(self.ic_time_scale) + 1e-4  # (5,)
        alpha = 1.0 - torch.exp(-t_norm / tau)       # (batch, seq, 5) via broadcast
        alpha = alpha.clamp(0.0, 1.0)
        return (1.0 - alpha) * ic + alpha * net

    def predict_endpoint(self, features: torch.Tensor) -> torch.Tensor:
        """Final-time concentrations (batch, 5) for comparison with v2 PINN API."""
        traj = self.forward(features)
        return traj[:, -1, :]

    def save(self, path) -> None:
        """Save config + weights so any loader can rebuild the exact architecture."""
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def _config_from_state_dict(cls, sd: dict) -> dict:
        """Infer architecture from tensor shapes (legacy bare checkpoints)."""
        head_w = sd["head.weight"]
        hidden_dim = int(head_w.shape[1])
        lstm_in = int(sd["lstm.weight_ih_l0"].shape[1])
        n_time_fourier = 0
        if "time_encoder.freqs" in sd:
            n_time_fourier = int(sd["time_encoder.freqs"].shape[0])
        # phi(1) + time(1 or 2*n_t) + energy(2*n_e) + ICs(5)
        n_time_out = 2 * n_time_fourier if n_time_fourier > 0 else 1
        n_energy_fourier = max(1, (lstm_in - 6 - n_time_out) // 2)
        num_layers = sum(1 for k in sd if k.startswith("lstm.weight_ih_l"))
        hard_ic = "ic_time_scale" in sd
        ic_time_scale = 0.02
        if hard_ic:
            ic_time_scale = float(sd["ic_time_scale"].detach().cpu().numpy().mean())
        return {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "n_energy_fourier": n_energy_fourier,
            "n_time_fourier": n_time_fourier,
            "dropout": 0.0,
            "hard_ic": hard_ic,
            "ic_time_scale": ic_time_scale,
            "t_ref_h": DEFAULT_T_REF_H,
        }

    @classmethod
    def load(cls, path, map_location="cpu") -> "PhysicsInformedLSTM":
        """Rebuild from a checkpoint (config+weights) or infer arch from legacy state_dict."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob and "config" in blob:
            model = cls(**blob["config"])
            model.load_state_dict(blob["state_dict"])
            return model
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        cfg = cls._config_from_state_dict(sd)
        import warnings
        warnings.warn(
            f"Legacy PI-LSTM checkpoint at {path} (inferred config={cfg}). "
            "Retrain with model.save() so config is embedded; arch mismatch causes "
            "train_summary vs compare_models discrepancies.",
            UserWarning,
            stacklevel=2,
        )
        model = cls(**cfg)
        model.load_state_dict(sd, strict=True)
        return model
