"""Vanilla LSTM baseline for 5-species isotope transmutation (ablation).

NO physics constraints, NO hard initial condition, NO distillation — the fair
"why physics at all?" comparator. Training/eval driver lives in
v3_pilstm/analysis/train_baseline.py; this module holds the architecture and
the parameter-matching helper (matched budget ±10% vs the PI-LSTM, as
required for the ISEF baseline comparison).
"""
from __future__ import annotations

import torch
from torch import nn


class BaselineLSTM(nn.Module):
    """
    Standard Baseline LSTM for 5-species isotope transmutation.
    This model contains NO physics constraints. It simply maps the inputs
    to outputs to serve as a baseline for the ablation study.
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, output_dim=5):
        super().__init__()
        # PyTorch LSTM expects input shape: (batch_size, seq_len, input_dim) when batch_first=True
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
        }

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, 8)
        Returns: (batch_size, seq_len, 5) -> The 5 predicted isotope concentrations
        """
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out)

        # Softplus to ensure outputs are non-negative (mass can't be negative)
        return torch.nn.functional.softplus(out)

    def save(self, path) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "BaselineLSTM":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and "config" in blob and "state_dict" in blob:
            model = cls(**blob["config"])
            model.load_state_dict(blob["state_dict"])
            return model
        # Legacy bare state_dict (default arch)
        model = cls()
        model.load_state_dict(blob)
        return model


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count (encoders' registered buffers excluded)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_matched_baseline(
    target_params: int,
    *,
    input_dim: int = 8,
    num_layers: int = 2,
    output_dim: int = 5,
    tolerance: float = 0.10,
) -> tuple[BaselineLSTM, int]:
    """Pick hidden_dim so the vanilla LSTM budget matches ``target_params``
    within ±``tolerance`` (default 10%), searching multiples of 8.

    For input_dim=8, num_layers=2, output_dim=5 the count is exactly
        12*H^2 + (4*input_dim + 4*num_layers + output_dim)*H + output_dim
      = 12*H^2 + 53*H + 5.
    Returns (model, actual_param_count). Raises if nothing fits the tolerance.
    """
    best: tuple[int, int] | None = None  # (hidden, count)
    for hidden in range(8, 1025, 8):
        model = BaselineLSTM(input_dim=input_dim, hidden_dim=hidden,
                             num_layers=num_layers, output_dim=output_dim)
        n = count_parameters(model)
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (hidden, n)
    assert best is not None
    hidden, n = best
    if abs(n - target_params) / max(target_params, 1) > tolerance:
        raise ValueError(
            f"No baseline hidden_dim within ±{tolerance:.0%} of {target_params} "
            f"(closest: hidden={hidden} -> {n})"
        )
    return BaselineLSTM(input_dim=input_dim, hidden_dim=hidden,
                        num_layers=num_layers, output_dim=output_dim), n
