"""
Local input attribution for the trained IsotopePINN (explainability).

Reference: Gevaert & Saeys, *PDD-SHAP: Fast Approximations for Shapley Values using
Functional Decomposition*, arXiv:2208.12595 (2022).

What that paper does
--------------------
It does **not** change how a PINN is trained. It speeds up **Shapley-value**
explanations for a black-box model by fitting an ANOVA / functional decomposition
surrogate, then estimating attributions from that surrogate. Useful when you must
explain **many** predictions and vanilla Shapley sampling is too expensive.

What this script does
---------------------
For the 8 public inputs, exact Shapley is already heavy for large batches, but *d=8*
is modest. This script implements **integrated gradients** (Sundararajan et al.,
*Integrated Gradients: Axiomatic Attribution for Deep Networks*, ICML 2017) in
pure PyTorch — same high-level goal (“how much did each input contribute to this
output?”) without extra packages.

If you later need **Shapley** specifically at scale, consider the PDD-SHAP
pipeline from 2208.12595 or libraries that implement it; integrated gradients
here are a practical ISEF/diagnostic layer.

Run (from repo root ``New folder``):

    python analysis/pinn_input_attribution.py
    python analysis/pinn_input_attribution.py --species ac227 --time-h 200 --energy-ev 14e6
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_model import (  # noqa: E402
    DEFAULT_N226_SCALE,
    DEFAULT_N225_SCALE,
    DEFAULT_NAC_SCALE,
    DEFAULT_N227_SCALE,
    DEFAULT_NAC227_SCALE,
    DEFAULT_PHI_SCALE,
    DEFAULT_T_REF_H,
    load_isotope_pinn_checkpoint,
    neutron_energy_ev_to_feature_numpy,
)

FEATURE_NAMES = (
    "t_nn (time / T_ref)",
    "phi_nn (flux / PHI_SCALE)",
    "E_nn (sqrt(E_ref/E))",
    "N_Ra226 / scale226",
    "N_Ra225 / scale225",
    "N_Ac225 / scale225",
    "N_Ra227 / scale227",
    "N_Ac227 / scale227",
)


def build_input_row(
    *,
    time_h: float,
    phi: float,
    energy_ev: float,
    n226: float,
    n225: float,
    nac225: float,
    n227: float,
    nac227: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    e_nn = float(neutron_energy_ev_to_feature_numpy(energy_ev))
    v = [
        time_h / DEFAULT_T_REF_H,
        phi / DEFAULT_PHI_SCALE,
        e_nn,
        n226 / DEFAULT_N226_SCALE,
        n225 / DEFAULT_N225_SCALE,
        nac225 / DEFAULT_NAC_SCALE,
        n227 / DEFAULT_N227_SCALE,
        nac227 / DEFAULT_NAC227_SCALE,
    ]
    return torch.tensor([v], dtype=dtype, device=device, requires_grad=True)


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    species_index: int,
    n_steps: int = 32,
    baseline: torch.Tensor | None = None,
    multiply_by_scale: float | None = None,
) -> torch.Tensor:
    """
    Returns attribution vector (8,) matching ``x``'s columns (atoms output if scale set).
    """
    if x.dim() != 2 or x.size(0) != 1 or x.size(1) != 8:
        raise ValueError("x must be shape (1, 8)")
    device = x.device
    dtype = x.dtype
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.to(device=device, dtype=dtype)

    x0 = baseline.detach()
    x1 = x.detach()
    grads_sum = torch.zeros_like(x1)
    model.eval()

    for k in range(1, n_steps + 1):
        alpha = float(k) / float(n_steps)
        x_interp = x0 + alpha * (x1 - x0)
        x_interp = x_interp.clone().requires_grad_(True)
        out = model(x_interp)
        if multiply_by_scale is not None:
            val = out[0, species_index] * multiply_by_scale
        else:
            val = out[0, species_index]
        (g_x,) = torch.autograd.grad(val, x_interp, retain_graph=False, create_graph=False)
        grads_sum += g_x.detach()

    avg_grads = grads_sum / float(n_steps)
    ig = (x1 - x0) * avg_grads
    return ig.squeeze(0)


def main() -> None:
    p = argparse.ArgumentParser(description="Integrated-gradients attribution for IsotopePINN")
    p.add_argument("--weights", type=str, default=str(ROOT / "weights" / "pinn_best_weights.pth"))
    p.add_argument(
        "--species",
        type=str,
        default="ac225",
        choices=("ac225", "ac227", "ra226", "ra225", "ra227"),
        help="which normalized output channel drives the scalar target",
    )
    p.add_argument("--time-h", type=float, default=250.0)
    p.add_argument("--phi", type=float, default=1e14)
    p.add_argument("--energy-ev", type=float, default=14e6)
    p.add_argument("--n226", type=float, default=DEFAULT_N226_SCALE)
    p.add_argument("--steps", type=int, default=32)
    args = p.parse_args()

    species_map = {
        "ra226": (0, DEFAULT_N226_SCALE),
        "ra225": (1, DEFAULT_N225_SCALE),
        "ac225": (2, DEFAULT_NAC_SCALE),
        "ra227": (3, DEFAULT_N227_SCALE),
        "ac227": (4, DEFAULT_NAC227_SCALE),
    }
    idx, scale = species_map[args.species]

    device = torch.device("cpu")
    model, _info = load_isotope_pinn_checkpoint(args.weights, map_location=device)
    model = model.to(device)

    x = build_input_row(
        time_h=args.time_h,
        phi=args.phi,
        energy_ev=args.energy_ev,
        n226=args.n226,
        n225=0.0,
        nac225=0.0,
        n227=0.0,
        nac227=0.0,
        dtype=torch.float32,
        device=device,
    )

    ig_norm = integrated_gradients(
        model, x, species_index=idx, n_steps=args.steps, multiply_by_scale=None
    )
    ig_atoms = integrated_gradients(
        model, x, species_index=idx, n_steps=args.steps, multiply_by_scale=scale
    )

    print("IsotopePINN input attribution (integrated gradients)")
    print("  Target scalar: predicted {} (normalized space for first block, atoms for second)".format(args.species))
    print("  Note: For Shapley-style credit with many queries, see Gevaert & Saeys arXiv:2208.12595 (PDD-SHAP).")
    print()
    for name, a_norm, a_atom in zip(FEATURE_NAMES, ig_norm.tolist(), ig_atoms.tolist()):
        print(f"  {name:32}  d_norm={a_norm: .6e}   d_atoms~{a_atom: .6e}")
    print()
    print(f"  Sum (norm space, sanity IC): {float(ig_norm.sum()):.6e}")


if __name__ == "__main__":
    main()
