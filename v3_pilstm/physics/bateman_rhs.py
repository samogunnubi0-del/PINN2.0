"""Bateman RHS helpers for PI-LSTM integrated physics loss (torch)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pinn_model import (  # noqa: E402
    DEFAULT_LAMBDA_226_H,
    DEFAULT_LAMBDA_225_H,
    DEFAULT_LAMBDA_AC_H,
    DEFAULT_LAMBDA_227_H,
    DEFAULT_LAMBDA_AC7_H,
    DEFAULT_N226_SCALE,
    DEFAULT_N225_SCALE,
    DEFAULT_NAC_SCALE,
    DEFAULT_N227_SCALE,
    DEFAULT_NAC227_SCALE,
    DEFAULT_PHI_SCALE,
    DEFAULT_SIGMA_N2N,
    DEFAULT_SIGMA_NGAMMA,
    SECONDS_PER_HOUR,
    THERMAL_REFERENCE_EV,
    _phi_sigma_per_hour,
    n2n_threshold_scale_torch,
)

R226_225 = DEFAULT_N226_SCALE / DEFAULT_N225_SCALE
R225_AC = DEFAULT_N225_SCALE / DEFAULT_NAC_SCALE
R226_227 = DEFAULT_N226_SCALE / DEFAULT_N227_SCALE
R227_AC7 = DEFAULT_N227_SCALE / DEFAULT_NAC227_SCALE

# Self-shielding factor used by the trajectory-generating ODE
# (IsotopeEnvironment with target_mass_g = 1.0): exp(-0.01 * 1.0).
ODE_TARGET_SHIELDING = 0.9900498337491681

# --- ODE_DATA_VERSION=v2 (evaluated nuclear data) ---------------------------
# Cached evaluated σ(E) tables as torch tensors, keyed by (device, dtype).
# Loaded lazily from data/evaluated/ only when ODE_DATA_VERSION=v2 so the
# default v1 training path never touches the new dependency.
_EVAL_TABLES_TORCH: dict = {}

# (n,γ): libraries tabulate ZERO capture below 1 keV; below the table's lower
# bound we use a 1/v tail anchored at the experimental 13.8 b thermal point
# (Bagheri 2015, EXFOR 31760). Mirrors ra226_ac225_transmutation.py.
_NG_TABLE_MIN_EV_V2 = 1.0e3
_NG_ANCHOR_B_V2 = 13.8
_NG_ANCHOR_EV_V2 = 0.0253


def _eval_tables_torch(device: torch.device, dtype: torch.dtype):
    key = (str(device), str(dtype))
    if key not in _EVAL_TABLES_TORCH:
        from ra226_ac225_transmutation import load_evaluated_nuclear_data
        ev = load_evaluated_nuclear_data()
        _EVAL_TABLES_TORCH[key] = tuple(
            torch.as_tensor(a, dtype=dtype, device=device)
            for a in (ev.n2n_energy_ev, ev.n2n_sigma_b, ev.ng_energy_ev, ev.ng_sigma_b)
        )
    return _EVAL_TABLES_TORCH[key]


def _linear_interp_clamped_torch(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation, clamped to the table ends (no extrapolation)."""
    xc = x.clamp(min=float(xp[0]), max=float(xp[-1]))
    idx = torch.searchsorted(xp, xc.contiguous()).clamp(min=1, max=xp.shape[0] - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = (xc - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def _sigma_eval_v2_torch(e_ev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(σ_n2n, σ_ngamma) in cm² from the evaluated tables (ODE_DATA_VERSION=v2)."""
    e_n2n, s_n2n, e_ng, s_ng = _eval_tables_torch(e_ev.device, e_ev.dtype)
    # σ(n,2n): first tabulated point is σ=0 at the 6.4218 MeV threshold, so the
    # clamped interpolation is exactly zero below threshold.
    sig_n2n_cm2 = _linear_interp_clamped_torch(e_ev, e_n2n, s_n2n) * 1e-24
    # σ(n,γ): evaluated table ≥ 1 keV; 1/v tail anchored at 13.8 b below 1 keV.
    sig_ng_tab = _linear_interp_clamped_torch(e_ev, e_ng, s_ng) * 1e-24
    sig_ng_tail = _NG_ANCHOR_B_V2 * 1e-24 * torch.sqrt(
        torch.as_tensor(_NG_ANCHOR_EV_V2, dtype=e_ev.dtype, device=e_ev.device)
        / e_ev.clamp(min=1e-30)
    )
    sig_ng_cm2 = torch.where(e_ev < _NG_TABLE_MIN_EV_V2, sig_ng_tail, sig_ng_tab)
    return sig_n2n_cm2, sig_ng_cm2


def reaction_rates(
    phi_norm: torch.Tensor,
    energy_feature: torch.Tensor,
    *,
    phi_scale: float = DEFAULT_PHI_SCALE,
    sigma_n2n: float = DEFAULT_SIGMA_N2N,
    sigma_ngamma: float = DEFAULT_SIGMA_NGAMMA,
    shielding: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (k_n2n, k_ng) per-hour rate tensors matching v2 PINN physics.

    ``shielding`` multiplies both channels. Default 1.0 preserves the legacy
    (pre-2026-07-18) behavior. Pass ``ODE_TARGET_SHIELDING`` (= e^-0.01) to
    match the data-generating ODE exactly: IsotopeEnvironment applies
    exp(-0.01 * target_mass_g) self-shielding with target_mass_g = 1.0 in the
    trajectory generator, and the exact propagator (expmix) loss is sensitive
    to rate mismatch at the sub-percent level.
    """
    phi_raw = phi_norm * phi_scale
    ng_scale = energy_feature.clamp(min=1e-8, max=1e6)
    e_ev = torch.as_tensor(THERMAL_REFERENCE_EV, dtype=phi_norm.dtype, device=phi_norm.device) / (
        energy_feature.clamp(min=1e-8) ** 2 + 1e-30
    )
    if os.environ.get("ODE_DATA_VERSION", "v1").strip().lower() == "v2":
        # v2: evaluated σ (JENDL-5 = ENDF/B-VIII.0) + EXFOR 31760 1/v thermal
        # tail for (n,γ); sigma_n2n/sigma_ngamma args inert. SPECTRUM_MODE
        # (mono|watt|twogroup, default mono) selects pointwise σ(E) vs
        # spectrum-averaged one-group σ (reactor scenarios).
        if os.environ.get("SPECTRUM_MODE", "mono").strip().lower() != "mono":
            from ra226_ac225_transmutation import spectrum_averaged_sigmas_b
            s_n2n_b, s_ng_b = spectrum_averaged_sigmas_b()
            sig_n2n_cm2 = torch.full_like(e_ev, s_n2n_b * 1e-24)
            sig_ng_cm2 = torch.full_like(e_ev, s_ng_b * 1e-24)
        else:
            sig_n2n_cm2, sig_ng_cm2 = _sigma_eval_v2_torch(e_ev)
        k_n2n = (phi_raw * sig_n2n_cm2 * SECONDS_PER_HOUR).clamp(min=0.0, max=5.0e3) * shielding
        k_ng = (phi_raw * sig_ng_cm2 * SECONDS_PER_HOUR).clamp(min=0.0, max=5.0e3) * shielding
        return k_n2n, k_ng
    n2n_thresh = n2n_threshold_scale_torch(e_ev)
    k_n2n = _phi_sigma_per_hour(phi_raw, sigma_n2n, n2n_thresh).clamp(min=0.0, max=5.0e3) * shielding
    k_ng = _phi_sigma_per_hour(phi_raw, sigma_ngamma, ng_scale).clamp(min=0.0, max=5.0e3) * shielding
    return k_n2n, k_ng


def bateman_rhs_normalized(
    n: torch.Tensor,
    k_n2n: torch.Tensor,
    k_ng: torch.Tensor,
    rate_scale: float = 1.0,
) -> torch.Tensor:
    """
    Normalized dN/dt (per hour) for five species.

    n: (..., 5) normalized inventories [Ra226, Ra225, Ac225, Ra227, Ac227]
    k_n2n, k_ng: (...) broadcastable rate constants
    rate_scale: >1 de-stiffens (all decay constants slowed by 1/rate_scale;
        reaction rates must be pre-scaled by the caller). Curriculum scaffold.
    """
    n226 = n[..., 0]
    n225 = n[..., 1]
    nac = n[..., 2]
    n227 = n[..., 3]
    nac7 = n[..., 4]

    dev = n.device
    dtype = n.dtype
    inv = 1.0 / float(rate_scale)
    lam226 = torch.as_tensor(DEFAULT_LAMBDA_226_H * inv, dtype=dtype, device=dev)
    lam225 = torch.as_tensor(DEFAULT_LAMBDA_225_H * inv, dtype=dtype, device=dev)
    lam_ac = torch.as_tensor(DEFAULT_LAMBDA_AC_H * inv, dtype=dtype, device=dev)
    lam227 = torch.as_tensor(DEFAULT_LAMBDA_227_H * inv, dtype=dtype, device=dev)
    lam_a7 = torch.as_tensor(DEFAULT_LAMBDA_AC7_H * inv, dtype=dtype, device=dev)

    if k_n2n.dim() < n226.dim():
        k_n2n = k_n2n.unsqueeze(-1)
    if k_ng.dim() < n226.dim():
        k_ng = k_ng.unsqueeze(-1)

    rhs226 = -(lam226 + k_n2n + k_ng) * n226
    rhs225 = k_n2n * n226 * R226_225 - lam225 * n225
    rhs_ac = lam225 * n225 * R225_AC - lam_ac * nac
    rhs227 = k_ng * n226 * R226_227 - lam227 * n227
    rhs_ac7 = lam227 * n227 * R227_AC7 - lam_a7 * nac7
    return torch.stack([rhs226, rhs225, rhs_ac, rhs227, rhs_ac7], dim=-1)


def jacobian_norms(k_n2n: torch.Tensor, k_ng: torch.Tensor, rate_scale: float = 1.0) -> torch.Tensor:
    """Per-species Jacobian row norms for Bento-style normalization (..., 5)."""
    if k_n2n.dim() == 1:
        k_n2n = k_n2n.unsqueeze(-1)
    if k_ng.dim() == 1:
        k_ng = k_ng.unsqueeze(-1)

    dev = k_n2n.device
    dtype = k_n2n.dtype
    inv = 1.0 / float(rate_scale)
    z = torch.zeros_like(k_n2n)
    lam226 = torch.as_tensor(DEFAULT_LAMBDA_226_H * inv, dtype=dtype, device=dev)
    lam225 = torch.as_tensor(DEFAULT_LAMBDA_225_H * inv, dtype=dtype, device=dev)
    lam_ac = torch.as_tensor(DEFAULT_LAMBDA_AC_H * inv, dtype=dtype, device=dev)
    lam227 = torch.as_tensor(DEFAULT_LAMBDA_227_H * inv, dtype=dtype, device=dev)
    lam_a7 = torch.as_tensor(DEFAULT_LAMBDA_AC7_H * inv, dtype=dtype, device=dev)

    j1 = torch.sqrt(1.0 + (lam226 + k_n2n + k_ng).pow(2))
    j2 = torch.sqrt(1.0 + (k_n2n * R226_225).pow(2) + lam225.pow(2))
    j3 = torch.sqrt(z + (lam225 * R225_AC).pow(2) + lam_ac.pow(2))
    j4 = torch.sqrt(1.0 + (k_ng * R226_227).pow(2) + lam227.pow(2))
    j5 = torch.sqrt(z + (lam227 * R227_AC7).pow(2) + lam_a7.pow(2))
    return torch.cat([j1, j2, j3, j4, j5], dim=-1)
