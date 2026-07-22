"""
Physics losses for PI-LSTM (Reduced-PINN style, Nasiri & Dargazany 2022).

Two collocation modes, selected by ``PI_LSTM_LOSS=trap|expmix``:

* ``trap`` (default, legacy): integrated trapezoidal residual
      N(t_{k+1}) - N(t_k) = dt/2 * [RHS(t_k) + RHS(t_{k+1})].
  Kept bit-for-bit compatible with all pre-2026-07-18 checkpoints/runs.
  Known weakness on log-spaced grids: final intervals reach ~50 h while
  Ra-227 has T1/2 = 42 min (lambda*dt ~ 50), where the trapezoid is far
  outside its stability/accuracy region, and the Bento-style Jacobian row
  normalization divides fast-species residuals by ~lambda (suppressing the
  very species the loss should police hardest).

* ``expmix``: piecewise-exponential (exponential-integrator) residual.
  The Bateman RHS is linear with (to excellent approximation) constant rates
  inside a scenario, so the EXACT propagator over each grid interval is the
  closed-form matrix exponential of the 5-species chain — the same idea as
  exponential time differencing (Cox & Matthews 2002) and CRAM
  (Pusa & Leppanen 2010) for stiff depletion systems. The residual
      r_k = n_{k+1} - P(dt_k; rates) n_k
  is exactly zero for any exact ODE trajectory sampled on ANY grid
  (log-spaced included), independent of dt. Per-species weighting uses a
  relative per-interval change scale (with an absolute floor), NOT the
  Jacobian norm, so fast species (Ra-227) are not normalized away.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .bateman_rhs import (
    ODE_TARGET_SHIELDING,
    R225_AC,
    R226_225,
    R226_227,
    R227_AC7,
    bateman_rhs_normalized,
    jacobian_norms,
    reaction_rates,
)
from pinn_model import (  # noqa: E402
    DEFAULT_LAMBDA_226_H,
    DEFAULT_LAMBDA_225_H,
    DEFAULT_LAMBDA_AC_H,
    DEFAULT_LAMBDA_227_H,
    DEFAULT_LAMBDA_AC7_H,
)

SPECIES_WEIGHTS = torch.tensor([1.0, 2.0, 5.0, 2.0, 3.0])

# expmix relative-scale floor (in normalized units): below this abundance the
# loss treats residuals as absolute, avoiding 0/0 blow-ups for trace species.
EXPMIX_REL_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Piecewise-exponential (exponential-integrator) propagator
# ---------------------------------------------------------------------------
def _phi1_diff(lam_a: torch.Tensor, lam_b: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """D(a,b;dt) = (exp(-a*dt) - exp(-b*dt)) / (b - a), numerically stable.

    Exact rewrite: with m=(a+b)/2, d=(b-a)/2,
        D = dt * exp(-m*dt) * sinhc(d*dt),   sinhc(x) = sinh(x)/x.
    The sinhc form removes the 0/0 singularity at a == b and avoids
    cancellation for |a-b|*dt << 1 (series branch below 1e-4).
    Computed elementwise; all inputs broadcastable, expected float64.
    """
    m = 0.5 * (lam_a + lam_b)
    d = 0.5 * (lam_b - lam_a)
    x = d * dt
    small = x.abs() < 1e-4
    # sinhc series: 1 + x^2/6 + x^4/120 (used where |x| is tiny)
    sinhc_series = 1.0 + x * x / 6.0 + x.pow(4) / 120.0
    # Full branch, guarded so the x==0 division is never selected.
    x_safe = torch.where(small, torch.ones_like(x), x)
    sinhc_full = torch.sinh(x_safe) / x_safe
    sinhc = torch.where(small, sinhc_series, sinhc_full)
    return dt * torch.exp(-m * dt) * sinhc


def _phi2_triple(lam0: torch.Tensor, lam1: torch.Tensor, lam2: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """B(a,b,c;dt) = (D(a,c) - D(b,c)) / (b - a) — the 3-link Bateman kernel.

    Equals sum_i exp(-lam_i dt) / prod_{j!=i}(lam_j - lam_i) but written as a
    divided difference of the stable _phi1_diff, so the removable singularity
    structure is inherited. Degenerate b ~ a uses the analytic derivative
    limit dD(a,c)/da (never triggered for this chain's half-life spread, but
    kept for safety).
    """
    d_ac = _phi1_diff(lam0, lam2, dt)
    d_bc = _phi1_diff(lam1, lam2, dt)
    denom = lam1 - lam0
    e0 = torch.exp(-lam0 * dt)
    # dD(a,c)/da = [-dt*e^{-a dt}(c-a) + (e^{-a dt} - e^{-c dt})] / (c-a)^2
    c_a = lam2 - lam0
    # Guard every division so unselected where-branches stay finite (NaN-safe).
    c_a_safe = torch.where(c_a.abs() < 1e-30, torch.ones_like(c_a), c_a)
    dD_da = (-dt * e0 * c_a + (e0 - torch.exp(-lam2 * dt))) / c_a_safe.pow(2)
    use_deriv = denom.abs() < 1e-10 * (1.0 + lam1.abs())
    denom_safe = torch.where(use_deriv, torch.ones_like(denom), denom)
    return torch.where(use_deriv, dD_da, (d_ac - d_bc) / denom_safe)


def bateman_interval_propagate(
    n: torch.Tensor,
    dt_h: torch.Tensor,
    k_n2n: torch.Tensor,
    k_ng: torch.Tensor,
    rate_scale: float = 1.0,
) -> torch.Tensor:
    """Exact constant-rate propagator over one interval for the 5-species chain.

    Solves dn/dt = A n analytically with A frozen over dt (rates are constant
    within a scenario), i.e. the matrix exponential of the two decay chains
      Ra-226 -> Ra-225 -> Ac-225   (production coeff k_n2n)
      Ra-226 -> Ra-227 -> Ac-227   (production coeff k_ng)
    in normalized units. Exact for the true ODE trajectory at ANY dt — this is
    what removes the log-grid trapezoid inconsistency for stiff Ra-227.

    Args:
        n:      (..., 5) normalized inventories at interval start.
        dt_h:   (...) interval length in HOURS (physical, not normalized).
        k_n2n:  (...) (n,2n) rate constant 1/h.
        k_ng:   (...) (n,gamma) rate constant 1/h.
        rate_scale: >1 de-stiffens (decay constants slowed by 1/rate_scale;
            reaction rates must be pre-scaled by the caller). Curriculum only.
    Returns:
        (..., 5) propagated inventories, float64 (cast back by caller).
    """
    n64 = n.to(torch.float64)
    # Normalize dt / rates to the species-slice shape (...) so elementwise
    # products never mis-broadcast (e.g. (B,K,1) * (B,K) -> (B,K,K)).
    dt = dt_h.to(torch.float64)
    if dt.dim() == n64.dim() and dt.size(-1) == 1:
        dt = dt.squeeze(-1)
    k2 = k_n2n.to(torch.float64)
    kg = k_ng.to(torch.float64)
    if k2.dim() == n64.dim() and k2.size(-1) == 1:
        k2 = k2.squeeze(-1)
    if kg.dim() == n64.dim() and kg.size(-1) == 1:
        kg = kg.squeeze(-1)

    dev = n.device
    inv = 1.0 / float(rate_scale)
    lam226 = torch.as_tensor(DEFAULT_LAMBDA_226_H * inv, dtype=torch.float64, device=dev)
    lam225 = torch.as_tensor(DEFAULT_LAMBDA_225_H * inv, dtype=torch.float64, device=dev)
    lam_ac = torch.as_tensor(DEFAULT_LAMBDA_AC_H * inv, dtype=torch.float64, device=dev)
    lam227 = torch.as_tensor(DEFAULT_LAMBDA_227_H * inv, dtype=torch.float64, device=dev)
    lam_a7 = torch.as_tensor(DEFAULT_LAMBDA_AC7_H * inv, dtype=torch.float64, device=dev)

    lam0 = lam226 + k2 + kg  # total Ra-226 removal rate

    n226 = n64[..., 0]
    n225 = n64[..., 1]
    nac = n64[..., 2]
    n227 = n64[..., 3]
    nac7 = n64[..., 4]

    e0 = torch.exp(-lam0 * dt)
    e225 = torch.exp(-lam225 * dt)
    eac = torch.exp(-lam_ac * dt)
    e227 = torch.exp(-lam227 * dt)
    ea7 = torch.exp(-lam_a7 * dt)

    # Chain A: 226 -> 225 -> Ac-225. c1 = k2*R226_225, c2 = lam225*R225_AC.
    c1a = k2 * R226_225
    c2a = lam225 * R225_AC
    out226 = n226 * e0
    out225 = n225 * e225 + c1a * n226 * _phi1_diff(lam0, lam225, dt)
    out_ac = (
        nac * eac
        + c2a * n225 * _phi1_diff(lam225, lam_ac, dt)
        + c1a * c2a * n226 * _phi2_triple(lam0, lam225, lam_ac, dt)
    )

    # Chain B: 226 -> 227 -> Ac-227. c1 = kg*R226_227, c2 = lam227*R227_AC7.
    c1b = kg * R226_227
    c2b = lam227 * R227_AC7
    out227 = n227 * e227 + c1b * n226 * _phi1_diff(lam0, lam227, dt)
    out_ac7 = (
        nac7 * ea7
        + c2b * n227 * _phi1_diff(lam227, lam_a7, dt)
        + c1b * c2b * n226 * _phi2_triple(lam0, lam227, lam_a7, dt)
    )

    out = torch.stack([out226, out225, out_ac, out227, out_ac7], dim=-1)
    if out.shape != n64.shape:
        raise RuntimeError(
            f"propagator shape mismatch: got {tuple(out.shape)} for input {tuple(n64.shape)}"
        )
    return out


def causal_time_weights(    t_norm: torch.Tensor,
    *,
    eps: float = 2.0,
    progress: float = 1.0,
) -> torch.Tensor:
    """Wang et al. (arXiv:2203.07404) causal weighting on interval midpoints.

    Early intervals get weight ~1; later intervals are down-weighted until the
    network fits the early dynamics. `progress` in [0,1] ramps causality off as
    training matures (progress=1 => nearly uniform weights).
    t_norm: (batch, seq). Returns (batch, seq-1) interval weights.
    """
    mid = 0.5 * (t_norm[:, 1:] + t_norm[:, :-1])           # (batch, seq-1)
    tmax = mid.max(dim=1, keepdim=True).values.clamp(min=1e-8)
    frac = mid / tmax
    strength = eps * (1.0 - float(progress))
    w = torch.exp(-strength * frac)
    return w / w.mean(dim=1, keepdim=True).clamp(min=1e-8)


def integrated_physics_loss(
    traj_pred: torch.Tensor,
    t_norm: torch.Tensor,
    phi_norm: torch.Tensor,
    energy_feature: torch.Tensor,
    *,
    t_ref_h: float = 500.0,
    physics_weight: float = 1.0,
    time_weights: torch.Tensor | None = None,
    mode: str = "trap",
    return_per_species: bool = False,
    rate_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Args:
        traj_pred: (batch, seq, 5) normalized species along time
        t_norm: (batch, seq) time / t_ref_h
        phi_norm: (batch,) or (batch, 1)
        energy_feature: (batch,) sqrt(E_ref/E)
        mode: "trap" (legacy trapezoid + Jacobian-norm scaling, default,
              reproduces old runs) | "expmix" (piecewise-exponential exact
              propagator residual + relative per-species scaling; see module
              docstring). Select via PI_LSTM_LOSS env in training scripts.
    """
    mode = mode.strip().lower()
    if mode not in ("trap", "expmix"):
        raise ValueError(f"Unknown physics loss mode {mode!r}; expected 'trap' or 'expmix'")
    if traj_pred.dim() != 3 or traj_pred.size(-1) != 5:
        raise ValueError("traj_pred must be (batch, seq, 5)")
    batch, seq, _ = traj_pred.shape
    if seq < 2:
        z = traj_pred.new_zeros(())
        return z, {"physics_mse": 0.0}

    if phi_norm.dim() == 1:
        phi_norm = phi_norm.unsqueeze(-1)
    if energy_feature.dim() == 1:
        energy_feature = energy_feature.unsqueeze(-1)

    k_n2n, k_ng = reaction_rates(phi_norm, energy_feature)
    inv_scale = 1.0 / float(rate_scale)
    k_n2n_b = k_n2n.view(batch, 1) * inv_scale
    k_ng_b = k_ng.view(batch, 1) * inv_scale

    dt_h = (t_norm[:, 1:] - t_norm[:, :-1]).unsqueeze(-1) * t_ref_h
    n_k = traj_pred[:, :-1, :]
    n_k1 = traj_pred[:, 1:, :]

    if mode == "expmix":
        # Rates must match the data-generating ODE (incl. its target
        # self-shielding) — the exact propagator is sensitive to sub-percent
        # rate mismatch. trap mode keeps legacy unshielded rates untouched.
        k_n2n_x, k_ng_x = reaction_rates(
            phi_norm, energy_feature, shielding=ODE_TARGET_SHIELDING
        )
        n_k1_hat = bateman_interval_propagate(
            n_k, dt_h, k_n2n_x.view(batch, 1) * inv_scale,
            k_ng_x.view(batch, 1) * inv_scale, rate_scale=rate_scale,
        ).to(dtype=traj_pred.dtype)
        residual = n_k1 - n_k1_hat
        # Per-species RELATIVE per-interval change scale (absolute floor):
        # divides by the species' own abundance level, not by the Jacobian
        # row norm ~lambda. Fast species (Ra-227, lambda ~ 0.99/h) therefore
        # keep full weight instead of being normalized away.
        scale = (0.5 * (n_k.abs() + n_k1.abs())).clamp(min=EXPMIX_REL_FLOOR)
        residual_norm = residual / scale
    else:
        rhs_k = bateman_rhs_normalized(n_k, k_n2n_b, k_ng_b, rate_scale=rate_scale)
        rhs_k1 = bateman_rhs_normalized(n_k1, k_n2n_b, k_ng_b, rate_scale=rate_scale)
        trap = 0.5 * dt_h * (rhs_k + rhs_k1)

        residual = (n_k1 - n_k) - trap
        jn = jacobian_norms(k_n2n_b, k_ng_b, rate_scale=rate_scale).unsqueeze(1)
        residual_norm = residual / jn.clamp(min=1e-8)

    w = SPECIES_WEIGHTS.to(device=traj_pred.device, dtype=traj_pred.dtype)
    huber = F.smooth_l1_loss(
        residual_norm * w,
        torch.zeros_like(residual_norm),
        reduction="none",
    )  # (batch, seq-1, 5)
    if time_weights is not None:
        tw = time_weights.to(device=traj_pred.device, dtype=traj_pred.dtype).unsqueeze(-1)
        huber = huber * tw
    physics_mse = huber.mean()
    total = physics_weight * physics_mse
    info: dict[str, float] = {"physics_mse": float(physics_mse.detach().cpu()), "physics_mode": mode}
    if return_per_species:
        # Per-species mean residual (batch/interval reduced). Returned as a
        # GRAPH tensor so adaptive-weight code can take per-species grads.
        info["per_species"] = huber.mean(dim=(0, 1))  # (5,)
        info["per_species_detached"] = [float(x) for x in info["per_species"].detach().cpu()]
    return total, info


def impurity_overshoot_loss(
    traj_pred: torch.Tensor,
    phi_norm: torch.Tensor,
    energy_feature: torch.Tensor,
    *,
    tolerance: float = 1.5,
) -> torch.Tensor:
    """P0-3: penalize Ra-227 exceeding its production/decay quasi-equilibrium.

    Ra-227 (T½=42 min) reaches n227_eq = k_ng * n226 * R226_227 / lam227 quickly.
    Predictions far above this are the source of the high-flux overshoot, so we
    penalize traj values beyond `tolerance * n227_eq`. Physics-based, ODE-free.
    """
    from pinn_model import DEFAULT_LAMBDA_227_H
    from .bateman_rhs import R226_227 as _R226_227

    if phi_norm.dim() == 1:
        phi_norm = phi_norm.unsqueeze(-1)
    if energy_feature.dim() == 1:
        energy_feature = energy_feature.unsqueeze(-1)
    k_n2n, k_ng = reaction_rates(phi_norm, energy_feature)
    k_ng_b = k_ng.view(-1, 1)                       # (batch, 1)
    lam227 = torch.as_tensor(DEFAULT_LAMBDA_227_H, dtype=traj_pred.dtype, device=traj_pred.device)

    n226 = traj_pred[:, :, 0]                        # (batch, seq)
    n227 = traj_pred[:, :, 3]
    n227_eq = (k_ng_b * n226 * _R226_227) / lam227.clamp(min=1e-12)
    excess = F.relu(n227 - tolerance * n227_eq - 1e-9)
    return excess.pow(2).mean()


def mass_conservation_loss(
    traj_pred: torch.Tensor,
    ic_norm: torch.Tensor,
    phi_norm: torch.Tensor,
) -> torch.Tensor:
    """Penalize net atom creation beyond initial budget (Ra226+Ra225+Ac225 channel)."""
    if phi_norm.dim() == 2:
        phi_norm = phi_norm.squeeze(-1)
    flux_on = phi_norm > 1e-6
    if not flux_on.any():
        return traj_pred.new_zeros(())
    budget = ic_norm[:, 0] + ic_norm[:, 1] + ic_norm[:, 2]
    total = traj_pred[:, :, 0] + traj_pred[:, :, 1] + traj_pred[:, :, 2]
    excess = F.relu(total - budget.unsqueeze(1) * 1.02)
    mask = flux_on.float().unsqueeze(1)
    return (excess * mask).pow(2).mean()


def data_trajectory_loss(
    traj_pred: torch.Tensor,
    traj_target: torch.Tensor,
    *,
    species_weights: torch.Tensor | None = None,
    log_weight: float = 2.0,
) -> torch.Tensor:
    w = species_weights
    if w is None:
        w = SPECIES_WEIGHTS
    w = w.to(device=traj_pred.device, dtype=traj_pred.dtype)
    diff = traj_pred - traj_target
    lin = F.smooth_l1_loss(diff * w, torch.zeros_like(diff))
    eps = 1e-12
    log_diff = torch.log(traj_pred.clamp(min=eps)) - torch.log(traj_target.clamp(min=eps))
    log_w = w.clone()
    log_w[0] = 0.0  # Ra-226: linear only
    log_loss = F.smooth_l1_loss(log_diff * log_w, torch.zeros_like(log_diff))
    return lin + log_weight * log_loss
