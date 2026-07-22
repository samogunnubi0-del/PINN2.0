"""
Physics-informed neural network backbone for five-species isotope transmutation.

Two competing neutron channels on Ra-226:
  (n,2n)  Ra-226 -> Ra-225 -> Ac-225   (desired product)
  (n,γ)   Ra-226 -> Ra-227 -> Ac-227   (impurity pathway)

SCALING CONTRACT (must match train.py exactly)
===============================================
Input columns (8 features, normalised):

  col 0   t_nn       = t_hours / T_REF_H         (T_REF_H   = 500)
  col 1   phi_nn     = phi / PHI_SCALE            (PHI_SCALE = 1e15)
  col 2   E_nn       = sqrt(E_ref / E_eV)         (1/v sigma ratio)
  col 3   n226_nn    = N_Ra226 / N226_SCALE        (6.022e23)
  col 4   n225_nn    = N_Ra225 / N225_SCALE        (1e20)
  col 5   nac_nn     = N_Ac225 / NAC_SCALE         (1e20)
  col 6   n227_nn    = N_Ra227 / N227_SCALE        (1e18)
  col 7   nac227_nn  = N_Ac227 / NAC227_SCALE      (1e18)

The model internally derives extra energy-regime features from col 2. The
external input contract remains 8 columns so training, validation, and the app
all call the same API.

Outputs (5): (n226, n225, nac225, n227, nac227) in normalised space.

NUCLEAR DATA (NNDC)
===================
Ra-226: 1600 y           Ra-225: 14.8 d       Ac-225: 9.920 d
Ra-227: 42.2 min         Ac-227: 21.772 y
sigma_n2n:  27 mb (spectrum-avg fast)    sigma_ngamma: 12.8 barn (thermal reference)

METHOD REFERENCES (this file)
=============================
The implementations below are **ordinary** PINN / NN building blocks tied to published
work — not ad‑hoc tricks. Where we use a **simplified** variant, it is labeled.

- **PINN (continuous formulation)**: Raissi, Perdikaris, Karniadakis, *Physics-
  informed neural networks: A deep learning framework for solving forward and
  inverse problems involving nonlinear partial differential equations*, J. Comput.
  Phys. 378 (2019) 686–707. (Bateman residuals / physics loss.)

- **Fourier features (deterministic bands on log-energy)**: Tancik et al.,
  *Fourier Features Let Networks Learn High Frequency Functions in Low
  Dimensional Domains*, NeurIPS 2020. We map log10(E [eV]) through fixed sin/cos
  features (log-spaced “bands”) — the standard random Fourier projection is i.i.d.
  Gaussian B; we freeze structured frequencies for reproducibility and checkpoint
  stability, which is a common practical variant of the same γ(x)=[sin(2πBx), …] idea.

- **Domain decomposition / interface regularization (“XPINN-style”)**: Jagtap &
  Karniadakis, *Extended physics-informed neural networks (XPINNs): A generalized
  space–time domain decomposition based deep learning framework for nonlinear partial
  differential equations*, Commun. Comput. Phys. **28** (5) (2020) 2002–2041,
  doi:10.4208/cicp.OA-2020-0164. Full XPINNs use separate nets per subdomain with
  interface conditions; here we approximate that idea with **two linear heads** on one
  shared trunk and a **weak penalty** that matches the two expert compositions of the
  hard-IC ansatz in a narrow energy window around the (n,2n) threshold — a
  stabilizer for the MoE router, **not** a literal continuity constraint on physical
  cross sections (which are discontinuous in our simplified threshold model).

- **Gradient magnitude balancing (two-term, one step)**: Wang, Teng, Perdikaris,
  *Understanding and mitigating gradient flow pathologies in physics-informed
  neural networks*, SIAM J. Sci. Comput. 43(5) (2021) A3055–A3081. Training code
  can reweight **supervised** vs **unsupervised** bundles so their gradient norms
  match (optional; doubles backward work).

- **Causal / time‑ordered training**: Wang et al., *Respecting causality is all
  you need for training physics-informed neural networks*, arXiv:2203.07404.
  Full causal PINNs use sequential window weights; we combine (i) **soft** emphasis
  toward small ``t_nn`` with (ii) **Krishnapriyan et al.** (*Characterizing possible
  failure modes in PINNs*, NeurIPS 2021, arXiv:2109.01050) **time-bin curriculum**
  that gradually equalizes emphasis from early-time bins to all bins as training
  progresses—see :func:`compose_physics_point_weights`.

- **Self‑adaptive residual emphasis (detached, focal)**: McClenny & Braga‑Neto,
  *Self-adaptive physics-informed neural networks*, J. Comput. Phys. **474**
  (2023) 111722 (arXiv:2009.04544). Full SA‑PINNs train separate attention weights;
  we use a **lightweight** detached focal factor ``∝‖r‖^α`` on normalized residuals
  so gradients do not flow through the weights (stable alternative to the saddle
  formulation). Composed **once** with causal weights + **one** mean normalization
  so nothing multi-counts against threshold boosting (which stays on ``fn`` before
  squaring).

- **MC Dropout (uncertainty)**: Gal & Ghahramani, *Dropout as a Bayesian
  approximation*, ICML 2016. :func:`predict_mcd` implements the standard
  train‑mode Monte‑Carlo sampling already present in this module.

- **Adaptive activation functions**: Jagtap, Kawaguchi & Karniadakis,
  *Adaptive activation functions accelerate convergence in deep and
  physics‑informed neural networks*, J. Comput. Phys. **404** (2020) 109136.
  Each hidden layer learns a scalar slope ``a`` so the activation is
  ``tanh(a * x)`` instead of ``tanh(x)`` — mitigates spectral bias at
  sharp threshold transitions.

- **Residual connections**: He et al., *Deep residual learning for image
  recognition*, CVPR 2016. Skip connections ``h + f(h)`` in the MLP
  improve gradient flow for deeper networks; standard in DeepXDE and
  similar PINN frameworks.

- **Exponential decay ansatz (Ra‑226)**: Lagaris, Likas & Fotiadis,
  *Artificial neural networks for solving ordinary and partial differential
  equations*, IEEE Trans. Neural Networks **9**(5) (1998) 987–1000. Species
  with **no source term** (Ra‑226) use ``N_0 * exp(-softplus(NN) * t)`` so
  they are **architecturally** monotone‑decreasing; species with source
  terms (Ra‑225 … Ac‑227) keep the signed‑rate ansatz.

- **Learnable loss balancing**: Kendall, Gal & Cipolla, *Multi‑task learning
  using uncertainty to weigh losses*, CVPR 2018. Implemented in ``train.py``
  as :class:`UncertaintyWeighter` — learns ``log(σ²_k)`` per loss group
  so the model self‑balances supervised vs. unsupervised objectives.

- **Cosine annealing with warm restarts**: Loshchilov & Hutter, *SGDR:
  Stochastic gradient descent with warm restarts*, ICLR 2017. Replaces
  ``ReduceLROnPlateau`` in ``train.py`` for more reliable PINN convergence.

- **Jacobian normalization of physics residuals**: Bento, Câmara, Rocha &
  Seabra, *Solving stiff dark matter equations via Jacobian Normalization with
  Physics-Informed Neural Networks*, arXiv:2602.21988 (2026). Each Bateman
  residual is divided by ``sqrt(1 + ||J_i||²)`` where ``J_i`` is the analytic
  row-Jacobian of the linear chain — prevents trace-daughter zero-collapse
  when ``N(t) ≈ 0`` (applied to dark-matter Boltzmann eqs. in the paper;
  we adapt the same normalization to the five-species transmutation ODE).
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# -- Physical constants (hours, NNDC verified) --------------------------------
_LN2 = math.log(2.0)

THERMAL_REFERENCE_EV = 0.025

DEFAULT_LAMBDA_226_H  = _LN2 / (1600.0 * 365.25 * 24.0)   # Ra-226: 1600 y
DEFAULT_LAMBDA_225_H  = _LN2 / (14.8  * 24.0)               # Ra-225: 14.8 d (NNDC NuDat3)
DEFAULT_LAMBDA_AC_H   = _LN2 / (9.920 * 24.0)               # Ac-225: 9.920 d
DEFAULT_LAMBDA_227_H  = _LN2 / (42.2 / 60.0)                # Ra-227: 42.2 min
DEFAULT_LAMBDA_AC7_H  = _LN2 / (21.772 * 365.25 * 24.0)     # Ac-227: 21.772 y

# Transient equilibrium ratio (correctly named — Ra-225/Ac-225 is transient, not secular)
TRANSIENT_EQ_RATIO = DEFAULT_LAMBDA_225_H / (DEFAULT_LAMBDA_AC_H - DEFAULT_LAMBDA_225_H)
SECULAR_EQ_RATIO   = TRANSIENT_EQ_RATIO   # legacy alias

# -- Scaling defaults (MUST match train.py) ------------------------------------
DEFAULT_N226_SCALE   = 6.022e23
DEFAULT_N225_SCALE   = 1e20
DEFAULT_NAC_SCALE    = 1e20
DEFAULT_N227_SCALE   = 1e18     # Ra-227 peaks much lower (short half-life)
DEFAULT_NAC227_SCALE = 1e18     # Ac-227 accumulates slowly
DEFAULT_PHI_SCALE    = 1e15
DEFAULT_T_REF_H      = 500.0

DEFAULT_SIGMA_N2N    = 27e-27    # (n,2n) 27 mb spectrum-avg fast reactor (JENDL-5)
                                 # threshold ~6.42 MeV — ZERO for thermal neutrons
DEFAULT_SIGMA_NGAMMA = 12.8e-24  # (n,γ) 12.8 barn thermal reference (ENDF/B-VIII.0)
E_THRESHOLD_N2N_EV   = 6.42e6   # (n,2n) threshold [eV] — ENDF/B-VIII.0
THRESHOLD_PHYSICS_WIDTH_EV = 5.0e5
THRESHOLD_HEAD_GATE_WIDTH_EV = 1.25e5

SECONDS_PER_HOUR = 3_600.0
N_INPUT_FEATURES = 8
N_OUTPUT_SPECIES = 5
N_ENERGY_REGIME_FEATURES = 4

_ENERGY_EV_CLIP = (1e-12, 2e7)  # 20 MeV upper limit covers fast neutron range


def neutron_energy_ev_to_feature_torch(energy_ev: torch.Tensor) -> torch.Tensor:
    """Network input col 2: sqrt(E_ref / E) -- 1/v scaling for (n,γ) channel."""
    E = energy_ev.clamp(min=_ENERGY_EV_CLIP[0], max=_ENERGY_EV_CLIP[1])
    ref = torch.as_tensor(THERMAL_REFERENCE_EV, dtype=E.dtype, device=E.device)
    return torch.sqrt(ref / E)


def _integrate_bateman_ra225_ac225(
    n226_traj: torch.Tensor,
    t_eval: torch.Tensor,
    n225_0: torch.Tensor,
    nac_0: torch.Tensor,
    k_n2n: torch.Tensor,
    *,
    lam225: float,
    lam_ac: float,
    t_ref_h: float = DEFAULT_T_REF_H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semi-analytic Ra-225 / Ac-225 chain along the PINN time grid (atoms)."""
    batch, steps, _ = n226_traj.shape
    n225 = n225_0.unsqueeze(1).expand(batch, steps, 1).clone()
    nac = nac_0.unsqueeze(1).expand(batch, steps, 1).clone()
    lam225_t = n226_traj.new_tensor(lam225)
    lam_ac_t = n226_traj.new_tensor(lam_ac)
    k_n2n_b = k_n2n.view(batch, 1, 1)
    for i in range(1, steps):
        dt_h = (t_eval[:, i : i + 1, :] - t_eval[:, i - 1 : i, :]) * t_ref_h
        n226_prev = n226_traj[:, i - 1 : i, :]
        prod225 = k_n2n_b * n226_prev
        n225_prev = n225[:, i - 1 : i, :]
        nac_prev = nac[:, i - 1 : i, :]
        n225[:, i : i + 1, :] = n225_prev + dt_h * (prod225 - lam225_t * n225_prev)
        nac[:, i : i + 1, :] = nac_prev + dt_h * (lam225_t * n225_prev - lam_ac_t * nac_prev)
    n226_last = n226_traj[:, -1:, :]
    n225_last = n225[:, -1:, :]
    nac_last = nac[:, -1:, :]
    v225_last = k_n2n_b * n226_last - lam225_t * n225_last
    vac_last = lam225_t * n225_last - lam_ac_t * nac_last
    return n225, nac, v225_last, vac_last


def _integrate_bateman_ra227_ac227(
    n226_traj: torch.Tensor,
    t_eval: torch.Tensor,
    n227_0: torch.Tensor,
    nac7_0: torch.Tensor,
    k_ngamma: torch.Tensor,
    *,
    lam227: float,
    lam_ac7: float,
    t_ref_h: float = DEFAULT_T_REF_H,
    max_substep_h: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Semi-analytic Ra-227 / Ac-227 impurity chain along the PINN time grid (atoms)."""
    batch, steps, _ = n226_traj.shape
    n227 = n227_0.unsqueeze(1).expand(batch, steps, 1).clone()
    nac7 = nac7_0.unsqueeze(1).expand(batch, steps, 1).clone()
    lam227_t = n226_traj.new_tensor(lam227)
    lam_ac7_t = n226_traj.new_tensor(lam_ac7)
    k_ng_b = k_ngamma.view(batch, 1, 1)
    for i in range(1, steps):
        dt_h = (t_eval[:, i : i + 1, :] - t_eval[:, i - 1 : i, :]) * t_ref_h
        n226_prev = n226_traj[:, i - 1 : i, :]
        n227_prev = n227[:, i - 1 : i, :].clone()
        nac7_prev = nac7[:, i - 1 : i, :].clone()
        n_sub = torch.clamp((dt_h / max_substep_h).ceil().to(torch.long), min=1)
        n_sub_max = int(n_sub.max().item())
        sub_dt = dt_h / float(n_sub_max)
        prod227 = k_ng_b * n226_prev
        for _ in range(n_sub_max):
            e227 = torch.exp(-lam227_t * sub_dt)
            n227_prev = n227_prev * e227 + prod227 / lam227_t.clamp(min=1e-30) * (1.0 - e227)
            feed_ac7 = lam227_t * n227_prev
            eac7 = torch.exp(-lam_ac7_t * sub_dt)
            nac7_prev = nac7_prev * eac7 + feed_ac7 / lam_ac7_t.clamp(min=1e-30) * (1.0 - eac7)
        n227[:, i : i + 1, :] = n227_prev.clamp(min=0.0)
        nac7[:, i : i + 1, :] = nac7_prev.clamp(min=0.0)
    n226_last = n226_traj[:, -1:, :]
    n227_last = n227[:, -1:, :]
    nac7_last = nac7[:, -1:, :]
    v227_last = k_ng_b * n226_last - lam227_t * n227_last
    vac7_last = lam227_t * n227_last - lam_ac7_t * nac7_last
    return n227, nac7, v227_last, vac7_last


def n2n_threshold_scale_torch(
    energy_ev: torch.Tensor,
    *,
    width_ev: float = THRESHOLD_PHYSICS_WIDTH_EV,
) -> torch.Tensor:
    """
    Smooth sigmoid threshold for (n,2n) reaction — fires above 6.42 MeV only.
    Returns value in [0, 1]; zero for thermal neutrons.
    """
    E_thresh = torch.as_tensor(float(E_THRESHOLD_N2N_EV), dtype=energy_ev.dtype, device=energy_ev.device)
    width    = torch.as_tensor(float(width_ev), dtype=energy_ev.dtype, device=energy_ev.device)
    return torch.sigmoid((energy_ev - E_thresh) / width)


def energy_regime_features_from_input_torch(energy_feature: torch.Tensor) -> torch.Tensor:
    """
    Threshold-aware features derived from the existing energy input column.

    The raw energy column is sqrt(E_ref/E), which is excellent for thermal
    capture but compresses the entire fast-neutron range into tiny values.
    These features give the network an explicit handle on the 6.42 MeV onset
    and the D-T fast-neutron regime without changing the public 8-column input.
    """
    e_feature = energy_feature.clamp(min=1e-8, max=1e6)
    dtype = e_feature.dtype
    dev = e_feature.device
    e_ev = torch.as_tensor(THERMAL_REFERENCE_EV, dtype=dtype, device=dev) / (e_feature.square() + 1e-30)
    e_ev = e_ev.clamp(min=_ENERGY_EV_CLIP[0], max=_ENERGY_EV_CLIP[1])

    threshold = torch.as_tensor(E_THRESHOLD_N2N_EV, dtype=dtype, device=dev)
    fast_center = torch.as_tensor(14.0e6, dtype=dtype, device=dev)
    fast_width = torch.as_tensor(3.0e6, dtype=dtype, device=dev)
    high_span = torch.as_tensor(_ENERGY_EV_CLIP[1] - E_THRESHOLD_N2N_EV, dtype=dtype, device=dev)

    log_energy = torch.log10(e_ev).sub(math.log10(E_THRESHOLD_N2N_EV)).div(4.0).clamp(min=-1.5, max=1.5)
    threshold_gate = n2n_threshold_scale_torch(e_ev, width_ev=THRESHOLD_HEAD_GATE_WIDTH_EV)
    above_threshold = ((e_ev - threshold) / high_span.clamp(min=1.0)).clamp(min=0.0, max=1.0)
    fast14_bump = torch.exp(-0.5 * ((e_ev - fast_center) / fast_width).square())
    return torch.cat([log_energy, threshold_gate, above_threshold, fast14_bump], dim=-1)


class FourierTimeEncoder(nn.Module):
    """
    Log-spaced Fourier features for the time input.
    Replaces sigmoid(t*1000) step function with smooth multi-frequency encoding.
    Covers stiffness ratio λ_Ra227/λ_Ac225 ≈ 338 → needs ≥8 octaves of frequency.
    16 frequencies (32 features: sin+cos) span from 0.01 to 10 rad per t_ref.
    """
    def __init__(self, n_freqs: int = 16, t_ref_h: float = 500.0) -> None:
        super().__init__()
        freqs = torch.logspace(-2, 1, n_freqs) * math.pi / t_ref_h
        self.register_buffer('freqs', freqs)
        self.out_dim = 2 * n_freqs

    def forward(self, t_nn: torch.Tensor) -> torch.Tensor:
        # t_nn: (batch, 1) normalized time
        angles = t_nn * self.freqs.unsqueeze(0)   # (batch, n_freqs)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch, 32)


class FourierEnergyEncoder(nn.Module):
    """
    Low-dimensional Fourier / sinusoidal embedding of log10(neutron energy).

    Tancik et al., NeurIPS 2020 — improves MLP approximation of sharp spectral
    features (here: the fast rise of (n,2n) above threshold) in low input
    dimension. Uses the same public energy column as :func:`energy_regime_features_from_input_torch`.

    We use **fixed log-spaced** frequency coefficients (registered buffer), not a
    refreshed random matrix B, so checkpoints stay deterministic. This follows
    eq. (4)–(5) in Tancik *et al.*, with a structured choice of B analogous to
    the "generalized Fourier features" construction.
    """

    def __init__(self, n_freqs: int = 8, *, e_thresh_ev: float = E_THRESHOLD_N2N_EV) -> None:
        super().__init__()
        if n_freqs < 1:
            raise ValueError("n_freqs must be >= 1 for FourierEnergyEncoder")
        # Roughly one decade below / above the (n,2n) threshold on the log10 scale
        freqs = torch.logspace(-0.7, 0.9, n_freqs) * math.pi
        self.register_buffer("freqs", freqs)
        self.out_dim = 2 * n_freqs
        # pyrefly: ignore [unnecessary-type-conversion]
        self.e_thresh_ev = float(e_thresh_ev)

    def forward(self, energy_feature_col: torch.Tensor) -> torch.Tensor:
        """energy_feature_col: sqrt(E_ref / E_nn) — same as inputs[:, 2:3]."""
        e_feature = energy_feature_col.clamp(min=1e-8, max=1e6)
        dtype = e_feature.dtype
        dev = e_feature.device
        e_ev = torch.as_tensor(THERMAL_REFERENCE_EV, dtype=dtype, device=dev) / (e_feature.square() + 1e-30)
        e_ev = e_ev.clamp(min=_ENERGY_EV_CLIP[0], max=_ENERGY_EV_CLIP[1])
        z = torch.log10(e_ev)
        z0 = math.log10(self.e_thresh_ev)
        z = ((z - z0) / 1.5).clamp(min=-4.0, max=4.0)
        # pyrefly: ignore [not-callable]
        angles = z * self.freqs.to(device=dev, dtype=dtype).unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class AdaptiveActivation(nn.Module):
    """Tanh with learnable slope: ``tanh(a * x)``.

    Jagtap, Kawaguchi & Karniadakis, J. Comput. Phys. **404** (2020) 109136.
    Initialised at ``a = 1`` (standard tanh); the optimiser is free to sharpen
    (``a > 1``) or flatten (``a < 1``) the activation per layer.
    """

    def __init__(self, initial_a: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(initial_a))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.a * x)


class ResidualBlock(nn.Module):
    """Post-activation residual block: ``h + dropout(act(linear(h)))``.

    He et al., CVPR 2016. Skip connections improve gradient flow for deeper
    PINN architectures. Each block uses :class:`AdaptiveActivation`.
    """

    def __init__(
        self,
        dim: int,
        *,
        dropout_p: float = 0.0,
        initial_a: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = AdaptiveActivation(initial_a)
        self.drop: nn.Module = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.linear(x)))


def neutron_energy_ev_to_feature_numpy(energy_ev: np.ndarray | float) -> np.ndarray:
    """Same mapping for NumPy pipelines."""
    a = np.asarray(energy_ev, dtype=np.float64)
    a = np.clip(a, _ENERGY_EV_CLIP[0], _ENERGY_EV_CLIP[1])
    return np.sqrt(THERMAL_REFERENCE_EV / a)


def _state_dict_key(state: dict[str, torch.Tensor], suffix: str) -> str | None:
    """Find a checkpoint key even if torch.compile prefixed module names."""
    for key in state:
        if key == suffix or key.endswith(f".{suffix}"):
            return key
    return None


def normalize_isotope_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip common wrapper prefixes from IsotopePINN checkpoints."""
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("_orig_mod.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


# ==============================================================================
# Model
# ==============================================================================
class IsotopePINN(nn.Module):
    """
    MLP with hard initial-condition (IC) constraint for 5 species::

        N_nn(t) = N_nn_0 + t_nn * NN(x)

    8 inputs, 5 outputs: Ra-226, Ra-225, Ac-225, Ra-227, Ac-227.
    forward_raw: physics training (allows slight negatives for gradient flow).
    forward:     inference with hard budget cap and non-negativity clamp.

    MC Dropout: set ``dropout_p > 0`` to insert dropout between hidden layers.
    Call :func:`predict_mcd` for uncertainty-aware inference.

    Optional ``FourierEnergyEncoder`` (Tancik et al., 2020) appends sin/cos features
    of log10(E) to help capture threshold shapes without changing the 8-column API.
    """

    def __init__(
        self,
        hidden_dim:  int   = 128,
        n_hidden:    int   = 4,
        *,
        signed_rate: bool  = True,
        t_zero_eps:  float = 0.001,
        dropout_p:   float = 0.0,
        n_fourier_freqs: int = 16,
        n_energy_fourier_freqs: int = 8,
        n_energy_features: int = N_ENERGY_REGIME_FEATURES,
        regime_gated: bool = True,
        t_ref_h: float = 500.0,
        use_residual: bool = True,
        use_exponential_decay: bool = True,
    ) -> None:
        super().__init__()
        if n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")

        self.time_encoder = FourierTimeEncoder(n_freqs=n_fourier_freqs, t_ref_h=t_ref_h)
        self.n_energy_features = int(n_energy_features)
        self.n_energy_fourier_freqs = int(n_energy_fourier_freqs)
        if self.n_energy_features not in (0, N_ENERGY_REGIME_FEATURES):
            raise ValueError(
                f"n_energy_features must be 0 or {N_ENERGY_REGIME_FEATURES}, got {n_energy_features}"
            )
        if self.n_energy_fourier_freqs < 0:
            raise ValueError("n_energy_fourier_freqs must be >= 0")
        self.regime_gated = bool(regime_gated)
        if self.regime_gated and self.n_energy_features == 0:
            raise ValueError("regime_gated=True requires threshold energy features.")

        self.energy_fourier_encoder: FourierEnergyEncoder | None
        if self.n_energy_fourier_freqs > 0:
            self.energy_fourier_encoder = FourierEnergyEncoder(n_freqs=self.n_energy_fourier_freqs)
        else:
            self.energy_fourier_encoder = None

        # Network input: 8 + scalar regime features + optional Fourier-on-E + Fourier time
        aug = self.n_energy_features + self.time_encoder.out_dim
        if self.energy_fourier_encoder is not None:
            aug += self.energy_fourier_encoder.out_dim
        in_dim_total = N_INPUT_FEATURES + aug

        # --- Hidden layers: residual (v3) or flat sequential (legacy) ---
        self.use_residual = bool(use_residual)
        self.use_exponential_decay = bool(use_exponential_decay)
        if self.use_residual:
            # Projection: map augmented input to hidden_dim
            self.projection = nn.Linear(in_dim_total, hidden_dim)
            self.projection_act = AdaptiveActivation(1.0)
            # Residual blocks for layers 2..n_hidden (He et al. CVPR 2016)
            self.res_blocks = nn.Sequential(*[
                ResidualBlock(hidden_dim, dropout_p=dropout_p)
                for _ in range(n_hidden - 1)
            ])
        else:
            # Legacy flat sequential (for loading old checkpoints)
            layers: list[nn.Module] = []
            in_dim = in_dim_total
            for i in range(n_hidden):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if n_fourier_freqs == 0:
                    layers.append(nn.SiLU())
                else:
                    layers.append(nn.Tanh())
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(p=dropout_p))
                in_dim = hidden_dim
            self.hidden = nn.Sequential(*layers)
        if self.regime_gated:
            self.low_head  = nn.Linear(hidden_dim, N_OUTPUT_SPECIES)
            self.high_head = nn.Linear(hidden_dim, N_OUTPUT_SPECIES)
            # Learnable temperature for regime gate: softplus-based smooth transition
            # Lowered from 4.0 → 2.0 so the optimizer learns the right sharpness
            self.gate_temperature = nn.Parameter(torch.tensor(2.0))
        else:
            self.head = nn.Linear(hidden_dim, N_OUTPUT_SPECIES)  # legacy single-head checkpoints
        self.signed_rate  = signed_rate
        self.t_zero_eps   = t_zero_eps
        self.n_fourier_freqs = n_fourier_freqs
        self.is_legacy_no_fourier = n_fourier_freqs == 0
        self.rate_abs_max = 10.0 if self.is_legacy_no_fourier else 1.0
        self.dropout_p    = dropout_p

        # Initialize heads so the model starts at near-zero production rates.
        # Daughter species (channels 1..4) use tanh(raw) for the rate;
        # tanh(0) = 0 means zero initial production/decay rate (steady state).
        # Ra-226 (channel 0) uses softplus(raw) for the decay constant; softplus(-10) ≈ 4.5e-5
        # → out226 ≈ n0 * 0.99996 at init (correct: T_1/2 = 1600 yr, negligible burnup over hours).
        _RA226_INIT_BIAS = -10.0
        _DAUGHTER_INIT_BIAS = 0.0
        with torch.no_grad():
            if self.regime_gated:
                self.low_head.weight.fill_(0.0)
                self.low_head.bias.fill_(_DAUGHTER_INIT_BIAS)
                self.low_head.bias[0] = _RA226_INIT_BIAS
                self.high_head.weight.fill_(0.0)
                self.high_head.bias.fill_(_DAUGHTER_INIT_BIAS)
                self.high_head.bias[0] = _RA226_INIT_BIAS
            else:
                self.head.weight.fill_(0.0)
                self.head.bias.fill_(_DAUGHTER_INIT_BIAS)
                self.head.bias[0] = _RA226_INIT_BIAS

        # Per-species learnable rate scale multiplier for daughter species (Ra-225, Ac-225, Ra-227, Ac-227).
        # We parameterize via log-scale so the multiplier is always positive. Initialize to 0.0 (multiplier=1.0).
        # The raw parameter is stored as _raw_log_scales; the property clamps it to [-3, 3] via tanh
        # (exp range [0.05, 20]) so the optimizer can never fully suppress a daughter channel.
        self._raw_log_scales = nn.Parameter(torch.full((4,), 0.0, dtype=torch.float32))

    @property
    def daughter_rate_log_scales(self) -> torch.Tensor:
        """Log-scales clamped to [0, 3] so multipliers stay in [1, 20] — never suppress daughters."""
        return torch.clamp(torch.tanh(self._raw_log_scales) * 3.0, min=0.0)

    def _energy_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_energy_features == 0:
            return x.new_zeros((x.size(0), 0))
        return energy_regime_features_from_input_torch(x[:, 2:3])

    def _threshold_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_energy_features == 0:
            return x.new_zeros((x.size(0), 1))
        return self._energy_features(x)[:, 1:2]

    def _raw_to_rate(self, raw: torch.Tensor) -> torch.Tensor:
        # Reduced-PINN: network output IS the derivative — no clamp.
        return raw if self.signed_rate else F.softplus(raw)

    def forward_raw(
        self,
        x: torch.Tensor,
        *,
        return_experts: bool = False,
        return_derivatives: bool = False,
    ) -> torch.Tensor | tuple:
        """Forward pass using Reduced-PINN Integral formulation.

        If return_derivatives is True, appends v_pred to the output tuple.
        """
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        t_nn_max = x[:, 0:1].contiguous()
        n0 = x[:, 3:8]

        # Integral-Based PINN formulation: time grid for trapezoidal integration
        # The number of steps is kept small (20) to maintain training speed
        steps = 20
        t_grid = torch.linspace(0.0, 1.0, steps, device=x.device, dtype=x.dtype).view(1, steps, 1)
        t_eval = t_nn_max.unsqueeze(1) * t_grid  # (batch, steps, 1)

        x_expand = x.unsqueeze(1).repeat(1, steps, 1)
        x_expand[:, :, 0:1] = t_eval

        x_flat = x_expand.view(-1, 8)
        
        e_features = self._energy_features(x_flat)
        parts = [x_flat, e_features]
        if self.energy_fourier_encoder is not None:
            parts.append(self.energy_fourier_encoder(x_flat[:, 2:3]))
        parts.append(self.time_encoder(x_flat[:, 0:1]))
        x_aug = torch.cat(parts, dim=-1)

        h = self.projection_act(self.projection(x_aug)) if self.use_residual else self.hidden(x_aug)
        if self.use_residual:
            h = self.res_blocks(h)

        if self.regime_gated:
            gate_base = e_features[:, 1:2]
            gate_e = torch.sigmoid(self.gate_temperature * (gate_base - 0.5) * 6.0)
            raw_low = self.low_head(h)
            raw_high = self.high_head(h)
            raw = raw_low * (1.0 - gate_e) + raw_high * gate_e
            
            # --- Hybrid Integral-Exponential PINN formulation (ISEF Research Grade) ---
            rate_low_flat  = raw_low
            rate_high_flat = raw_high
            rate_blend_flat = raw
        else:
            raw = self.head(h)
            rate_blend_flat = raw
            rate_low_flat = rate_high_flat = raw

        def _assemble_state(r_flat, x_in, t_ev, n0_in, steps_in):
            # Reshape to (batch, steps, 5)
            r_grid = r_flat.view(x_in.shape[0], steps_in, 5)
            
            # Species 0 (Ra-226): predict positive rate constant k(t)
            # Use softplus for k >= 0
            k226 = F.softplus(r_grid[:, :, 0:1])
            
            # Species 1-4: predict signed net derivative v(t).
            # We use dynamic, input-dependent physics-based scaling: S_i(x) * tanh(raw).
            # tanh allows BOTH positive (ingrowth) and negative (decay) derivatives,
            # which is essential — Ra-225 decays, Ac-225 decays. softplus would
            # force all-positive derivatives, causing alchemy (atoms created from nothing).
            phi_nn = x_in[:, 1:2]
            init226 = x_in[:, 3:4]
            init225 = x_in[:, 4:5]
            init_ac = x_in[:, 5:6]
            init227 = x_in[:, 6:7]
            init_ac7 = x_in[:, 7:8]

            # Raw physical values
            phi_raw = phi_nn * float(DEFAULT_PHI_SCALE)
            n226_0 = init226 * float(DEFAULT_N226_SCALE)
            n225_0 = init225 * float(DEFAULT_N225_SCALE)
            nac_0 = init_ac * float(DEFAULT_NAC_SCALE)
            n227_0 = init227 * float(DEFAULT_N227_SCALE)
            nac227_0 = init_ac7 * float(DEFAULT_NAC227_SCALE)

            # Constants
            sigma_n2n = float(DEFAULT_SIGMA_N2N)
            sigma_ngamma = float(DEFAULT_SIGMA_NGAMMA)
            lam225 = float(DEFAULT_LAMBDA_225_H)
            lam_ac = float(DEFAULT_LAMBDA_AC_H)
            lam227 = float(DEFAULT_LAMBDA_227_H)
            lam_ac7 = float(DEFAULT_LAMBDA_AC7_H)
            T = float(DEFAULT_T_REF_H) # max irradiation time: 500.0 hours

            # Max potential inventories over the time interval (conservative bounds)
            # flux-based reaction rate is in s^-1, convert to h^-1 by multiplying by SECONDS_PER_HOUR (3600.0)
            flux_rate_n2n = phi_raw * sigma_n2n * float(SECONDS_PER_HOUR)
            flux_rate_ngamma = phi_raw * sigma_ngamma * float(SECONDS_PER_HOUR)

            max_226 = n226_0
            max_225 = n225_0 + (flux_rate_n2n * max_226) * T
            max_ac  = nac_0 + (lam225 * max_225) * T
            max_227 = n227_0 + (flux_rate_ngamma * max_226) * T
            max_ac7 = nac227_0 + (lam227 * max_227) * T

            # Max physical rates (atoms/hour)
            rate_225 = flux_rate_n2n * max_226 + lam225 * max_225
            rate_ac  = lam225 * max_225 + lam_ac * max_ac
            rate_227 = flux_rate_ngamma * max_226 + lam227 * max_227
            rate_ac7 = lam227 * max_227 + lam_ac7 * max_ac7

            # Convert to normalized units/hour and scale by T_REF (500) for t_nn derivative
            scale_225 = T * rate_225 / float(DEFAULT_N225_SCALE)
            scale_ac  = T * rate_ac / float(DEFAULT_NAC_SCALE)
            scale_227 = T * rate_227 / float(DEFAULT_N227_SCALE)
            scale_ac7 = T * rate_ac7 / float(DEFAULT_NAC227_SCALE)

            # Combine and add per-species floors to prevent zero-scale problems.
            dyn_scales = torch.cat([scale_225, scale_ac, scale_227, scale_ac7], dim=1)
            min_floors = dyn_scales.new_tensor([1e-2, 1e-2, 1e-4, 1e-4])
            dyn_scales = torch.max(dyn_scales, min_floors.view(1, 4))

            multipliers = torch.exp(self.daughter_rate_log_scales).to(dtype=r_grid.dtype, device=r_grid.device)
            scales = multipliers.view(1, 1, 4) * dyn_scales.unsqueeze(1)

            v_signed = scales * torch.tanh(r_grid[:, :, 1:5])
            v_others = v_signed

            t_eval_expand = t_ev.expand_as(k226)
            integral_k = torch.trapezoid(k226, t_eval_expand, dim=1)
            k_cum = torch.cat(
                [torch.zeros_like(k226[:, :1, :]), torch.cumsum(k226[:, :-1, :] * (t_eval[:, 1:, :] - t_eval[:, :-1, :]), dim=1)],
                dim=1,
            ).clamp(max=80.0)
            n226_traj = n0_in[:, 0:1].unsqueeze(1) * torch.exp(-k_cum)
            out226 = n0_in[:, 0:1] * torch.exp(-integral_k)

            use_bateman = os.environ.get("PINN_BATEMAN_BACKBONE", "1").strip().lower() in ("1", "true", "yes", "on")
            if use_bateman:
                e_feature = x_in[:, 2:3].clamp(min=1e-8, max=1e6)
                e_ev = torch.as_tensor(THERMAL_REFERENCE_EV, dtype=r_grid.dtype, device=r_grid.device) / (
                    e_feature**2 + 1e-30
                )
                n2n_thresh = n2n_threshold_scale_torch(e_ev)
                k_n2n = _phi_sigma_per_hour(phi_raw, sigma_n2n, n2n_thresh).clamp(min=0.0, max=5.0e3)
                n225_traj, nac_traj, v225_atoms, vac_atoms = _integrate_bateman_ra225_ac225(
                    n226_traj * float(DEFAULT_N226_SCALE),
                    t_eval,
                    n225_0,
                    nac_0,
                    k_n2n,
                    lam225=lam225,
                    lam_ac=lam_ac,
                )
                # Shared log-correction preserves Ra-225 / Ac-225 budget ratio (no independent alchemy).
                raw_pair = 0.5 * (r_grid[:, :, 1:2] + r_grid[:, :, 2:3])
                corr = torch.exp(torch.clamp(torch.tanh(raw_pair) * 0.5, -0.5, 0.5))
                out225_atoms = n225_traj[:, -1, :] * corr[:, -1, :]
                out_ac_atoms = nac_traj[:, -1, :] * corr[:, -1, :]
                pair_budget_0 = n225_0 + nac_0
                pair_sum = out225_atoms + out_ac_atoms
                phi_zero = phi_raw < 1.0
                no226 = n226_0 <= 1.0
                clamp_decay = phi_zero & no226
                max_budget = pair_budget_0 * (1.0 - 1e-9)
                over = clamp_decay & (pair_sum > max_budget)
                scale = torch.where(over, max_budget / (pair_sum + 1e-30), torch.ones_like(pair_sum))
                out225_atoms = out225_atoms * scale
                out_ac_atoms = out_ac_atoms * scale
                out225 = out225_atoms / float(DEFAULT_N225_SCALE)
                out_ac = out_ac_atoms / float(DEFAULT_NAC_SCALE)

                imp_budget_0 = n227_0 + nac227_0
                has226_feed = n226_0 > 1.0e16
                ng_scale = x_in[:, 2:3].clamp(min=1e-8, max=1e6)
                k_ng = _phi_sigma_per_hour(phi_raw, sigma_ngamma, ng_scale).clamp(min=0.0, max=5.0e3)
                n227_traj, nac7_traj, v227_atoms, vac7_atoms = _integrate_bateman_ra227_ac227(
                    n226_traj * float(DEFAULT_N226_SCALE),
                    t_eval,
                    n227_0,
                    nac227_0,
                    k_ng,
                    lam227=lam227,
                    lam_ac7=lam_ac7,
                )
                raw_227_pair = 0.5 * (r_grid[:, :, 3:4] + r_grid[:, :, 4:5])
                corr227 = torch.exp(torch.clamp(torch.tanh(raw_227_pair) * 0.5, -0.5, 0.5))
                out227_atoms = n227_traj[:, -1, :] * corr227[:, -1, :]
                out_ac7_atoms = nac7_traj[:, -1, :] * corr227[:, -1, :]
                imp_sum = out227_atoms + out_ac7_atoms
                clamp_imp = phi_zero & no226
                max_imp = imp_budget_0 * (1.0 - 1e-9)
                over_imp = clamp_imp & (imp_sum > max_imp)
                scale_imp = torch.where(over_imp, max_imp / (imp_sum + 1e-30), torch.ones_like(imp_sum))
                out227_atoms = out227_atoms * scale_imp
                out_ac7_atoms = out_ac7_atoms * scale_imp
                out227_norm = out227_atoms / float(DEFAULT_N227_SCALE)
                out_ac7_norm = out_ac7_atoms / float(DEFAULT_NAC227_SCALE)
                combined_grid = torch.cat([k226, v_others], dim=2)
                integral_grid = torch.trapezoid(combined_grid, t_eval_expand.expand_as(combined_grid), dim=1)
                out227_int = n0_in[:, 3:5] + integral_grid[:, 3:5]
                out_227_bateman = torch.cat([out227_norm, out_ac7_norm], dim=1)
                out_227_fallback = torch.where(
                    (no226 & (imp_budget_0 <= 1.0)).view(-1, 1),
                    torch.zeros_like(out227_int),
                    out227_int,
                )
                feed_mask2 = has226_feed.view(-1, 1).expand(-1, 2)
                out_227 = torch.where(feed_mask2, out_227_bateman, out_227_fallback)
                out = torch.cat([out226, out225, out_ac, out_227], dim=1)
                v226_last = -k226[:, -1, :] * out226
                v225_last = (v225_atoms[:, 0, :] / float(DEFAULT_N225_SCALE)) * corr[:, -1, :]
                vac_last = (vac_atoms[:, 0, :] / float(DEFAULT_NAC_SCALE)) * corr[:, -1, :]
                v227_last = (v227_atoms[:, 0, :] / float(DEFAULT_N227_SCALE)) * corr227[:, -1, :]
                vac7_last = (vac7_atoms[:, 0, :] / float(DEFAULT_NAC227_SCALE)) * corr227[:, -1, :]
                feed_mask = has226_feed.view(-1, 1)
                v227_blend = torch.where(feed_mask, v227_last, v_others[:, -1, 2:3])
                vac7_blend = torch.where(feed_mask, vac7_last, v_others[:, -1, 3:4])
                v_others_last = torch.cat([v225_last, vac_last, v227_blend, vac7_blend], dim=1)
                v_p = torch.cat([v226_last, v_others_last], dim=1)
            else:
                combined_grid = torch.cat([k226, v_others], dim=2)
                integral_grid = torch.trapezoid(combined_grid, t_eval_expand.expand_as(combined_grid), dim=1)
                out226_fb = n0_in[:, 0:1] * torch.exp(-integral_grid[:, 0:1])
                out_others = n0_in[:, 1:5] + integral_grid[:, 1:5]
                out = torch.cat([out226_fb, out_others], dim=1)
                v226_last = -k226[:, -1, :] * out226_fb
                v_others_last = v_others[:, -1, :]
                v_p = torch.cat([v226_last, v_others_last], dim=1)
            
            return out, v_p

        output, v_pred = _assemble_state(rate_blend_flat, x, t_eval, n0, steps)

        if return_experts and self.regime_gated:
            out_low, _ = _assemble_state(rate_low_flat, x, t_eval, n0, steps)
            out_high, _ = _assemble_state(rate_high_flat, x, t_eval, n0, steps)
            
            if squeeze:
                if return_derivatives:
                    return output.squeeze(0), out_low.squeeze(0), out_high.squeeze(0), v_pred.squeeze(0)
                return output.squeeze(0), out_low.squeeze(0), out_high.squeeze(0)
            else:
                if return_derivatives:
                    return output, out_low, out_high, v_pred
                return output, out_low, out_high

        if return_derivatives:
            return (output.squeeze(0), v_pred.squeeze(0)) if squeeze else (output, v_pred)

        return output.squeeze(0) if squeeze else output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward: non-negativity clamp only.
        Hard budget cap REMOVED — it was a discontinuous step function that caused
        MC Dropout saw-blade jitter. Mass conservation enforced via soft penalty in loss.
        """
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        res = self.forward_raw(x)
        out = res[0] if isinstance(res, tuple) else res
        
        if out.dim() == 1:
            out = out.unsqueeze(0)
        out = out.clamp(min=0.0)
        empty_start = x[:, 3:8].abs().sum(dim=1, keepdim=True) < 1e-12
        out = torch.where(empty_start, torch.zeros_like(out), out)
        if self.is_legacy_no_fourier:
            scales = x.new_tensor([
                DEFAULT_N226_SCALE, DEFAULT_N225_SCALE, DEFAULT_NAC_SCALE,
                DEFAULT_N227_SCALE, DEFAULT_NAC227_SCALE,
            ])
            total_start = (x[:, 3:8] * scales).sum(dim=1, keepdim=True)
            total_pred = (out * scales).sum(dim=1, keepdim=True)
            factor = torch.minimum(torch.ones_like(total_pred), total_start / total_pred.clamp(min=1e-30))
            out = out * factor
        return out.squeeze(0) if squeeze else out


def infer_n_fourier_freqs_from_state_dict(state: dict[str, torch.Tensor]) -> int:
    """Infer the Fourier encoder width needed by a saved checkpoint."""
    freq_key = _state_dict_key(state, "time_encoder.freqs")
    if freq_key is not None:
        return int(state[freq_key].numel())
    # New residual arch: first linear is 'projection.weight'
    key = _state_dict_key(state, "projection.weight")
    if key is None:
        # Legacy flat sequential
        key = _state_dict_key(state, "hidden.0.weight")
    if key is None:
        raise ValueError("Checkpoint is missing projection/hidden.0 weight; cannot infer model layout.")
    input_width = int(state[key].shape[1])
    extra_width = input_width - N_INPUT_FEATURES
    if extra_width < 0 or extra_width % 2 != 0:
        raise ValueError(
            f"Checkpoint first-layer weight has unsupported input width {input_width}."
        )
    return extra_width // 2


def checkpoint_uses_regime_gates(state: dict[str, torch.Tensor]) -> bool:
    """New checkpoints have separate low/high output heads."""
    return _state_dict_key(state, "low_head.weight") is not None


def infer_n_energy_fourier_freqs_from_state_dict(state: dict[str, torch.Tensor]) -> int:
    key = _state_dict_key(state, "energy_fourier_encoder.freqs")
    return int(state[key].numel()) if key is not None else 0


def infer_n_energy_features_from_state_dict(
    state: dict[str, torch.Tensor],
    n_fourier_freqs: int,
    n_energy_fourier_freqs: int,
) -> int:
    """Infer scalar regime energy feature width (0 or 4) given spectral layouts."""
    key = _state_dict_key(state, "projection.weight")
    if key is None:
        key = _state_dict_key(state, "hidden.0.weight")
    if key is None:
        raise ValueError("Checkpoint is missing projection/hidden.0 weight; cannot infer model layout.")
    input_width = int(state[key].shape[1])
    n_energy_features = (
        input_width - N_INPUT_FEATURES - 2 * int(n_fourier_freqs) - 2 * int(n_energy_fourier_freqs)
    )
    if n_energy_features not in (0, N_ENERGY_REGIME_FEATURES):
        raise ValueError(
            f"Checkpoint first-layer weight has unsupported derived energy layout "
            f"(scalar width {n_energy_features}; input_width={input_width}, "
            f"n_time_fourier={n_fourier_freqs}, n_energy_fourier={n_energy_fourier_freqs})."
        )
    return n_energy_features


def checkpoint_has_dropout_layout(state: dict[str, torch.Tensor]) -> bool:
    """Dropout changes Linear layer indices in the checkpoint key names."""
    return _state_dict_key(state, "hidden.3.weight") is not None


def checkpoint_has_residual_layout(state: dict[str, torch.Tensor]) -> bool:
    """New v3 checkpoints use projection + ResidualBlock architecture."""
    return _state_dict_key(state, "projection.weight") is not None


def make_isotope_pinn_for_state_dict(
    state: dict[str, torch.Tensor],
    *,
    dropout_p: float = 0.0,
) -> tuple[IsotopePINN, dict[str, Any]]:
    """
    Build an IsotopePINN matching a checkpoint.

    Detects architecture version from checkpoint keys:
    - v3 (residual): ``projection.weight`` present → residual + exponential decay
    - v2 (flat sequential): ``hidden.0.weight`` present → flat + linear ansatz
    """
    n_fourier_freqs = infer_n_fourier_freqs_from_state_dict(state)
    n_energy_fourier_freqs = infer_n_energy_fourier_freqs_from_state_dict(state)
    n_energy_features = infer_n_energy_features_from_state_dict(
        state, n_fourier_freqs, n_energy_fourier_freqs
    )
    regime_gated = checkpoint_uses_regime_gates(state)
    has_residual = checkpoint_has_residual_layout(state)
    has_dropout_layout = checkpoint_has_dropout_layout(state)
    effective_dropout_p = float(dropout_p) if (has_dropout_layout or has_residual) else 0.0
    model = IsotopePINN(
        dropout_p=effective_dropout_p,
        n_fourier_freqs=n_fourier_freqs,
        n_energy_fourier_freqs=n_energy_fourier_freqs,
        n_energy_features=n_energy_features,
        regime_gated=regime_gated,
        use_residual=has_residual,
        use_exponential_decay=has_residual,
    )
    return model, {
        "n_fourier_freqs": n_fourier_freqs,
        "n_energy_fourier_freqs": n_energy_fourier_freqs,
        "n_energy_features": n_energy_features,
        "regime_gated": regime_gated,
        "use_residual": has_residual,
        "use_exponential_decay": has_residual,
        "dropout_p": effective_dropout_p,
        "checkpoint_has_dropout_layout": has_dropout_layout,
    }


def load_isotope_pinn_checkpoint(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
    dropout_p: float = 0.0,
) -> tuple[IsotopePINN, dict[str, Any]]:
    """Load an IsotopePINN checkpoint and return the matching model plus layout info."""
    raw_state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    state = normalize_isotope_state_dict(raw_state)
    # Remap old 'daughter_rate_log_scales' key to '_raw_log_scales' for backward compat.
    # Old checkpoints stored the raw parameter directly; new checkpoints use a property
    # backed by _raw_log_scales (bounded via tanh*3).
    if "daughter_rate_log_scales" in state and "_raw_log_scales" not in state:
        state["_raw_log_scales"] = state.pop("daughter_rate_log_scales")
    model, info = make_isotope_pinn_for_state_dict(state, dropout_p=dropout_p)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Allow missing keys that are new architecture features not in old checkpoints
    allowed_missing = set()
    if info["n_fourier_freqs"] == 0:
        allowed_missing.add("time_encoder.freqs")
    # gate_temperature is a new learnable parameter - old checkpoints won't have it
    if "gate_temperature" in missing:
        allowed_missing.add("gate_temperature")
    # _raw_log_scales is the renamed/reparameterized version of daughter_rate_log_scales.
    # Old checkpoints get remapped above; brand-new checkpoints initialize to zeros (multiplier=1.0).
    if "_raw_log_scales" in missing:
        allowed_missing.add("_raw_log_scales")
    real_missing = [key for key in missing if key not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint did not match inferred model. Missing={real_missing}, "
            f"unexpected={list(unexpected)}"
        )
    model.eval()
    return model, info


# ==============================================================================
# Vanilla NN (ablation control): same MLP, NO physics structure
# ==============================================================================
class VanillaIsotopeNN(nn.Module):
    """
    Plain MLP for ablation comparison.

    Same hidden layers / width as IsotopePINN but:
      - NO initial-condition constraint (N0 + t * rate)
      - NO hard mass-budget cap
      - NO non-negativity clamp at inference
    Trained purely on data MSE — no physics losses.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_hidden:   int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = N_INPUT_FEATURES
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)
        self.head   = nn.Linear(hidden_dim, N_OUTPUT_SPECIES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        out = self.head(self.hidden(x))
        return out.squeeze(0) if squeeze else out


# ==============================================================================
# Monte Carlo Dropout inference
# ==============================================================================
def predict_mcd(
    model: IsotopePINN,
    x: torch.Tensor,
    n_samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run ``n_samples`` stochastic forward passes with dropout **active**.

    Returns (mean, std) each of shape matching a single ``model(x)`` call.
    Dropout layers fire because we temporarily set model.train(); the
    non-dropout layers (BatchNorm, etc.) are not affected because we only
    toggle the Dropout modules.
    """
    was_training = model.training

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x))
    stack = torch.stack(preds, dim=0)
    mean = stack.mean(dim=0)
    std  = stack.std(dim=0)

    model.train(was_training)
    return mean, std


# -- Helpers -------------------------------------------------------------------
def _phi_sigma_per_hour(
    phi_raw:      torch.Tensor,
    sigma_cm2:    float,
    energy_scale: torch.Tensor,
) -> torch.Tensor:
    """Transmutation rate k [h^-1] = phi * sigma(E) * 3600."""
    return phi_raw * (sigma_cm2 * energy_scale) * SECONDS_PER_HOUR


def time_derivatives_finite_diff(
    forward_fn:  Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    pred:   torch.Tensor,
    *,
    eps:         float = 5e-4,
    t_nn_max:    float = 1.0e3,
) -> tuple[torch.Tensor, ...]:
    """dN_nn/dt_input by central finite difference for all 5 species."""
    t0 = inputs[:, 0:1]
    dt = inputs.new_tensor(eps)

    t_low_unclamped = t0 - dt
    t_high_unclamped = t0 + dt

    at_lower_bound = t_low_unclamped <= 0.0

    t_low_clamped = t_low_unclamped.clamp(min=0.0)
    t_high_clamped = t_high_unclamped.clamp(max=t_nn_max)
    denom_central = (t_high_clamped - t_low_clamped).clamp(min=dt * 0.5)

    low_in = torch.cat([t_low_clamped, inputs[:, 1:]], dim=1)
    high_in = torch.cat([t_high_clamped, inputs[:, 1:]], dim=1)
    pred_low = forward_fn(low_in)
    pred_high = forward_fn(high_in)

    d_central = (pred_high - pred_low) / denom_central

    t_fwd = t0 + 2.0 * dt
    t_fwd_clamped = t_fwd.clamp(max=t_nn_max)
    denom_fwd = (t_fwd_clamped - t0).clamp(min=dt)

    fwd_in = torch.cat([t_fwd_clamped, inputs[:, 1:]], dim=1)
    pred_fwd = forward_fn(fwd_in)

    d_fwd = (pred_fwd - pred) / denom_fwd

    d_all = torch.where(at_lower_bound.expand_as(d_central), d_fwd, d_central)
    d_all = d_all.clamp(min=-5.0e3, max=5.0e3)

    return tuple(d_all[:, i:i+1] for i in range(N_OUTPUT_SPECIES))


def _time_derivatives_wrt_hours(
    pred: torch.Tensor,
    inputs: torch.Tensor,
    t_ref_h: float,
) -> tuple[torch.Tensor, ...]:
    """
    ∂N_nn/∂t_hours from autograd w.r.t. normalized time t_nn = t_hours / t_ref_h.

    torch.autograd.grad yields ∂N/∂t_nn; chain rule gives
    ∂N/∂t_hours = (∂N/∂t_nn) / t_ref_h.
    """
    dtype = pred.dtype
    dev = pred.device
    nsp = pred.size(1)
    tr = torch.as_tensor(float(t_ref_h), dtype=dtype, device=dev)
    tr_safe = tr + torch.as_tensor(1e-8, dtype=dtype, device=dev)
    derivs: list[torch.Tensor] = []
    for s in range(nsp):
        # We completely removed torch.autograd.grad here.
        # This function is now deprecated in favor of Reduced-PINN v_pred output.
        # But we return zeros just in case a legacy test calls it.
        derivs.append(torch.zeros_like(pred[:, 0:1]))
    return tuple(derivs)


def compose_physics_point_weights(
    t_nn: torch.Tensor,
    fn1: torch.Tensor,
    fn2: torch.Tensor,
    fn3: torch.Tensor,
    fn4: torch.Tensor,
    fn5: torch.Tensor,
    *,
    causal_physics_time_scale: float,
    causal_curriculum_bins: int,
    causal_curriculum_progress: float,
    sa_physics_alpha: float,
    sa_mag_eps: float = 0.05,
) -> torch.Tensor:
    """
    Single composed per-point nonnegative weight for physics MSE (multiply ``fn_k^2``).

    **Composition policy (avoids contradictory triple-counting):**

    1. ``w_soft``: Wang et al. (arXiv:2203.07404) soft emphasis on small normalized
       time — encourages earlier irradiation dynamics before late-time dominance.
    2. ``w_curr``: Krishnapriyan et al. (NeurIPS 2021) **curriculum** — partition
       ``t_nn∈[0,1]`` into ``K`` bins; at progress ``p=0`` emphasize early bins,
       at ``p→1`` flatten toward uniform so late-time physics is not permanently
       starved.
    3. ``w_sa``: detached focal factor from residual RMS (McClenny & Braga-Neto
       SA-PINN spirit; arXiv:2009.04544) — up-weight stubborn high-residual points
       without backprop through the weights.

    Energy threshold boosting is applied **separately** to ``fn`` (before squaring);
    this tensor only handles **time / adaptation**. Final ``w`` is mean-normalized
    with a **detached** denominator so overall loss scale stays stable.
    """
    dtype = t_nn.dtype
    dev = t_nn.device
    w = torch.ones_like(t_nn)

    if causal_physics_time_scale > 0.0:
        w = w * (1.0 + causal_physics_time_scale * (1.0 - t_nn))

    kb = int(causal_curriculum_bins)
    if kb > 0:
        p = float(min(1.0, max(0.0, causal_curriculum_progress)))
        idx = (t_nn.clamp(0.0, 1.0) * float(kb)).long().clamp(max=kb - 1)
        k_float = idx.to(dtype=dtype)
        kf = float(kb)
        raw = p + (1.0 - p) * (kf - k_float) / kf
        w = w * raw

    if sa_physics_alpha > 0.0:
        mag_sq = (fn1.pow(2) + fn2.pow(2) + fn3.pow(2) + fn4.pow(2) + fn5.pow(2)) * (1.0 / 5.0)
        mag = torch.sqrt(mag_sq + sa_mag_eps)
        focus = mag.detach().pow(sa_physics_alpha).clamp(0.25, 8.0)
        w = w * focus

    denom = w.detach().mean().clamp(min=torch.as_tensor(1e-8, dtype=dtype, device=dev))
    w = w / denom
    return w


# ==============================================================================
# Combined loss: physics + data + mass + non-neg + secular eq (5-species)
# ==============================================================================
def compute_physics_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    pred: torch.Tensor,
    targets: torch.Tensor,
    *,
    v_pred: torch.Tensor | None = None,
    physics_weight: float = 1.0,
    sigma_n2n_cm2:         float               = DEFAULT_SIGMA_N2N,
    sigma_ngamma_cm2:      float               = DEFAULT_SIGMA_NGAMMA,
    energy_scale:          torch.Tensor | None = None,
    use_one_over_v_energy: bool                = True,
    lambda_226_h:          float               = DEFAULT_LAMBDA_226_H,
    lambda_225_h:          float               = DEFAULT_LAMBDA_225_H,
    lambda_ac_h:           float               = DEFAULT_LAMBDA_AC_H,
    lambda_227_h:          float               = DEFAULT_LAMBDA_227_H,
    lambda_ac7_h:          float               = DEFAULT_LAMBDA_AC7_H,
    data_weight:           float               = 1.0,
    mass_weight:           float               = 10.0,
    fuel_anchor_weight:    float               = 0.0,
    non_neg_weight:        float               = 50.0,
    secular_eq_weight:     float               = 25.0,
    ra225_physics_weight:  float               = 5.0,
    pred_for_data:         torch.Tensor | None = None,
    data_species_weights:  tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    log_data_weight:       float               = 1.0,
    log_species_weights:   tuple[float, ...] = (0.0, 2.0, 5.0, 2.0, 5.0),
    impurity_log_weight:   float               = 0.2,
    target_inventory_weight: float             = 2.0,
    trace_relative_weight: float               = 0.0,
    chain_consistency_weight: float            = 0.0,
    virgin_ac225_weight:   float               = 0.0,
    ac227_chain_weight:    float               = 0.0,
    f5_signal_weight:      float               = 0.0,
    empty_output_weight:   float               = 0.0,
    n226_scale:            float               = DEFAULT_N226_SCALE,
    n225_scale:            float               = DEFAULT_N225_SCALE,
    nac_scale:             float               = DEFAULT_NAC_SCALE,
    n227_scale:            float               = DEFAULT_N227_SCALE,
    nac227_scale:          float               = DEFAULT_NAC227_SCALE,
    phi_scale:             float               = DEFAULT_PHI_SCALE,
    t_ref_h:               float               = DEFAULT_T_REF_H,
    d_t_input_d_t_hours:   float               = 1.0 / DEFAULT_T_REF_H,
    atom_scale:            float | None        = None,
    # Energy-conditioned physics weighting for threshold boundary
    threshold_physics_boost: float             = 5.0,
    threshold_energy_low_ev: float              = 6.0e6,
    threshold_energy_high_ev: float             = 7.0e6,
    # XPINN-style expert agreement (Jagtap & Karniadakis, CICP 2020) — see module docstring
    pred_expert_low:      torch.Tensor | None    = None,
    pred_expert_high:     torch.Tensor | None    = None,
    xpinn_interface_weight: float               = 0.0,
    xpinn_interface_low_ev: float               = 6.35e6,
    xpinn_interface_high_ev: float            = 6.55e6,
    # Causal PINN (Wang & Perdikaris, arXiv:2203.07404): full method uses sequential
    # weights over time windows; composed with curriculum + SA in compose_physics_point_weights.
    causal_physics_time_scale: float            = 0.0,
    # Krishnapriyan et al. (NeurIPS 2021): time-bin curriculum along t_nn (0 = disabled).
    causal_curriculum_bins: int                 = 0,
    causal_curriculum_progress: float           = 1.0,
    # McClenny & Braga-Neto (arXiv:2009.04544): detached focal emphasis (0 = off).
    sa_physics_alpha: float                     = 0.0,
    sa_mag_eps: float                           = 0.05,
    # Legacy compatibility: callers may pass sigma_cm2 instead of sigma_n2n_cm2
    sigma_cm2:             float | None        = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Five-species PINN loss with both (n,2n) and (n,γ) channels.

    Time derivatives use ``t_ref_h`` (default 500 h) for ∂N/∂t_hours; ``d_t_input_d_t_hours``
    is kept only for backward-compatible call sites.
    """
    if sigma_cm2 is not None:
        sigma_n2n_cm2 = sigma_cm2
    _ = (model, d_t_input_d_t_hours, atom_scale)  # API / legacy kwargs; autograd uses ``t_ref_h``

    if inputs.dim() != 2 or inputs.size(-1) != N_INPUT_FEATURES:
        raise ValueError(
            f"inputs must be (batch, {N_INPUT_FEATURES}): "
            "[t, phi, E, n226_0, n225_0, nac_0, n227_0, nac227_0]"
        )
    nsp = N_OUTPUT_SPECIES
    if pred.dim() != 2 or pred.size(-1) != nsp or pred.size(0) != inputs.size(0):
        raise ValueError(f"pred must be (batch, {nsp}) matching inputs batch dim.")
    pred_data = pred
    if pred_for_data is not None:
        if pred_for_data.shape != pred.shape:
            raise ValueError(f"pred_for_data must match pred shape (batch, {nsp}).")
        pred_data = pred_for_data
    if pred_expert_low is not None or pred_expert_high is not None:
        if pred_expert_low is None or pred_expert_high is None:
            raise ValueError("pred_expert_low and pred_expert_high must be provided together.")
        for name, t in (("pred_expert_low", pred_expert_low), ("pred_expert_high", pred_expert_high)):
            if t.dim() != 2 or t.size(-1) != nsp or t.size(0) != inputs.size(0):
                raise ValueError(f"{name} must be (batch, {nsp}) matching inputs.")
    # Time derivatives come from v_pred (forward_raw with return_derivatives=True).
    # Requiring inputs.requires_grad=True forced an unnecessary autograd graph on collocation
    # tensors when v_pred was already supplied.

    dev   = pred.device
    dtype = pred.dtype

    phi_raw = inputs[:, 1:2] * phi_scale

    if energy_scale is None:
        if use_one_over_v_energy:
            energy_scale = inputs[:, 2:3].clamp(min=1e-8, max=1e6)
        else:
            energy_scale = torch.ones_like(phi_raw)

    n226   = pred[:, 0:1]
    n225   = pred[:, 1:2]
    nac    = pred[:, 2:3]
    n227   = pred[:, 3:4]
    nac227 = pred[:, 4:5]

    # -- time derivatives (5 species): from network output directly (Reduced-PINN)
    # The derivatives are passed directly from the forward_raw output!
    # No autograd required.
    # Note: v_pred must be passed into compute_physics_loss. 
    # If not passed (legacy call), we use zeros.
    if v_pred is not None:
        # The neural network outputs v(t_nn), which is ∂N/∂t_nn.
        # We must apply the chain rule: ∂N/∂t_hours = (∂N/∂t_nn) / t_ref_h.
        tr_safe = torch.as_tensor(float(t_ref_h), dtype=dtype, device=dev) + 1e-8
        d226 = v_pred[:, 0:1] / tr_safe
        d225 = v_pred[:, 1:2] / tr_safe
        dac  = v_pred[:, 2:3] / tr_safe
        d227 = v_pred[:, 3:4] / tr_safe
        dac7 = v_pred[:, 4:5] / tr_safe
    else:
        d226 = d225 = dac = d227 = dac7 = torch.zeros_like(pred[:, 0:1])

    # -- rate constants --------------------------------------------------------
    # (n,γ): 1/v scaling is correct physics for thermal capture
    # (n,2n): threshold model — zero below 6.42 MeV, sigmoid rise above
    if use_one_over_v_energy:
        ng_scale  = inputs[:, 2:3].clamp(min=1e-8, max=1e6)   # sqrt(E_ref/E) for (n,γ)
        # Convert nn energy feature back to eV for threshold calc: feature = sqrt(E_ref/E)
        # => E_eV = E_ref / feature^2
        e_feature = inputs[:, 2:3].clamp(min=1e-8, max=1e6)
        e_ev_approx = torch.as_tensor(THERMAL_REFERENCE_EV, dtype=dtype, device=dev) / (e_feature ** 2 + 1e-30)
        n2n_thresh = n2n_threshold_scale_torch(e_ev_approx, width_ev=THRESHOLD_PHYSICS_WIDTH_EV)
    else:
        ng_scale  = torch.ones_like(phi_raw)
        # Fallback for energy when one_over_v is off (should still use energy column for boost)
        e_ev_approx = inputs[:, 2] 
        n2n_thresh = torch.ones_like(phi_raw)
    k_n2n = _phi_sigma_per_hour(phi_raw, sigma_n2n_cm2, n2n_thresh).clamp(min=0.0, max=5.0e3)
    k_ng  = _phi_sigma_per_hour(phi_raw, sigma_ngamma_cm2, ng_scale).clamp(min=0.0, max=5.0e3)

    lam226 = torch.as_tensor(lambda_226_h, dtype=dtype, device=dev)
    lam225 = torch.as_tensor(lambda_225_h, dtype=dtype, device=dev)
    lam_ac = torch.as_tensor(lambda_ac_h,  dtype=dtype, device=dev)
    lam227 = torch.as_tensor(lambda_227_h, dtype=dtype, device=dev)
    lam_a7 = torch.as_tensor(lambda_ac7_h, dtype=dtype, device=dev)

    r226_225 = n226_scale / n225_scale
    r225_ac  = n225_scale / nac_scale
    r226_227 = n226_scale / n227_scale
    r227_ac7 = n227_scale / nac227_scale

    # -- Bateman residuals (5 equations) ---------------------------------------
    f1 = d226 + (lam226 + k_n2n + k_ng) * n226
    f2 = d225 - (k_n2n * n226 * r226_225 - lam225 * n225)
    f3 = dac  - (lam225 * n225 * r225_ac  - lam_ac * nac)
    f4 = d227 - (k_ng  * n226 * r226_227  - lam227 * n227)
    f5 = dac7 - (lam227 * n227 * r227_ac7 - lam_a7 * nac227)

    # -- Jacobian Normalization (Bento et al., arXiv:2602.21988, 2026) ----------
    # Previous approach divided residuals by N(t). When N(t) ≈ 0 for trace
    # daughters (Ac-225 from virgin IC), the divisor collapsed to epsilon,
    # making those residuals either vanish or explode — the "zero-collapse" bug.
    #
    # Jacobian normalization divides each residual by sqrt(1 + ||J_i||²),
    # where J_i is the i-th row of the ODE system Jacobian ∂f_i/∂y_j.
    # For our linear Bateman system, the Jacobian entries are known analytically
    # and depend only on decay constants, cross-sections, and scale ratios —
    # NOT on the predicted populations. This guarantees a stable, nonzero
    # denominator even when trace-daughter inventories are exactly zero.
    #
    # Effect: Ra-227 (λ≈0.985 h⁻¹, massive Jacobian) gets a large divisor,
    # while Ac-225 (λ≈2.9e-3 h⁻¹, small Jacobian) keeps a divisor near 1.0.
    # The optimizer now "sees" Ac-225 gradients instead of them being drowned.

    # Eq 1: r1 = d226 + (λ226 + k_n2n + k_ng) * n226
    #   ∂r1/∂n226 = (λ226 + k_n2n + k_ng)
    j1_sq = (lam226 + k_n2n + k_ng).pow(2)
    jn1 = torch.sqrt(1.0 + j1_sq)

    # Eq 2: r2 = d225 - k_n2n * n226 * r226_225 + λ225 * n225
    #   ∂r2/∂n226 = -k_n2n * r226_225,  ∂r2/∂n225 = λ225
    j2_sq = (k_n2n * r226_225).pow(2) + lam225.pow(2)
    jn2 = torch.sqrt(1.0 + j2_sq)

    # Eq 3: r3 = dac - λ225 * n225 * r225_ac + λ_ac * nac
    #   ∂r3/∂n225 = -λ225 * r225_ac,  ∂r3/∂nac = λ_ac
    j3_sq = (lam225 * r225_ac) ** 2 + lam_ac.pow(2)
    jn3 = torch.sqrt(1.0 + j3_sq)

    # Eq 4: r4 = d227 - k_ng * n226 * r226_227 + λ227 * n227
    #   ∂r4/∂n226 = -k_ng * r226_227,  ∂r4/∂n227 = λ227
    j4_sq = (k_ng * r226_227).pow(2) + lam227.pow(2)
    jn4 = torch.sqrt(1.0 + j4_sq)

    # Eq 5: r5 = dac7 - λ227 * n227 * r227_ac7 + λ_a7 * nac227
    #   ∂r5/∂n227 = -λ227 * r227_ac7,  ∂r5/∂nac227 = λ_a7
    j5_sq = (lam227 * r227_ac7) ** 2 + lam_a7.pow(2)
    jn5 = torch.sqrt(1.0 + j5_sq)

    fn1 = f1 / jn1
    fn2 = f2 / jn2
    fn3 = f3 / jn3
    fn4 = f4 / jn4
    fn5 = f5 / jn5

    tr = torch.as_tensor(t_ref_h, dtype=dtype, device=dev)

    # Energy-conditioned physics weighting: boost loss near threshold boundary
    if use_one_over_v_energy and threshold_physics_boost > 1.0:
        e_low = torch.as_tensor(threshold_energy_low_ev, dtype=dtype, device=dev)
        e_high = torch.as_tensor(threshold_energy_high_ev, dtype=dtype, device=dev)
        in_threshold_band = (e_ev_approx >= e_low) & (e_ev_approx <= e_high)
        if bool(in_threshold_band.any().item()):
            boost = torch.as_tensor(threshold_physics_boost, dtype=dtype, device=dev)
            # Apply boost only to points in the threshold band
            fn1 = torch.where(in_threshold_band, fn1 * boost, fn1)
            fn2 = torch.where(in_threshold_band, fn2 * boost, fn2)
            fn3 = torch.where(in_threshold_band, fn3 * boost, fn3)
            fn4 = torch.where(in_threshold_band, fn4 * boost, fn4)
            fn5 = torch.where(in_threshold_band, fn5 * boost, fn5)

    # All residuals now O(1) — equal contribution to loss
    _ = ra225_physics_weight  # kept for API compat; normalization replaces it

    t_nn_col = inputs[:, 0:1]
    w_t = compose_physics_point_weights(
        t_nn_col,
        fn1,
        fn2,
        fn3,
        fn4,
        fn5,
        causal_physics_time_scale=causal_physics_time_scale,
        causal_curriculum_bins=causal_curriculum_bins,
        causal_curriculum_progress=causal_curriculum_progress,
        sa_physics_alpha=sa_physics_alpha,
        sa_mag_eps=sa_mag_eps,
    )

    def _weighted_residual_mse(r: torch.Tensor) -> torch.Tensor:
        # Huber loss with delta=1.0: quadratic for small errors, linear for outliers.
        # This prevents gradient explosions while ensuring a persistent gradient for correction.
        h = F.huber_loss(r, torch.zeros_like(r), reduction='none', delta=1.0)
        return (h * w_t).mean()

    f1_mse = _weighted_residual_mse(fn1)
    f2_mse = _weighted_residual_mse(fn2)
    f3_mse = _weighted_residual_mse(fn3)
    f4_mse = _weighted_residual_mse(fn4)
    f5_mse = _weighted_residual_mse(fn5)
    f5_signal = ((inputs[:, 6:7] > 1e-12) | (inputs[:, 7:8] > 1e-12)).squeeze(1)
    if bool(f5_signal.any().item()):
        w_sig = w_t[f5_signal]
        r_sig = fn5[f5_signal]
        h_sig = F.huber_loss(r_sig, torch.zeros_like(r_sig), reduction='none', delta=1.0)
        f5_signal_loss = (h_sig * w_sig).mean()
    else:
        f5_signal_loss = torch.zeros((), device=dev, dtype=dtype)
    physics_mse = f1_mse + f2_mse + f3_mse + f4_mse + f5_mse + f5_signal_weight * f5_signal_loss

    xpinn_interface_loss = torch.zeros((), device=dev, dtype=dtype)
    if (
        xpinn_interface_weight > 0.0
        and pred_expert_low is not None
        and pred_expert_high is not None
        and use_one_over_v_energy
    ):
        e_lo = torch.as_tensor(xpinn_interface_low_ev, dtype=dtype, device=dev)
        e_hi = torch.as_tensor(xpinn_interface_high_ev, dtype=dtype, device=dev)
        in_if_band = (e_ev_approx >= e_lo) & (e_ev_approx <= e_hi)
        if bool(in_if_band.any().item()):
            xpinn_interface_loss = F.mse_loss(
                pred_expert_low[in_if_band.squeeze(-1)],
                pred_expert_high[in_if_band.squeeze(-1)],
            )

    # -- Ra-226 burnup anchor --------------------------------------------------
    n226_0_in = inputs[:, 3:4]
    s226_mass = torch.as_tensor(n226_scale, dtype=dtype, device=dev)
    raw226_0 = n226_0_in * s226_mass
    t_h = inputs[:, 0:1] * tr
    burn_exp = (-(lam226 + k_n2n + k_ng) * t_h).clamp(min=-80.0, max=80.0)
    n226_burn = n226_0_in * torch.exp(burn_exp)
    m_fuel = raw226_0.squeeze(-1) > 1.0e16
    if bool(m_fuel.any().item()):
        fuel_anchor_loss = F.mse_loss(n226[m_fuel], n226_burn[m_fuel])
    else:
        fuel_anchor_loss = torch.zeros((), device=dev, dtype=dtype)

    # -- data loss (5 species) — Robust normalized Huber loss ------------------
    # Huber loss on O(1) normalized data: avoids infinite log10 gradients near 0.
    # Quadratic for small errors, linear for outliers. Suppresses whale-tail.
    if targets is not None:
        if targets.shape != pred.shape:
            raise ValueError(f"targets must match pred shape (batch, {nsp}).")
        scales_t = torch.as_tensor(
            [n226_scale, n225_scale, nac_scale, n227_scale, nac227_scale],
            dtype=dtype, device=dev,
        )
        pred_data_c = pred_data.clamp(min=0.0)
        true_data_c = (targets.clamp(min=0.0) / scales_t)
        
        # Mask out NaN targets (RAR collocation points have no data target)
        valid_mask = ~torch.isnan(targets).any(dim=1)
        
        w = torch.as_tensor(data_species_weights[:nsp], dtype=dtype, device=dev).view(1, nsp)
        if valid_mask.any():
            # Huber loss in O(1) normalized space. delta=0.1 means errors > 10% transition to linear.
            huber = F.huber_loss(pred_data_c[valid_mask], true_data_c[valid_mask], reduction='none', delta=0.1)
            data_mse = (huber * w.expand_as(huber)).mean()

            # Log-space loss keeps trace daughters from being numerically invisible.
            true_atoms_v = targets[valid_mask].clamp(min=0.0)
            pred_atoms_v = (pred_data_c[valid_mask] * scales_t).clamp(min=0.0)
            atom_floor = torch.clamp(scales_t * 1e-16, min=1.0).view(1, nsp)
            log_true = torch.log10(true_atoms_v + atom_floor)
            log_pred = torch.log10(pred_atoms_v + atom_floor)
            log_w = torch.as_tensor(log_species_weights[:nsp], dtype=dtype, device=dev).view(1, nsp)
            log_huber = F.huber_loss(log_pred, log_true, reduction='none', delta=0.25)
            log_weight_sum = log_w.sum().clamp(min=1.0)
            log_data_loss = (log_huber * log_w.expand_as(log_huber)).sum() / (log_weight_sum * log_huber.size(0))

            lam_ac_s = lam_ac / torch.as_tensor(3600.0, dtype=dtype, device=dev)
            lam_a7_s = lam_a7 / torch.as_tensor(3600.0, dtype=dtype, device=dev)
            true_ac225_a = true_atoms_v[:, 2] * lam_ac_s
            pred_ac225_a = pred_atoms_v[:, 2] * lam_ac_s
            true_ac227_a = true_atoms_v[:, 4] * lam_a7_s
            pred_ac227_a = pred_atoms_v[:, 4] * lam_a7_s
            activity_floor = torch.as_tensor(1e-12, dtype=dtype, device=dev)
            true_imp = true_ac227_a / (true_ac225_a + true_ac227_a + activity_floor)
            pred_imp = pred_ac227_a / (pred_ac225_a + pred_ac227_a + activity_floor)
            imp_signal = (true_ac225_a + true_ac227_a) > activity_floor
            if imp_signal.any():
                impurity_log_loss = F.huber_loss(
                    torch.log10(pred_imp[imp_signal].clamp(min=1e-7)),
                    torch.log10(true_imp[imp_signal].clamp(min=1e-7)),
                    reduction='mean',
                    delta=0.5,
                )
            else:
                impurity_log_loss = torch.zeros((), device=dev, dtype=dtype)

            trace_relative_loss = torch.zeros((), device=dev, dtype=dtype)
            trace_mask = true_atoms_v[:, 2:5] > atom_floor[:, 2:5]
            if trace_mask.any():
                trace_rel = (pred_atoms_v[:, 2:5] - true_atoms_v[:, 2:5]).abs() / true_atoms_v[:, 2:5].clamp(
                    min=atom_floor[:, 2:5]
                )
                trace_relative_loss = F.huber_loss(
                    torch.log10(trace_rel[trace_mask] + 1.0),
                    torch.zeros_like(trace_rel[trace_mask]),
                    reduction='mean',
                    delta=0.25,
                )

            chain_target_loss = torch.zeros((), device=dev, dtype=dtype)
            ra225_ac225_total_true = true_atoms_v[:, 1] + true_atoms_v[:, 2]
            ra225_ac225_total_pred = pred_atoms_v[:, 1] + pred_atoms_v[:, 2]
            ra227_ac227_total_true = true_atoms_v[:, 3] + true_atoms_v[:, 4]
            ra227_ac227_total_pred = pred_atoms_v[:, 3] + pred_atoms_v[:, 4]
            chain_true = torch.stack([ra225_ac225_total_true, ra227_ac227_total_true], dim=1)
            chain_pred = torch.stack([ra225_ac225_total_pred, ra227_ac227_total_pred], dim=1)
            chain_floor = torch.stack([atom_floor[:, 1] + atom_floor[:, 2], atom_floor[:, 3] + atom_floor[:, 4]], dim=1)
            chain_signal = chain_true > chain_floor
            if chain_signal.any():
                chain_target_loss = F.huber_loss(
                    torch.log10(chain_pred[chain_signal] + chain_floor.expand_as(chain_true)[chain_signal]),
                    torch.log10(chain_true[chain_signal] + chain_floor.expand_as(chain_true)[chain_signal]),
                    reduction='mean',
                    delta=0.25,
                )

            virgin_ac225_loss = torch.zeros((), device=dev, dtype=dtype)
            valid_inputs_v = inputs[valid_mask]
            e_feature_v = valid_inputs_v[:, 2].clamp(min=1e-8, max=1e6)
            e_ev_v = torch.as_tensor(THERMAL_REFERENCE_EV, dtype=dtype, device=dev) / (e_feature_v ** 2 + 1e-30)
            daughter_ic_zero = valid_inputs_v[:, 4:8].abs().sum(dim=1) < 1e-12
            virgin_high_energy = (
                (valid_inputs_v[:, 3] > 1e-8)
                & daughter_ic_zero
                & (e_ev_v > torch.as_tensor(E_THRESHOLD_N2N_EV, dtype=dtype, device=dev))
                & (true_atoms_v[:, 2] > atom_floor[:, 2])
            )
            if virgin_high_energy.any():
                log_true_ac = torch.log10(true_atoms_v[virgin_high_energy, 2] + atom_floor[:, 2])
                log_pred_ac = torch.log10(pred_atoms_v[virgin_high_energy, 2] + atom_floor[:, 2])
                under_log = F.relu(log_true_ac - log_pred_ac)
                virgin_ac225_loss = F.huber_loss(
                    under_log,
                    torch.zeros_like(under_log),
                    reduction='mean',
                    delta=0.25,
                )

            ac227_chain_transfer_loss = torch.zeros((), device=dev, dtype=dtype)
            ac227_signal = (true_atoms_v[:, 3] > atom_floor[:, 3]) & (true_atoms_v[:, 4] > atom_floor[:, 4])
            if ac227_signal.any():
                true_ratio = true_atoms_v[ac227_signal, 4] / (true_atoms_v[ac227_signal, 3] + true_atoms_v[ac227_signal, 4] + atom_floor[:, 4])
                pred_ratio = pred_atoms_v[ac227_signal, 4] / (pred_atoms_v[ac227_signal, 3] + pred_atoms_v[ac227_signal, 4] + atom_floor[:, 4])
                ac227_chain_transfer_loss = F.huber_loss(
                    torch.log10(pred_ratio.clamp(min=1e-7)),
                    torch.log10(true_ratio.clamp(min=1e-7)),
                    reduction='mean',
                    delta=0.5,
                )

            target_total = true_atoms_v.sum(dim=1)
            pred_total = pred_atoms_v.sum(dim=1)
            total_signal = target_total > 1.0
            if total_signal.any():
                target_inventory_loss = F.huber_loss(
                    pred_total[total_signal] / n226_scale,
                    target_total[total_signal] / n226_scale,
                    reduction='mean',
                    delta=0.01,
                )
            else:
                target_inventory_loss = torch.zeros((), device=dev, dtype=dtype)
        else:
            data_mse = torch.zeros((), device=dev, dtype=dtype)
            log_data_loss = torch.zeros((), device=dev, dtype=dtype)
            impurity_log_loss = torch.zeros((), device=dev, dtype=dtype)
            target_inventory_loss = torch.zeros((), device=dev, dtype=dtype)
            trace_relative_loss = torch.zeros((), device=dev, dtype=dtype)
            chain_target_loss = torch.zeros((), device=dev, dtype=dtype)
            virgin_ac225_loss = torch.zeros((), device=dev, dtype=dtype)
            ac227_chain_transfer_loss = torch.zeros((), device=dev, dtype=dtype)
        
        # Relative error diagnostic (not used in loss) - only on valid targets
        with torch.no_grad():
            if valid_mask.any():
                true_atoms = targets[valid_mask].clamp(min=0.0)
                pred_atoms = pred_data_c[valid_mask] * scales_t
                signal_floor = torch.clamp(scales_t * 1e-12, min=1.0)
                signal_mask = true_atoms > signal_floor
                if signal_mask.any():
                    rel_err = ((pred_atoms - true_atoms).abs() / true_atoms.clamp(min=signal_floor))[signal_mask].mean()
                else:
                    rel_err = torch.zeros((), device=dev, dtype=dtype)
            else:
                rel_err = torch.zeros((), device=dev, dtype=dtype)
    else:
        data_mse = torch.zeros((), device=dev, dtype=dtype)
        log_data_loss = torch.zeros((), device=dev, dtype=dtype)
        impurity_log_loss = torch.zeros((), device=dev, dtype=dtype)
        target_inventory_loss = torch.zeros((), device=dev, dtype=dtype)
        trace_relative_loss = torch.zeros((), device=dev, dtype=dtype)
        chain_target_loss = torch.zeros((), device=dev, dtype=dtype)
        virgin_ac225_loss = torch.zeros((), device=dev, dtype=dtype)
        ac227_chain_transfer_loss = torch.zeros((), device=dev, dtype=dtype)
        rel_err  = torch.zeros((), device=dev, dtype=dtype)

    # Softer non-negativity: dead zone at -1e-3 prevents mode collapse to zero
    non_neg_loss = F.relu(-pred - 1e-3).pow(2).mean()

    # -- secular equilibrium ceiling for Ac-225 --------------------------------
    _eps_eq = torch.as_tensor(1e-8, dtype=dtype, device=dev)
    eq_ratio = lam225 / (lam_ac - lam225 + _eps_eq)
    scale_ratio = float(n225_scale) / (float(nac_scale) + 1e-8)
    nac_ceiling = eq_ratio * scale_ratio * n225.detach().clamp(min=0.0)
    secular_eq_loss = F.relu(nac - nac_ceiling).pow(2).mean()

    # -- mass conservation (TWO-SIDED) -----------------------------------------
    # Previous version only penalized excess atoms (relu(pred - start)); under-prediction
    # was free, and per-species drift was invisible. The model exploited this by
    # over-predicting Ac-225 to ~1e18 atoms while under-predicting Ra-226 (mass-stealing).
    # Now we penalize total imbalance in BOTH directions, plus a Ra-226 overshoot guard
    # and a small per-species drift regularizer.
    n0_all = inputs[:, 3:8]
    scales_all = torch.as_tensor(
        [n226_scale, n225_scale, nac_scale, n227_scale, nac227_scale],
        dtype=dtype, device=dev,
    )
    total_start_atoms = (n0_all * scales_all).sum(dim=1, keepdim=True)
    total_pred_atoms  = (pred_data * scales_all).sum(dim=1, keepdim=True)

    # Total imbalance in O(1) moles (Ra-226 scale = 6.022e23 ~ 1 mol).
    # Huber on both directions: gentle quadratic for small drift, linear for outliers.
    n226_scale_t = torch.as_tensor(n226_scale, dtype=dtype, device=dev)
    total_imbalance_moles = (total_pred_atoms - total_start_atoms) / n226_scale_t
    mass_cons_loss = F.huber_loss(
        total_imbalance_moles,
        torch.zeros_like(total_imbalance_moles),
        reduction='mean',
        delta=0.01,
    )

    # Ra-226 cannot grow during irradiation (no source for Ra-226 in this chain).
    # Architecturally enforced by the exp-decay ansatz, but this catches numerical drift
    # if the ansatz is ever bypassed and acts as a hard guard against the failure where
    # the regime gate routes Ra-226 into a head that produces growth.
    ra226_pred_norm = pred_data[:, 0:1]
    ra226_ic_norm = inputs[:, 3:4]
    ra226_overshoot_loss = F.relu(ra226_pred_norm - ra226_ic_norm).pow(2).mean()

    # Per-species drift regularizer (very small weight in the combined loss).
    # Only penalize drift for species with nonzero initial inventory. Penalizing
    # zero-IC daughters (virgin fuel) actively fights correct ingrowth — the root
    # cause of Ra-225/Ac-225 zero-collapse during training.
    ic_norm = inputs[:, 3:8]
    drift_mask = (ic_norm.abs() > 1e-8).to(dtype=pred_data.dtype)
    drift_denom = drift_mask.sum().clamp(min=1.0)
    species_drift_loss = ((pred_data - ic_norm).pow(2) * drift_mask).sum() / drift_denom

    # -- zero-injection (empty start: closed system must remain empty) ----------
    empty_mask = (n0_all.abs().sum(dim=1) < 1e-7)
    zero_injection_loss = torch.zeros((), device=dev, dtype=dtype)
    empty_output_loss = torch.zeros((), device=dev, dtype=dtype)
    if empty_mask.any():
        nonzero_outputs = pred_data[empty_mask].clamp(min=0.0)
        zero_injection_loss = nonzero_outputs.sum(dim=1).mean()
        empty_atoms = nonzero_outputs * scales_all
        empty_atom_floor = torch.clamp(scales_all * 1e-16, min=1.0)
        empty_output_loss = F.huber_loss(
            torch.log10(empty_atoms + empty_atom_floor),
            torch.log10(empty_atom_floor).expand_as(empty_atoms),
            reduction='mean',
            delta=0.25,
        )

    # -- combine ---------------------------------------------------------------
    zero_injection_weight = 20.0

    # Split for optional per-epoch gradient balancing (Wang et al., SISC 2021).
    supervised_total = (
        data_weight * data_mse
        + log_data_weight * log_data_loss
        + impurity_log_weight * impurity_log_loss
        + target_inventory_weight * target_inventory_loss
        + trace_relative_weight * trace_relative_loss
        + chain_consistency_weight * chain_target_loss
        + virgin_ac225_weight * virgin_ac225_loss
        + ac227_chain_weight * ac227_chain_transfer_loss
        + empty_output_weight * empty_output_loss
    )
    unsupervised_total = (
        physics_weight * physics_mse
        + mass_weight * mass_cons_loss
        + (mass_weight * 2.0) * ra226_overshoot_loss
        + (mass_weight * 0.1) * species_drift_loss
        + fuel_anchor_weight * fuel_anchor_loss
        + non_neg_weight * non_neg_loss
        + secular_eq_weight * secular_eq_loss
        + zero_injection_weight * zero_injection_loss
        + xpinn_interface_weight * xpinn_interface_loss
    )
    total = supervised_total + unsupervised_total

    return total, {
        "total_loss":          total,
        "data_mse":            data_mse,
        "log_data_loss":       log_data_loss,
        "impurity_log_loss":   impurity_log_loss,
        "target_inventory_loss": target_inventory_loss,
        "trace_relative_loss": trace_relative_loss,
        "chain_target_loss":   chain_target_loss,
        "virgin_ac225_loss":   virgin_ac225_loss,
        "ac227_chain_transfer_loss": ac227_chain_transfer_loss,
        "f5_signal_loss":      f5_signal_loss,
        "empty_output_loss":   empty_output_loss,
        "physics_mse":         physics_mse,
        "physics_f1":           f1_mse,
        "physics_f2":           f2_mse,
        "physics_f3":           f3_mse,
        "physics_f4":           f4_mse,
        "physics_f5":           f5_mse,
        "mass_cons_loss":      mass_cons_loss,
        "ra226_overshoot_loss": ra226_overshoot_loss,
        "species_drift_loss":  species_drift_loss,
        "fuel_anchor_loss":    fuel_anchor_loss,
        "non_neg_loss":        non_neg_loss,
        "secular_eq_loss":     secular_eq_loss,
        "zero_injection_loss": zero_injection_loss,
        "xpinn_interface_loss": xpinn_interface_loss,
        "supervised_total":    supervised_total,
        "unsupervised_total":  unsupervised_total,
        "rel_err":             rel_err if targets is not None else torch.zeros((), device=dev, dtype=dtype),
    }


# ==============================================================================
# Peak Yield Finder: optimal reactor shutdown time
# ==============================================================================
def find_peak_yield(
    model: IsotopePINN,
    flux_phi: float,
    energy_ev: float,
    ra226_0: float = DEFAULT_N226_SCALE,
    ra225_0: float = 0.0,
    ac225_0: float = 0.0,
    ra227_0: float = 0.0,
    ac227_0: float = 0.0,
    t_min_h: float = 1.0,
    t_max_h: float = 500.0,
    n_points: int = 500,
) -> dict[str, float]:
    """
    Sweep irradiation times with the trained PINN to find the hour
    that maximizes Ac-225 yield for a given flux and initial inventory.

    Returns dict with keys:
        optimal_time_h:   best shutdown time (hours)
        max_ac225_atoms:  peak Ac-225 inventory (atoms)
        purity_pct:       Ac-225 / (Ac-225 + Ac-227) * 100 at peak
        ra226_remaining:  Ra-226 left at peak time (atoms)
    """
    model.eval()
    times = torch.linspace(float(t_min_h), float(t_max_h), n_points)

    phi_nn = flux_phi / DEFAULT_PHI_SCALE
    e_nn = float(neutron_energy_ev_to_feature_torch(
        torch.tensor([energy_ev], dtype=torch.float32)
    )[0].item())

    rows = []
    for t_val in times:
        rows.append([
            float(t_val) / DEFAULT_T_REF_H, phi_nn, e_nn,
            ra226_0 / DEFAULT_N226_SCALE,
            ra225_0 / DEFAULT_N225_SCALE,
            ac225_0 / DEFAULT_NAC_SCALE,
            ra227_0 / DEFAULT_N227_SCALE,
            ac227_0 / DEFAULT_NAC227_SCALE,
        ])

    x = torch.tensor(rows, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x)

    ac225 = pred[:, 2] * DEFAULT_NAC_SCALE
    ac227 = pred[:, 4] * DEFAULT_NAC227_SCALE
    ra226 = pred[:, 0] * DEFAULT_N226_SCALE

    idx = int(ac225.argmax().item())
    peak_ac225 = float(ac225[idx].item())
    peak_ac227 = float(ac227[idx].item())
    ac_total = peak_ac225 + peak_ac227

    return {
        "optimal_time_h": float(times[idx].item()),
        "max_ac225_atoms": peak_ac225,
        "purity_pct": (peak_ac225 / ac_total * 100.0) if ac_total > 0 else 100.0,
        "ra226_remaining": float(ra226[idx].item()),
        "times_h": times.numpy(),
        "ac225_curve": ac225.numpy(),
    }
