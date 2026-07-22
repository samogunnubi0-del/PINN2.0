"""
Train IsotopePINN on pinn_training_data.csv with Adam and physics-informed loss.

Max-Fix v2: 12,000 total epochs (2,000 physics pretrain + 10,000 joint),
1/v energy scaling, Ra-225 physics residual x5, non-negativity penalty,
secular equilibrium ceiling, 30% empty-tank collocation, diverse ICs enabled,
and Time/Flux normalized to [0, 1].

Speed env vars (hardware only, no data/epoch changes):
  PINN_AMP=1              CUDA mixed precision (fp16 forward/loss); default off for PINN stability
  PINN_FLOAT64=1          float64 model + training tensors (better tiny-N vs 1e20 scales; slower on GPU)
  PINN_COLLOCATION_POINTS=N  physics pretrain collocation pts/epoch (default 300); raise on GPU to saturate SM
  PINN_FAST_CPU=1         full-batch joint step on CPU (needs more RAM)
  PINN_JOINT_CHUNK=0      full-batch joint step (needs ~16GB RAM); best GPU utilization if VRAM allows
  PINN_ODE_PARALLEL=0     disable multi-process ODE cache on Windows
  PINN_ODE_PREP_WORKERS=N cap ODE reference-cache processes (Kaggle default 1 unless set; local default all CPUs unless ODE_PREP_MAX_WORKERS)
  PINN_DATA_CACHE=1       cache augmented ODE training rows between runs (default)
  PINN_LBFGS_MAX_ITER=100 full-quality L-BFGS fine-tune iteration count
  PINN_COMPILE=1          opt into torch.compile (off on CPU; off by default on Kaggle; on by default for local CUDA)
  PINN_RESUME=1           resume from pinn_training_resume.pt if present (syncs pinn_best_weights.pth
                          to the resumed model so final plot reload cannot pick up a stale dataset .pth)
  PINN_WARM_START=1       load pinn_best_weights.pth and skip pretrain
  PINN_GRAD_BALANCE=1     Wang et al. SISC 2021-style supervised vs unsupervised grad
                          balance (requires full-batch joint: PINN_JOINT_CHUNK=0)
  PINN_ENERGY_FOURIER=8   Tancik et al. Fourier features on log10(E); 0 = off
  PINN_XPINN_IFACE=0.2    XPINN-style expert agreement weight at threshold
  PINN_CAUSAL_PHYS=0.35   Wang et al. soft emphasis on small t_nn (arXiv:2203.07404)
  PINN_CURRIC_BINS=4      Krishnapriyan-style time bins along t_nn (0=off)
  PINN_CURRIC_RAMP=4000   joint epochs to ramp curriculum progress to 1
  PINN_MEDIUM_TRAIN=1     ISEF short budget: 600 pretrain + 3400 joint (4000 total), scaled
                          warmups + boosted daughter losses + virgin oversampling
  PINN_SA_PHYSICS=0.35    McClenny-style detached focal on physics residuals (0=off)
  PINN_UW=1               enable Kendall et al. CVPR 2018 uncertainty loss weighting
  PINN_UW=0               default when unset: fixed loss weights
  PINN_COS_T0=2000        CosineAnnealingWarmRestarts T_0 (first cycle length)
  PINN_LOG_EVERY=N        print every N epochs (default 1000); lower for denser cloud logs
  PINN_OUTPUT_ROOT=path   on Kaggle, all outputs (graphs/, results/, weights/) go under this directory (default /kaggle/working); matches extra_plots.py. Also used as graph_provenance project_root when train.py is run from a read-only path under /kaggle/input.
  PINN_KAGGLE_SKIP_CUDA_HIDE=1  skip auto-hiding GPU on Kaggle (default: hide sm<7 for PyTorch cu128)

Kaggle multi-dataset:
  PINN_DATA_INPUT_SUBSTRING=name      required when multiple pinn_training_data.csv exist under /kaggle/input
  (CSV discovery checks /kaggle/input/datasets first, then full /kaggle/input, to avoid slow full-tree rglob when possible.)

Joint gradient circuit breaker (optional; most runs should rely on clip only):
  PINN_GRAD_BREAKER=0       default — never skip joint steps (clip_grad_norm_ bounds updates)
  PINN_GRAD_BREAKER=1       skip optimizer.step if pre-clip L2 norm > threshold after grace
  PINN_GRAD_BREAKER_THRESHOLD=500000  only when breaker on
  PINN_GRAD_BREAKER_GRACE=500         joint epochs before breaker can trip

Local (not Kaggle):
  cd to the folder containing train.py. Input CSV path is ./data/pinn_training_data.csv (created by repo layout).
  Writes ./weights/, ./graphs/, ./results/, ./data/pinn_* cache files. Same code path as cloud; no LOCAL_PINN_DATA in train.py.
  Use PINN_RESUME=0 unless you intentionally continue from weights/pinn_training_resume.pt (resume syncs pinn_best_weights.pth).
  PINN_FLOAT64=1 uses float64 tensors+model (helps trace inventories vs 1e20 scales; slower on GPU; disables AMP).

Training writes provenance to results/graph_manifest.json, results/last_training_run.json,
results/LAST_GRAPH_WRITE.txt (last train.py graph record).
Truncates results/loss_history.csv once per process at training start (unless PINN_RESUME loaded a checkpoint — then appends), then appends per-epoch rows (epoch, phase, data_mse, physics_mse, supervised_total, unsupervised_total, grad_norm) for graphs/loss_components.png and extra_plots.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _kaggle_maybe_hide_incompatible_cuda_before_torch() -> None:
    """Kaggle P100 (sm_60) + preinstalled PyTorch 2.10+cu128: hide GPU before torch import.

    PyTorch builds with min sm_70 cannot use P100; importing CUDA still warns and may break runs.
    Set ``PINN_KAGGLE_SKIP_CUDA_HIDE=1`` to skip (not recommended on P100).

    If ``CUDA_VISIBLE_DEVICES=99`` is set from an old cell run but this session has a sm>=7 GPU
    (e.g. T4), we clear ``99`` so ``torch.cuda.is_available()`` is True again.
    """
    if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
        return
    if os.environ.get("PINN_KAGGLE_SKIP_CUDA_HIDE", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    import shutil
    import subprocess as sp

    exe = shutil.which("nvidia-smi")
    if not exe:
        return
    try:
        r = sp.run(
            [exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return
        first = (r.stdout or "").strip().splitlines()[0].strip()
        parts = first.replace(" ", "").split(".")
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1][:1].isdigit():
            return
        major, minor = int(parts[0]), int(parts[1][0])
    except Exception:
        return
    # Modified to allow sm >= 6 so P100 works with downgraded PyTorch
    if major >= 6:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "99":
            del os.environ["CUDA_VISIBLE_DEVICES"]
            print(
                f"[train.py] Kaggle: GPU sm_{major}.{minor} is OK for this PyTorch; "
                "cleared stale CUDA_VISIBLE_DEVICES=99 so CUDA is visible."
            )
        return
    # sm < 6 (e.g. older than Pascal): hide from PyTorch unless user pinned a real device index.
    cur = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cur is not None and str(cur).strip() not in ("", "99"):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "99"
    print(
        f"[train.py] Kaggle: GPU compute capability {major}.{minor} < 6; "
        "set CUDA_VISIBLE_DEVICES=99 so PyTorch uses CPU. "
        "For CUDA training: Session → Change accelerator → GPU T4 (or L4), then restart session."
    )


_kaggle_maybe_hide_incompatible_cuda_before_torch()

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# ==============================================================================
# Muon Optimizer (DeepSeek-V4 style) - Optional for better convergence
# ==============================================================================
class MuonOptimizer(torch.optim.Optimizer):
    """
    Muon optimizer with Newton-Schulz orthogonalization (Jordan et al., 2024).

    For 2D weight matrices, the momentum-accumulated gradient is orthogonalized
    via iterative Newton-Schulz iterations before being used as the parameter
    update direction. This decorrelates gradient directions across neurons,
    improving convergence on anisotropic loss landscapes typical of PINNs.

    For 1D parameters (biases, layer-norm scales), standard Nesterov SGD is
    used since orthogonalization is not applicable to vectors.

    Reference: Jordan et al., "Muon: An optimizer for hidden layers in
    neural networks", 2024; Liu et al., 2025 (DeepSeek-V4 adoption).
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Approximate the closest orthogonal matrix to G using Newton-Schulz
        iteration:  X_{k+1} = X_k (aI + bX_k^T X_k + c(X_k^T X_k)^2)

        Coefficients (a, b, c) = (3, -3, 1) correspond to the quintic
        convergence variant from Bjoerck & Bowie (1971), scaled for the
        Polar decomposition:  G = U P  where U is orthogonal.

        Only applied to 2D tensors with rows <= cols (if rows > cols, we
        transpose, orthogonalize, and transpose back).
        """
        assert G.dim() == 2, "Newton-Schulz expects a 2D matrix"
        rows, cols = G.shape
        transposed = False
        if rows > cols:
            G = G.T
            transposed = True

        # Normalize to unit spectral norm (heuristic: Frobenius norm / sqrt(min(m,n)))
        m, n = G.shape
        scale = G.norm() / (m ** 0.5)
        if scale < 1e-12:
            return G if not transposed else G.T
        X = G / scale

        # Newton-Schulz quintic iteration coefficients
        a, b, c = 3.0, -3.0, 1.0
        I = torch.eye(m, device=G.device, dtype=G.dtype)
        for _ in range(steps):
            XtX = X.T @ X  # (n, n)
            X = X @ (a * I[:n, :n] + b * XtX + c * (XtX @ XtX))

        if transposed:
            X = X.T
        return X

    @torch.no_grad()
    # pyrefly: ignore [bad-override]
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad

                # Weight decay (decoupled, applied to param directly)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # Momentum buffer
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p)
                else:
                    buf = param_state["momentum_buffer"]

                buf.mul_(momentum).add_(d_p)

                if nesterov:
                    update = d_p.add(buf, alpha=momentum)
                else:
                    update = buf.clone()

                # Newton-Schulz orthogonalization for 2D weight matrices
                if update.dim() == 2 and min(update.shape) >= 2:
                    update = self._newton_schulz_orthogonalize(update, steps=ns_steps)

                # Update parameters
                p.add_(update, alpha=-lr)

        return loss


def get_optimizer(model: torch.nn.Module, lr: float = 1e-4, use_muon: bool = False):
    """Factory for optimizer selection: Adam (default) or Muon (DeepSeek-V4 style)."""
    if use_muon:
        print(f"Using Muon optimizer (lr={lr:.2e}) - DeepSeek-V4 style")
        return MuonOptimizer(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute total gradient norm for debugging training instability (DeepSeek-V4 style logging)."""
    total_sq = None
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float().norm(2) ** 2
        total_sq = g if total_sq is None else total_sq + g
    if total_sq is None:
        return 0.0
    return float(torch.sqrt(total_sq).item())


def _grad_tuple_l2_norm(grads: tuple) -> float:
    """L2 norm of a torch.autograd.grad() tuple (None slots ignored)."""
    s = 0.0
    for g in grads:
        if g is None:
            continue
        s += float(g.detach().float().norm().item() ** 2)
    return s ** 0.5


class UncertaintyWeighter(torch.nn.Module):
    """Kendall et al. CVPR 2018: multi-task loss balancing.

    Learns ``log(sigma^2_k)`` per task group so the effective loss is::

        total = sum_k  exp(-s_k) * L_k  +  0.5 * s_k

    where ``s_k = log(sigma^2_k)``.

    NOTE: the unconstrained Kendall formulation lets the optimizer drive
    ``s_unsup -> +inf`` (which sends ``exp(-s_unsup) -> 0``), silently zeroing
    out the physics gradient. We clamp ``log_vars`` to ``[-3, 3]`` so each
    task weight stays in ``[exp(-3), exp(3)] ~ [0.05, 20]`` -- still adaptive,
    but cannot fully kill any task.
    """

    LOG_VAR_CLAMP_MIN = -3.0
    LOG_VAR_CLAMP_MAX = 3.0

    def __init__(self, n_tasks: int = 2) -> None:
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=losses[0].device, dtype=losses[0].dtype)
        safe_log_vars = self.log_vars.clamp(min=self.LOG_VAR_CLAMP_MIN, max=self.LOG_VAR_CLAMP_MAX)
        for s, l in zip(safe_log_vars, losses):
            total = total + torch.exp(-s) * l + 0.5 * s
        return total


from pinn_model import (
    IsotopePINN,
    THERMAL_REFERENCE_EV,
    compute_physics_loss,
    neutron_energy_ev_to_feature_torch,
)
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

import graph_provenance


def _kaggle_find_pinn_training_csv_paths() -> list[pathlib.Path]:
    """Prefer /kaggle/input/datasets/** (typical mount); fall back to full tree if needed."""
    input_dir = pathlib.Path("/kaggle/input")
    ds = input_dir / "datasets"
    if ds.is_dir():
        found = sorted(ds.rglob("pinn_training_data.csv"))
        if found:
            return found
    return sorted(input_dir.rglob("pinn_training_data.csv"))


def _resolve_path(rel_path: str) -> pathlib.Path:
    """Intelligent path resolution for local vs Kaggle environments."""
    is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    root = pathlib.Path(__file__).resolve().parent

    if is_kaggle:
        working_dir = pathlib.Path(os.environ.get("PINN_OUTPUT_ROOT", "/kaggle/working")).resolve()
        if "pinn_training_data.csv" in rel_path:
            matches = _kaggle_find_pinn_training_csv_paths()
            if not matches:
                return root / rel_path
            if len(matches) > 1:
                hint = os.environ.get("PINN_DATA_INPUT_SUBSTRING", "").strip()
                if hint:
                    filt = [p for p in matches if hint in str(p)]
                    if len(filt) == 1:
                        print(f"PINN_DATA_INPUT_SUBSTRING matched single CSV: {filt[0]}")
                        return filt[0]
                    raise RuntimeError(
                        f"PINN_DATA_INPUT_SUBSTRING={hint!r} matched {len(filt)} paths "
                        f"(need exactly 1). Candidates:\n"
                        + "\n".join(f"  {p}" for p in matches)
                    )
                raise RuntimeError(
                    "Multiple pinn_training_data.csv files under /kaggle/input:\n"
                    + "\n".join(f"  {p}" for p in matches)
                    + "\nSet env PINN_DATA_INPUT_SUBSTRING to a unique substring of the path you want "
                    "(e.g. dataset folder name), or mount only one dataset."
                )
            return matches[0]
        else:
            # Outputs MUST go to /kaggle/working to be persistent/downloadable
            p = working_dir / rel_path
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
    else:
        p = root / rel_path
        # Ensure directory exists locally too
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

DATA_PATH = _resolve_path("data/pinn_training_data.csv")
WEIGHTS_PATH = _resolve_path("weights/pinn_trained_weights.pth")
LOSS_PLOT_PATH = _resolve_path("graphs/pinn_loss_history.png")
LOSS_COMPONENTS_PATH = _resolve_path("graphs/loss_components.png")
LOSS_HISTORY_CSV_PATH = _resolve_path("results/loss_history.csv")
AC225_SCATTER_PATH = _resolve_path("graphs/pinn_ac225_pred_vs_true.png")
AC225_SCATTER_VIRGIN_PATH = _resolve_path("graphs/pinn_ac225_pred_vs_true_virgin_ic.png")
VALIDATION_REPORT_PATH = _resolve_path("data/pinn_validation_summary.csv")
AUGMENTED_DATA_CACHE_PATH = _resolve_path("data/pinn_augmented_training_cache.csv")
AUGMENTED_DATA_CACHE_META_PATH = _resolve_path("data/pinn_augmented_training_cache.meta.json")
RESUME_CKPT_PATH = _resolve_path("weights/pinn_training_resume.pt")
BEST_CKPT_PATH = _resolve_path("weights/pinn_best_weights.pth")
PINN_ARCHITECTURE = "regime-gated-pinn-v3-adaptive-residual-decay-kendall"

# --- Scaling (MUST match pinn_model.py; 0-1 normalised for Time & Flux) ------
N226_SCALE = 6.022e23
N225_SCALE = 1e20
NAC_SCALE = 1e20
N227_SCALE = 1e18
NAC227_SCALE = 1e18
PHI_SCALE = 1e15       # max flux -> phi_nn in [0, 1]
TIME_SCALE_H = 500.0   # max irradiation time -> t_nn in [0, 1]
TRAIN_INIT_RA226 = 6.022e23
TRAIN_INIT_RA225 = 0.0
TRAIN_INIT_AC225 = 0.0
TRAIN_INIT_RA227 = 0.0
TRAIN_INIT_AC227 = 0.0

# Time-shift augmentation
USE_TIME_SHIFT_AUGMENT = True
AUGMENT_PER_ROW = 2
MIN_T_REM_H = 0.05
TRAJ_CACHE_MAX = 4096
AUGMENT_BASE_ROW_LIMIT = 0
ODE_PREP_MAX_WORKERS = 0

# Enable diverse ICs so the network learns decay-only and mixed scenarios
SINGLE_SUPPLY_MODE = False
INVERTED_IC_N_EXTRA = 400
DIVERSE_IC_N_EXTRA = 500
SPECTRUM_SWEEP_N_EXTRA = 400
TRACE_RECYCLED_N_EXTRA = 500
TRACE_TINY_N_EXTRA = 300
HIGH_ENERGY_VIRGIN_N_EXTRA = 700
THRESHOLD_EDGE_VIRGIN_N_EXTRA = 300
AC227_LONG_CHAIN_N_EXTRA = 350


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# Short-epoch ISEF profile: oversample virgin/fast14 daughter signal, trim low-value aug.
if _env_on("PINN_MEDIUM_TRAIN"):
    INVERTED_IC_N_EXTRA = 250
    DIVERSE_IC_N_EXTRA = 300
    SPECTRUM_SWEEP_N_EXTRA = 250
    TRACE_RECYCLED_N_EXTRA = 400
    TRACE_TINY_N_EXTRA = 250
    HIGH_ENERGY_VIRGIN_N_EXTRA = 900
    THRESHOLD_EDGE_VIRGIN_N_EXTRA = 450
    AC227_LONG_CHAIN_N_EXTRA = 200

THERMAL_ENERGY_RANGE_EV = (0.015, 0.08)
EPITHERMAL_ENERGY_RANGE_EV = (0.1, 1.0e4)
THRESHOLD_ENERGY_RANGE_EV = (5.8e6, 7.5e6)
FAST_ENERGY_RANGE_EV = (6.5e6, 2.0e7)

# --- Default full run: 12,000 total epochs (2,000 pretrain + 10,000 joint) ---
EPOCHS = int(os.environ.get("PINN_EPOCHS", "10000"))
PHYS_PRETRAIN_EPOCHS = int(os.environ.get("PINN_PRETRAIN_EPOCHS", "2000"))
COLLOCATION_POINTS = max(32, int(os.environ.get("PINN_COLLOCATION_POINTS", "300")))
LOG_EVERY = max(1, int(os.environ.get("PINN_LOG_EVERY", "1000")))
# LR_PLATEAU_PATIENCE = 500  # unused while ReduceLROnPlateau is disabled
# LR_PLATEAU_FACTOR = 0.5

# Loss weights: data_mse was 113 vs physics_mse 1.7 — optimizer ignored data.
# DATA_WEIGHT boosted 10x to force the network to fit trace daughters.
PHYSICS_WEIGHT = 1.0
DATA_WEIGHT = float(os.environ.get("PINN_DATA_WEIGHT", "10.0"))
DATA_SPECIES_WEIGHTS = (1.0, 100.0, 500.0, 100.0, 500.0)
LOG_DATA_WEIGHT = float(os.environ.get("PINN_LOG_DATA_WEIGHT", "3.0"))
LOG_SPECIES_WEIGHTS = (0.0, 8.0, 20.0, 8.0, 18.0)
IMPURITY_LOG_WEIGHT = 5.0       # was 3.0
TARGET_INVENTORY_WEIGHT = 3.0   # was 2.0
TRACE_RELATIVE_WEIGHT = 2.0     # was 1.0
CHAIN_CONSISTENCY_WEIGHT = 2.0  # was 1.0
VIRGIN_AC225_WEIGHT = 3.0       # was 1.5: critical for virgin IC zero-collapse
AC227_CHAIN_WEIGHT = 4.0        # was 2.0
F5_SIGNAL_WEIGHT = 1.0          # was 0.5
EMPTY_OUTPUT_WEIGHT = 8.0
RA225_PHYSICS_WEIGHT = 5.0      # legacy: ignored by normalized residuals

# Energy-conditioned physics weighting for threshold boundary (6.0-7.0 MeV)
THRESHOLD_PHYSICS_BOOST = 5.0       # 5x physics loss at the (n,2n) cliff edge

# Spectral / interface / causal extensions — defaults ON; see pinn_model.py citations.
N_ENERGY_FOURIER_FREQS = max(0, int(os.environ.get("PINN_ENERGY_FOURIER", "8")))
XPINN_INTERFACE_WEIGHT = float(os.environ.get("PINN_XPINN_IFACE", "0.2"))
CAUSAL_PHYSICS_TIME_SCALE = float(os.environ.get("PINN_CAUSAL_PHYS", "0.35"))

# Krishnapriyan et al. (NeurIPS 2021) time-bin curriculum + McClenny et al. (JCP 2023) SA-style focal
CAUSAL_CURRICULUM_BINS = max(0, int(os.environ.get("PINN_CURRIC_BINS", "4")))
CAUSAL_CURRICULUM_RAMP_EPOCHS = float(os.environ.get("PINN_CURRIC_RAMP", "4000"))
SA_PHYSICS_ALPHA = float(os.environ.get("PINN_SA_PHYSICS", "0.35"))

NON_NEG_WEIGHT = 10.0               # reduced: prevents mode collapse to zero
SECULAR_EQ_WEIGHT = 10.0            # reduced from 25
MAX_GRAD_NORM = 1.0
D_T_INPUT_D_T_HOURS = 1.0 / TIME_SCALE_H

# Joint training stability
WARMUP_EPOCHS = 1_000
TRACE_SUPERVISED_WARMUP_EPOCHS = 500
VIRGIN_AC225_WARMUP_EPOCHS = 2_000
TRACE_CHUNK_FRACTION = 0.25
RAR_MAX_TARGETLESS_FRACTION = 0.20
GRAD_CLIP_NORM = 10.0
_grad_breaker_env = os.environ.get("PINN_GRAD_BREAKER", "0").strip().lower()
GRAD_BREAKER_ENABLED = _grad_breaker_env in ("1", "true", "yes", "on")
GRAD_BREAKER_THRESHOLD = float(os.environ.get("PINN_GRAD_BREAKER_THRESHOLD", "500000"))
GRAD_BREAKER_GRACE_JOINT_EPOCHS = max(0, int(os.environ.get("PINN_GRAD_BREAKER_GRACE", "500")))
PINN_TRAIN_SCRIPT_REV = "2026-05-03h-float64-option"
NAN_PATIENCE = 3
# BEST_CKPT_PATH handled by _resolve_path helper above
MASS_WEIGHT = 100.0                 # reduced from 350
PHYS_PRETRAIN_PHYS_WEIGHT = 10.0
PHYS_PRETRAIN_MASS_WEIGHT = 300.0   # reduced from 800
PHYS_PRETRAIN_FUEL_ANCHOR_WEIGHT = 20.0  # reduced from 40
FUEL_ANCHOR_WEIGHT = 50.0           # reduced from 100
EMPTY_FEED_FRACTION = 0.30          # 30% empty-tank collocation points
EMPTY_FEED_HIGH_FLUX = 1.0e15

# Joint chunk size
_jc_env = os.environ.get("PINN_JOINT_CHUNK", "").strip()
if _jc_env == "":
    JOINT_CHUNK_SIZE = 8192
elif _jc_env.lower() in ("0", "full", "none"):
    JOINT_CHUNK_SIZE = 0
else:
    JOINT_CHUNK_SIZE = max(32, int(_jc_env))


def sample_neutron_energy_ev(rng: np.random.Generator) -> float:
    """Sample thermal, epithermal, and fast spectra for training coverage."""
    u = float(rng.uniform())
    if u < 0.30:
        return float(rng.uniform(*THERMAL_ENERGY_RANGE_EV))
    if u < 0.52:
        lo, hi = EPITHERMAL_ENERGY_RANGE_EV
        return float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))
    if u < 0.70:
        return float(rng.uniform(*THRESHOLD_ENERGY_RANGE_EV))
    if u < 0.85:
        # Pin many samples near the common D-T fast neutron benchmark.
        return float(np.clip(rng.normal(14.0e6, 1.0e6), *FAST_ENERGY_RANGE_EV))
    return float(rng.uniform(*FAST_ENERGY_RANGE_EV))


def sample_time_hours(rng: np.random.Generator, low_h: float, high_h: float) -> float:
    """Mix log and linear time sampling to cover early stiff transients and long buildup."""
    low = max(float(low_h), MIN_T_REM_H)
    high = max(float(high_h), low + MIN_T_REM_H)
    if rng.uniform() < 0.70:
        return float(10.0 ** rng.uniform(np.log10(low), np.log10(high)))
    return float(rng.uniform(low, high))


def sample_validation_regime_energy_ev(rng: np.random.Generator) -> float:
    """Sample the same named regimes used by held-out validation."""
    u = float(rng.uniform())
    if u < 0.25:
        return float(rng.uniform(*THERMAL_ENERGY_RANGE_EV))
    if u < 0.50:
        return float(10.0 ** rng.uniform(np.log10(EPITHERMAL_ENERGY_RANGE_EV[0]), np.log10(EPITHERMAL_ENERGY_RANGE_EV[1])))
    if u < 0.75:
        return float(rng.uniform(*THRESHOLD_ENERGY_RANGE_EV))
    return float(rng.uniform(12.5e6, 15.5e6))


def _append_ode_row(
    rows_out: list[dict[str, float]],
    *,
    phi: float,
    energy_ev: float,
    time_h: float,
    ra226_0: float,
    ra225_0: float,
    ac_0: float,
    ra227_0: float,
    ac227_0: float,
    min_points: int = 250,
    points_per_hour: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    t_h, Y = run_simulation(
        env,
        t_end_h=time_h,
        n_points=max(min_points, int(time_h * points_per_hour)),
        N_ra0=ra226_0,
        N_ra225_0=ra225_0,
        N_ac0=ac_0,
        N_ra227_0=ra227_0,
        N_ac227_0=ac227_0,
    )
    nf = Y[-1].astype(np.float64)
    rows_out.append({
        "phi": phi, "energy": energy_ev, "time": time_h,
        "init_N226": ra226_0, "init_N225": ra225_0, "init_NAc": ac_0,
        "init_N227": ra227_0, "init_NAc227": ac227_0,
        "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
        "N_Ra227": nf[3], "N_Ac227": nf[4],
    })
    return t_h, Y


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _training_cache_config(df_raw: pd.DataFrame, augment_per_row: int) -> dict[str, object]:
    stat = DATA_PATH.stat()
    return {
        "cache_version": "trace-predictor-v7-adaptive-residual-decay",
        "model_architecture": PINN_ARCHITECTURE,
        "data_path": str(DATA_PATH.resolve()),
        "data_mtime_ns": stat.st_mtime_ns,
        "data_size": stat.st_size,
        "raw_rows": int(len(df_raw)),
        "use_time_shift": bool(USE_TIME_SHIFT_AUGMENT),
        "augment_per_row": int(augment_per_row),
        "single_supply_mode": bool(SINGLE_SUPPLY_MODE),
        "inverted_ic_n_extra": int(INVERTED_IC_N_EXTRA),
        "diverse_ic_n_extra": int(DIVERSE_IC_N_EXTRA),
        "spectrum_sweep_n_extra": int(SPECTRUM_SWEEP_N_EXTRA),
        "trace_recycled_n_extra": int(TRACE_RECYCLED_N_EXTRA),
        "trace_tiny_n_extra": int(TRACE_TINY_N_EXTRA),
        "high_energy_virgin_n_extra": int(HIGH_ENERGY_VIRGIN_N_EXTRA),
        "threshold_edge_virgin_n_extra": int(THRESHOLD_EDGE_VIRGIN_N_EXTRA),
        "ac227_long_chain_n_extra": int(AC227_LONG_CHAIN_N_EXTRA),
        "thermal_energy_range_ev": list(THERMAL_ENERGY_RANGE_EV),
        "epithermal_energy_range_ev": list(EPITHERMAL_ENERGY_RANGE_EV),
        "threshold_energy_range_ev": list(THRESHOLD_ENERGY_RANGE_EV),
        "fast_energy_range_ev": list(FAST_ENERGY_RANGE_EV),
        "empty_rows": 300,
        "seed": 42,
    }


def _load_augmented_training_cache(config: dict[str, object]) -> pd.DataFrame | None:
    if not _env_flag("PINN_DATA_CACHE", default=True):
        return None
    if not AUGMENTED_DATA_CACHE_PATH.exists() or not AUGMENTED_DATA_CACHE_META_PATH.exists():
        return None
    try:
        meta = json.loads(AUGMENTED_DATA_CACHE_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if meta != config:
        return None
    print(f"Loading augmented training cache: {AUGMENTED_DATA_CACHE_PATH.name}")
    return pd.read_csv(AUGMENTED_DATA_CACHE_PATH, dtype="float64", engine="python")


def _write_augmented_training_cache(train_dat: pd.DataFrame, config: dict[str, object]) -> None:
    if not _env_flag("PINN_DATA_CACHE", default=True):
        return
    try:
        train_dat.to_csv(AUGMENTED_DATA_CACHE_PATH, index=False)
        AUGMENTED_DATA_CACHE_META_PATH.write_text(
            json.dumps(config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved augmented training cache: {AUGMENTED_DATA_CACHE_PATH.name}")
    except OSError as exc:
        print(f"Warning: could not write augmented training cache ({exc!r})")


def _model_state_dict_for_checkpoint(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a normal state dict even when torch.compile wraps the model."""
    inner = getattr(model, "_orig_mod", model)
    return inner.state_dict()


def _load_model_state_for_training(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a training checkpoint into a compiled or uncompiled model."""
    inner = getattr(model, "_orig_mod", model)
    inner.load_state_dict(state)


def _try_load_model_state_for_training(model: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> bool:
    """Best-effort checkpoint load; architecture changes should fall back to fresh training."""
    try:
        _load_model_state_for_training(model, state)
        return True
    except RuntimeError as exc:
        print(f"{label}: skipped incompatible checkpoint for {PINN_ARCHITECTURE}: {exc}")
        return False


def _checkpoint_state_is_finite(state: dict[str, torch.Tensor]) -> bool:
    """True only if every tensor in a state dict is finite (guards NaN/Inf corruption)."""
    try:
        for value in state.values():
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                return False
        return True
    except Exception:
        return False


def _resume_checkpoint_is_usable(ckpt: dict, train_epochs: int, label: str) -> bool:
    """Reject corrupt/truncated/finished resume checkpoints so they cannot clobber good weights.

    The historical failure mode: PINN_RESUME=1 loaded a truncated pinn_training_resume.pt,
    overwrote pinn_best_weights.pth with the broken model, then trained from a diverged state.
    """
    if not isinstance(ckpt, dict):
        print(f"{label}: resume checkpoint is not a dict; starting fresh.")
        return False
    required = ("model_state", "optimizer_state", "inputs", "targets")
    missing = [k for k in required if k not in ckpt]
    if missing:
        print(f"{label}: resume checkpoint missing keys {missing}; starting fresh.")
        return False
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict) or len(model_state) == 0:
        print(f"{label}: resume checkpoint has empty model_state; starting fresh.")
        return False
    if not _checkpoint_state_is_finite(model_state):
        print(f"{label}: resume checkpoint contains NaN/Inf weights; starting fresh.")
        return False
    joint_done = int(ckpt.get("joint_epoch", 0))
    if joint_done >= train_epochs:
        print(
            f"{label}: resume joint_epoch={joint_done} >= train_epochs={train_epochs}; "
            "nothing left to train. Starting fresh (raise PINN_EPOCHS to truly continue)."
        )
        return False
    return True


def _backup_existing_checkpoint(path: pathlib.Path) -> None:
    """Copy an existing checkpoint to *.prev before it is overwritten (insurance for good weights)."""
    import shutil

    try:
        if path.exists():
            backup = path.with_name(path.name + ".prev")
            shutil.copy2(path, backup)
            print(f"  Backed up existing {path.name} -> {backup.name}")
    except Exception as exc:
        print(f"  Warning: could not back up {path.name}: {exc}")


def _save_training_resume_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    phase: str,
    epoch: int,
    joint_epoch: int,
    train_pre: int,
    train_epochs: int,
    best_loss: float,
    best_epoch: int,
    best_joint_epoch: int,
    adaptive_sec_eq_weight: float,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epoch_list: list[int],
    hist_data: list[float],
    hist_phys: list[float],
    uw_state: dict | None = None,
) -> None:
    """Save enough state to continue after Ctrl+C without losing RAR points."""
    ckpt = {
        "phase": phase,
        "epoch": int(epoch),
        "joint_epoch": int(joint_epoch),
        "train_pre": int(train_pre),
        "train_epochs": int(train_epochs),
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "best_joint_epoch": int(best_joint_epoch),
        "adaptive_sec_eq_weight": float(adaptive_sec_eq_weight),
        "model_state": _model_state_dict_for_checkpoint(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "inputs": inputs.detach().cpu(),
        "targets": targets.detach().cpu(),
        "epoch_list": list(epoch_list),
        "hist_data": list(hist_data),
        "hist_phys": list(hist_phys),
    }
    if uw_state is not None:
        ckpt["uw_state"] = uw_state
    torch.save(ckpt, RESUME_CKPT_PATH)


def _species_error_snapshot(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_rows: int = 2048,
) -> str:
    """Compact per-species median/p95 relative error on a deterministic subset."""
    valid_mask = ~torch.isnan(targets).any(dim=1)
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        return "species_err unavailable"
    if valid_idx.numel() > max_rows:
        pick = torch.linspace(0, valid_idx.numel() - 1, max_rows, device=valid_idx.device).long()
        valid_idx = valid_idx[pick]

    x = inputs[valid_idx].detach()
    y = targets[valid_idx].detach().clamp(min=0.0)
    scales = torch.tensor(
        [N226_SCALE, N225_SCALE, NAC_SCALE, N227_SCALE, NAC227_SCALE],
        dtype=x.dtype,
        device=x.device,
    )
    labels = ["Ra226", "Ra225", "Ac225", "Ra227", "Ac227"]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred_atoms = model(x).clamp(min=0.0) * scales
    model.train(was_training)

    parts: list[str] = []
    for i, label in enumerate(labels):
        floor = max(1.0, float(scales[i].item()) * 1e-12)
        signal = y[:, i] > floor
        if not bool(signal.any().item()):
            continue
        rel = (pred_atoms[signal, i] - y[signal, i]).abs() / y[signal, i].clamp(min=floor)
        parts.append(
            f"{label} med={float(rel.median().cpu()):.2e} "
            f"p95={float(torch.quantile(rel, 0.95).cpu()):.2e}"
        )
    return " | ".join(parts) if parts else "species_err no-signal"


def _traj_cache_key(phi: float, e_ev: float, t_end: float) -> tuple[float, float, float]:
    return (round(phi, 8), round(float(e_ev), 12), round(t_end, 8))


def _reference_traj_worker(
    args: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[np.ndarray, np.ndarray]]:
    phi, e_ev, t_end = args
    key = _traj_cache_key(phi, e_ev, t_end)
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=e_ev)
    n_pts = max(400, int(np.clip(t_end * 8, 400, 15000)))
    t_h, Y = run_simulation(
        env, t_end_h=t_end, n_points=n_pts,
        N_ra0=TRAIN_INIT_RA226, N_ra225_0=TRAIN_INIT_RA225, N_ac0=TRAIN_INIT_AC225,
    )
    return key, (t_h, Y)


def _collect_unique_reference_triples(df: pd.DataFrame) -> list[tuple[float, float, float]]:
    pe_t = df[["phi", "energy", "time"]]
    seen_keys: set[tuple[float, float, float]] = set()
    out: list[tuple[float, float, float]] = []
    for row in pe_t.itertuples(index=False, name=None):
        phi, e_ev, t_full = float(row[0]), float(row[1]), float(row[2])
        k = _traj_cache_key(phi, e_ev, t_full)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        out.append((phi, e_ev, t_full))
    return out


def build_reference_traj_cache_parallel(
    df: pd.DataFrame, *, max_workers: int,
) -> dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]]:
    triples = _collect_unique_reference_triples(df)
    if not triples:
        return {}
    n = len(triples)
    workers = max(1, min(max_workers, n))
    cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    report_every = max(1, n // 20)

    def _report(done: int) -> None:
        if done == 1 or done == n or done % report_every == 0:
            print(f"  ODE cache progress: {done}/{n} trajectories", flush=True)

    if workers == 1:
        for i, args in enumerate(triples, start=1):
            key, val = _reference_traj_worker(args)
            cache[key] = val
            _report(i)
        return cache

    chunksize = max(1, n // (workers * 4))
    print(f"  ODE cache: {n} unique trajectories, {workers} worker(s), chunksize={chunksize}", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = 0
            for key, val in ex.map(_reference_traj_worker, triples, chunksize=chunksize):
                cache[key] = val
                done += 1
                _report(done)
    except BrokenProcessPool:
        cache.clear()
        for i, args in enumerate(triples, start=1):
            key, val = _reference_traj_worker(args)
            cache[key] = val
            _report(i)
    return cache


def get_trajectory(
    phi: float, e_ev: float, t_end: float,
    cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key = _traj_cache_key(phi, e_ev, t_end)
    if key in cache:
        return cache[key]
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=e_ev)
    n_pts = max(400, int(np.clip(t_end * 8, 400, 15000)))
    t_h, Y = run_simulation(
        env, t_end_h=t_end, n_points=n_pts,
        N_ra0=TRAIN_INIT_RA226, N_ra225_0=TRAIN_INIT_RA225, N_ac0=TRAIN_INIT_AC225,
    )
    if len(cache) < TRAJ_CACHE_MAX:
        cache[key] = (t_h, Y)
    return t_h, Y


def state_at_time(t_h: np.ndarray, Y: np.ndarray, t_query: float) -> np.ndarray:
    t_q = float(np.clip(t_query, float(t_h[0]), float(t_h[-1])))
    n_sp = Y.shape[1]
    return np.array([float(np.interp(t_q, t_h, Y[:, i])) for i in range(n_sp)], dtype=np.float64)


def augment_rows_time_shift(
    df: pd.DataFrame,
    rng: np.random.Generator,
    augment_per_row: int,
    include_unshifted_base: bool = True,
    *,
    initial_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    traj_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    if initial_cache:
        traj_cache.update(initial_cache)
    rows_out: list[dict[str, float]] = []

    pe_t = df[["phi", "energy", "time"]]
    for row in pe_t.itertuples(index=False, name=None):
        phi = float(row[0])
        e_ev = float(row[1])
        t_full = float(row[2])

        t_h, Y = get_trajectory(phi, e_ev, t_full, traj_cache)
        n_at_T = Y[-1].astype(np.float64)

        if include_unshifted_base:
            rows_out.append({
                "phi": phi, "energy": e_ev, "time": t_full,
                "init_N226": TRAIN_INIT_RA226, "init_N225": TRAIN_INIT_RA225,
                "init_NAc": TRAIN_INIT_AC225,
                "init_N227": TRAIN_INIT_RA227, "init_NAc227": TRAIN_INIT_AC227,
                "N_Ra226": n_at_T[0], "N_Ra225": n_at_T[1], "N_Ac225": n_at_T[2],
                "N_Ra227": n_at_T[3], "N_Ac227": n_at_T[4],
            })

        for _ in range(augment_per_row):
            delta = float(rng.uniform(0.0, max(t_full - MIN_T_REM_H, 1e-9)))
            t_rem = t_full - delta
            if t_rem < MIN_T_REM_H:
                continue
            n_delta = state_at_time(t_h, Y, delta)
            rows_out.append({
                "phi": phi, "energy": e_ev, "time": t_rem,
                "init_N226": n_delta[0], "init_N225": n_delta[1], "init_NAc": n_delta[2],
                "init_N227": n_delta[3], "init_NAc227": n_delta[4],
                "N_Ra226": n_at_T[0], "N_Ra225": n_at_T[1], "N_Ac225": n_at_T[2],
                "N_Ra227": n_at_T[3], "N_Ac227": n_at_T[4],
            })
    return pd.DataFrame(rows_out)


def augment_inverted_ic_scenarios(rng: np.random.Generator, n_extra: int = 400) -> pd.DataFrame:
    """Ra-225 dominant, Ra-226 low: pure decay / low-flux scenarios."""
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        ra225_0 = float(rng.uniform(5e17, 2e19))
        ra226_0 = float(rng.uniform(0.0, 1e17))
        ac_0 = 0.0
        phi = 0.0 if rng.uniform() < 0.7 else float(rng.uniform(1e12, 1e13))
        energy_ev = sample_neutron_energy_ev(rng)
        time_h = sample_time_hours(rng, 0.1, 300.0)

        env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
        t_h, Y = run_simulation(
            env, t_end_h=time_h, n_points=max(200, int(time_h * 5)),
            N_ra0=ra226_0, N_ra225_0=ra225_0, N_ac0=ac_0,
        )
        nf = Y[-1].astype(np.float64)
        rows_out.append({
            "phi": phi, "energy": energy_ev, "time": time_h,
            "init_N226": ra226_0, "init_N225": ra225_0, "init_NAc": ac_0,
            "init_N227": 0.0, "init_NAc227": 0.0,
            "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
            "N_Ra227": nf[3], "N_Ac227": nf[4],
        })
        for shift_frac in [0.25, 0.5]:
            t_shift = time_h * shift_frac
            ns = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi, "energy": energy_ev, "time": t_remaining,
                    "init_N226": ns[0], "init_N225": ns[1], "init_NAc": ns[2],
                    "init_N227": ns[3], "init_NAc227": ns[4],
                    "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
                    "N_Ra227": nf[3], "N_Ac227": nf[4],
                })
    return pd.DataFrame(rows_out)


def augment_diverse_ic_scenarios(rng: np.random.Generator, n_extra: int = 500) -> pd.DataFrame:
    """Diverse ICs: Ra225-dominant, Ra226-dominant, mixed, Ac-dominant."""
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        regime_choice = float(rng.uniform())
        if regime_choice < 0.5:
            regime = "ra225_dom"
        elif regime_choice < 0.75:
            regime = "ra226_dom"
        elif regime_choice < 0.9:
            regime = "mixed"
        else:
            regime = "ac_dom"

        if regime == "ra226_dom":
            ra226_0 = float(rng.uniform(1e23, 1e24))
            ra225_0 = float(rng.uniform(0.0, 1e18))
            ac_0 = float(rng.uniform(0.0, 1e17))
        elif regime == "ra225_dom":
            ra225_0 = float(rng.uniform(1e18, 5e19))
            ra226_0 = float(rng.uniform(0.0, 1e17))
            ac_0 = float(rng.uniform(0.0, 1e17))
        elif regime == "ac_dom":
            ac_0 = float(rng.uniform(1e18, 1e20))
            ra225_0 = float(rng.uniform(0.0, 1e18))
            ra226_0 = float(rng.uniform(0.0, 1e17))
        else:
            ra226_0 = float(rng.uniform(1e17, 1e24))
            ra225_0 = float(rng.uniform(1e17, 1e20))
            ac_0 = float(rng.uniform(1e16, 1e20))

        phi = float(rng.uniform(0.0, 1e15))
        energy_ev = sample_neutron_energy_ev(rng)
        time_h = sample_time_hours(rng, 0.1, 300.0)

        env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
        t_h, Y = run_simulation(
            env, t_end_h=time_h, n_points=max(200, int(time_h * 5)),
            N_ra0=ra226_0, N_ra225_0=ra225_0, N_ac0=ac_0,
        )
        nf = Y[-1].astype(np.float64)
        rows_out.append({
            "phi": phi, "energy": energy_ev, "time": time_h,
            "init_N226": ra226_0, "init_N225": ra225_0, "init_NAc": ac_0,
            "init_N227": 0.0, "init_NAc227": 0.0,
            "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
            "N_Ra227": nf[3], "N_Ac227": nf[4],
        })
        for shift_frac in [0.3, 0.6]:
            t_shift = time_h * shift_frac
            ns = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi, "energy": energy_ev, "time": t_remaining,
                    "init_N226": ns[0], "init_N225": ns[1], "init_NAc": ns[2],
                    "init_N227": ns[3], "init_NAc227": ns[4],
                    "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
                    "N_Ra227": nf[3], "N_Ac227": nf[4],
                })
    return pd.DataFrame(rows_out)


def augment_spectrum_sweep_scenarios(rng: np.random.Generator, n_extra: int = 400) -> pd.DataFrame:
    """
    Add explicit thermal/epithermal/fast-neutron cases so the PINN can learn
    both the Ac-225 production channel and the Ac-227 impurity channel.
    """
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        regime = float(rng.uniform())
        if regime < 0.50:
            # Desired fast production: Ra-226(n,2n)Ra-225 above threshold.
            energy_ev = float(np.clip(rng.normal(14.0e6, 1.0e6), 12.5e6, 15.5e6))
            phi = float(10.0 ** rng.uniform(13.0, 15.0))
            ra226_0 = float(10.0 ** rng.uniform(21.0, np.log10(TRAIN_INIT_RA226)))
            ra225_0 = 0.0
            ac_0 = 0.0
            ra227_0 = 0.0
            ac227_0 = 0.0
            time_h = float(10.0 ** rng.uniform(np.log10(0.25), np.log10(500.0)))
        elif regime < 0.80:
            # Impurity stress cases: thermal/epithermal capture to Ra-227/Ac-227.
            energy_ev = float(rng.uniform(*THERMAL_ENERGY_RANGE_EV)) if rng.uniform() < 0.6 else float(
                10.0 ** rng.uniform(np.log10(EPITHERMAL_ENERGY_RANGE_EV[0]), np.log10(EPITHERMAL_ENERGY_RANGE_EV[1]))
            )
            phi = float(10.0 ** rng.uniform(12.0, 15.0))
            ra226_0 = float(10.0 ** rng.uniform(20.0, np.log10(TRAIN_INIT_RA226)))
            ra225_0 = 0.0
            ac_0 = 0.0
            ra227_0 = float(10.0 ** rng.uniform(8.0, 15.0)) if rng.uniform() < 0.25 else 0.0
            ac227_0 = float(10.0 ** rng.uniform(6.0, 13.0)) if rng.uniform() < 0.25 else 0.0
            time_h = sample_time_hours(rng, 0.25, 300.0)
        elif regime < 0.92:
            # Threshold cases: hardest region for the (n,2n) turn-on.
            energy_ev = float(rng.uniform(*THRESHOLD_ENERGY_RANGE_EV))
            phi = float(10.0 ** rng.uniform(12.0, 15.0))
            ra226_0 = float(10.0 ** rng.uniform(20.0, np.log10(TRAIN_INIT_RA226)))
            ra225_0 = float(10.0 ** rng.uniform(12.0, 18.0)) if rng.uniform() < 0.35 else 0.0
            ac_0 = float(10.0 ** rng.uniform(10.0, 17.0)) if rng.uniform() < 0.35 else 0.0
            ra227_0 = float(10.0 ** rng.uniform(10.0, 16.0)) if rng.uniform() < 0.20 else 0.0
            ac227_0 = float(10.0 ** rng.uniform(8.0, 14.0)) if rng.uniform() < 0.20 else 0.0
            time_h = sample_time_hours(rng, 0.1, 500.0)
        else:
            # Recycled/interrupted targets with nonzero daughters.
            energy_ev = sample_validation_regime_energy_ev(rng)
            phi = float(10.0 ** rng.uniform(11.0, 15.0))
            ra226_0 = float(10.0 ** rng.uniform(18.0, np.log10(TRAIN_INIT_RA226)))
            ra225_0 = float(10.0 ** rng.uniform(15.0, 20.0))
            ac_0 = float(10.0 ** rng.uniform(14.0, 19.0))
            ra227_0 = float(10.0 ** rng.uniform(10.0, 17.0))
            ac227_0 = float(10.0 ** rng.uniform(8.0, 15.0))
            time_h = sample_time_hours(rng, 0.25, 500.0)

        t_h, Y = _append_ode_row(
            rows_out,
            phi=phi, energy_ev=energy_ev, time_h=time_h,
            ra226_0=ra226_0, ra225_0=ra225_0, ac_0=ac_0,
            ra227_0=ra227_0, ac227_0=ac227_0,
        )
        nf = Y[-1].astype(np.float64)

        for shift_frac in [0.1, 0.35, 0.7]:
            t_shift = time_h * shift_frac
            ns = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi, "energy": energy_ev, "time": t_remaining,
                    "init_N226": ns[0], "init_N225": ns[1], "init_NAc": ns[2],
                    "init_N227": ns[3], "init_NAc227": ns[4],
                    "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
                    "N_Ra227": nf[3], "N_Ac227": nf[4],
                })
    return pd.DataFrame(rows_out)


def augment_recycled_trace_inventory_scenarios(
    rng: np.random.Generator,
    *,
    n_recycled: int = 500,
    n_tiny: int = 300,
) -> pd.DataFrame:
    """
    Stress the exact trace-daughter cases that held-out validation probes:
    nonzero recycled Ra-227/Ac-227 starts and tiny-but-nonzero daughter signals.
    """
    rows_out: list[dict[str, float]] = []
    n_total = int(n_recycled) + int(n_tiny)
    for i in range(n_total):
        energy_ev = sample_validation_regime_energy_ev(rng)
        phi = float(10.0 ** rng.uniform(12.0, 15.0))
        time_h = sample_time_hours(rng, 0.1, 500.0)

        if i < n_recycled:
            ra226_0 = float(10.0 ** rng.uniform(18.0, np.log10(TRAIN_INIT_RA226)))
            ra225_0 = float(10.0 ** rng.uniform(15.0, 20.0))
            ac_0 = float(10.0 ** rng.uniform(14.0, 19.0))
            ra227_0 = float(10.0 ** rng.uniform(10.0, 17.0))
            ac227_0 = float(10.0 ** rng.uniform(8.0, 15.0))
        else:
            # Tiny nonzero daughters teach the network not to collapse traces to zero.
            ra226_0 = float(10.0 ** rng.uniform(20.0, np.log10(TRAIN_INIT_RA226)))
            ra225_0 = float(10.0 ** rng.uniform(10.0, 16.0))
            ac_0 = float(10.0 ** rng.uniform(8.0, 15.0))
            ra227_0 = float(10.0 ** rng.uniform(6.0, 13.0))
            ac227_0 = float(10.0 ** rng.uniform(4.0, 12.0))

        t_h, Y = _append_ode_row(
            rows_out,
            phi=phi, energy_ev=energy_ev, time_h=time_h,
            ra226_0=ra226_0, ra225_0=ra225_0, ac_0=ac_0,
            ra227_0=ra227_0, ac227_0=ac227_0,
            min_points=300,
            points_per_hour=8,
        )
        nf = Y[-1].astype(np.float64)
        for shift_frac in [0.2, 0.55]:
            t_shift = time_h * shift_frac
            ns = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi, "energy": energy_ev, "time": t_remaining,
                    "init_N226": ns[0], "init_N225": ns[1], "init_NAc": ns[2],
                    "init_N227": ns[3], "init_NAc227": ns[4],
                    "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
                    "N_Ra227": nf[3], "N_Ac227": nf[4],
                })
    return pd.DataFrame(rows_out)


def augment_high_energy_virgin_scenarios(
    rng: np.random.Generator,
    n_extra: int = 700,
) -> pd.DataFrame:
    """Fresh Ra-226 high-energy cases that prevent virgin Ac-225 zero-collapse."""
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        if rng.uniform() < 0.55:
            energy_ev = float(np.clip(rng.normal(14.0e6, 0.9e6), 12.5e6, 15.5e6))
        else:
            energy_ev = float(rng.uniform(*THRESHOLD_ENERGY_RANGE_EV))
        phi = float(10.0 ** rng.uniform(12.0, 15.0))
        time_h = sample_time_hours(rng, 0.1, 500.0)
        ra226_0 = float(10.0 ** rng.uniform(20.0, np.log10(TRAIN_INIT_RA226)))
        ra225_0 = 0.0
        ac_0 = 0.0
        ra227_0 = 0.0
        ac227_0 = 0.0

        t_h, Y = _append_ode_row(
            rows_out,
            phi=phi, energy_ev=energy_ev, time_h=time_h,
            ra226_0=ra226_0, ra225_0=ra225_0, ac_0=ac_0,
            ra227_0=ra227_0, ac227_0=ac227_0,
            min_points=350,
            points_per_hour=10,
        )
        nf = Y[-1].astype(np.float64)
        for shift_frac in [0.15, 0.35, 0.65, 0.85]:
            t_shift = time_h * shift_frac
            ns = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi, "energy": energy_ev, "time": t_remaining,
                    "init_N226": ns[0], "init_N225": ns[1], "init_NAc": ns[2],
                    "init_N227": ns[3], "init_NAc227": ns[4],
                    "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
                    "N_Ra227": nf[3], "N_Ac227": nf[4],
                })
    return pd.DataFrame(rows_out)


def augment_threshold_edge_virgin_scenarios(
    rng: np.random.Generator,
    n_extra: int = 300,
) -> pd.DataFrame:
    """
    Fresh-target cases pinned to the 6.42 MeV cliff edge.

    The regime-gated model needs direct evidence at the transition, not only
    broad threshold samples, or the high-energy head learns a large baseline.
    """
    rows_out: list[dict[str, float]] = []
    edge_energies = np.array([
        6.20e6,
        6.32e6,
        6.38e6,
        6.42e6,
        6.46e6,
        6.52e6,
        6.65e6,
    ], dtype=np.float64)
    for i in range(n_extra):
        if rng.uniform() < 0.60:
            energy_ev = float(edge_energies[i % len(edge_energies)])
        else:
            energy_ev = float(np.clip(rng.normal(6.42e6, 1.0e5), 6.15e6, 6.75e6))

        phi = float(10.0 ** rng.uniform(12.0, 15.0))
        time_h = sample_time_hours(rng, 0.1, 500.0)
        ra226_0 = float(10.0 ** rng.uniform(20.0, np.log10(TRAIN_INIT_RA226)))
        t_h, Y = _append_ode_row(
            rows_out,
            phi=phi,
            energy_ev=energy_ev,
            time_h=time_h,
            ra226_0=ra226_0,
            ra225_0=0.0,
            ac_0=0.0,
            ra227_0=0.0,
            ac227_0=0.0,
            min_points=350,
            points_per_hour=10,
        )
        nf = Y[-1].astype(np.float64)
        for shift_frac in [0.2, 0.5, 0.8]:
            t_shift = time_h * shift_frac
            ns = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi,
                    "energy": energy_ev,
                    "time": t_remaining,
                    "init_N226": ns[0],
                    "init_N225": ns[1],
                    "init_NAc": ns[2],
                    "init_N227": ns[3],
                    "init_NAc227": ns[4],
                    "N_Ra226": nf[0],
                    "N_Ra225": nf[1],
                    "N_Ac225": nf[2],
                    "N_Ra227": nf[3],
                    "N_Ac227": nf[4],
                })
    return pd.DataFrame(rows_out)


def augment_ac227_long_chain_scenarios(
    rng: np.random.Generator,
    n_extra: int = 350,
) -> pd.DataFrame:
    """Long thermal/epithermal runs that teach slow Ra-227 -> Ac-227 ingrowth."""
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        energy_ev = float(rng.uniform(*THERMAL_ENERGY_RANGE_EV)) if rng.uniform() < 0.6 else float(
            10.0 ** rng.uniform(np.log10(EPITHERMAL_ENERGY_RANGE_EV[0]), np.log10(EPITHERMAL_ENERGY_RANGE_EV[1]))
        )
        phi = float(10.0 ** rng.uniform(12.0, 15.0))
        time_h = sample_time_hours(rng, 24.0, 500.0)
        ra226_0 = float(10.0 ** rng.uniform(20.0, np.log10(TRAIN_INIT_RA226)))
        ra225_0 = float(10.0 ** rng.uniform(12.0, 18.0)) if rng.uniform() < 0.25 else 0.0
        ac_0 = float(10.0 ** rng.uniform(10.0, 16.0)) if rng.uniform() < 0.25 else 0.0
        ra227_0 = float(10.0 ** rng.uniform(12.0, 18.0)) if rng.uniform() < 0.5 else 0.0
        ac227_0 = float(10.0 ** rng.uniform(8.0, 15.0)) if rng.uniform() < 0.5 else 0.0

        t_h, Y = _append_ode_row(
            rows_out,
            phi=phi, energy_ev=energy_ev, time_h=time_h,
            ra226_0=ra226_0, ra225_0=ra225_0, ac_0=ac_0,
            ra227_0=ra227_0, ac227_0=ac227_0,
            min_points=400,
            points_per_hour=10,
        )
        nf = Y[-1].astype(np.float64)
        for shift_frac in [0.25, 0.5, 0.75]:
            t_shift = time_h * shift_frac
            ns = state_at_time(t_h, Y, t_shift)
            t_remaining = time_h - t_shift
            if t_remaining > MIN_T_REM_H:
                rows_out.append({
                    "phi": phi, "energy": energy_ev, "time": t_remaining,
                    "init_N226": ns[0], "init_N225": ns[1], "init_NAc": ns[2],
                    "init_N227": ns[3], "init_NAc227": ns[4],
                    "N_Ra226": nf[0], "N_Ra225": nf[1], "N_Ac225": nf[2],
                    "N_Ra227": nf[3], "N_Ac227": nf[4],
                })
    return pd.DataFrame(rows_out)


def augment_empty_tank_rows(rng: np.random.Generator, n_extra: int = 300) -> pd.DataFrame:
    """Empty initial inventories at various flux/time/energy -> all outputs must be 0."""
    rows_out: list[dict[str, float]] = []
    for _ in range(n_extra):
        phi = float(10.0 ** rng.uniform(13.0, 15.0))
        energy_ev = sample_neutron_energy_ev(rng)
        time_h = sample_time_hours(rng, 0.1, 500.0)
        rows_out.append({
            "phi": phi, "energy": energy_ev, "time": time_h,
            "init_N226": 0.0, "init_N225": 0.0, "init_NAc": 0.0,
            "init_N227": 0.0, "init_NAc227": 0.0,
            "N_Ra226": 0.0, "N_Ra225": 0.0, "N_Ac225": 0.0,
            "N_Ra227": 0.0, "N_Ac227": 0.0,
        })
    return pd.DataFrame(rows_out)


def prepare_training_tensors(
    df: pd.DataFrame, device: torch.device, dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    time_h = torch.tensor(df["time"].values, dtype=dtype, device=device)
    phi = torch.tensor(df["phi"].values, dtype=dtype, device=device)
    energy_raw = torch.tensor(df["energy"].values, dtype=dtype, device=device)
    energy_nn = neutron_energy_ev_to_feature_torch(energy_raw)
    init226 = torch.tensor(df["init_N226"].values, dtype=dtype, device=device)
    init225 = torch.tensor(df["init_N225"].values, dtype=dtype, device=device)
    init_ac = torch.tensor(df["init_NAc"].values, dtype=dtype, device=device)
    init227 = torch.tensor(df.get("init_N227", pd.Series(0.0, index=df.index)).values, dtype=dtype, device=device)
    init_ac7 = torch.tensor(df.get("init_NAc227", pd.Series(0.0, index=df.index)).values, dtype=dtype, device=device)

    t_input = (time_h / TIME_SCALE_H).clone()
    phi_nn = (phi / PHI_SCALE).clone()
    inputs = torch.cat([
        t_input.unsqueeze(1),
        phi_nn.unsqueeze(1),
        energy_nn.unsqueeze(1).detach(),
        (init226 / N226_SCALE).unsqueeze(1).detach(),
        (init225 / N225_SCALE).unsqueeze(1).detach(),
        (init_ac / NAC_SCALE).unsqueeze(1).detach(),
        (init227 / N227_SCALE).unsqueeze(1).detach(),
        (init_ac7 / NAC227_SCALE).unsqueeze(1).detach(),
    ], dim=1).detach()
    inputs.requires_grad_(True)

    target_cols = ["N_Ra226", "N_Ra225", "N_Ac225"]
    for col, default in [("N_Ra227", 0.0), ("N_Ac227", 0.0)]:
        if col not in df.columns:
            df[col] = default
        target_cols.append(col)
    targets = torch.tensor(
        df[target_cols].values,
        dtype=dtype, device=device,
    )
    return inputs, targets


def apply_empirical_flux_jitter(train_dat: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Optional log-normal phi jitter from empirical flux logs (PINN_FLUX_JITTER_SIGMA)."""
    sigma = float(os.environ.get("PINN_FLUX_JITTER_SIGMA", "0") or "0")
    if sigma <= 0:
        return train_dat
    out = train_dat.copy()
    mask = out["phi"].astype(float) > 0.0
    if not mask.any():
        return out
    jitter = rng.lognormal(mean=0.0, sigma=sigma, size=int(mask.sum()))
    out.loc[mask, "phi"] = out.loc[mask, "phi"].astype(float) * jitter
    print(f"Empirical flux jitter: sigma={sigma}, perturbed {int(mask.sum())} rows")
    return out


def print_trace_coverage(train_dat: pd.DataFrame) -> None:
    """Show whether training data contains the trace cases the predictor must learn."""
    n = max(1, len(train_dat))
    ac225_target = train_dat.get("N_Ac225", pd.Series(0.0, index=train_dat.index)).astype(float)
    ac227_target = train_dat.get("N_Ac227", pd.Series(0.0, index=train_dat.index)).astype(float)
    init227 = train_dat.get("init_N227", pd.Series(0.0, index=train_dat.index)).astype(float)
    init_ac227 = train_dat.get("init_NAc227", pd.Series(0.0, index=train_dat.index)).astype(float)
    fast14 = train_dat["energy"].between(12.5e6, 15.5e6)
    threshold = train_dat["energy"].between(THRESHOLD_ENERGY_RANGE_EV[0], THRESHOLD_ENERGY_RANGE_EV[1])
    virgin = (
        (train_dat["init_N226"].astype(float) > 1.0)
        & (train_dat["init_N225"].astype(float).abs() <= 1.0)
        & (train_dat["init_NAc"].astype(float).abs() <= 1.0)
        & (init227.abs() <= 1.0)
        & (init_ac227.abs() <= 1.0)
    )
    recycled_trace = (init227 > 1.0) | (init_ac227 > 1.0)
    print(
        "Trace coverage: "
        f"Ac225>floor={int((ac225_target > NAC_SCALE * 1e-12).sum())}/{n}, "
        f"Ac227>floor={int((ac227_target > NAC227_SCALE * 1e-12).sum())}/{n}, "
        f"recycled_227_ic={int(recycled_trace.sum())}/{n}, "
        f"fast14={int(fast14.sum())}/{n}, "
        f"virgin_fast14={int((virgin & fast14 & (ac225_target > NAC_SCALE * 1e-12)).sum())}/{n}, "
        f"virgin_threshold={int((virgin & threshold & (ac225_target > NAC_SCALE * 1e-12)).sum())}/{n}"
    )


def trace_balanced_epoch_order(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    *,
    trace_fraction: float = TRACE_CHUNK_FRACTION,
) -> torch.Tensor:
    """Interleave trace-positive rows so each mini-batch sees Ac-225/Ac-227 signal."""
    n = int(inputs.size(0))
    device = inputs.device
    if chunk_size <= 0 or chunk_size >= n:
        return torch.arange(n, device=device)

    valid = ~torch.isnan(targets).any(dim=1)
    target_trace = valid & ((targets[:, 2] > NAC_SCALE * 1e-12) | (targets[:, 4] > NAC227_SCALE * 1e-12))
    init_trace = (inputs[:, 4] > 1e-12) | (inputs[:, 5] > 1e-12) | (inputs[:, 6] > 1e-12) | (inputs[:, 7] > 1e-12)
    trace_mask = target_trace | init_trace
    trace_idx = torch.nonzero(trace_mask, as_tuple=False).flatten()
    other_idx = torch.nonzero(~trace_mask, as_tuple=False).flatten()
    if trace_idx.numel() == 0 or other_idx.numel() == 0:
        return torch.randperm(n, device=device)

    trace_idx = trace_idx[torch.randperm(trace_idx.numel(), device=device)]
    other_idx = other_idx[torch.randperm(other_idx.numel(), device=device)]
    min_trace_per_chunk = max(1, int(round(chunk_size * trace_fraction)))
    ordered: list[torch.Tensor] = []
    t_pos = 0
    o_pos = 0
    while t_pos < trace_idx.numel() or o_pos < other_idx.numel():
        take_t = min(min_trace_per_chunk, trace_idx.numel() - t_pos)
        take_o = min(chunk_size - take_t, other_idx.numel() - o_pos)
        if take_t > 0:
            ordered.append(trace_idx[t_pos:t_pos + take_t])
            t_pos += take_t
        if take_o > 0:
            ordered.append(other_idx[o_pos:o_pos + take_o])
            o_pos += take_o
        if take_o <= 0 and t_pos < trace_idx.numel():
            take_extra = min(chunk_size, trace_idx.numel() - t_pos)
            ordered.append(trace_idx[t_pos:t_pos + take_extra])
            t_pos += take_extra
    return torch.cat(ordered)[:n]


def build_pretrain_collocation(
    n: int, rng: np.random.Generator, device: torch.device, *, t_max_h: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Three-slice pretrain collocation that exposes the model to all physics regimes.

    The previous implementation (``build_zero_flux_collocation``) had a critical bug:
    100% of points started with phi=0, then the EMPTY_FEED_FRACTION slice (~30%) was
    overwritten with phi=high AND ICs zeroed. Result: NO point in pretrain ever had
    both flux AND fuel, so the model had zero prior on flux-driven Bateman dynamics
    when joint training began.

    New distribution:
      - 60%: log-uniform flux in [1e-3, 1.0] normalized (1e12-1e15 raw) WITH non-zero
             Ra-226 IC. This is the actual transmutation regime the model must learn.
      -  20%: flux=0 with non-zero ICs (decay-only boundary case; preserves the
             original useful coverage of pure-decay dynamics).
      -  20%: flux=high (1.0 normalized) with empty ICs (former EMPTY_FEED slice;
             closed-system constraint that empty in -> empty out).
    """
    n = int(n)
    n_xmute = int(round(0.60 * n))
    n_decay = int(round(0.20 * n))
    n_empty = max(1, n - n_xmute - n_decay)
    # If rounding overshot, trim from the transmutation slice (largest).
    if n_xmute + n_decay + n_empty != n:
        n_xmute = n - n_decay - n_empty

    time_h = torch.tensor(rng.uniform(1e-4, float(t_max_h), n), dtype=dtype, device=device)

    # Energy: same spectral mix as before (thermal/epithermal/threshold/fast).
    energy_raw = torch.tensor(
        [sample_neutron_energy_ev(rng) for _ in range(n)],
        dtype=dtype,
        device=device,
    )
    energy_nn = neutron_energy_ev_to_feature_torch(energy_raw)

    # Per-point flux assignment.
    # Slices are defined by INDEX RANGES into a randomly-permuted view so the assignment
    # is deterministic given rng but uncorrelated with time/energy sampling above.
    phi_nn = torch.zeros((n,), dtype=dtype, device=device)
    perm = torch.from_numpy(rng.permutation(n).astype(np.int64)).to(device=device)
    idx_xmute = perm[:n_xmute]
    idx_decay = perm[n_xmute:n_xmute + n_decay]
    idx_empty = perm[n_xmute + n_decay:]

    # Transmutation slice: log-uniform phi_nn in [1e-3, 1.0]  (matches joint distribution)
    if n_xmute > 0:
        log_phi = torch.from_numpy(
            rng.uniform(np.log(1e-3), np.log(1.0), size=n_xmute).astype(np.float64)
        ).to(device=device, dtype=dtype)
        phi_nn[idx_xmute] = torch.exp(log_phi)

    # Decay-only slice: phi_nn already 0 from initialization.
    # Empty-feed slice: phi_nn = 1.0 (i.e. PHI_SCALE = 1e15 raw).
    if n_empty > 0:
        phi_nn[idx_empty] = float(EMPTY_FEED_HIGH_FLUX / PHI_SCALE)

    # Per-point IC assignment.
    # Default = log-uniform Ra-226 in [1, 1.1 * N226_SCALE], plus optional small Ra-225 / Ac-225.
    # Empty slice gets all-zero ICs.
    lo = 1.0
    hi_226 = N226_SCALE * 1.1
    hi_225 = N225_SCALE * 10.0
    hi_ac = NAC_SCALE * 10.0
    u = rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float64)
    init226_raw = np.exp(np.log(lo) + u[:, 0] * (np.log(hi_226) - np.log(lo)))
    init225_raw = np.exp(np.log(lo) + u[:, 1] * (np.log(hi_225) - np.log(lo)))
    initac_raw = np.exp(np.log(lo) + u[:, 2] * (np.log(hi_ac) - np.log(lo)))
    init226 = torch.tensor(init226_raw, dtype=dtype, device=device)
    init225 = torch.tensor(init225_raw, dtype=dtype, device=device)
    init_ac = torch.tensor(initac_raw, dtype=dtype, device=device)
    zeros_n = torch.zeros(n, dtype=dtype, device=device)

    t_input = (time_h / TIME_SCALE_H).clone()
    colloc = torch.cat([
        t_input.unsqueeze(1),
        phi_nn.unsqueeze(1),
        energy_nn.unsqueeze(1),
        (init226 / N226_SCALE).unsqueeze(1).detach(),
        (init225 / N225_SCALE).unsqueeze(1).detach(),
        (init_ac / NAC_SCALE).unsqueeze(1).detach(),
        zeros_n.unsqueeze(1),   # init Ra-227
        zeros_n.unsqueeze(1),   # init Ac-227
    ], dim=1)

    # Empty slice: zero out all ICs (closed-system boundary case).
    if n_empty > 0:
        colloc[idx_empty, 3:8] = 0.0

    return colloc.detach().requires_grad_(True)


# Backward-compatible alias. Old callers (analysis scripts, RAR sampling) keep working
# while we migrate. The implementation is now the three-slice flux+fuel mix above.
build_zero_flux_collocation = build_pretrain_collocation


LOSS_HISTORY_CSV_FIELDNAMES = (
    "epoch",
    "phase",
    "data_mse",
    "physics_mse",
    "supervised_total",
    "unsupervised_total",
    "grad_norm",
)


def truncate_loss_history_csv_for_run(path: pathlib.Path) -> None:
    """Start each training run with a fresh loss CSV (no append across separate invocations)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOSS_HISTORY_CSV_FIELDNAMES)
        w.writeheader()


def prepare_loss_history_df_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by epoch; if epoch restarts mid-file (new run concatenated), keep only the last segment."""
    if df is None or len(df) == 0:
        return df
    out = df.sort_values("epoch", kind="mergesort").reset_index(drop=True)
    ep = out["epoch"].to_numpy()
    if len(ep) < 2:
        return out
    last_split = 0
    for i in range(1, len(ep)):
        if ep[i] < ep[i - 1]:
            last_split = i
    if last_split > 0:
        out = out.iloc[last_split:].reset_index(drop=True)
    return out


def plot_loss_components_png(
    loss_csv_path: pathlib.Path,
    out_path: pathlib.Path,
    *,
    proj_root: pathlib.Path | None = None,
    run_id: str | None = None,
    record_provenance: bool = True,
    producer: str = "train.py",
    df: pd.DataFrame | None = None,
) -> bool:
    """Log-log loss components from loss_history.csv; writes PNG and optional graph_provenance record."""
    if df is None:
        if not loss_csv_path.is_file():
            return False
        try:
            raw = pd.read_csv(loss_csv_path)
        except Exception:
            return False
    else:
        raw = df
    if raw.empty:
        return False
    if {"epoch", "data_loss", "physics_loss"}.issubset(raw.columns) and not {
        "data_mse",
        "physics_mse",
    }.issubset(raw.columns):
        raw = raw.rename(columns={"data_loss": "data_mse", "physics_loss": "physics_mse"})
    if not {"epoch", "data_mse", "physics_mse"}.issubset(raw.columns):
        return False
    df = prepare_loss_history_df_for_plot(raw)
    if df.empty:
        return False

    phase_lower = (
        df["phase"].astype(str).str.lower()
        if "phase" in df.columns
        else pd.Series("", index=df.index)
    )
    is_pre = phase_lower.eq("pretrain")

    dcol = df["data_mse"].astype(float)
    if "phase" in df.columns:
        dcol = dcol.mask(is_pre & (dcol == 0.0), np.nan)
    else:
        dcol = dcol.replace(0.0, np.nan)

    pcol = df["physics_mse"].astype(float).replace(0.0, np.nan)
    ep = df["epoch"].astype(float)

    def ema(values: np.ndarray, alpha: float = 0.04) -> np.ndarray:
        if len(values) == 0:
            return values
        smoothed = []
        # Find first non-nan value to initialize current
        current = np.nan
        for v in values:
            if not np.isnan(v):
                current = v
                break
        for val in values:
            if np.isnan(val):
                smoothed.append(np.nan)
            else:
                if np.isnan(current):
                    current = val
                current = alpha * val + (1.0 - alpha) * current
                smoothed.append(current)
        return np.asarray(smoothed)

    dcol_smooth = ema(dcol.values, alpha=0.04)
    pcol_smooth = ema(pcol.values, alpha=0.04)

    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Raw components (light)
    ax.loglog(ep, dcol, color="#3498db", alpha=0.12)
    ax.loglog(ep, pcol, color="#e74c3c", alpha=0.12)

    # Smoothed components (bold)
    ax.loglog(ep, dcol_smooth, label="Data MSE (Smoothed)", color="#3498db", lw=2)
    ax.loglog(ep, pcol_smooth, label="Physics MSE (Smoothed)", color="#e74c3c", lw=2)

    if {"supervised_total", "unsupervised_total"}.issubset(df.columns):
        sup = df["supervised_total"].astype(float)
        uns = df["unsupervised_total"].astype(float)
        if "phase" in df.columns:
            sup_plot = sup.mask(is_pre, np.nan)
        else:
            sup_plot = sup.replace(0.0, np.nan)
        uns_plot = uns.replace(0.0, np.nan)
        
        sup_smooth = ema(sup_plot.values, alpha=0.04)
        uns_smooth = ema(uns_plot.values, alpha=0.04)
        
        # Raw totals (light)
        ax.loglog(ep, sup_plot, color="#2ecc71", alpha=0.10, ls="--")
        ax.loglog(ep, uns_plot, color="#9b59b6", alpha=0.10, ls="--")
        
        # Smoothed totals (bold dashed)
        ax.loglog(ep, sup_smooth, label="Supervised Total (Smoothed)", color="#2ecc71", ls="--", lw=1.5)
        ax.loglog(ep, uns_smooth, label="Unsupervised Total (Smoothed)", color="#9b59b6", ls="--", lw=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Components Convergence (EMA Smoothed)", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.tight_layout()
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if record_provenance and proj_root is not None and run_id is not None:
        graph_provenance.record_graph_write(
            proj_root,
            out_path.resolve(),
            producer=producer,
            run_id=run_id,
            extra={"loss_csv": str(pathlib.Path(loss_csv_path).resolve())},
        )
    return True


def append_loss_history_csv(
    path: pathlib.Path,
    *,
    epoch: int,
    phase: str,
    data_mse: float,
    physics_mse: float,
    supervised_total: float,
    unsupervised_total: float,
    grad_norm: float = 0.0,
) -> None:
    """Append one row for extra_plots loss_components / diagnostics (phase-separated training)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = LOSS_HISTORY_CSV_FIELDNAMES
    row = {
        "epoch": epoch,
        "phase": phase,
        "data_mse": data_mse,
        "physics_mse": physics_mse,
        "supervised_total": supervised_total,
        "unsupervised_total": unsupervised_total,
        "grad_norm": grad_norm,
    }
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def plot_loss_history(epochs, data_loss, physics_loss, path, train_pre: int) -> None:
    """Two-panel loss plot with EMA smoothing to showcase clean convergence."""
    ep = np.asarray(epochs, dtype=float)
    hd = np.asarray(data_loss, dtype=float)
    hp = np.asarray(physics_loss, dtype=float)
    tp = float(train_pre)
    pre_m = ep <= tp
    joint_m = ep > tp

    def ema(values: np.ndarray, alpha: float = 0.04) -> np.ndarray:
        if len(values) == 0:
            return values
        smoothed = []
        current = values[0]
        for val in values:
            if np.isnan(val):
                smoothed.append(np.nan)
            else:
                if np.isnan(current):
                    current = val
                current = alpha * val + (1.0 - alpha) * current
                smoothed.append(current)
        return np.asarray(smoothed)

    fig, (ax_pre, ax_joint) = plt.subplots(1, 2, figsize=(12, 5))

    if np.any(pre_m):
        epochs_pre = ep[pre_m]
        phys_pre = np.clip(hp[pre_m], 1e-30, None)
        phys_pre_smooth = ema(phys_pre, alpha=0.08)
        
        # Raw loss is light
        ax_pre.semilogy(epochs_pre, phys_pre, color="#e74c3c", alpha=0.15, label="Raw Physics MSE")
        # Smoothed loss is bold
        ax_pre.semilogy(epochs_pre, phys_pre_smooth, color="#e74c3c", lw=2, label="Smoothed Physics MSE")
        
        ax_pre.set_title("Phase 1: Physics Pre-training", fontsize=12, fontweight="bold", pad=10)
        ax_pre.set_xlabel("Epoch")
        ax_pre.set_ylabel("Loss (log scale)")
        ax_pre.legend(loc="upper right")
        ax_pre.grid(True, which="both", ls="-", alpha=0.25)
    else:
        ax_pre.text(0.5, 0.5, "No pretrain epochs logged", ha="center", va="center", transform=ax_pre.transAxes)
        ax_pre.set_axis_off()

    if np.any(joint_m):
        epochs_joint = ep[joint_m]
        data_joint = np.where(hd[joint_m] > 0.0, hd[joint_m], np.nan)
        phys_joint = np.clip(hp[joint_m], 1e-30, None)
        
        data_joint_smooth = ema(data_joint, alpha=0.04)
        phys_joint_smooth = ema(phys_joint, alpha=0.04)
        
        # Raw losses
        ax_joint.semilogy(epochs_joint, data_joint, color="#3498db", alpha=0.15, label="Raw Data MSE")
        ax_joint.semilogy(epochs_joint, phys_joint, color="#e74c3c", alpha=0.15, label="Raw Physics MSE")
        
        # Smoothed losses
        ax_joint.semilogy(epochs_joint, data_joint_smooth, color="#3498db", lw=2, label="Smoothed Data MSE")
        ax_joint.semilogy(epochs_joint, phys_joint_smooth, color="#e74c3c", lw=2, label="Smoothed Physics MSE")
        
        ax_joint.set_title("Phase 2: Joint Training", fontsize=12, fontweight="bold", pad=10)
        ax_joint.set_xlabel("Epoch")
        ax_joint.legend(loc="upper right")
        ax_joint.grid(True, which="both", ls="-", alpha=0.25)
    else:
        ax_joint.text(0.5, 0.5, "No joint epochs logged", ha="center", va="center", transform=ax_joint.transAxes)
        ax_joint.set_axis_off()

    fig.suptitle("Physics-Informed Neural Network (PINN) Loss Convergence", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def virgin_ic_mask(inputs: torch.Tensor) -> torch.Tensor:
    """Same inventory rule as print_trace_coverage (atoms): fresh Ra-226, no recycled trace ICs."""
    n226 = inputs[:, 3] * N226_SCALE
    n225 = inputs[:, 4] * N225_SCALE
    nac = inputs[:, 5] * NAC_SCALE
    n227 = inputs[:, 6] * N227_SCALE
    nac7 = inputs[:, 7] * NAC227_SCALE
    return (
        (n226 > 1.0)
        & (n225.abs() <= 1.0)
        & (nac.abs() <= 1.0)
        & (n227.abs() <= 1.0)
        & (nac7.abs() <= 1.0)
    )


def plot_ac225_pred_vs_true(
    n_ac_true_atoms: np.ndarray,
    n_ac_pred_atoms: np.ndarray,
    path: str | pathlib.Path,
    *,
    title: str | None = None,
    metrics_line: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    mask = (n_ac_true_atoms > 0) & (n_ac_pred_atoms > 0)
    true_m = n_ac_true_atoms[mask]
    pred_m = n_ac_pred_atoms[mask]
    if len(true_m) == 0:
        plt.close(fig)
        return
    lo = max(float(np.nanmin(np.minimum(true_m, pred_m))), 1e-300) * 0.5
    hi = float(np.nanmax(np.maximum(true_m, pred_m))) * 2.0
    log_err = np.abs(np.log10(pred_m + 1e-300) - np.log10(true_m + 1e-300))
    sc = ax.scatter(
        true_m, pred_m, c=log_err, cmap="cool", s=28, alpha=0.75,
        edgecolors="white", linewidths=0.3, vmin=0, vmax=max(1.0, float(np.percentile(log_err, 95))),
        zorder=3,
    )
    ax.plot([lo, hi], [lo, hi], color="#10b981", ls="--", lw=2, label="Perfect agreement", zorder=2)
    ax.fill_between(
        [lo, hi], [lo * 0.5, hi * 0.5], [lo * 2, hi * 2],
        color="#10b981", alpha=0.07, label="2x envelope", zorder=1,
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"ODE Simulated $^{225}$Ac (atoms)", fontsize=12, color="white")
    ax.set_ylabel(r"PINN Predicted $^{225}$Ac (atoms)", fontsize=12, color="white")
    ax.set_title(
        title if title is not None else r"Ac-225 Parity: PINN vs ODE",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=12,
    )
    if metrics_line:
        ax.text(
            0.02, 0.98, metrics_line, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#334155", alpha=0.9),
        )
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, which="major", ls="-", alpha=0.15, color="white")
    ax.grid(True, which="minor", ls=":", alpha=0.08, color="white")
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Log10 Absolute Error", fontsize=10, color="white")
    cb.ax.tick_params(colors="white")
    ax.legend(fontsize=10, loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="white")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="#0f172a", bbox_inches="tight")
    plt.close(fig)


def write_validation_summary(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pred_norm: torch.Tensor,
    path: pathlib.Path,
) -> None:
    """Write compact hold-in validation diagnostics after training."""
    valid_mask = ~torch.isnan(targets).any(dim=1)
    if not bool(valid_mask.any().item()):
        pd.DataFrame([{"metric": "valid_rows", "value": 0}]).to_csv(path, index=False)
        return

    species = ["Ra-226", "Ra-225", "Ac-225", "Ra-227", "Ac-227"]
    scales = torch.tensor(
        [N226_SCALE, N225_SCALE, NAC_SCALE, N227_SCALE, NAC227_SCALE],
        dtype=pred_norm.dtype,
        device=pred_norm.device,
    )
    inp_v = inputs[valid_mask].detach()
    tgt_atoms = targets[valid_mask].detach().clamp(min=0.0)
    pred_atoms = (pred_norm[valid_mask].detach().clamp(min=0.0) * scales)

    rows: list[dict[str, float | str | int]] = []
    for i, name in enumerate(species):
        true_i = tgt_atoms[:, i]
        pred_i = pred_atoms[:, i]
        signal_floor = max(1.0, float(scales[i].item()) * 1e-12)
        signal = true_i > signal_floor
        abs_err = (pred_i - true_i).abs()
        if bool(signal.any().item()):
            rel = abs_err[signal] / true_i[signal].clamp(min=signal_floor)
            rows.append({
                "metric": "species_error",
                "species": name,
                "n_signal": int(signal.sum().item()),
                "mae_atoms": float(abs_err[signal].mean().cpu()),
                "median_rel_error": float(rel.median().cpu()),
                "p95_rel_error": float(torch.quantile(rel, 0.95).cpu()),
                "max_rel_error": float(rel.max().cpu()),
            })
        else:
            rows.append({
                "metric": "species_error",
                "species": name,
                "n_signal": 0,
                "mae_atoms": float(abs_err.mean().cpu()),
                "median_rel_error": np.nan,
                "p95_rel_error": np.nan,
                "max_rel_error": np.nan,
            })

    start_atoms = inp_v[:, 3:8] * scales
    start_total = start_atoms.sum(dim=1)
    pred_total = pred_atoms.sum(dim=1)
    mass_signal = start_total > 1.0
    if bool(mass_signal.any().item()):
        drift = (pred_total[mass_signal] - start_total[mass_signal]).abs() / start_total[mass_signal]
        rows.append({
            "metric": "inventory_drift",
            "species": "tracked_total",
            "n_signal": int(mass_signal.sum().item()),
            "mae_atoms": float((pred_total[mass_signal] - start_total[mass_signal]).abs().mean().cpu()),
            "median_rel_error": float(drift.median().cpu()),
            "p95_rel_error": float(torch.quantile(drift, 0.95).cpu()),
            "max_rel_error": float(drift.max().cpu()),
        })

    pd.DataFrame(rows).to_csv(path, index=False)


def _log_train_script_identity() -> None:
    """Log which train.py file is executing (Kaggle often runs a stale copy from /kaggle/input)."""
    import hashlib
    from datetime import datetime, timezone

    p = pathlib.Path(__file__).resolve()
    try:
        st = p.stat()
        tip = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        mutc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(
            f"train.py identity: path={p} mtime_utc={mutc} sha256[:12]={tip}",
            flush=True,
        )
    except OSError as exc:
        print(f"train.py identity: path={p} (stat/hash failed: {exc})", flush=True)
    try:
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ and graph_provenance.under_kaggle_input_tree(p.parent):
            print(
                "WARNING: train.py lives under read-only /kaggle/input — prefer a bundle under "
                "/kaggle/working from your pushed kernel, or publish an updated dataset so you "
                "run the code revision you expect.",
                flush=True,
            )
    except Exception:
        pass


def main() -> None:
    compute_dtype = torch.float64 if _env_flag("PINN_FLOAT64") else torch.float32
    _log_train_script_identity()
    print(
        f"train.py revision {PINN_TRAIN_SCRIPT_REV} | grad_clip={GRAD_CLIP_NORM} | "
        f"PINN_GRAD_BREAKER={'on' if GRAD_BREAKER_ENABLED else 'off (default)'}"
        + (f" threshold={GRAD_BREAKER_THRESHOLD:.0f} grace={GRAD_BREAKER_GRACE_JOINT_EPOCHS}" if GRAD_BREAKER_ENABLED else "")
    )
    print(
        "Stage: load CSV + build augmented table (CUDA/cuDNN init deferred until tensor move).",
        flush=True,
    )

    _proj_root = pathlib.Path(__file__).resolve().parent
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ and graph_provenance.under_kaggle_input_tree(_proj_root):
        # graph_provenance writes project_root/results — input datasets are read-only
        _proj_root = pathlib.Path(os.environ.get("PINN_OUTPUT_ROOT", "/kaggle/working")).resolve()
        print(f"Kaggle: code under /kaggle/input — writable project_root for provenance: {_proj_root}")
    _run_id = graph_provenance.new_run_id()
    print(f"RUN_ID={_run_id}")
    print(f"DATA_PATH resolved: {DATA_PATH.resolve()} exists={DATA_PATH.is_file()}")
    graph_provenance.training_run_start(
        _proj_root,
        run_id=_run_id,
        train_script_rev=PINN_TRAIN_SCRIPT_REV,
        data_path=DATA_PATH,
        loss_plot_path=LOSS_PLOT_PATH,
        parity_plot_path=AC225_SCATTER_PATH,
    )

    _csv_header = pd.read_csv(DATA_PATH, nrows=0, engine="python").columns.tolist()
    _usecols = ["phi", "energy", "time", "N_Ra226", "N_Ra225", "N_Ac225"]
    for _c in ("N_Ra227", "N_Ac227"):
        if _c in _csv_header:
            _usecols.append(_c)
    _dtype_map = {c: "float64" for c in _usecols}

    df_raw = pd.read_csv(
        DATA_PATH,
        usecols=_usecols,
        dtype=_dtype_map,
        engine="python",
    )
    for _c in ("N_Ra227", "N_Ac227"):
        if _c not in df_raw.columns:
            df_raw[_c] = 0.0
    df_raw["energy"] = df_raw["energy"].replace([np.inf, -np.inf], np.nan)
    df_raw["energy"] = df_raw["energy"].fillna(THERMAL_REFERENCE_EV)
    df_raw["energy"] = df_raw["energy"].clip(lower=1e-6, upper=1e8)
    if AUGMENT_BASE_ROW_LIMIT > 0:
        df_raw = df_raw.iloc[:AUGMENT_BASE_ROW_LIMIT].copy()
        print(f"AUGMENT_BASE_ROW_LIMIT={AUGMENT_BASE_ROW_LIMIT}: using first {len(df_raw)} CSV rows.")

    rng = np.random.default_rng(42)
    aug_n = AUGMENT_PER_ROW
    _ap = os.environ.get("PINN_AUGMENT_PER_ROW", "").strip()
    if _ap.isdigit():
        aug_n = max(0, int(_ap))
    cache_config = _training_cache_config(df_raw, aug_n)
    train_dat = _load_augmented_training_cache(cache_config)

    if train_dat is None:
        if USE_TIME_SHIFT_AUGMENT:
            n_cpu = os.cpu_count() or 1
            _ode_env = os.environ.get("PINN_ODE_PREP_WORKERS", "").strip()
            if _ode_env.isdigit():
                prep_workers = max(1, min(int(_ode_env), n_cpu))
            elif ODE_PREP_MAX_WORKERS > 0:
                prep_workers = max(1, min(ODE_PREP_MAX_WORKERS, n_cpu))
            elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
                prep_workers = 1
            else:
                prep_workers = n_cpu
            prep_workers = max(1, min(prep_workers, n_cpu))
            if sys.platform == "win32":
                ode_par = os.environ.get("PINN_ODE_PARALLEL", "").strip().lower()
                if ode_par in ("0", "false", "no"):
                    prep_workers = 1
                elif ode_par not in ("1", "true", "yes"):
                    prep_workers = max(1, min(n_cpu - 1, 8)) if n_cpu >= 4 else 1
            print(f"ODE trajectory cache: {prep_workers} worker(s), {n_cpu} logical CPU(s)...")
            try:
                ref_cache = build_reference_traj_cache_parallel(df_raw, max_workers=prep_workers)
            except (BrokenProcessPool, OSError) as exc:
                print(f"Parallel ODE prep failed ({exc!r}); retrying single-process.")
                ref_cache = build_reference_traj_cache_parallel(df_raw, max_workers=1)
            print(f"  Distinct (phi, E, T) paths: {len(ref_cache)}")
            train_dat = augment_rows_time_shift(
                df_raw, rng, augment_per_row=aug_n,
                include_unshifted_base=True, initial_cache=ref_cache,
            )
            print(f"Time-shift: {len(df_raw)} base -> {len(train_dat)} samples ({aug_n} shifts/row + base)")
        else:
            train_dat = pd.DataFrame({
                "phi": df_raw["phi"], "energy": df_raw["energy"], "time": df_raw["time"],
                "init_N226": TRAIN_INIT_RA226, "init_N225": TRAIN_INIT_RA225,
                "init_NAc": TRAIN_INIT_AC225,
                "N_Ra226": df_raw["N_Ra226"], "N_Ra225": df_raw["N_Ra225"],
                "N_Ac225": df_raw["N_Ac225"],
            })

        # Diverse ICs for pure-decay / mixed / Ac-dominant scenarios
        if not SINGLE_SUPPLY_MODE:
            print("Generating inverted-IC training data...")
            inverted_rows = augment_inverted_ic_scenarios(rng, n_extra=INVERTED_IC_N_EXTRA)
            train_dat = pd.concat([train_dat, inverted_rows], ignore_index=True)
            print(f"  Added {len(inverted_rows)} inverted-IC rows. Total: {len(train_dat)}")
            print("Generating diverse-IC training data...")
            diverse_rows = augment_diverse_ic_scenarios(rng, n_extra=DIVERSE_IC_N_EXTRA)
            train_dat = pd.concat([train_dat, diverse_rows], ignore_index=True)
            print(f"  Added {len(diverse_rows)} diverse-IC rows. Total: {len(train_dat)}")
            print("Generating spectrum-sweep training data...")
            spectrum_rows = augment_spectrum_sweep_scenarios(rng, n_extra=SPECTRUM_SWEEP_N_EXTRA)
            train_dat = pd.concat([train_dat, spectrum_rows], ignore_index=True)
            print(f"  Added {len(spectrum_rows)} spectrum-sweep rows. Total: {len(train_dat)}")
            print("Generating recycled trace-inventory training data...")
            trace_rows = augment_recycled_trace_inventory_scenarios(
                rng,
                n_recycled=TRACE_RECYCLED_N_EXTRA,
                n_tiny=TRACE_TINY_N_EXTRA,
            )
            train_dat = pd.concat([train_dat, trace_rows], ignore_index=True)
            print(f"  Added {len(trace_rows)} recycled/tiny trace rows. Total: {len(train_dat)}")
            print("Generating high-energy virgin training data...")
            virgin_rows = augment_high_energy_virgin_scenarios(rng, n_extra=HIGH_ENERGY_VIRGIN_N_EXTRA)
            train_dat = pd.concat([train_dat, virgin_rows], ignore_index=True)
            print(f"  Added {len(virgin_rows)} high-energy virgin rows. Total: {len(train_dat)}")
            print("Generating threshold-edge virgin training data...")
            threshold_edge_rows = augment_threshold_edge_virgin_scenarios(
                rng,
                n_extra=THRESHOLD_EDGE_VIRGIN_N_EXTRA,
            )
            train_dat = pd.concat([train_dat, threshold_edge_rows], ignore_index=True)
            print(f"  Added {len(threshold_edge_rows)} threshold-edge virgin rows. Total: {len(train_dat)}")
            print("Generating long Ac-227 chain training data...")
            ac227_chain_rows = augment_ac227_long_chain_scenarios(rng, n_extra=AC227_LONG_CHAIN_N_EXTRA)
            train_dat = pd.concat([train_dat, ac227_chain_rows], ignore_index=True)
            print(f"  Added {len(ac227_chain_rows)} long Ac-227 chain rows. Total: {len(train_dat)}")
        else:
            print("SINGLE_SUPPLY_MODE: skipping inverted/diverse synthetic rows.")

        # Empty-tank training rows (output must be all zeros)
        print("Generating empty-tank training rows...")
        empty_rows = augment_empty_tank_rows(rng, n_extra=300)
        train_dat = pd.concat([train_dat, empty_rows], ignore_index=True)
        print(f"  Added {len(empty_rows)} empty-tank rows. Total: {len(train_dat)}")
        _write_augmented_training_cache(train_dat, cache_config)

    train_dat = apply_empirical_flux_jitter(train_dat, rng)

    e_vals = train_dat["energy"].astype(float)
    n_thermal = int((e_vals < 0.1).sum())
    n_epi = int(((e_vals >= 0.1) & (e_vals < THRESHOLD_ENERGY_RANGE_EV[0])).sum())
    n_threshold = int(((e_vals >= THRESHOLD_ENERGY_RANGE_EV[0]) & (e_vals < 7.5e6)).sum())
    n_fast = int((e_vals >= 7.5e6).sum())
    print(
        "Energy coverage: "
        f"thermal={n_thermal}, epithermal={n_epi}, threshold={n_threshold}, fast={n_fast}"
    )
    print_trace_coverage(train_dat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training device: {device} (cuda available: {torch.cuda.is_available()}) | compute_dtype={compute_dtype}",
        flush=True,
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if compute_dtype == torch.float32:
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    else:
        n_thr = max(1, int(os.environ.get("PINN_CPU_THREADS", os.cpu_count() or 1)))
        interop = os.environ.get("PINN_CPU_INTEROP", "").strip()
        n_interop = max(1, int(interop)) if interop.isdigit() else max(1, min(4, n_thr // 2))
        torch.set_num_threads(n_thr)
        try:
            torch.set_num_interop_threads(n_interop)
        except RuntimeError:
            pass
        print(f"CPU: torch threads={n_thr}, interop={n_interop}")

    inputs, targets = prepare_training_tensors(train_dat, device, dtype=compute_dtype)
    t_max_h = max(float(train_dat["time"].max()), 1.0)

    train_epochs = EPOCHS
    train_pre = PHYS_PRETRAIN_EPOCHS
    medium_train = _env_on("PINN_MEDIUM_TRAIN")
    if medium_train:
        # 4000 total ≈ 33% of original 12000 — fits Kaggle GPU time limits with float64.
        train_pre = int(os.environ.get("PINN_PRETRAIN_EPOCHS", "600"))
        train_epochs = int(os.environ.get("PINN_EPOCHS", "3400"))
        print(
            f"PINN_MEDIUM_TRAIN=1: ISEF short budget {train_pre} pretrain + "
            f"{train_epochs} joint = {train_pre + train_epochs} total epochs"
        )
    if os.environ.get("PINN_QUICK_TRAIN", "").lower() in ("1", "true", "yes"):
        train_epochs = min(train_epochs, 400)
        train_pre = min(train_pre, 120)
        print("PINN_QUICK_TRAIN=1: short run for debugging.")
    if os.environ.get("PINN_SMOKE", "").lower() in ("1", "true", "yes"):
        train_epochs = min(train_epochs, 3)
        train_pre = min(train_pre, 3)
        print("PINN_SMOKE=1: tiny run for CI only.")
    # Auto-reduce epochs on Kaggle CPU so training finishes within the 9-hour limit.
    # CPU is ~50x slower per epoch; 500 pretrain + 2000 joint ≈ 5-6 hours on 4-core CPU.
    if (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        and device.type == "cpu"
        and os.environ.get("PINN_EPOCHS", "") == ""
        and not medium_train
    ):
        train_epochs = min(train_epochs, 2000)
        train_pre = min(train_pre, 500)
        print(
            "KAGGLE CPU FALLBACK: reduced to 500 pretrain + 2000 joint to fit 9h limit. "
            "Set PINN_EPOCHS to override."
        )

    total_epochs = train_pre + train_epochs
    budget_scale = max(0.25, min(1.0, total_epochs / 12000.0))
    eff_warmup_epochs = max(150, int(WARMUP_EPOCHS * budget_scale))
    eff_virgin_warmup_epochs = max(300, int(VIRGIN_AC225_WARMUP_EPOCHS * budget_scale))
    eff_curric_ramp = max(
        400,
        int(float(os.environ.get("PINN_CURRIC_RAMP", str(CAUSAL_CURRICULUM_RAMP_EPOCHS))) * budget_scale),
    )
    eff_phys_anneal_epochs = max(100, int(500.0 * budget_scale))
    eff_trace_fraction = TRACE_CHUNK_FRACTION
    eff_data_weight = DATA_WEIGHT
    eff_log_data_weight = LOG_DATA_WEIGHT
    eff_log_species_weights = LOG_SPECIES_WEIGHTS
    eff_virgin_ac225_weight = VIRGIN_AC225_WEIGHT
    eff_collocation = COLLOCATION_POINTS
    if medium_train:
        eff_trace_fraction = max(TRACE_CHUNK_FRACTION, 0.50)
        eff_data_weight = DATA_WEIGHT * 1.5
        eff_log_data_weight = LOG_DATA_WEIGHT * 1.5
        eff_log_species_weights = (0.0, 10.0, 25.0, 10.0, 22.0)
        eff_virgin_ac225_weight = VIRGIN_AC225_WEIGHT * 2.0
        if os.environ.get("PINN_COLLOCATION_POINTS", "") == "":
            eff_collocation = max(COLLOCATION_POINTS, 500)
        print(
            f"  Medium-train scaling: warmup={eff_warmup_epochs}, virgin_warmup="
            f"{eff_virgin_warmup_epochs}, curric_ramp={eff_curric_ramp}, "
            f"trace_chunk={eff_trace_fraction:.0%}, colloc={eff_collocation}"
        )
    print(f"\nMax-Fix v2: {train_pre} pretrain + {train_epochs} joint = {total_epochs} total epochs")
    print(f"  Ra-225 physics weight: {RA225_PHYSICS_WEIGHT}x")
    print(f"  Log daughter loss:     {LOG_DATA_WEIGHT}x")
    print(f"  Impurity log loss:     {IMPURITY_LOG_WEIGHT}x")
    print(f"  Target inventory loss: {TARGET_INVENTORY_WEIGHT}x")
    print(f"  Trace relative loss:   {TRACE_RELATIVE_WEIGHT}x")
    print(f"  Chain consistency:     {CHAIN_CONSISTENCY_WEIGHT}x")
    print(f"  Virgin Ac-225 loss:    {VIRGIN_AC225_WEIGHT}x")
    print(f"  Ac-227 chain loss:     {AC227_CHAIN_WEIGHT}x")
    print(f"  Empty output loss:     {EMPTY_OUTPUT_WEIGHT}x")
    # Optimizer selection BEFORE logging
    use_muon = _env_flag("PINN_MUON", default=False)
    use_grad_balance = _env_flag("PINN_GRAD_BALANCE", default=False)

    print(f"  Threshold physics boost: {THRESHOLD_PHYSICS_BOOST}x (6.0-7.0 MeV)")
    print(f"  Energy Fourier feats:    {N_ENERGY_FOURIER_FREQS} freq pairs (Tancik et al. 2020); PINN_ENERGY_FOURIER=0 off")
    print(f"  XPINN interface weight: {XPINN_INTERFACE_WEIGHT} (Jagtap & Karniadakis, CICP 2020)")
    print(f"  Causal physics scale:   {CAUSAL_PHYSICS_TIME_SCALE} (Wang et al. arXiv:2203.07404)")
    print(f"  Time curriculum:       {CAUSAL_CURRICULUM_BINS} bins, ramp {eff_curric_ramp} j.ep. (Krishnapriyan et al. 2021)")
    print(f"  SA physics alpha:      {SA_PHYSICS_ALPHA} (McClenny & Braga-Neto; PINN_SA_PHYSICS=0 off)")
    print(f"  Grad balance:            {'on (Wang et al. SISC 2021; full-batch only)' if use_grad_balance else 'off'}")
    print(f"  Non-negativity weight: {NON_NEG_WEIGHT}")
    print(f"  Secular eq weight:     {SECULAR_EQ_WEIGHT}")
    print(f"  Empty-tank colloc:     {EMPTY_FEED_FRACTION*100:.0f}%")
    print(f"  Input norm: t/={TIME_SCALE_H}, phi/={PHI_SCALE:.0e}")
    print(f"  Architecture:          {PINN_ARCHITECTURE}")
    print(f"  Optimizer:             {'Muon (DeepSeek-V4)' if use_muon else 'Adam (default)'}")
    print()

    model = IsotopePINN(n_energy_fourier_freqs=N_ENERGY_FOURIER_FREQS).to(device=device, dtype=compute_dtype)
    _is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    compile_default = device.type == "cuda" and not _is_kaggle
    should_compile = _env_flag("PINN_COMPILE", default=compile_default)
    if hasattr(torch, 'compile') and should_compile:
        compile_backend = os.environ.get("PINN_COMPILE_BACKEND", "aot_eager")
        print(
            "torch.compile: compiling model (first steps can take many minutes; "
            f"backend={compile_backend})...",
            flush=True,
        )
        try:
            model = torch.compile(model, mode="reduce-overhead", backend=compile_backend)
            print(f"torch.compile enabled ({compile_backend} backend)", flush=True)
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    elif hasattr(torch, 'compile'):
        _hint = "set PINN_COMPILE=1 to enable"
        if _is_kaggle:
            _hint = "Kaggle default off; " + _hint + " (avoids long silent compile warmup)"
        print(f"torch.compile skipped ({_hint})")

    # Optimizer: Adam (default) or Muon (DeepSeek-V4 style for better stability)
    lr_init = 1e-3
    # Accelerate learning for daughter rate scales (which cover 4+ orders of magnitude)
    param_groups = []
    base_params = []
    scale_params = []
    for name, p in model.named_parameters():
        if "daughter_rate_log_scales" in name:
            scale_params.append(p)
        else:
            base_params.append(p)
    
    param_groups.append({"params": base_params, "lr": lr_init})
    if scale_params:
        param_groups.append({"params": scale_params, "lr": 1e-2})
        print(f"  Daughter rate scales: Accelerated (LR=1e-2)")

    if use_muon:
        print(f"Using Muon optimizer (lr={lr_init:.2e}) - DeepSeek-V4 style")
        optimizer = MuonOptimizer(param_groups, lr=lr_init, momentum=0.9, weight_decay=0.0)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=lr_init)

    # Uncertainty weighter (Kendall et al. CVPR 2018): 2 tasks (supervised, unsupervised).
    # Default OFF: when ON, the optimizer can silently drive log_vars upward to ignore the
    # unsupervised (physics) loss, overriding all per-loss weights set elsewhere in this file.
    # When ON, log_vars are clamped to [-3, 3] inside UncertaintyWeighter as a safety floor.
    use_uw = _env_flag("PINN_UW", default=False)
    uncertainty_weighter = UncertaintyWeighter(n_tasks=2).to(device=device, dtype=compute_dtype)
    if use_uw:
        # Co-optimize UW params with model params
        for p in uncertainty_weighter.parameters():
            optimizer.add_param_group({"params": p, "lr": 1e-3})
        print(f"  Uncertainty weighter: ON (Kendall et al. 2018, 2 tasks)")
    else:
        print(f"  Uncertainty weighter: OFF")

    # Cosine annealing with warm restarts (Loshchilov & Hutter ICLR 2017)
    cos_t0 = max(100, int(os.environ.get("PINN_COS_T0", str(max(100, int(2000 * budget_scale))))))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cos_t0, T_mult=2, eta_min=1e-5)
    print(f"  LR scheduler: CosineAnnealingWarmRestarts(T_0={cos_t0}, T_mult=2)")

    use_amp = (
        device.type == "cuda"
        and _env_flag("PINN_AMP", default=False)
        and compute_dtype == torch.float32
    )
    grad_scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if device.type == "cuda" and _env_flag("PINN_AMP", default=False) and compute_dtype == torch.float64:
        print("  AMP skipped: PINN_FLOAT64=1 (fp16 autocast is incompatible with float64).")
    elif use_amp:
        print("  AMP (fp16): ON (PINN_AMP=1); disable if physics loss becomes unstable")

    epoch_list: list[int] = []
    hist_data: list[float] = []
    hist_phys: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_joint_epoch = 0
    joint_epoch = 0
    adaptive_sec_eq_weight = SECULAR_EQ_WEIGHT
    resume_phase: str | None = None
    resume_epoch = 0
    loss_history_keep_existing = False

    if _env_flag("PINN_RESUME") and RESUME_CKPT_PATH.exists():
        ckpt = torch.load(RESUME_CKPT_PATH, map_location=device, weights_only=False)
        if _resume_checkpoint_is_usable(ckpt, train_epochs, "PINN_RESUME=1") and _try_load_model_state_for_training(
            model, ckpt["model_state"], "PINN_RESUME=1"
        ):
            optimizer.load_state_dict(ckpt["optimizer_state"])
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except (KeyError, ValueError, RuntimeError):
                print("  Scheduler state incompatible (old ReduceLROnPlateau?); fresh cosine schedule.")
            if use_uw and "uw_state" in ckpt:
                try:
                    uncertainty_weighter.load_state_dict(ckpt["uw_state"])
                except Exception:
                    pass
            inputs = ckpt["inputs"].to(device=device, dtype=compute_dtype)
            targets = ckpt["targets"].to(device=device, dtype=compute_dtype)
            best_loss = float(ckpt.get("best_loss", best_loss))
            best_epoch = int(ckpt.get("best_epoch", 0))
            best_joint_epoch = int(ckpt.get("best_joint_epoch", 0))
            joint_epoch = int(ckpt.get("joint_epoch", 0))
            adaptive_sec_eq_weight = float(ckpt.get("adaptive_sec_eq_weight", adaptive_sec_eq_weight))
            epoch_list = list(ckpt.get("epoch_list", []))
            hist_data = list(ckpt.get("hist_data", []))
            hist_phys = list(ckpt.get("hist_phys", []))
            resume_phase = str(ckpt.get("phase", "joint"))
            resume_epoch = int(ckpt.get("epoch", 0))
            print(
                f"PINN_RESUME=1: loaded {RESUME_CKPT_PATH.name} "
                f"(phase={resume_phase}, epoch={resume_epoch}, joint_epoch={joint_epoch})"
            )
            # Final plotting reloads BEST_CKPT_PATH from disk. Without this sync, a dataset-shipped
            # pinn_best_weights.pth (unrelated to the resumed model) would overwrite in-memory weights
            # whenever pretrain/joint run zero iterations — identical parity PNGs across every run.
            # Back up the existing best weights first so a resume can never destroy a good checkpoint.
            _backup_existing_checkpoint(BEST_CKPT_PATH)
            torch.save(_model_state_dict_for_checkpoint(model), BEST_CKPT_PATH)
            print(f"PINN_RESUME=1: wrote {BEST_CKPT_PATH.name} from resumed model (keep disk aligned).")
            loss_history_keep_existing = True
        else:
            print(
                "PINN_RESUME=1: resume checkpoint rejected (corrupt/incompatible/finished). "
                "Starting fresh; existing best weights left untouched."
            )
            loss_history_keep_existing = False
    elif _env_flag("PINN_WARM_START") and BEST_CKPT_PATH.exists():
        loaded_warm = _try_load_model_state_for_training(
            model,
            torch.load(BEST_CKPT_PATH, map_location=device, weights_only=True),
            "PINN_WARM_START=1",
        )
        if loaded_warm:
            if not _env_flag("PINN_WARM_START_PRETRAIN"):
                train_pre = 0
                total_epochs = train_epochs
                print("PINN_WARM_START=1: loaded best weights and skipped pretrain.")
            else:
                print("PINN_WARM_START=1: loaded best weights; pretrain still enabled.")
        else:
            print("PINN_WARM_START=1: starting fresh because the best weights use an older architecture.")

    rng_colloc = np.random.default_rng(2026)

    # Pre-allocate large collocation pool once — subsample each epoch (avoid per-epoch alloc)
    _POOL_SIZE = eff_collocation * 20
    colloc_pool = build_pretrain_collocation(_POOL_SIZE, rng_colloc, device, t_max_h=t_max_h, dtype=compute_dtype)

    if not loss_history_keep_existing or not LOSS_HISTORY_CSV_PATH.is_file():
        truncate_loss_history_csv_for_run(LOSS_HISTORY_CSV_PATH)
        print(f"Fresh loss history CSV: {LOSS_HISTORY_CSV_PATH}")
    else:
        print(f"PINN_RESUME: appending to existing loss history CSV: {LOSS_HISTORY_CSV_PATH}")

    # ---- Physics-only pretrain ----
    print(
        f"Physics pretrain: {train_pre} epochs, {eff_collocation} collocation pts/epoch, "
        f"lr={optimizer.param_groups[0]['lr']:.0e}"
    )
    phys_skip_streak = 0
    pretrain_start = 1
    if resume_phase == "pretrain":
        pretrain_start = min(train_pre + 1, resume_epoch + 1)
    elif resume_phase == "joint":
        pretrain_start = train_pre + 1
    for epoch in range(pretrain_start, train_pre + 1):
        # Subsample from pre-allocated pool each epoch
        perm = torch.randperm(_POOL_SIZE, device=device)[:eff_collocation]
        colloc = colloc_pool[perm].detach().clone().requires_grad_(True)

        optimizer.zero_grad(set_to_none=True)
        cm_amp = device.type == "cuda" and use_amp
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cm_amp)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            rg = getattr(model, "regime_gated", True)
            if rg:
                pred, pred_low, pred_high, v_pred = model.forward_raw(colloc, return_experts=True, return_derivatives=True)
            else:
                pred, v_pred = model.forward_raw(colloc, return_derivatives=True)
                pred_low = pred_high = None
            if not torch.isfinite(pred).all():
                phys_skip_streak += 1
                if phys_skip_streak <= 2 or epoch % LOG_EVERY == 0:
                    print(f"  !! [phys] epoch {epoch}: non-finite pred (streak {phys_skip_streak})")
                continue
            cc_prog_pre = min(1.0, (epoch - 1) / max(1.0, train_pre * 0.85))
            loss, info = compute_physics_loss(
                model, colloc, pred, targets=None,
                v_pred=v_pred,
                physics_weight=PHYS_PRETRAIN_PHYS_WEIGHT, data_weight=0.0,
                mass_weight=PHYS_PRETRAIN_MASS_WEIGHT,
                fuel_anchor_weight=PHYS_PRETRAIN_FUEL_ANCHOR_WEIGHT,
                non_neg_weight=NON_NEG_WEIGHT, secular_eq_weight=adaptive_sec_eq_weight,
                ra225_physics_weight=RA225_PHYSICS_WEIGHT,
                n226_scale=N226_SCALE, n225_scale=N225_SCALE, nac_scale=NAC_SCALE,
                n227_scale=N227_SCALE, nac227_scale=NAC227_SCALE,
                phi_scale=PHI_SCALE, d_t_input_d_t_hours=D_T_INPUT_D_T_HOURS,
                use_one_over_v_energy=True,
                threshold_physics_boost=THRESHOLD_PHYSICS_BOOST,
                pred_expert_low=pred_low,
                pred_expert_high=pred_high,
                xpinn_interface_weight=XPINN_INTERFACE_WEIGHT,
                causal_physics_time_scale=CAUSAL_PHYSICS_TIME_SCALE,
                causal_curriculum_bins=CAUSAL_CURRICULUM_BINS,
                causal_curriculum_progress=cc_prog_pre,
                sa_physics_alpha=SA_PHYSICS_ALPHA,
            )
        if not torch.isfinite(loss):
            phys_skip_streak += 1
            if phys_skip_streak <= 2 or epoch % LOG_EVERY == 0:
                print(f"  !! [phys] epoch {epoch}: non-finite loss (streak {phys_skip_streak})")
            continue
        phys_skip_streak = 0

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_norm_pre = compute_gradient_norm(model)
        clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.update()
            continue
        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        epoch_list.append(epoch)
        hist_data.append(float(info["data_mse"].detach().cpu()))
        hist_phys.append(float(info["physics_mse"].detach().cpu()))
        append_loss_history_csv(
            LOSS_HISTORY_CSV_PATH,
            epoch=epoch,
            phase="pretrain",
            data_mse=float(info["data_mse"].detach().cpu()),
            physics_mse=float(info["physics_mse"].detach().cpu()),
            supervised_total=float(info["supervised_total"].detach().cpu()),
            unsupervised_total=float(info["unsupervised_total"].detach().cpu()),
            grad_norm=float(grad_norm_pre)
        )

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == train_pre:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:6d} [phys] | lr {lr:.2e} | total {info['total_loss'].item():.4e} "
                f"| phys {info['physics_mse'].item():.4e} | mass {info['mass_cons_loss'].item():.4e} "
                f"| fuel {info['fuel_anchor_loss'].item():.4e} | neg {info['non_neg_loss'].item():.4e} "
                f"| sec_eq {info['secular_eq_loss'].item():.4e}"
            )
            _save_training_resume_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                phase="pretrain", epoch=epoch, joint_epoch=joint_epoch,
                train_pre=train_pre, train_epochs=train_epochs,
                best_loss=best_loss, best_epoch=best_epoch, best_joint_epoch=best_joint_epoch,
                adaptive_sec_eq_weight=adaptive_sec_eq_weight,
                inputs=inputs, targets=targets, epoch_list=epoch_list,
                hist_data=hist_data, hist_phys=hist_phys,
                uw_state=uncertainty_weighter.state_dict() if use_uw else None,
            )

    # ---- Joint training ----
    nan_streak = 0
    if best_loss == float("inf"):
        # Fresh run: about to seed BEST_CKPT_PATH with the pretrained (not-yet-joint) model.
        # Back up any prior good weights first so a new run cannot silently destroy them.
        _backup_existing_checkpoint(BEST_CKPT_PATH)
        torch.save(_model_state_dict_for_checkpoint(model), BEST_CKPT_PATH)
        if train_pre > 0:
            best_epoch = train_pre
            best_joint_epoch = 0

    n_train = int(inputs.size(0))
    joint_chunk_sz = JOINT_CHUNK_SIZE
    if (
        device.type == "cpu"
        and os.environ.get("PINN_FAST_CPU", "").lower() in ("1", "true", "yes")
        and not os.environ.get("PINN_JOINT_CHUNK", "").strip()
        and joint_chunk_sz > 0
    ):
        joint_chunk_sz = 0
        print("PINN_FAST_CPU: full-batch joint step")
    chunk_sz = n_train if joint_chunk_sz <= 0 else min(joint_chunk_sz, n_train)
    print(f"Joint training: {train_epochs} epochs, chunks {chunk_sz}/{n_train}")
    if GRAD_BREAKER_ENABLED:
        print(
            f"  Grad circuit breaker: ON — skip pre-clip norm > {GRAD_BREAKER_THRESHOLD:.0f} "
            f"after {GRAD_BREAKER_GRACE_JOINT_EPOCHS} joint epochs (PINN_GRAD_BREAKER=0 recommended)"
        )
    else:
        print("  Grad circuit breaker: OFF — clip_grad_norm only (default; avoids Kaggle stall)")
    if use_grad_balance and chunk_sz < n_train:
        print("PINN_GRAD_BALANCE: disabled for this run (mini-batch chunks). Set PINN_JOINT_CHUNK=0 to enable.")
        use_grad_balance = False

    joint_start = joint_epoch + 1 if resume_phase == "joint" else 1
    if joint_start > train_epochs:
        print(
            "\n!! PINN_RESUME: joint_epoch >= train_epochs — running ZERO joint epochs. "
            "Plots still run; weights are those from the resume file (and synced best checkpoint). "
            "For a full retrain, remove weights/pinn_training_resume.pt from the dataset/working "
            "or set PINN_RESUME=0.\n"
        )
    for j in range(joint_start, train_epochs + 1):
        epoch = train_pre + j
        joint_epoch += 1
        cc_prog_joint = min(1.0, (joint_epoch - 1) / max(1.0, eff_curric_ramp))
        # Re-calc boost ramp every epoch: start at 1.0x, ramp to full value over scaled window.
        boost_ramp_progress = min(1.0, joint_epoch / max(1.0, eff_warmup_epochs))
        cur_boost = 1.0 + (THRESHOLD_PHYSICS_BOOST - 1.0) * boost_ramp_progress
        model.train()
        
        ramp = min(1.0, joint_epoch / float(eff_warmup_epochs))
        cur_fuel_w = ramp * FUEL_ANCHOR_WEIGHT

        # Geometric weight ramping: drop physics_weight from pretrain level -> 1.0.
        phys_anneal = min(1.0, joint_epoch / float(eff_phys_anneal_epochs))
        cur_phys_w = PHYS_PRETRAIN_PHYS_WEIGHT * ((PHYSICS_WEIGHT / max(1e-12, PHYS_PRETRAIN_PHYS_WEIGHT)) ** phys_anneal)
        
        cur_data_w = eff_data_weight
        if joint_epoch < 50:
            # Gentle data warm-up to prevent initial IC residuals from drowning physics
            cur_data_w = eff_data_weight * (joint_epoch / 50.0)
            
        cur_log_w = eff_log_data_weight
        cur_impurity_w = IMPURITY_LOG_WEIGHT
        cur_trace_rel_w = TRACE_RELATIVE_WEIGHT
        cur_chain_w = CHAIN_CONSISTENCY_WEIGHT
        # virgin_warm and fuel-anchor ramp are slower curricula and do not create cliffs; kept as-is.
        virgin_warm = max(0.0, 1.0 - (joint_epoch - 1) / max(1.0, float(eff_virgin_warmup_epochs)))
        cur_virgin_ac225_w = eff_virgin_ac225_weight * (0.5 + virgin_warm)
        cur_ac227_chain_w = AC227_CHAIN_WEIGHT

        optimizer.zero_grad(set_to_none=True)

        z = torch.zeros((), device=device, dtype=compute_dtype)
        total_loss_acc = torch.zeros((), device=device, dtype=compute_dtype)
        wsum_phys_t = z.clone()
        wsum_data_t = z.clone()
        wsum_mass_t = z.clone()
        wsum_fuel_t = z.clone()
        wsum_zero_t = z.clone()
        wsum_log_t = z.clone()
        wsum_impurity_t = z.clone()
        wsum_target_inventory_t = z.clone()
        wsum_trace_rel_t = z.clone()
        wsum_chain_t = z.clone()
        wsum_virgin_ac225_t = z.clone()
        wsum_ac227_chain_t = z.clone()
        wsum_f5_signal_t = z.clone()
        wsum_empty_output_t = z.clone()
        wsum_f1_t = z.clone()
        wsum_f2_t = z.clone()
        wsum_f3_t = z.clone()
        wsum_f4_t = z.clone()
        wsum_f5_t = z.clone()
        wsum_neg_t = z.clone()
        wsum_sec_t = z.clone()
        wsum_rel_err_t = z.clone()
        wsum_sup_t = z.clone()
        wsum_unsup_t = z.clone()
        joint_chunks_failed = False
        cm_amp = device.type == "cuda" and use_amp
        amp_ctx_joint = (
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cm_amp)
            if device.type == "cuda"
            else nullcontext()
        )

        epoch_order = trace_balanced_epoch_order(inputs, targets, chunk_sz, trace_fraction=eff_trace_fraction)
        for start in range(0, n_train, chunk_sz):
            end = min(start + chunk_sz, n_train)
            batch_idx = epoch_order[start:end]
            n_b = int(batch_idx.numel())
            w = float(n_b) / float(n_train)
            inp = inputs[batch_idx].detach().clone()
            inp.requires_grad_(True)
            tgt = targets[batch_idx]

            with amp_ctx_joint:
                rg = getattr(model, "regime_gated", True)
                if rg:
                    pred_raw, pred_low, pred_high, v_pred = model.forward_raw(inp, return_experts=True, return_derivatives=True)
                else:
                    pred_raw, v_pred = model.forward_raw(inp, return_derivatives=True)
                    pred_low = pred_high = None
                # NOTE: pred_capped = model(inp) was previously used for the data loss, but
                # (a) it ran forward_raw a second time per joint step (~2x compute) and
                # (b) it created a separate autograd subgraph for data vs physics (the inference
                # clamp + empty-zeroing in model.forward zeros gradients for empty-IC points,
                # so data loss and physics loss optimized different versions of the prediction).
                # forward_raw already enforces the right architectural constraints (exp decay
                # for Ra-226, integration for daughters); use it as the single source of truth.

                loss_b, info_b = compute_physics_loss(
                    model, inp, pred_raw, targets=tgt,
                    v_pred=v_pred,
                    physics_weight=cur_phys_w, data_weight=cur_data_w,
                    mass_weight=MASS_WEIGHT, fuel_anchor_weight=cur_fuel_w,
                    non_neg_weight=NON_NEG_WEIGHT, secular_eq_weight=adaptive_sec_eq_weight,
                    ra225_physics_weight=RA225_PHYSICS_WEIGHT,
                    pred_for_data=None, data_species_weights=DATA_SPECIES_WEIGHTS,
                    log_data_weight=cur_log_w,
                    log_species_weights=eff_log_species_weights,
                    impurity_log_weight=cur_impurity_w,
                    target_inventory_weight=TARGET_INVENTORY_WEIGHT,
                    trace_relative_weight=cur_trace_rel_w,
                    chain_consistency_weight=cur_chain_w,
                    virgin_ac225_weight=cur_virgin_ac225_w,
                    ac227_chain_weight=cur_ac227_chain_w,
                    f5_signal_weight=F5_SIGNAL_WEIGHT,
                    empty_output_weight=EMPTY_OUTPUT_WEIGHT,
                    n226_scale=N226_SCALE, n225_scale=N225_SCALE, nac_scale=NAC_SCALE,
                    n227_scale=N227_SCALE, nac227_scale=NAC227_SCALE,
                    phi_scale=PHI_SCALE, d_t_input_d_t_hours=D_T_INPUT_D_T_HOURS,
                    use_one_over_v_energy=True,
                    threshold_physics_boost=cur_boost,
                    pred_expert_low=pred_low,
                    pred_expert_high=pred_high,
                    xpinn_interface_weight=XPINN_INTERFACE_WEIGHT,
                    causal_physics_time_scale=CAUSAL_PHYSICS_TIME_SCALE,
                    causal_curriculum_bins=CAUSAL_CURRICULUM_BINS,
                    causal_curriculum_progress=cc_prog_joint,
                    sa_physics_alpha=SA_PHYSICS_ALPHA,
                )
            if not torch.isfinite(loss_b):
                joint_chunks_failed = True
                optimizer.zero_grad(set_to_none=True)
                break
            if use_grad_balance and chunk_sz == n_train:
                sup = info_b["supervised_total"]
                unsup = info_b["unsupervised_total"]
                gs = torch.autograd.grad(
                    sup, model.parameters(), retain_graph=True, allow_unused=True,
                )
                ns = _grad_tuple_l2_norm(gs)
                gu = torch.autograd.grad(
                    unsup, model.parameters(), retain_graph=True, allow_unused=True,
                )
                nu = _grad_tuple_l2_norm(gu)
                ns = max(ns, 1e-12)
                nu = max(nu, 1e-12)
                alpha_s = (ns + nu) / (2.0 * ns)
                alpha_u = (ns + nu) / (2.0 * nu)
                balanced = alpha_s * sup + alpha_u * unsup
                bt = balanced * w
                if grad_scaler is not None:
                    grad_scaler.scale(bt).backward()
                else:
                    bt.backward()
            else:
                if use_uw:
                    sup = info_b["supervised_total"]
                    unsup = info_b["unsupervised_total"]
                    reweighted = uncertainty_weighter(sup, unsup)
                    rt = reweighted * w
                    if grad_scaler is not None:
                        grad_scaler.scale(rt).backward()
                    else:
                        rt.backward()
                else:
                    lt = loss_b * w
                    if grad_scaler is not None:
                        grad_scaler.scale(lt).backward()
                    else:
                        lt.backward()
            total_loss_acc = total_loss_acc + loss_b.detach() * w
            wsum_phys_t = wsum_phys_t + info_b["physics_mse"].detach() * w
            wsum_f1_t = wsum_f1_t + info_b["physics_f1"].detach() * w
            wsum_f2_t = wsum_f2_t + info_b["physics_f2"].detach() * w
            wsum_f3_t = wsum_f3_t + info_b["physics_f3"].detach() * w
            wsum_f4_t = wsum_f4_t + info_b["physics_f4"].detach() * w
            wsum_f5_t = wsum_f5_t + info_b["physics_f5"].detach() * w
            wsum_sup_t = wsum_sup_t + info_b["supervised_total"].detach() * w
            wsum_unsup_t = wsum_unsup_t + info_b["unsupervised_total"].detach() * w
            wsum_data_t = wsum_data_t + info_b["data_mse"].detach() * w
            wsum_log_t = wsum_log_t + info_b["log_data_loss"].detach() * w
            wsum_impurity_t = wsum_impurity_t + info_b["impurity_log_loss"].detach() * w
            wsum_target_inventory_t = wsum_target_inventory_t + info_b["target_inventory_loss"].detach() * w
            wsum_trace_rel_t = wsum_trace_rel_t + info_b["trace_relative_loss"].detach() * w
            wsum_chain_t = wsum_chain_t + info_b["chain_target_loss"].detach() * w
            wsum_virgin_ac225_t = wsum_virgin_ac225_t + info_b["virgin_ac225_loss"].detach() * w
            wsum_ac227_chain_t = wsum_ac227_chain_t + info_b["ac227_chain_transfer_loss"].detach() * w
            wsum_f5_signal_t = wsum_f5_signal_t + info_b["f5_signal_loss"].detach() * w
            wsum_empty_output_t = wsum_empty_output_t + info_b["empty_output_loss"].detach() * w
            wsum_mass_t = wsum_mass_t + info_b["mass_cons_loss"].detach() * w
            wsum_fuel_t = wsum_fuel_t + info_b["fuel_anchor_loss"].detach() * w
            wsum_zero_t = wsum_zero_t + info_b["zero_injection_loss"].detach() * w
            wsum_neg_t = wsum_neg_t + info_b["non_neg_loss"].detach() * w
            wsum_sec_t = wsum_sec_t + info_b["secular_eq_loss"].detach() * w
            rel_b = info_b.get("rel_err")
            if rel_b is None:
                rel_b = z
            wsum_rel_err_t = wsum_rel_err_t + rel_b.detach() * w

        if joint_chunks_failed or not torch.isfinite(total_loss_acc):
            total_loss_val = float("nan")
        else:
            total_loss_val = float(total_loss_acc.item())

        if not np.isfinite(total_loss_val):
            nan_streak += 1
            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.update()
            if nan_streak >= NAN_PATIENCE:
                print(f"  !! {NAN_PATIENCE} NaN streak -- restore best, halve LR")
                _load_model_state_for_training(model, torch.load(BEST_CKPT_PATH, map_location=device, weights_only=True))
                optimizer.state.clear()
                for pg in optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, 1.0e-7)
                nan_streak = 0
            continue

        nan_streak = 0
        if grad_scaler is not None:
            grad_scaler.unscale_(optimizer)
        grad_norm_pre_clip = compute_gradient_norm(model)
        # Skip *before* clip only when optional breaker is on (dataset mismatch / legacy runs often stall here).
        if (
            GRAD_BREAKER_ENABLED
            and grad_norm_pre_clip > GRAD_BREAKER_THRESHOLD
            and joint_epoch > GRAD_BREAKER_GRACE_JOINT_EPOCHS
        ):
            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.update()
            if joint_epoch % 20 == 0:
                print(
                    f"  !! Skipped step: pre-clip grad_norm={grad_norm_pre_clip:.1e} "
                    f"> {GRAD_BREAKER_THRESHOLD:.0f}"
                )
            continue

        clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        grad_norm = compute_gradient_norm(model)

        info = {
            "data_mse": wsum_data_t,
            "log_data_loss": wsum_log_t,
            "impurity_log_loss": wsum_impurity_t,
            "target_inventory_loss": wsum_target_inventory_t,
            "trace_relative_loss": wsum_trace_rel_t,
            "chain_target_loss": wsum_chain_t,
            "virgin_ac225_loss": wsum_virgin_ac225_t,
            "ac227_chain_transfer_loss": wsum_ac227_chain_t,
            "f5_signal_loss": wsum_f5_signal_t,
            "empty_output_loss": wsum_empty_output_t,
            "physics_mse": wsum_phys_t,
            "physics_f1": wsum_f1_t,
            "physics_f2": wsum_f2_t,
            "physics_f3": wsum_f3_t,
            "physics_f4": wsum_f4_t,
            "physics_f5": wsum_f5_t,
            "mass_cons_loss": wsum_mass_t,
            "fuel_anchor_loss": wsum_fuel_t,
            "zero_injection_loss": wsum_zero_t,
            "non_neg_loss": wsum_neg_t,
            "secular_eq_loss": wsum_sec_t,
            "rel_err": wsum_rel_err_t,
            "grad_norm": torch.as_tensor(grad_norm, device=device, dtype=wsum_data_t.dtype),
        }
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            nan_streak += 1
            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None:
                grad_scaler.update()
            if nan_streak >= NAN_PATIENCE:
                print(f"  !! {NAN_PATIENCE} non-finite grads -- restore best, halve LR")
                _load_model_state_for_training(model, torch.load(BEST_CKPT_PATH, map_location=device, weights_only=True))
                optimizer.state.clear()
                for pg in optimizer.param_groups:
                    pg["lr"] = max(pg["lr"] * 0.5, 1.0e-7)
                nan_streak = 0
            continue

        nan_streak = 0
        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        wsum_phys = float(wsum_phys_t.item())
        wsum_data = float(wsum_data_t.item())
        wsum_mass = float(wsum_mass_t.item())
        wsum_fuel = float(wsum_fuel_t.item())
        wsum_zero = float(wsum_zero_t.item())
        wsum_log = float(wsum_log_t.item())
        wsum_impurity = float(wsum_impurity_t.item())
        wsum_target_inventory = float(wsum_target_inventory_t.item())
        wsum_trace_rel = float(wsum_trace_rel_t.item())
        wsum_chain = float(wsum_chain_t.item())
        wsum_virgin_ac225 = float(wsum_virgin_ac225_t.item())
        wsum_ac227_chain = float(wsum_ac227_chain_t.item())
        wsum_f5_signal = float(wsum_f5_signal_t.item())
        wsum_empty_output = float(wsum_empty_output_t.item())
        wsum_f1 = float(wsum_f1_t.item())
        wsum_f2 = float(wsum_f2_t.item())
        wsum_f3 = float(wsum_f3_t.item())
        wsum_f4 = float(wsum_f4_t.item())
        wsum_f5 = float(wsum_f5_t.item())
        wsum_neg = float(wsum_neg_t.item())
        wsum_sec = float(wsum_sec_t.item())
        wsum_rel_err = float(wsum_rel_err_t.item())

        # ---- Adaptive secular equilibrium weighting (anti tug-of-war) ----
        # If secular eq violation grows too large, boost its weight temporarily
        SEC_EQ_VIOLATION_THRESHOLD = 0.5  # mean squared violation threshold
        SEC_EQ_WEIGHT_BOOST = 2.0         # multiplicative boost
        SEC_EQ_WEIGHT_MAX = 50.0          # ceiling to prevent runaway
        if wsum_sec > SEC_EQ_VIOLATION_THRESHOLD:
            adaptive_sec_eq_weight = min(SEC_EQ_WEIGHT_MAX, adaptive_sec_eq_weight * SEC_EQ_WEIGHT_BOOST ** 0.01)
        elif wsum_sec < SEC_EQ_VIOLATION_THRESHOLD * 0.1 and adaptive_sec_eq_weight > SECULAR_EQ_WEIGHT:
            # Gradually decay back toward baseline if violation is under control
            adaptive_sec_eq_weight = max(SECULAR_EQ_WEIGHT, adaptive_sec_eq_weight * 0.9999)

        loss_val = total_loss_val

        epoch_list.append(epoch)
        hist_data.append(float(info["data_mse"].detach().cpu()))
        hist_phys.append(float(info["physics_mse"].detach().cpu()))
        append_loss_history_csv(
            LOSS_HISTORY_CSV_PATH,
            epoch=epoch,
            phase="joint",
            data_mse=float(info["data_mse"].detach().cpu()),
            physics_mse=float(info["physics_mse"].detach().cpu()),
            supervised_total=float(wsum_sup_t.item()),
            unsupervised_total=float(wsum_unsup_t.item()),
            grad_norm=float(grad_norm_pre_clip)
        )

        # --- Safety Net: Halt if physics/data ratio exceeds threshold (default 100:1) ---
        # PINN_SAFETY_RATIO=0 disables halt; MEDIUM train uses 250 via notebook env.
        safety_ratio = float(os.environ.get("PINN_SAFETY_RATIO", "100"))
        if safety_ratio > 0 and wsum_data > 1e-6:
            ratio = wsum_unsup_t.item() / wsum_data
            if ratio > safety_ratio and joint_epoch > 1000:
                print(f"\n[SAFETY HALT] Physics/Data ratio ({ratio:.1f}) exceeded {safety_ratio}:1 threshold.")
                print(f"Saving emergency weights to {BEST_CKPT_PATH} and stopping.")
                torch.save(_model_state_dict_for_checkpoint(model), BEST_CKPT_PATH)
                break

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = epoch
            best_joint_epoch = joint_epoch
            torch.save(_model_state_dict_for_checkpoint(model), BEST_CKPT_PATH)

        # CosineAnnealingWarmRestarts step (epoch-based)
        scheduler.step()

        # --- Adaptive Residual Refinement (RAR) every 500 joint epochs ---
        # Finds top-10% highest-residual collocation points and adds to training pool
        if j % 500 == 0:
            try:
                rar_coll = build_pretrain_collocation(2000, rng_colloc, device, t_max_h=t_max_h, dtype=compute_dtype)
                with torch.no_grad():
                    rar_pred = model.forward_raw(rar_coll)
                # Simple per-point residual: L2 norm of output deviation from IC
                rar_res = (rar_pred - rar_coll[:, 3:8]).pow(2).sum(dim=1)
                targetless_now = int(torch.isnan(targets).any(dim=1).sum().item())
                max_targetless = int(max(0, round((n_train + 100) * RAR_MAX_TARGETLESS_FRACTION)))
                add_limit = min(100, max(0, max_targetless - targetless_now))
                if add_limit <= 0:
                    continue
                top_k = rar_res.argsort(descending=True)[:add_limit]
                hard_pts = rar_coll[top_k].detach()
                # Mark RAR points with NaN targets so data loss can skip them (physics-only points)
                hard_tgts = torch.full((hard_pts.size(0), 5), float('nan'), device=device)
                # CRITICAL: Keep requires_grad=True for physics loss autograd!
                # Each batch re-clones with requires_grad in the training loop
                inputs  = torch.cat([inputs.detach(), hard_pts], dim=0)
                targets = torch.cat([targets, hard_tgts], dim=0)
                n_train = int(inputs.size(0))
                chunk_sz = n_train if joint_chunk_sz <= 0 else min(joint_chunk_sz, n_train)
                if j % 1000 == 0:
                    print(f"  [RAR] Added {hard_pts.size(0)} hard colloc pts at epoch {epoch}. Pool: {n_train}")
                _save_training_resume_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    phase="joint", epoch=epoch, joint_epoch=joint_epoch,
                    train_pre=train_pre, train_epochs=train_epochs,
                    best_loss=best_loss, best_epoch=best_epoch, best_joint_epoch=best_joint_epoch,
                    adaptive_sec_eq_weight=adaptive_sec_eq_weight,
                    inputs=inputs, targets=targets, epoch_list=epoch_list,
                    hist_data=hist_data, hist_phys=hist_phys,
                    uw_state=uncertainty_weighter.state_dict() if use_uw else None,
                )
            except Exception:
                pass

        if j == 1 or j % LOG_EVERY == 0 or j == train_epochs:
            lr = optimizer.param_groups[0]["lr"]
            rel_e = float(info["rel_err"].detach().cpu())
            grad_norm_log = float(info["grad_norm"].detach().cpu())
            species_snapshot = _species_error_snapshot(model, inputs, targets)
            print(
                f"epoch {epoch:6d} | lr {lr:.2e} | total {loss_val:.4e} "
                f"| data {wsum_data:.4e} (w_d={cur_data_w:.1f} w_p={cur_phys_w:.1f}) "
                f"| log {wsum_log:.4e} | imp {wsum_impurity:.4e} | inv {wsum_target_inventory:.4e} "
                f"| trace {wsum_trace_rel:.4e} | chain {wsum_chain:.4e} "
                f"| virg {wsum_virgin_ac225:.4e} | ac227chain {wsum_ac227_chain:.4e} "
                f"| empty {wsum_empty_output:.4e} "
                f"| phys {wsum_phys:.4e} | mass {wsum_mass:.4e} | zero {wsum_zero:.4e} "
                f"| neg {wsum_neg:.4e} | sec_eq {wsum_sec:.4e} "
                f"(w_s={adaptive_sec_eq_weight:.1f}) | rel_err {rel_e:.3f} | grad_norm {grad_norm_log:.2e} (pre={grad_norm_pre_clip:.2e})"
            )
            print(
                "  physics residuals: "
                f"f226={wsum_f1:.2e} f225={wsum_f2:.2e} fac225={wsum_f3:.2e} "
                f"f227={wsum_f4:.2e} fac227={wsum_f5:.2e}"
            )
            print(f"  species rel_err: {species_snapshot}")
            _save_training_resume_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                phase="joint", epoch=epoch, joint_epoch=joint_epoch,
                train_pre=train_pre, train_epochs=train_epochs,
                best_loss=best_loss, best_epoch=best_epoch, best_joint_epoch=best_joint_epoch,
                adaptive_sec_eq_weight=adaptive_sec_eq_weight,
                inputs=inputs, targets=targets, epoch_list=epoch_list,
                hist_data=hist_data, hist_phys=hist_phys,
                uw_state=uncertainty_weighter.state_dict() if use_uw else None,
            )

    _load_model_state_for_training(model, torch.load(BEST_CKPT_PATH, map_location=device, weights_only=True))

    # ---- L-BFGS Fine-Tuning Phase ----
    # Filter out RAR collocation points (NaN targets) for L-BFGS - it needs valid data
    lbfgs_max_iter = max(0, int(os.environ.get("PINN_LBFGS_MAX_ITER", "100")))
    valid_mask_lbfgs = ~torch.isnan(targets).any(dim=1)
    if lbfgs_max_iter == 0:
        print("\nL-BFGS skipped: PINN_LBFGS_MAX_ITER=0")
    elif valid_mask_lbfgs.any():
        print(
            f"\nStarting L-BFGS fine-tuning (max {lbfgs_max_iter} iterations, "
            f"{valid_mask_lbfgs.sum().item()} valid data points)..."
        )
        lbfgs_params = list(model.parameters())
        if use_uw:
            lbfgs_params += list(uncertainty_weighter.parameters())
        lbfgs = torch.optim.LBFGS(lbfgs_params, lr=1.0, max_iter=lbfgs_max_iter,
                                  line_search_fn="strong_wolfe", tolerance_change=1e-7)
        
        lbfgs_inputs = inputs[valid_mask_lbfgs].detach().requires_grad_(True)
        lbfgs_targets = targets[valid_mask_lbfgs]
        
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            # Use only valid data points for L-BFGS step (no NaN targets from RAR)
            rg = getattr(model, "regime_gated", True)
            if rg:
                p, pl, ph = model.forward_raw(lbfgs_inputs, return_experts=True)
            else:
                p = model.forward_raw(lbfgs_inputs)
                pl = ph = None
            # Same Fix 4 (Opus audit) as the joint loop: drop the redundant model() forward pass
            # and use forward_raw output as the single source of truth for both physics and data.
            loss_val, info_lbfgs = compute_physics_loss(
                model, lbfgs_inputs, p, targets=lbfgs_targets,
                physics_weight=cur_phys_w, data_weight=cur_data_w,
                mass_weight=MASS_WEIGHT, fuel_anchor_weight=FUEL_ANCHOR_WEIGHT,
                non_neg_weight=NON_NEG_WEIGHT, secular_eq_weight=adaptive_sec_eq_weight,
                ra225_physics_weight=RA225_PHYSICS_WEIGHT,
                pred_for_data=None, data_species_weights=DATA_SPECIES_WEIGHTS,
                log_data_weight=LOG_DATA_WEIGHT,
                log_species_weights=LOG_SPECIES_WEIGHTS,
                impurity_log_weight=IMPURITY_LOG_WEIGHT,
                target_inventory_weight=TARGET_INVENTORY_WEIGHT,
                trace_relative_weight=TRACE_RELATIVE_WEIGHT,
                chain_consistency_weight=CHAIN_CONSISTENCY_WEIGHT,
                virgin_ac225_weight=VIRGIN_AC225_WEIGHT * 0.5,
                ac227_chain_weight=AC227_CHAIN_WEIGHT,
                f5_signal_weight=F5_SIGNAL_WEIGHT,
                empty_output_weight=EMPTY_OUTPUT_WEIGHT,
                n226_scale=N226_SCALE, n225_scale=N225_SCALE, nac_scale=NAC_SCALE,
                n227_scale=N227_SCALE, nac227_scale=NAC227_SCALE,
                phi_scale=PHI_SCALE, d_t_input_d_t_hours=D_T_INPUT_D_T_HOURS,
                use_one_over_v_energy=True,
                threshold_physics_boost=THRESHOLD_PHYSICS_BOOST,
                pred_expert_low=pl,
                pred_expert_high=ph,
                xpinn_interface_weight=XPINN_INTERFACE_WEIGHT,
                causal_physics_time_scale=CAUSAL_PHYSICS_TIME_SCALE,
                causal_curriculum_bins=CAUSAL_CURRICULUM_BINS,
                causal_curriculum_progress=1.0,
                sa_physics_alpha=min(SA_PHYSICS_ALPHA, 0.25),
            )
            if torch.isfinite(loss_val):
                if use_uw:
                    sup = info_lbfgs["supervised_total"]
                    unsup = info_lbfgs["unsupervised_total"]
                    reweighted = uncertainty_weighter(sup, unsup)
                    reweighted.backward()
                else:
                    loss_val.backward()
            return loss_val

        try:
            final_loss = lbfgs.step(closure)
            print(f"L-BFGS finished with total loss: {float(final_loss):.4e}")
            if float(final_loss) < best_loss:
                best_loss = float(final_loss)
                best_epoch = epoch
                best_joint_epoch = joint_epoch
                torch.save(_model_state_dict_for_checkpoint(model), BEST_CKPT_PATH)
                print(
                    f"L-BFGS found a better minimum at epoch {best_epoch} "
                    f"(joint {best_joint_epoch}). Model updated."
                )
        except Exception as e:
            print(f"L-BFGS skipped/failed: {e}")
    else:
        print("\nL-BFGS skipped: no valid data points (all NaN targets from RAR)")

    # Load absolute best before plotting
    _load_model_state_for_training(model, torch.load(BEST_CKPT_PATH, map_location=device, weights_only=True))

    print("\nTraining completed.")
    print(f"Best recorded Loss: {best_loss:.6e}")
    if best_epoch > 0:
        print(
            f"Best checkpoint: epoch {best_epoch} (joint {best_joint_epoch}) "
            f"-> {BEST_CKPT_PATH.name}"
        )
    last_epoch = int(epoch_list[-1]) if epoch_list else int(resume_epoch)
    print(
        f"Last epoch:        {last_epoch} (joint {joint_epoch}) "
        f"-> {pathlib.Path(WEIGHTS_PATH).name}"
    )

    torch.save(_model_state_dict_for_checkpoint(model), WEIGHTS_PATH)
    print(f"Saved state dict to {WEIGHTS_PATH}")

    plot_loss_history(epoch_list, hist_data, hist_phys, LOSS_PLOT_PATH, train_pre)
    print(f"Saved loss plot to {LOSS_PLOT_PATH}")
    graph_provenance.record_graph_write(
        _proj_root,
        pathlib.Path(LOSS_PLOT_PATH),
        producer="train.py",
        run_id=_run_id,
        extra={"data_path": str(DATA_PATH.resolve()), "train_script_rev": PINN_TRAIN_SCRIPT_REV},
    )

    if plot_loss_components_png(
        LOSS_HISTORY_CSV_PATH,
        LOSS_COMPONENTS_PATH,
        proj_root=_proj_root,
        run_id=_run_id,
        record_provenance=True,
    ):
        print(f"Saved loss components plot to {LOSS_COMPONENTS_PATH}")
    else:
        print("Skipping loss components plot (missing or empty loss_history.csv).")

    model.eval()
    with torch.no_grad():
        pred_final = model(inputs.detach())
    # Filter out RAR points (NaN targets) for final scatter plot
    valid_mask_final = ~torch.isnan(targets).any(dim=1)
    if valid_mask_final.any():
        n_ac_true = targets[valid_mask_final, 2].detach().cpu().numpy()
        n_ac_pred = (pred_final[valid_mask_final, 2] * NAC_SCALE).detach().cpu().numpy()
        plot_ac225_pred_vs_true(n_ac_true, n_ac_pred, AC225_SCATTER_PATH)
        print(f"Saved Ac-225 scatter to {AC225_SCATTER_PATH} ({len(n_ac_true)} valid points)")
        if pathlib.Path(AC225_SCATTER_PATH).is_file():
            graph_provenance.record_graph_write(
                _proj_root,
                pathlib.Path(AC225_SCATTER_PATH),
                producer="train.py",
                run_id=_run_id,
                extra={"data_path": str(DATA_PATH.resolve()), "train_script_rev": PINN_TRAIN_SCRIPT_REV},
            )

        vm = virgin_ic_mask(inputs.detach()) & valid_mask_final
        if bool(vm.any().item()):
            nt = targets[vm, 2].detach().cpu().numpy()
            pr = (pred_final[vm, 2] * NAC_SCALE).detach().cpu().numpy()
            ok = (nt > 0) & (pr > 0)
            if ok.any():
                mape = float(np.mean(np.abs(pr[ok] - nt[ok]) / nt[ok]) * 100.0)
                med_rel = float(np.median(np.abs(pr[ok] - nt[ok]) / nt[ok]) * 100.0)
                metrics = f"Virgin IC only (train.py rule)\nMAPE = {mape:.2f}% | median rel = {med_rel:.2f}%\nn = {int(ok.sum())}"
            else:
                metrics = "Virgin IC only — no positive Ac-225 pairs to score"
            plot_ac225_pred_vs_true(
                nt, pr, AC225_SCATTER_VIRGIN_PATH,
                title=r"Ac-225 parity (virgin inventory IC)",
                metrics_line=metrics,
            )
            print(f"Saved virgin-IC Ac-225 scatter to {AC225_SCATTER_VIRGIN_PATH}")
            if pathlib.Path(AC225_SCATTER_VIRGIN_PATH).is_file():
                graph_provenance.record_graph_write(
                    _proj_root,
                    pathlib.Path(AC225_SCATTER_VIRGIN_PATH),
                    producer="train.py",
                    run_id=_run_id,
                    extra={
                        "data_path": str(DATA_PATH.resolve()),
                        "train_script_rev": PINN_TRAIN_SCRIPT_REV,
                        "subset": "virgin_ic",
                    },
                )
    else:
        print("Warning: No valid targets for final scatter plot (all NaN)")
    write_validation_summary(inputs.detach(), targets.detach(), pred_final.detach(), VALIDATION_REPORT_PATH)
    print(f"Saved validation summary to {VALIDATION_REPORT_PATH}")

    manifest = graph_provenance.load_manifest(_proj_root)
    arts = manifest.get("artifacts", {})
    fin_extra: dict = {
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "best_joint_epoch": int(best_joint_epoch),
        "last_epoch": int(epoch_list[-1]) if epoch_list else int(resume_epoch),
        "last_joint_epoch": int(joint_epoch),
    }
    for rel in (
        "graphs/pinn_loss_history.png",
        "graphs/loss_components.png",
        "graphs/pinn_ac225_pred_vs_true.png",
        "graphs/pinn_ac225_pred_vs_true_virgin_ic.png",
    ):
        ent = arts.get(rel)
        if isinstance(ent, dict) and ent.get("sha256"):
            short = rel.replace("graphs/", "").replace(".png", "")
            fin_extra[f"{short}_sha256"] = ent["sha256"]
    graph_provenance.training_run_finalize(
        _proj_root,
        run_id=_run_id,
        status="graphs_saved",
        loss_plot_path=pathlib.Path(LOSS_PLOT_PATH),
        parity_plot_path=pathlib.Path(AC225_SCATTER_PATH),
        extra=fin_extra or None,
    )


if __name__ == "__main__":
    main()
