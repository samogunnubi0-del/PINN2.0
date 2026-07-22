"""
Train PI-LSTM v3 with integrated trapezoidal physics loss + v2 distillation.

Key upgrades (all tunable via env vars):
  - Full-trajectory Ac-225 eval + checkpoint metric (P0-1)
  - Canonical held-out val/test shared with compare_models (P0-2)
  - Ra-227 overshoot penalty (P0-3)
  - Structured 800-scenario training pool, log-spaced time grid (P1-4/5)
  - Hard initial condition ansatz (P1-6, in the model)
  - v2 distillation teacher (P1-7)
  - Two-phase (physics pretrain -> joint), long schedule + LR decay (P2-9/10)
  - Bigger model 256/8 (P2-11), causal time weighting (P2-12)
  - L-BFGS polish (P3-13), lightweight grad balancing (P3-14), log_weight (P3-15)

Usage:
    python v3_pilstm/train_pi_lstm.py
    PILSTM_QUICK=1 python v3_pilstm/train_pi_lstm.py   # fast smoke test
    PILSTM_EPOCHS=12000 python v3_pilstm/train_pi_lstm.py

Reproducibility / physics knobs (2026-07-18):
    PI_LSTM_SEED=42            deterministic seeding (python/numpy/torch + loader order)
    PI_LSTM_LOSS=trap|expmix   physics collocation (trap = legacy trapezoid;
                               expmix = exact piecewise-exponential propagator)
    SCENARIO_VERSION=v1|v2     inventory scale (v1 = legacy 6.022e23 atoms "226 g";
                               v2 = true 1 g = 2.664e21 atoms)
    PI_LSTM_ADAPTIVE_WEIGHTS=1 self-adaptive per-species physics weights
                               (McClenny & Braga-Neto 2023; grad-norm-ratio,
                                refreshed every PI_LSTM_ADAPTIVE_EVERY epochs)
    PI_LSTM_CURRICULUM=1       stiffness curriculum (Seiler et al. 2025):
                               de-stiffened rates ladder 100,10,1 annealed over
                               training; or explicit ladder e.g. =50,5,1
    PILSTM_WEIGHTS_PATH / PILSTM_STATE_PATH / PILSTM_PROGRESS_PATH /
    PILSTM_RESULTS_PATH        redirect artifacts (smoke runs; protects real checkpoints)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

V3_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = V3_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402
from data.trajectory_dataset import build_dataloaders  # noqa: E402
from physics.integrated_loss import (  # noqa: E402
    causal_time_weights,
    data_trajectory_loss,
    impurity_overshoot_loss,
    integrated_physics_loss,
    mass_conservation_loss,
)
from physics.distill import V2Teacher, distillation_loss  # noqa: E402
from analysis.endpoint_eval import (  # noqa: E402
    ac225_endpoint_median,
    evaluate_endpoints,
    pilstm_endpoint,
)
from data.trajectory_dataset import canonical_heldout_scenarios  # noqa: E402
from seed_utils import seed_everything  # noqa: E402

WEIGHTS_DIR = V3_ROOT / "weights"


def _out_path(env_name: str, default: Path) -> Path:
    """Allow smoke/CI runs to redirect artifact paths via env (keeps the
    student's real checkpoints/summaries untouched)."""
    v = os.environ.get(env_name, "").strip()
    return Path(v) if v else default


WEIGHTS_PATH = _out_path("PILSTM_WEIGHTS_PATH", WEIGHTS_DIR / "pi_lstm_best.pth")
STATE_PATH = _out_path("PILSTM_STATE_PATH", WEIGHTS_DIR / "pi_lstm_train_state.pth")
PROGRESS_PATH = _out_path("PILSTM_PROGRESS_PATH", V3_ROOT / "results" / "train_progress.json")
RESULTS_PATH = _out_path("PILSTM_RESULTS_PATH", V3_ROOT / "results" / "train_summary.json")
V2_WEIGHTS = PROJECT_ROOT / "weights" / "pinn_best_weights.pth"

AC225_IDX = 2


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes")


def _batch_to_model(batch: dict, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.is_floating_point():
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device=device)
    return out


def _grad_norm(loss: torch.Tensor, params) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum())
    return total ** 0.5


def _scheduled_distill_w(epoch: int, epochs: int, base_w: float, pretrain_frac: float) -> float:
    """Ramp distill weight from base_w to 0 over epochs 60%-100% (full during pretrain)."""
    if base_w <= 0.0:
        return 0.0
    pretrain_end = int(pretrain_frac * epochs)
    ramp_start = int(0.6 * epochs)
    if epoch <= pretrain_end:
        return base_w
    if epoch < ramp_start:
        return base_w
    if epoch >= epochs:
        return 0.0
    progress = (epoch - ramp_start) / max(epochs - ramp_start, 1)
    return base_w * (1.0 - progress)


def train_one_epoch(
    model, loader, optimizer, device, dtype, *,
    data_w, phys_w, mass_w, distill_w, overshoot_w, log_weight,
    teacher, progress, causal_eps, grad_balance, loss_mode="trap",
    adaptive_weighter=None, epoch=0, rate_scale=1.0,
):
    model.train()
    totals = {"loss": 0.0, "data": 0.0, "physics": 0.0, "distill": 0.0}
    n_batches = 0
    for batch in loader:
        batch = _batch_to_model(batch, device, dtype)
        features = batch["features"]
        target = batch["target"]
        t_norm = batch["t_norm"]
        phi_norm = batch["phi_norm"]
        energy_feature = batch["energy_feature"]
        ic_norm = batch["ic_norm"]

        optimizer.zero_grad(set_to_none=True)
        pred = model(features)

        loss_data = data_trajectory_loss(pred, target, log_weight=log_weight)
        tw = causal_time_weights(t_norm, eps=causal_eps, progress=progress)
        want_ps = adaptive_weighter is not None
        loss_phys, pinfo = integrated_physics_loss(
            pred, t_norm, phi_norm, energy_feature, physics_weight=1.0,
            time_weights=tw, mode=loss_mode, return_per_species=want_ps,
            rate_scale=rate_scale,
        )
        if want_ps:
            # PI_LSTM_ADAPTIVE_WEIGHTS=1: grad-norm-ratio per-species weights
            # (McClenny & Braga-Neto 2023 / Wang, Yu & Perdikaris 2022) —
            # refreshed on the first batch of the epoch, mean-normalized so
            # the global physics scale is unchanged.
            if n_batches == 0 and adaptive_weighter.should_update(epoch):
                adaptive_weighter.update(model, pinfo["per_species"], loss_phys, epoch)
            w_species = adaptive_weighter.tensor(device, dtype)
            loss_phys = (pinfo["per_species"] * w_species).mean()
        loss_mass = mass_conservation_loss(pred, ic_norm, phi_norm)
        loss_over = impurity_overshoot_loss(pred, phi_norm, energy_feature)

        loss_distill = pred.new_zeros(())
        if teacher is not None and distill_w > 0.0:
            teacher_traj = teacher.predict_traj(features)
            loss_distill = distillation_loss(pred, teacher_traj, log_weight=log_weight)

        # P3-14: rebalance physics weight so its grad norm tracks the data term.
        eff_phys_w = phys_w
        if grad_balance and n_batches == 0:
            gd = _grad_norm(data_w * loss_data, list(model.parameters()))
            gp = _grad_norm(loss_phys, list(model.parameters()))
            if gp > 1e-12:
                eff_phys_w = float(np.clip(gd / gp, 0.1, 100.0))

        loss = (
            data_w * loss_data
            + eff_phys_w * loss_phys
            + mass_w * loss_mass
            + overshoot_w * loss_over
            + distill_w * loss_distill
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["data"] += float(loss_data.detach())
        totals["physics"] += pinfo["physics_mse"]
        totals["distill"] += float(loss_distill.detach())
        n_batches += 1

    for k in totals:
        totals[k] /= max(n_batches, 1)
    return totals


@torch.no_grad()
def evaluate(model, loader, device, dtype) -> dict[str, float]:
    """P0-1: full-trajectory relative error (all timesteps), per species."""
    model.eval()
    species = ["Ra-226", "Ra-225", "Ac-225", "Ra-227", "Ac-227"]
    errs: dict[int, list[float]] = {i: [] for i in range(5)}
    for batch in loader:
        batch = _batch_to_model(batch, device, dtype)
        pred = model(batch["features"])
        target = batch["target"]
        for i in range(5):
            t_true = target[:, :, i]
            t_pred = pred[:, :, i]
            mask = t_true > 1e-10
            if not mask.any():
                continue
            denom = t_true[mask].abs().clamp(min=1e-8)
            errs[i].extend(((t_pred[mask] - t_true[mask]).abs() / denom).cpu().numpy().tolist())

    out: dict[str, float] = {}
    for i, name in enumerate(species):
        arr = np.array(errs[i], dtype=np.float64) if errs[i] else np.array([1.0])
        out[f"{name}_median_rel"] = float(np.median(arr))
    out["ac225_median_rel"] = out["Ac-225_median_rel"]
    return out


def _lbfgs_polish(model, loader, device, dtype, *, data_w, log_weight, max_iter):
    """P3-13: short full-batch L-BFGS fine-tune on the weighted training data term.

    Data-only polish can worsen the endpoint checkpoint metric; callers MUST
    score with the same ckpt metric and reject (reload best) when post >= pre.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")
    batches = [_batch_to_model(b, device, dtype) for b in loader]
    w = float(data_w)

    def closure():
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device, dtype=dtype)
        for b in batches:
            pred = model(b["features"])
            total = total + w * data_trajectory_loss(pred, b["target"], log_weight=log_weight)
        total = total / max(len(batches), 1)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        return total

    model.train()
    opt.step(closure)


def main() -> None:
    # P1 reproducibility: seed everything FIRST (PI_LSTM_SEED env, default 42).
    seed = seed_everything()
    quick = _env_flag("PILSTM_QUICK", False)
    epochs = 40 if quick else _env_int("PILSTM_EPOCHS", 6000)
    n_train = 60 if quick else _env_int("PILSTM_N_TRAIN", 1400)
    n_val = 10 if quick else _env_int("PILSTM_N_VAL", 22)
    n_test = 10 if quick else _env_int("PILSTM_N_TEST", 22)
    batch_size = 8 if quick else _env_int("PILSTM_BATCH", 16)
    n_steps = 30 if quick else _env_int("PILSTM_N_STEPS", 64)
    dense_steps_raw = os.environ.get("PILSTM_DENSE_STEPS", "").strip()
    dense_steps = int(dense_steps_raw) if dense_steps_raw else None

    # Defaults match Results-6 (winning held-out recipe), not the slower R7 quality path.
    data_w = _env_float("PILSTM_DATA_WEIGHT", 35.0)
    phys_w = _env_float("PILSTM_PHYSICS_WEIGHT", 20.0)
    mass_w = _env_float("PILSTM_MASS_WEIGHT", 10.0)
    distill_w = _env_float("PILSTM_DISTILL_WEIGHT", 5.0)
    overshoot_w = _env_float("PILSTM_OVERSHOOT_WEIGHT", 20.0)
    log_weight = _env_float("PILSTM_LOG_WEIGHT", 2.0)
    causal_eps = _env_float("PILSTM_CAUSAL_EPS", 2.5)

    pretrain_frac = _env_float("PILSTM_PRETRAIN_FRAC", 0.20)  # P2-9 physics-first fraction
    grad_balance = _env_flag("PILSTM_GRAD_BALANCE", False)
    use_distill = _env_flag("PILSTM_DISTILL", True) and V2_WEIGHTS.exists()
    lbfgs_iter = 0 if quick else _env_int("PILSTM_LBFGS_ITER", 60)
    ckpt_metric = os.environ.get("PILSTM_CKPT_METRIC", "endpoint_ac225").strip().lower()
    # Physics collocation: "trap" (legacy trapezoid) | "expmix" (exact
    # piecewise-exponential propagator; see physics/integrated_loss.py).
    loss_mode = os.environ.get("PI_LSTM_LOSS", "trap").strip().lower()
    # Inventory scale for scenario generation: "v1" legacy_226g | "v2" true_1g.
    scenario_version = os.environ.get("SCENARIO_VERSION", "v1").strip().lower()
    # Self-adaptive per-species physics weights (default OFF = legacy uniform).
    adaptive_weights = _env_flag("PI_LSTM_ADAPTIVE_WEIGHTS", False)
    adaptive_every = _env_int("PI_LSTM_ADAPTIVE_EVERY", 5)
    adaptive_ema = _env_float("PI_LSTM_ADAPTIVE_EMA", 0.9)
    # Stiffness curriculum (Seiler et al. 2025, arXiv:2501.17281, stiff
    # transfer learning): PI_LSTM_CURRICULUM=1 -> default ladder 100,10,1
    # (train stage 1 on rates/100, anneal to full stiffness); or an explicit
    # comma ladder e.g. PI_LSTM_CURRICULUM=50,5,1. Off/0 = legacy.
    curriculum_raw = os.environ.get("PI_LSTM_CURRICULUM", "").strip().lower()
    if curriculum_raw in ("", "0", "false", "no"):
        curriculum_scales: list[float] = []
    elif curriculum_raw in ("1", "true", "yes"):
        curriculum_scales = [100.0, 10.0, 1.0]
    else:
        curriculum_scales = [float(x) for x in curriculum_raw.split(",") if x.strip()]
    log_every = max(1, _env_int("PILSTM_LOG_EVERY", 25))
    # Eval/checkpoint every N epochs (25 = Results-6 / fast; 1 = slow quality path that lost held-out).
    eval_every = max(1, _env_int("PILSTM_EVAL_EVERY", 25))
    # Optional early stop when best checkpoint score drops below this (e.g. 0.01 = 1%).
    early_stop = _env_float("PILSTM_EARLY_STOP", 0.01)
    resume = _env_flag("PILSTM_RESUME", False)
    # How often to write full train state (model+opt+sched) for Kaggle timeout recovery.
    state_every = max(1, _env_int("PILSTM_STATE_EVERY", 25))

    hidden_dim = 64 if quick else _env_int("PILSTM_HIDDEN", 256)
    n_fourier = 4 if quick else _env_int("PILSTM_FOURIER", 8)
    n_time_fourier = 0 if quick else _env_int("PILSTM_TIME_FOURIER", 16)
    hard_ic = _env_flag("PILSTM_HARD_IC", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if _env_flag("PILSTM_FLOAT64", False) else torch.float32

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    (V3_ROOT / "results").mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_med = float("inf")
    best_epoch = 0
    resume_blob = None
    if resume:
        if STATE_PATH.exists():
            resume_blob = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
            start_epoch = int(resume_blob.get("epoch", 0)) + 1
            best_med = float(resume_blob.get("best_med", float("inf")))
            best_epoch = int(resume_blob.get("best_epoch", 0))
            print(f"Resume: full state from {STATE_PATH} -> start_epoch={start_epoch}")
        elif WEIGHTS_PATH.exists() and PROGRESS_PATH.exists():
            prog = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
            start_epoch = int(prog.get("epoch", 0)) + 1
            if prog.get("best_score") is not None:
                best_med = float(prog["best_score"])
            best_epoch = int(prog.get("best_epoch", 0))
            print(
                f"Resume: best weights + {PROGRESS_PATH.name} -> "
                f"start_epoch={start_epoch} best={best_med:.4f}@{best_epoch}"
            )
        elif WEIGHTS_PATH.exists():
            start_epoch = max(1, _env_int("PILSTM_START_EPOCH", 1))
            print(f"Resume: weights only -> start_epoch={start_epoch}")
        else:
            raise FileNotFoundError(
                "PILSTM_RESUME=1 but no pi_lstm_train_state.pth / pi_lstm_best.pth found"
            )
        start_override = os.environ.get("PILSTM_START_EPOCH", "").strip()
        if start_override:
            start_epoch = max(1, int(start_override))
            print(f"Resume: PILSTM_START_EPOCH override -> {start_epoch}")

    if start_epoch > epochs:
        print(f"Nothing to do: start_epoch={start_epoch} > epochs={epochs}")
        return

    print(
        f"PI-LSTM training | device={device} dtype={dtype} epochs={epochs} "
        f"start_epoch={start_epoch} "
        f"n_train={n_train} n_steps={n_steps} hidden={hidden_dim} "
        f"fourier={n_fourier} time_fourier={n_time_fourier} "
        f"hard_ic={hard_ic} distill={use_distill} grad_balance={grad_balance} "
        f"ckpt_metric={ckpt_metric} log_every={log_every} eval_every={eval_every} "
        f"seed={seed} loss_mode={loss_mode} scenario_version={scenario_version} "
        f"adaptive_weights={adaptive_weights} curriculum={curriculum_scales or 'off'}"
    )

    def _make_loaders(train_rate_scale: float = 1.0):
        return build_dataloaders(
            n_train=n_train, n_val=n_val, n_test=n_test, n_steps=n_steps,
            batch_size=batch_size, dense_steps=dense_steps, seed=seed,
            scenario_version=scenario_version, loader_seed=seed,
            train_rate_scale=train_rate_scale,
        )

    # Stiffness curriculum (Seiler et al. 2025, arXiv:2501.17281): start on the
    # de-stiffened ODE (rates/scale), anneal to full stiffness; each stage gets
    # an equal share of the epoch budget and a rebuilt training set.
    current_scale = curriculum_scales[0] if curriculum_scales else 1.0
    train_loader, val_loader, test_loader, _ = _make_loaders(current_scale)
    if curriculum_scales:
        print(f"Curriculum stage 1/{len(curriculum_scales)}: rate_scale={current_scale:g}")

    adaptive_weighter = None
    if adaptive_weights:
        from physics.weights import GradNormSpeciesWeighter
        adaptive_weighter = GradNormSpeciesWeighter(
            n_species=5, update_every=adaptive_every, ema=adaptive_ema
        )

    if resume and WEIGHTS_PATH.exists() and resume_blob is None:
        model = PhysicsInformedLSTM.load(WEIGHTS_PATH, map_location=device).to(
            device=device, dtype=dtype
        )
        print(f"Loaded resume weights from {WEIGHTS_PATH}")
    elif resume and resume_blob is not None and "model_state_dict" in resume_blob:
        cfg = resume_blob.get("model_config") or {
            "hidden_dim": hidden_dim,
            "num_layers": 2,
            "n_energy_fourier": n_fourier,
            "n_time_fourier": n_time_fourier,
            "hard_ic": hard_ic,
        }
        model = PhysicsInformedLSTM(**{
            k: cfg[k] for k in (
                "hidden_dim", "num_layers", "n_energy_fourier",
                "n_time_fourier", "hard_ic",
            ) if k in cfg
        }).to(device=device, dtype=dtype)
        model.load_state_dict(resume_blob["model_state_dict"])
        model.config.update({k: cfg[k] for k in cfg if k in model.config})
        print(f"Loaded resume model state from {STATE_PATH}")
    else:
        model = PhysicsInformedLSTM(
            hidden_dim=hidden_dim,
            num_layers=2,
            n_energy_fourier=n_fourier,
            n_time_fourier=n_time_fourier,
            hard_ic=hard_ic,
        ).to(device=device, dtype=dtype)

    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    if resume_blob is not None and "optimizer_state_dict" in resume_blob:
        optimizer.load_state_dict(resume_blob["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group.get("lr", 1e-3))
        print("Restored optimizer state")
    if resume_blob is not None and "scheduler_state_dict" in resume_blob:
        scheduler.load_state_dict(resume_blob["scheduler_state_dict"])
        print("Restored scheduler state")
    elif start_epoch > 1:
        # Fast-forward cosine schedule to match absolute epoch (no optimizer momentum).
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"Advanced LR schedule to epoch {start_epoch - 1}, lr={scheduler.get_last_lr()}")

    teacher = None
    if use_distill:
        teacher = V2Teacher(V2_WEIGHTS, device=device, dtype=dtype)
        print(f"Distillation teacher loaded from {V2_WEIGHTS}")

    pretrain_epochs = int(pretrain_frac * epochs)
    t0 = time.time()
    ckpt_scenarios = canonical_heldout_scenarios(n_val, seed=2025, scenario_version=scenario_version)

    def _save_train_state(epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_med": best_med,
                "best_epoch": best_epoch,
                "model_config": dict(model.config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            STATE_PATH,
        )

    def _endpoint_score(scenarios) -> tuple[float, dict[str, float]]:
        def _pred(sc):
            return pilstm_endpoint(model, sc, device=device, dtype=dtype, n_steps=n_steps)

        errs = evaluate_endpoints(scenarios, _pred)
        med = ac225_endpoint_median(errs)
        return med, {"endpoint_ac225_median_rel": med}

    def _checkpoint_score() -> tuple[float, dict[str, float]]:
        """Metric for best checkpoint (lower is better)."""
        if ckpt_metric == "traj_median":
            va = evaluate(model, val_loader, device, dtype)
            return va["ac225_median_rel"], {"traj_ac225_median_rel": va["ac225_median_rel"]}
        return _endpoint_score(ckpt_scenarios)

    for epoch in range(start_epoch, epochs + 1):
        # Curriculum anneal: switch to the next stiffness stage when crossing
        # the stage boundary (rebuilds the training set at the new rate scale).
        if curriculum_scales:
            stage_idx = min(
                len(curriculum_scales) - 1,
                (epoch - 1) * len(curriculum_scales) // max(epochs, 1),
            )
            stage_scale = curriculum_scales[stage_idx]
            if stage_scale != current_scale:
                current_scale = stage_scale
                train_loader, _, _, _ = _make_loaders(current_scale)
                print(
                    f"Curriculum stage {stage_idx + 1}/{len(curriculum_scales)}: "
                    f"rate_scale={current_scale:g} (epoch {epoch})"
                )
        causal_progress = min(1.0, epoch / max(1, int(0.6 * epochs)))
        # P2-9: physics/distill-first pretrain de-emphasizes the raw data term early.
        if epoch <= pretrain_epochs:
            phase_data_w = 0.3 * data_w
            phase_phys_w = phys_w * 1.5
        else:
            phase_data_w = data_w
            phase_phys_w = phys_w

        tr = train_one_epoch(
            model, train_loader, optimizer, device, dtype,
            data_w=phase_data_w, phys_w=phase_phys_w, mass_w=mass_w,
            distill_w=_scheduled_distill_w(epoch, epochs, distill_w, pretrain_frac),
            overshoot_w=overshoot_w, log_weight=log_weight,
            teacher=teacher, progress=causal_progress, causal_eps=causal_eps,
            grad_balance=grad_balance, loss_mode=loss_mode,
            adaptive_weighter=adaptive_weighter, epoch=epoch,
            rate_scale=current_scale,
        )
        scheduler.step()

        # Colab Run C (0.51%): eval every epoch. Kaggle/Vast: raise PILSTM_EVAL_EVERY.
        do_eval = epoch == start_epoch or epoch % eval_every == 0 or epoch == epochs
        if do_eval:
            va = evaluate(model, val_loader, device, dtype)
            med, _ = _checkpoint_score()
            if med < best_med:
                best_med = med
                best_epoch = epoch
                model.save(WEIGHTS_PATH)
        else:
            va = {"ac225_median_rel": float("nan")}

        if epoch == start_epoch or epoch % log_every == 0 or epoch == epochs:
            val_str = (
                f"{va['ac225_median_rel']:.4f}"
                if va["ac225_median_rel"] == va["ac225_median_rel"]
                else "n/a"
            )
            print(
                f"epoch {epoch:5d}/{epochs} | loss={tr['loss']:.4e} "
                f"data={tr['data']:.4e} phys={tr['physics']:.4e} | "
                f"val Ac225 med={val_str} "
                f"best={best_med:.4f}@{best_epoch}"
            )

        # Live progress for Vast/Colab/Kaggle monitoring
        live = {
            "epoch": epoch,
            "total_epochs": epochs,
            "loss": tr["loss"],
            "best_epoch": best_epoch,
            "best_score": best_med if best_med < float("inf") else None,
            "val_ac225_median_rel": (
                float(va["ac225_median_rel"])
                if va["ac225_median_rel"] == va["ac225_median_rel"]
                else None
            ),
            "device": str(device),
            "elapsed_s": time.time() - t0,
            "eval_every": eval_every,
            "resumed_from": start_epoch if resume else None,
        }
        PROGRESS_PATH.write_text(json.dumps(live, indent=2), encoding="utf-8")

        if epoch % state_every == 0 or epoch == epochs:
            _save_train_state(epoch)

        if early_stop > 0.0 and best_med < early_stop:
            print(
                f"Early stop at epoch {epoch}: best={best_med:.4f} < {early_stop}"
            )
            _save_train_state(epoch)
            break

    # Reload best checkpoint, then optional L-BFGS polish (checkpoint again if better).
    if WEIGHTS_PATH.exists():
        model = PhysicsInformedLSTM.load(WEIGHTS_PATH, map_location=device).to(device=device, dtype=dtype)

    if lbfgs_iter > 0:
        pre, _ = _checkpoint_score()
        _lbfgs_polish(model, train_loader, device, dtype, data_w=data_w, log_weight=log_weight, max_iter=lbfgs_iter)
        post, _ = _checkpoint_score()
        print(f"L-BFGS polish: ckpt({ckpt_metric}) {pre:.4f} -> {post:.4f}")
        # Data-only polish often worsens endpoint_ac225; never overwrite a better best.
        if post < best_med and post < pre:
            best_med = post
            model.save(WEIGHTS_PATH)
            print(f"L-BFGS polish: accepted_polish (saved best ckpt={best_med:.4f})")
        else:
            model = PhysicsInformedLSTM.load(WEIGHTS_PATH, map_location=device).to(device=device, dtype=dtype)
            print(
                f"L-BFGS polish: kept_best (rejected polish; "
                f"best={best_med:.4f}, pre={pre:.4f}, post={post:.4f})"
            )

    te = evaluate(model, test_loader, device, dtype)
    test_ckpt_sc = canonical_heldout_scenarios(n_test, seed=2024, scenario_version=scenario_version)

    def _test_pred(sc):
        return pilstm_endpoint(model, sc, device=device, dtype=dtype, n_steps=n_steps)

    endpoint_errs = evaluate_endpoints(test_ckpt_sc, _test_pred)
    endpoint_medians = {k: float(np.median(v)) for k, v in endpoint_errs.items()}
    test_ep = endpoint_medians.get("Ac-225", float("nan"))
    summary = {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "seed": seed,
        "physics_loss_mode": loss_mode,
        "scenario_version": scenario_version,
        "checkpoint_metric": ckpt_metric,
        "best_checkpoint_score": best_med,
        "best_val_ac225_median_rel": best_med if ckpt_metric == "traj_median" else None,
        "test_ac225_median_rel": te["ac225_median_rel"],
        "test_endpoint_ac225_median_rel": endpoint_medians.get("Ac-225"),
        "test_endpoint_species_median_rel": endpoint_medians,
        "test_species_median_rel": {k: v for k, v in te.items() if k.endswith("_median_rel")},
        "config": {
            "n_train": n_train, "n_steps": n_steps, "hidden_dim": hidden_dim,
            "n_fourier": n_fourier, "n_time_fourier": n_time_fourier, "hard_ic": hard_ic,
            "distill": use_distill,
            "seed": seed,
            "physics_loss_mode": loss_mode,
            "scenario_version": scenario_version,
            "adaptive_weights": adaptive_weights,
            "adaptive_weight_history": (adaptive_weighter.history if adaptive_weighter else None),
            "curriculum_scales": curriculum_scales or None,
            "data_w": data_w, "phys_w": phys_w, "mass_w": mass_w,
            "distill_w": distill_w, "overshoot_w": overshoot_w,
            "pretrain_frac": pretrain_frac, "grad_balance": grad_balance,
            "lbfgs_iter": lbfgs_iter, "ckpt_metric": ckpt_metric,
            "dense_steps": dense_steps, "eval_every": eval_every,
            "early_stop": early_stop,
        },
        "weights": str(WEIGHTS_PATH),
        "elapsed_s": time.time() - t0,
        "quick_mode": quick,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Test Ac-225 median rel error: {test_ep:.4f}")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
