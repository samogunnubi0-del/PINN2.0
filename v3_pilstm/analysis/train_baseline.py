"""Train + evaluate the vanilla LSTM baseline (fair-baseline ablation, P0-2.4).

Matched protocol vs the PI-LSTM (v3):
  * same dataset (build_dataloaders, same seed, same scenario_version)
  * same epoch budget (default 6000) and same data loss term
  * parameter budget matched to the PI-LSTM within ±10%
  * NO physics loss, NO hard IC, NO distillation, NO mass/overshoot penalties
  * checkpoint selection on the SAME canonical held-out endpoint metric
  * evaluated on the SAME canonical 22 held-out scenarios (seed 2024)

Usage (from project root):
    python v3_pilstm/analysis/train_baseline.py
    BASELINE_QUICK=1 python v3_pilstm/analysis/train_baseline.py   # smoke
    BASELINE_EPOCHS=6000 SCENARIO_VERSION=v2 python v3_pilstm/analysis/train_baseline.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3_pilstm"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

from baseline_lstm import BaselineLSTM, build_matched_baseline, count_parameters  # noqa: E402
from data.trajectory_dataset import build_dataloaders, canonical_heldout_scenarios  # noqa: E402
from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402
from physics.integrated_loss import data_trajectory_loss  # noqa: E402
from analysis.endpoint_eval import (  # noqa: E402
    ac225_endpoint_median,
    baseline_endpoint,
    evaluate_endpoints,
)
from seed_utils import seed_everything  # noqa: E402

PILSTM_WEIGHTS = V3_ROOT / "weights" / "pi_lstm_best.pth"


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


def _pilstm_target_params() -> tuple[int, dict]:
    """Parameter budget of the PI-LSTM this baseline must match.

    Uses the shipped checkpoint config when available (the real comparator),
    else the default v3 architecture (256/8/16).
    """
    if PILSTM_WEIGHTS.exists():
        try:
            model = PhysicsInformedLSTM.load(PILSTM_WEIGHTS, map_location="cpu")
            return count_parameters(model), dict(model.config)
        except Exception:
            pass
    model = PhysicsInformedLSTM()
    return count_parameters(model), dict(model.config)


@torch.no_grad()
def evaluate_traj(model, loader, device, dtype) -> dict[str, float]:
    """Full-trajectory per-species median relative error (same as PI-LSTM eval)."""
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


def main() -> None:
    seed = seed_everything()
    quick = _env_flag("BASELINE_QUICK", False)
    epochs = 5 if quick else _env_int("BASELINE_EPOCHS", 6000)
    n_train = 24 if quick else _env_int("BASELINE_N_TRAIN", 1400)
    n_val = 10 if quick else _env_int("BASELINE_N_VAL", 22)
    n_test = 10 if quick else _env_int("BASELINE_N_TEST", 22)
    batch_size = 8 if quick else _env_int("BASELINE_BATCH", 16)
    n_steps = 30 if quick else _env_int("BASELINE_N_STEPS", 64)
    lr = _env_float("BASELINE_LR", 1e-3)
    log_weight = _env_float("BASELINE_LOG_WEIGHT", 2.0)  # same data term as PI-LSTM
    eval_every = max(1, _env_int("BASELINE_EVAL_EVERY", 25))
    log_every = max(1, _env_int("BASELINE_LOG_EVERY", 25))
    scenario_version = os.environ.get("SCENARIO_VERSION", "v1").strip().lower()

    weights_path = Path(os.environ.get(
        "BASELINE_WEIGHTS_PATH", str(V3_ROOT / "weights" / "baseline_lstm_best.pth")))
    results_path = Path(os.environ.get(
        "BASELINE_RESULTS_PATH", str(V3_ROOT / "results" / "baseline_lstm_summary.json")))
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    target_params, pilstm_cfg = _pilstm_target_params()
    if os.environ.get("BASELINE_HIDDEN", "").strip():
        model = BaselineLSTM(hidden_dim=_env_int("BASELINE_HIDDEN", 264))
        n_params = count_parameters(model)
    else:
        model, n_params = build_matched_baseline(target_params)
    match_frac = abs(n_params - target_params) / max(target_params, 1)
    model = model.to(device=device, dtype=dtype)

    print(
        f"Baseline LSTM training | device={device} epochs={epochs} n_train={n_train} "
        f"n_steps={n_steps} hidden={model.config['hidden_dim']} params={n_params} "
        f"target_params={target_params} match_delta={match_frac:.2%} "
        f"seed={seed} scenario_version={scenario_version}"
    )

    train_loader, val_loader, test_loader, _ = build_dataloaders(
        n_train=n_train, n_val=n_val, n_test=n_test, n_steps=n_steps,
        batch_size=batch_size, seed=seed,
        scenario_version=scenario_version, loader_seed=seed,
    )

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)
    ckpt_scenarios = canonical_heldout_scenarios(n_val, seed=2025, scenario_version=scenario_version)

    best_med = float("inf")
    best_epoch = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in train_loader:
            batch = _batch_to_model(batch, device, dtype)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["features"])
            loss = data_trajectory_loss(pred, batch["target"], log_weight=log_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot += float(loss.detach())
            nb += 1
        scheduler.step()

        do_eval = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        if do_eval:
            errs = evaluate_endpoints(
                ckpt_scenarios,
                lambda sc: baseline_endpoint(model, sc, device=device, dtype=dtype, n_steps=n_steps),
            )
            med = ac225_endpoint_median(errs)
            if med < best_med:
                best_med = med
                best_epoch = epoch
                model.save(weights_path)
        else:
            med = float("nan")

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            med_str = f"{med:.4f}" if med == med else "n/a"
            print(
                f"epoch {epoch:5d}/{epochs} | data_loss={tot / max(nb, 1):.4e} | "
                f"val endpoint Ac225 med={med_str} best={best_med:.4f}@{best_epoch}"
            )

    # Final evaluation on the canonical held-out test set (same 22 as compare_models).
    if weights_path.exists():
        model = BaselineLSTM.load(weights_path, map_location=device).to(device=device, dtype=dtype)
    te = evaluate_traj(model, test_loader, device, dtype)
    test_sc = canonical_heldout_scenarios(n_test, seed=2024, scenario_version=scenario_version)
    endpoint_errs = evaluate_endpoints(
        test_sc,
        lambda sc: baseline_endpoint(model, sc, device=device, dtype=dtype, n_steps=n_steps),
    )
    endpoint_medians = {k: float(np.median(v)) for k, v in endpoint_errs.items()}

    summary = {
        "model": "BaselineLSTM (vanilla, no physics loss, no hard IC, no distillation)",
        "seed": seed,
        "scenario_version": scenario_version,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_endpoint_ac225_median_rel": best_med,
        "test_endpoint_species_median_rel": endpoint_medians,
        "test_species_median_rel": {k: v for k, v in te.items() if k.endswith("_median_rel")},
        "param_budget": {
            "baseline_params": n_params,
            "pilstm_params": target_params,
            "match_delta_fraction": match_frac,
            "within_10pct": bool(match_frac <= 0.10),
            "pilstm_config": pilstm_cfg,
        },
        "config": {
            "n_train": n_train, "n_steps": n_steps, "batch_size": batch_size,
            "hidden_dim": model.config["hidden_dim"], "lr": lr,
            "log_weight": log_weight, "eval_every": eval_every,
        },
        "weights": str(weights_path),
        "elapsed_s": time.time() - t0,
        "quick_mode": quick,
    }
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Test endpoint Ac-225 median rel: {endpoint_medians.get('Ac-225'):.4f}")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
