"""End-to-end smoke validation for the 2026-07-18 PI-LSTM upgrade.

Checks (each recorded into v3_pilstm/results/smoke_20260718.json):
  a. loss_exact:      expmix residual ~0 on exact ODE trajectories (and
                      propagator == scipy matrix exponential); trap recorded
                      for comparison.
  b. reproducibility: same seed -> bit-identical model init, batch order, and
                      one training step; different seed -> different init.
  c. baseline_trains: BASELINE_QUICK subprocess trains a few epochs and writes
                      its summary JSON.
  d. speed_benchmark: quick subprocess run writes results JSON.
  e. pilstm_expmix_trains: tiny 2-epoch PI-LSTM run with PI_LSTM_LOSS=expmix
                      completes and writes a summary (temp paths; the real
                      checkpoints are never touched).
  f. checkpoint_compat: the pre-existing pi_lstm_best.pth still loads and
                      produces a finite endpoint prediction.

Run parts selectively (CI-friendly):
    python v3_pilstm/scripts/smoke_checks.py --parts loss,repro
    python v3_pilstm/scripts/smoke_checks.py                  # everything
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3_pilstm"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

OUT_JSON = Path(os.environ.get(
    "SMOKE_OUT_JSON", str(V3_ROOT / "results" / "smoke_20260718.json")))
TMP_DIR = V3_ROOT / "results" / "smoke_tmp"
PYTHON = sys.executable

ALL_PARTS = ["loss", "repro", "baseline", "speed", "pilstm", "compat",
             "jackknife", "adaptive", "curriculum", "ensemble"]


def _run_subprocess(env_extra: dict, script: str, timeout_s: int = 280) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, script],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return {
        "returncode": proc.returncode,
        "elapsed_s": time.time() - t0,
        "stdout_tail": proc.stdout.strip().splitlines()[-5:],
        "stderr_tail": proc.stderr.strip().splitlines()[-5:],
    }


def check_loss() -> dict:
    from scipy.linalg import expm

    from data.trajectory_dataset import canonical_heldout_scenarios, integrate_scenario
    from physics.bateman_rhs import R226_225, R225_AC, R226_227, R227_AC7
    from physics.integrated_loss import bateman_interval_propagate, integrated_physics_loss
    from pinn_model import (
        DEFAULT_LAMBDA_226_H,
        DEFAULT_LAMBDA_225_H,
        DEFAULT_LAMBDA_227_H,
        DEFAULT_LAMBDA_AC7_H,
        DEFAULT_LAMBDA_AC_H,
        DEFAULT_PHI_SCALE,
        DEFAULT_T_REF_H,
        neutron_energy_ev_to_feature_numpy,
    )

    # 1) propagator == scipy.linalg.expm on randomized rates/dt
    def A_matrix(k2, kg):
        lam0 = DEFAULT_LAMBDA_226_H + k2 + kg
        A = np.diag([-lam0, -DEFAULT_LAMBDA_225_H, -DEFAULT_LAMBDA_AC_H,
                     -DEFAULT_LAMBDA_227_H, -DEFAULT_LAMBDA_AC7_H])
        A[1, 0] = k2 * R226_225
        A[2, 1] = DEFAULT_LAMBDA_225_H * R225_AC
        A[3, 0] = kg * R226_227
        A[4, 3] = DEFAULT_LAMBDA_227_H * R227_AC7
        return A

    torch.manual_seed(0)
    max_rel = 0.0
    for k2, kg, dt in [(1e-7, 4.6e-5, 50.0), (2.7e-6, 1e-6, 0.02), (0.0, 0.0, 500.0)]:
        n0 = torch.rand(5, dtype=torch.float64)
        mine = bateman_interval_propagate(
            n0, torch.tensor(dt, dtype=torch.float64),
            torch.tensor(k2, dtype=torch.float64), torch.tensor(kg, dtype=torch.float64),
        ).numpy()
        ref = expm(A_matrix(k2, kg) * dt) @ n0.numpy()
        max_rel = max(max_rel, float(np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-30))))

    # 2) expmix ~ 0 on exact ODE trajectories (log-spaced grid, stiff included)
    per_mode = {"trap": [], "expmix": []}
    for sc in canonical_heldout_scenarios(6, seed=2024):
        t_norm, y_norm = integrate_scenario(sc, n_steps=64)
        traj = torch.from_numpy(y_norm[None].astype(np.float64))
        t = torch.from_numpy(t_norm[None].astype(np.float64))
        phi = torch.tensor([sc.phi / DEFAULT_PHI_SCALE], dtype=torch.float64)
        ef = torch.tensor([float(neutron_energy_ev_to_feature_numpy(sc.energy_ev))], dtype=torch.float64)
        for mode in per_mode:
            _, info = integrated_physics_loss(traj, t, phi, ef, mode=mode, t_ref_h=DEFAULT_T_REF_H)
            per_mode[mode].append(info["physics_mse"])

    expmix_max = float(max(per_mode["expmix"]))
    trap_max = float(max(per_mode["trap"]))
    return {
        "propagator_vs_scipy_expm_max_rel": max_rel,
        "expmix_physics_mse_on_exact_traj": per_mode["expmix"],
        "trap_physics_mse_on_exact_traj": per_mode["trap"],
        "expmix_max": expmix_max,
        "trap_max": trap_max,
        "pass": bool(max_rel < 1e-10 and expmix_max < 1e-8),
    }


def check_repro() -> dict:
    from seed_utils import seed_everything, seeded_torch_generator
    from models.pi_lstm import PhysicsInformedLSTM

    def _init_bytes():
        seed_everything(123)
        m = PhysicsInformedLSTM(hidden_dim=32, num_layers=2, n_energy_fourier=2, n_time_fourier=2)
        return m.state_dict()

    sd1, sd2 = _init_bytes(), _init_bytes()
    seed_everything(999)
    sd3 = PhysicsInformedLSTM(hidden_dim=32, num_layers=2, n_energy_fourier=2, n_time_fourier=2).state_dict()
    same_init = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
    diff_init = any(not torch.equal(sd1[k], sd3[k]) for k in sd1)

    # DataLoader shuffle order reproducibility
    g1 = seeded_torch_generator(42)
    order1 = torch.randperm(50, generator=g1).tolist()
    g2 = seeded_torch_generator(42)
    order2 = torch.randperm(50, generator=g2).tolist()

    # One deterministic training step, twice
    def _one_step():
        seed_everything(7)
        m = PhysicsInformedLSTM(hidden_dim=16, num_layers=1, n_energy_fourier=2, n_time_fourier=0)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x = torch.randn(2, 5, 8)
        loss = m(x).pow(2).mean()
        loss.backward()
        opt.step()
        return float(loss), torch.cat([p.flatten() for p in m.parameters()]).clone()

    loss_a, params_a = _one_step()
    loss_b, params_b = _one_step()
    bit_identical_step = bool(loss_a == loss_b and torch.equal(params_a, params_b))

    return {
        "same_seed_identical_init": bool(same_init),
        "different_seed_differs": bool(diff_init),
        "loader_order_reproducible": order1 == order2,
        "one_train_step_bit_identical": bit_identical_step,
        "deterministic_algorithms_warn_only": torch.are_deterministic_algorithms_enabled(),
        "pass": bool(same_init and diff_init and order1 == order2 and bit_identical_step),
    }


def check_baseline() -> dict:
    out = TMP_DIR / "baseline_summary_test.json"
    weights = TMP_DIR / "baseline_lstm_test.pth"
    res = _run_subprocess(
        {
            "BASELINE_QUICK": "1",
            "BASELINE_EVAL_EVERY": "2",
            "BASELINE_LOG_EVERY": "5",
            "BASELINE_WEIGHTS_PATH": str(weights),
            "BASELINE_RESULTS_PATH": str(out),
        },
        str(V3_ROOT / "analysis" / "train_baseline.py"),
    )
    ok = res["returncode"] == 0 and out.exists()
    payload = {}
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        ok = ok and "test_endpoint_species_median_rel" in payload
    return {
        "subprocess": res,
        "summary_exists": out.exists(),
        "epochs_ran": payload.get("epochs"),
        "param_budget": payload.get("param_budget", {}).get("within_10pct"),
        "pass": bool(ok),
        "note": "5-epoch smoke only; accuracy numbers are meaningless here by design.",
    }


def check_speed() -> dict:
    out = TMP_DIR / "speed_benchmark_test.json"
    res = _run_subprocess(
        {
            "SPEED_N_SCENARIOS": "4",
            "SPEED_BATCH_SIZES": "1,8",
            "SPEED_REPEATS": "2",
            "SPEED_WARMUP": "1",
            "SPEED_OUT_JSON": str(out),
        },
        str(V3_ROOT / "scripts" / "speed_benchmark.py"),
    )
    ok = res["returncode"] == 0 and out.exists()
    payload = {}
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        ok = ok and "latency_ms_per_scenario" in payload and "throughput_scenarios_per_s" in payload
    return {
        "subprocess": res,
        "json_exists": out.exists(),
        "latency_ms_per_scenario": payload.get("latency_ms_per_scenario"),
        "pass": bool(ok),
        "note": "tiny-n smoke timing (4 scenarios, 2 repeats) — not a publishable benchmark.",
    }


def check_pilstm_expmix() -> dict:
    real_weights = V3_ROOT / "weights" / "pi_lstm_best.pth"
    real_stat = real_weights.stat().st_mtime_ns if real_weights.exists() else None
    weights = TMP_DIR / "pi_lstm_expmix_smoke.pth"
    state = TMP_DIR / "pi_lstm_expmix_smoke_state.pth"
    progress = TMP_DIR / "pi_lstm_expmix_smoke_progress.json"
    results = TMP_DIR / "pi_lstm_expmix_smoke_summary.json"
    res = _run_subprocess(
        {
            "PI_LSTM_LOSS": "expmix",
            "PILSTM_EPOCHS": "2",
            "PILSTM_N_TRAIN": "8",
            "PILSTM_N_VAL": "4",
            "PILSTM_N_TEST": "4",
            "PILSTM_N_STEPS": "24",
            "PILSTM_BATCH": "4",
            "PILSTM_HIDDEN": "32",
            "PILSTM_FOURIER": "2",
            "PILSTM_TIME_FOURIER": "0",
            "PILSTM_LBFGS_ITER": "0",
            "PILSTM_EVAL_EVERY": "1",
            "PILSTM_LOG_EVERY": "1",
            "PILSTM_DISTILL": "0",
            "PILSTM_EARLY_STOP": "0",
            "PILSTM_WEIGHTS_PATH": str(weights),
            "PILSTM_STATE_PATH": str(state),
            "PILSTM_PROGRESS_PATH": str(progress),
            "PILSTM_RESULTS_PATH": str(results),
        },
        str(V3_ROOT / "train_pi_lstm.py"),
    )
    ok = res["returncode"] == 0 and results.exists()
    payload = {}
    if results.exists():
        payload = json.loads(results.read_text(encoding="utf-8"))
        ok = ok and payload.get("physics_loss_mode") == "expmix" and payload.get("seed") == 42
    real_after = real_weights.stat().st_mtime_ns if real_weights.exists() else None
    return {
        "subprocess": res,
        "summary_exists": results.exists(),
        "physics_loss_mode": payload.get("physics_loss_mode"),
        "seed_recorded": payload.get("seed"),
        "real_checkpoint_untouched": real_stat == real_after,
        "pass": bool(ok and real_stat == real_after),
        "note": "2-epoch tiny run; verifies the expmix training path end-to-end only.",
    }


def check_compat() -> dict:
    from analysis.endpoint_eval import pilstm_endpoint
    from data.trajectory_dataset import canonical_heldout_scenarios
    from models.pi_lstm import PhysicsInformedLSTM

    weights = V3_ROOT / "weights" / "pi_lstm_best.pth"
    if not weights.exists():
        return {"pass": False, "reason": f"missing {weights}"}
    model = PhysicsInformedLSTM.load(weights, map_location="cpu").eval()
    sc = canonical_heldout_scenarios(1, seed=2024)[0]
    pred = pilstm_endpoint(model, sc, device=torch.device("cpu"), dtype=torch.float32, n_steps=32)
    finite = bool(np.all(np.isfinite(pred)) and np.all(pred >= 0))
    return {
        "checkpoint": str(weights),
        "config": dict(model.config),
        "endpoint_pred_atoms": pred.tolist(),
        "finite_and_nonneg": finite,
        "pass": finite,
    }


def check_jackknife() -> dict:
    """CV+ and jackknife modes run and report coverage + width + stability."""
    outs = {}
    ok = True
    for mode, ncal, ntest in (("cv+", "12", "8"), ("jackknife", "10", "6")):
        out = TMP_DIR / f"conformal_{mode.replace('+', 'plus')}_test.json"
        res = _run_subprocess(
            {
                "CONFORMAL_MODE": mode,
                "CONFORMAL_MODEL": "v2",
                "CONFORMAL_N_CAL": ncal,
                "CONFORMAL_N_TEST": ntest,
                "CONFORMAL_N_BOOT": "20",
                "CONFORMAL_OUT_JSON": str(out),
            },
            str(V3_ROOT / "analysis" / "run_conformal_validation.py"),
        )
        good = res["returncode"] == 0 and out.exists()
        entry = {"subprocess": res, "json_exists": out.exists()}
        if out.exists():
            payload = json.loads(out.read_text(encoding="utf-8"))
            ac = payload.get("per_species", {}).get("Ac-225", {})
            entry["mode"] = payload.get("mode")
            entry["k_folds"] = payload.get("k_folds")
            entry["ac225_absolute_coverage"] = ac.get("absolute", {}).get("test_coverage")
            entry["ac225_width"] = ac.get("absolute", {}).get("median_relative_width")
            entry["ac225_degenerates_to_split"] = ac.get("absolute", {}).get(
                "degenerates_to_split_conformal"
            )
            good = good and entry["ac225_degenerates_to_split"] is True
        outs[mode] = entry
        ok = ok and good
    return {"modes": outs, "pass": bool(ok)}


def check_adaptive() -> dict:
    """PI_LSTM_ADAPTIVE_WEIGHTS=1 trains and records weight updates."""
    weights = TMP_DIR / "pi_lstm_adaptive_smoke.pth"
    results = TMP_DIR / "pi_lstm_adaptive_smoke_summary.json"
    res = _run_subprocess(
        {
            "PI_LSTM_ADAPTIVE_WEIGHTS": "1",
            "PI_LSTM_ADAPTIVE_EVERY": "1",
            "PI_LSTM_LOSS": "expmix",
            "PILSTM_EPOCHS": "2",
            "PILSTM_N_TRAIN": "8",
            "PILSTM_N_VAL": "4",
            "PILSTM_N_TEST": "4",
            "PILSTM_N_STEPS": "24",
            "PILSTM_BATCH": "4",
            "PILSTM_HIDDEN": "32",
            "PILSTM_FOURIER": "2",
            "PILSTM_TIME_FOURIER": "0",
            "PILSTM_LBFGS_ITER": "0",
            "PILSTM_EVAL_EVERY": "1",
            "PILSTM_LOG_EVERY": "1",
            "PILSTM_DISTILL": "0",
            "PILSTM_EARLY_STOP": "0",
            "PILSTM_WEIGHTS_PATH": str(weights),
            "PILSTM_STATE_PATH": str(TMP_DIR / "pi_lstm_adaptive_state.pth"),
            "PILSTM_PROGRESS_PATH": str(TMP_DIR / "pi_lstm_adaptive_progress.json"),
            "PILSTM_RESULTS_PATH": str(results),
        },
        str(V3_ROOT / "train_pi_lstm.py"),
    )
    ok = res["returncode"] == 0 and results.exists()
    history_len = 0
    if results.exists():
        payload = json.loads(results.read_text(encoding="utf-8"))
        hist = payload.get("config", {}).get("adaptive_weight_history") or []
        history_len = len(hist)
        ok = ok and payload.get("config", {}).get("adaptive_weights") is True and history_len >= 2
    return {
        "subprocess": res,
        "weight_updates_recorded": history_len,
        "pass": bool(ok),
    }


def check_curriculum() -> dict:
    """Curriculum: rate-scaled data differs, loss exact at scale, trainer anneals."""
    from data.trajectory_dataset import canonical_heldout_scenarios, integrate_scenario
    from physics.integrated_loss import integrated_physics_loss
    from pinn_model import DEFAULT_PHI_SCALE, DEFAULT_T_REF_H, neutron_energy_ev_to_feature_numpy

    sc = canonical_heldout_scenarios(2, seed=2024)[0]
    t1, y1 = integrate_scenario(sc, n_steps=24, rate_scale=1.0)
    t2, y2 = integrate_scenario(sc, n_steps=24, rate_scale=50.0)
    data_differs = not np.allclose(y1, y2)
    traj = torch.from_numpy(y2[None].astype(np.float64))
    t = torch.from_numpy(t2[None].astype(np.float64))
    phi = torch.tensor([sc.phi / DEFAULT_PHI_SCALE], dtype=torch.float64)
    ef = torch.tensor([float(neutron_energy_ev_to_feature_numpy(sc.energy_ev))], dtype=torch.float64)
    _, info = integrated_physics_loss(
        traj, t, phi, ef, mode="expmix", t_ref_h=DEFAULT_T_REF_H, rate_scale=50.0
    )
    loss_exact_at_scale = info["physics_mse"] < 1e-8

    weights = TMP_DIR / "pi_lstm_curriculum_smoke.pth"
    results = TMP_DIR / "pi_lstm_curriculum_smoke_summary.json"
    res = _run_subprocess(
        {
            "PI_LSTM_CURRICULUM": "10,1",
            "PI_LSTM_LOSS": "expmix",
            "PILSTM_EPOCHS": "2",
            "PILSTM_N_TRAIN": "8",
            "PILSTM_N_VAL": "4",
            "PILSTM_N_TEST": "4",
            "PILSTM_N_STEPS": "24",
            "PILSTM_BATCH": "4",
            "PILSTM_HIDDEN": "32",
            "PILSTM_FOURIER": "2",
            "PILSTM_TIME_FOURIER": "0",
            "PILSTM_LBFGS_ITER": "0",
            "PILSTM_EVAL_EVERY": "1",
            "PILSTM_LOG_EVERY": "1",
            "PILSTM_DISTILL": "0",
            "PILSTM_EARLY_STOP": "0",
            "PILSTM_WEIGHTS_PATH": str(weights),
            "PILSTM_STATE_PATH": str(TMP_DIR / "pi_lstm_curriculum_state.pth"),
            "PILSTM_PROGRESS_PATH": str(TMP_DIR / "pi_lstm_curriculum_progress.json"),
            "PILSTM_RESULTS_PATH": str(results),
        },
        str(V3_ROOT / "train_pi_lstm.py"),
    )
    stage_switched = any("Curriculum stage 2/2" in line for line in res["stdout_tail"]) or any(
        "Curriculum stage" in line for line in res["stdout_tail"]
    )
    ok = res["returncode"] == 0 and results.exists()
    if results.exists():
        payload = json.loads(results.read_text(encoding="utf-8"))
        ok = ok and payload.get("config", {}).get("curriculum_scales") == [10.0, 1.0]
    return {
        "destiffened_data_differs": bool(data_differs),
        "expmix_exact_at_scale_50": bool(loss_exact_at_scale),
        "subprocess": res,
        "curriculum_scales_recorded": (
            json.loads(results.read_text(encoding="utf-8")).get("config", {}).get("curriculum_scales")
            if results.exists() else None
        ),
        "pass": bool(ok and data_differs and loss_exact_at_scale),
    }


def check_ensemble() -> dict:
    """Tiny-K ensemble runner trains members and reports spread."""
    workdir = TMP_DIR / "ensemble"
    out = TMP_DIR / "ensemble_summary_test.json"
    res = _run_subprocess(
        {
            "ENSEMBLE_K": "2",
            "ENSEMBLE_SEED_BASE": "700",
            "ENSEMBLE_EPOCHS": "2",
            "ENSEMBLE_N_TRAIN": "8",
            "ENSEMBLE_N_VAL": "4",
            "ENSEMBLE_N_TEST": "4",
            "ENSEMBLE_N_STEPS": "24",
            "ENSEMBLE_BATCH": "4",
            "ENSEMBLE_HIDDEN": "32",
            "ENSEMBLE_FOURIER": "2",
            "ENSEMBLE_TIME_FOURIER": "0",
            "PILSTM_N_STEPS": "24",
            "PILSTM_LBFGS_ITER": "0",
            "PILSTM_EVAL_EVERY": "1",
            "PILSTM_LOG_EVERY": "1",
            "PILSTM_DISTILL": "0",
            "PILSTM_EARLY_STOP": "0",
            "ENSEMBLE_WORKDIR": str(workdir),
            "ENSEMBLE_OUT_JSON": str(out),
        },
        str(V3_ROOT / "scripts" / "run_ensemble.py"),
        timeout_s=280,
    )
    ok = res["returncode"] == 0 and out.exists()
    k_trained = 0
    if out.exists():
        payload = json.loads(out.read_text(encoding="utf-8"))
        k_trained = payload.get("k_trained", 0)
        ok = ok and k_trained >= 1 and "ensemble" in payload
    return {
        "subprocess": res,
        "k_trained": k_trained,
        "pass": bool(ok),
        "note": "K=2 x 2-epoch smoke; spread values meaningless by design.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", default=",".join(ALL_PARTS))
    args = parser.parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"smoke_date": "2026-07-18", "parts_requested": parts}
    if OUT_JSON.exists():
        try:
            results["previous"] = json.loads(OUT_JSON.read_text(encoding="utf-8")).get("checks")
        except Exception:
            pass

    checks: dict = {}
    t0 = time.time()
    if "loss" in parts:
        checks["loss_exact"] = check_loss()
    if "repro" in parts:
        checks["reproducibility"] = check_repro()
    if "compat" in parts:
        checks["checkpoint_compat"] = check_compat()
    if "baseline" in parts:
        checks["baseline_trains"] = check_baseline()
    if "pilstm" in parts:
        checks["pilstm_expmix_trains"] = check_pilstm_expmix()
    if "speed" in parts:
        checks["speed_benchmark"] = check_speed()
    if "jackknife" in parts:
        checks["jackknife_cvplus"] = check_jackknife()
    if "adaptive" in parts:
        checks["adaptive_weights"] = check_adaptive()
    if "curriculum" in parts:
        checks["stiffness_curriculum"] = check_curriculum()
    if "ensemble" in parts:
        checks["deep_ensemble"] = check_ensemble()

    # merge with previous parts if any
    merged = dict(results.get("previous") or {})
    merged.update(checks)
    results = {
        "smoke_date": "2026-07-18",
        "elapsed_s": time.time() - t0,
        "checks": merged,
        "all_pass": all(v.get("pass", False) for v in merged.values()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
