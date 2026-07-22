"""
Additive speed harness for frozen Results-6 PI-LSTM (no teacher retrain).

Measures eager vs batched vs optional torch.compile on the same 22 held-out
scenarios. Gates numeric match to eager (Ac-225 median within 0.1 pt).

Usage (from project root):
    python v3_pilstm/scripts/speed_harness.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3_pilstm"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

from analysis.endpoint_eval import (  # noqa: E402
    SPECIES,
    evaluate_endpoints,
    median_species_errors,
    pilstm_endpoint,
)
from data.trajectory_dataset import canonical_heldout_scenarios, integrate_scenario  # noqa: E402
from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402
from pinn_model import DEFAULT_PHI_SCALE, neutron_energy_ev_to_feature_numpy  # noqa: E402
from analysis.endpoint_eval import SCALES  # noqa: E402

WEIGHTS = V3_ROOT / "weights" / "pi_lstm_best.pth"
OUT_JSON = V3_ROOT / "results" / "speed_harness.json"
N_STEPS = int(os.environ.get("PILSTM_N_STEPS", "64"))
WARMUP = int(os.environ.get("SPEED_WARMUP", "3"))
REPEATS = int(os.environ.get("SPEED_REPEATS", "5"))
AC225_GATE = 0.001  # 0.1 percentage point absolute on median rel error


def _features_for_scenario(sc, n_steps: int) -> np.ndarray:
    t_norm, _ = integrate_scenario(sc, n_steps=n_steps)
    e_feat = float(neutron_energy_ev_to_feature_numpy(sc.energy_ev))
    ic_norm = sc.ic / SCALES
    seq_len = len(t_norm)
    feats = np.zeros((1, seq_len, 8), dtype=np.float32)
    for k in range(seq_len):
        feats[0, k, 0] = t_norm[k]
        feats[0, k, 1] = sc.phi / DEFAULT_PHI_SCALE
        feats[0, k, 2] = e_feat
        feats[0, k, 3:8] = ic_norm
    return feats


def _batch_features(scenarios, n_steps: int) -> torch.Tensor:
    # Pad/truncate to common length (integrate_scenario uses fixed n_steps).
    mats = [_features_for_scenario(sc, n_steps) for sc in scenarios]
    return torch.from_numpy(np.concatenate(mats, axis=0))


def _endpoint_from_traj(traj: torch.Tensor) -> np.ndarray:
    # traj: (B, T, 5) normalized -> physical endpoint
    return (traj[:, -1, :].detach().cpu().numpy() * SCALES)


def _time_ms(fn, repeats: int) -> float:
    # Warmup outside.
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(times))


def main() -> None:
    device = torch.device("cuda" if os.environ.get("SPEED_CUDA", "0") == "1" and torch.cuda.is_available() else "cpu")
    use_f64 = os.environ.get("PILSTM_FLOAT64", "0").lower() in ("1", "true", "yes")
    dtype = torch.float64 if use_f64 else torch.float32

    scenarios = canonical_heldout_scenarios(22, seed=2024)
    model = PhysicsInformedLSTM.load(WEIGHTS, map_location=device)
    model.to(device=device, dtype=dtype).eval()

    def eager_loop():
        outs = []
        with torch.inference_mode():
            for sc in scenarios:
                outs.append(pilstm_endpoint(model, sc, device=device, dtype=dtype, n_steps=N_STEPS))
        return outs

    for _ in range(WARMUP):
        eager_loop()

    eager_ms = _time_ms(eager_loop, REPEATS)
    eager_outs = eager_loop()
    eager_errs = {sp: [] for sp in SPECIES}
    from analysis.endpoint_eval import ode_endpoint, rel_err
    for sc, pred in zip(scenarios, eager_outs):
        e = rel_err(pred, ode_endpoint(sc))
        for i, sp in enumerate(SPECIES):
            eager_errs[sp].append(float(e[i]))
    eager_med = median_species_errors(eager_errs)
    eager_ac = eager_med["Ac-225"]

    batch_x = _batch_features(scenarios, N_STEPS).to(device=device, dtype=dtype)

    def batched_forward():
        with torch.inference_mode():
            return model(batch_x)

    for _ in range(WARMUP):
        batched_forward()
    batch_ms = _time_ms(batched_forward, REPEATS)
    with torch.inference_mode():
        batch_traj = batched_forward()
    batch_preds = _endpoint_from_traj(batch_traj)
    # numeric gate vs eager endpoints
    max_abs = float(np.max(np.abs(batch_preds - np.stack(eager_outs, axis=0))))
    eager_stack = np.stack(eager_outs, axis=0)
    rel_delta = np.abs(batch_preds - eager_stack) / np.maximum(np.abs(eager_stack), 1e-30)
    max_rel = float(np.max(rel_delta))
    batch_errs = {sp: [] for sp in SPECIES}
    from analysis.endpoint_eval import ode_endpoint, rel_err
    for sc, pred in zip(scenarios, batch_preds):
        e = rel_err(pred, ode_endpoint(sc))
        for i, sp in enumerate(SPECIES):
            batch_errs[sp].append(float(e[i]))
    batch_med = median_species_errors(batch_errs)
    batch_ac = batch_med["Ac-225"]
    batch_gate = abs(batch_ac - eager_ac) <= AC225_GATE

    compile_info = {"attempted": False, "ok": False, "ms_22": None, "ac225": None, "gate": None, "error": None}
    if os.environ.get("SPEED_COMPILE", "1") != "0":
        compile_info["attempted"] = True
        try:
            compiled = torch.compile(model, fullgraph=False)
            def compiled_forward():
                with torch.inference_mode():
                    return compiled(batch_x)
            for _ in range(max(WARMUP, 5)):
                compiled_forward()
            comp_ms = _time_ms(compiled_forward, REPEATS)
            with torch.inference_mode():
                comp_traj = compiled_forward()
            comp_preds = _endpoint_from_traj(comp_traj)
            comp_errs = {sp: [] for sp in SPECIES}
            for sc, pred in zip(scenarios, comp_preds):
                e = rel_err(pred, ode_endpoint(sc))
                for i, sp in enumerate(SPECIES):
                    comp_errs[sp].append(float(e[i]))
            comp_med = median_species_errors(comp_errs)
            comp_ac = comp_med["Ac-225"]
            compile_info.update({
                "ok": True,
                "ms_22": comp_ms,
                "ms_per_scenario": comp_ms / len(scenarios),
                "ac225": comp_ac,
                "gate": abs(comp_ac - eager_ac) <= AC225_GATE,
                "max_abs_endpoint_delta_vs_eager": float(np.max(np.abs(comp_preds - np.stack(eager_outs, axis=0)))),
                "speedup_vs_eager": (eager_ms / comp_ms) if comp_ms > 0 else None,
            })
        except Exception as exc:  # noqa: BLE001
            compile_info["error"] = f"{type(exc).__name__}: {exc}"

    summary = {
        "weights": str(WEIGHTS),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "n_scenarios": len(scenarios),
        "n_steps": N_STEPS,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "eager": {
            "ms_22": eager_ms,
            "ms_per_scenario": eager_ms / len(scenarios),
            "ac225_median_rel": eager_ac,
            "species_median_rel": eager_med,
        },
        "batched": {
            "ms_22": batch_ms,
            "ms_per_scenario": batch_ms / len(scenarios),
            "ac225_median_rel": batch_ac,
            "gate_vs_eager_ac225": batch_gate,
            "max_abs_endpoint_delta_vs_eager": max_abs,
            "max_rel_endpoint_delta_vs_eager": max_rel,
            "speedup_vs_eager": (eager_ms / batch_ms) if batch_ms > 0 else None,
            "species_median_rel": batch_med,
        },
        "compile": compile_info,
        "kd_student": {
            "status": "not_trained",
            "note": "Teacher frozen per plan; optional KD student deferred (no teacher retrain).",
        },
        "gate_threshold_ac225_abs": AC225_GATE,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
