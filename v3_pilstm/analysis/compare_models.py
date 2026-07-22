"""
Compare IsotopePINN v2 (frozen) vs PI-LSTM v3 (integrated loss).

Usage (from project root):
    python v3_pilstm/analysis/compare_models.py
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

from pinn_model import (  # noqa: E402
    load_isotope_pinn_checkpoint,
)
from analysis.endpoint_eval import (  # noqa: E402
    SPECIES,
    baseline_endpoint,
    evaluate_endpoints,
    median_species_errors,
    ode_endpoint,
    pilstm_endpoint,
    v2_endpoint,
)
from baseline_lstm import BaselineLSTM  # noqa: E402
from data.trajectory_dataset import TrajectoryScenario, canonical_heldout_scenarios  # noqa: E402
from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402
from pinn_model import DEFAULT_PHI_SCALE, neutron_energy_ev_to_feature_numpy  # noqa: E402
from analysis.endpoint_eval import SCALES  # noqa: E402
from seed_utils import seed_everything  # noqa: E402

V2_WEIGHTS = PROJECT_ROOT / "weights" / "pinn_best_weights.pth"
PILSTM_WEIGHTS = V3_ROOT / "weights" / "pi_lstm_best.pth"
BASELINE_WEIGHTS = Path(os.environ.get(
    "BASELINE_WEIGHTS_PATH", str(V3_ROOT / "weights" / "baseline_lstm_best.pth")))
OUT_JSON = V3_ROOT / "results" / "compare_v2_pilstm.json"
N_STEPS = int(os.environ.get("PILSTM_N_STEPS", "64"))


def _high_flux_ra227_scenario() -> TrajectoryScenario:
    return TrajectoryScenario(
        phi=1.0e15,
        energy_ev=14.0e6,
        t_end_h=250.0,
        ic=np.array([1.0e22, 0.0, 0.0, 0.0, 0.0]),
        scenario_id=9999,
    )


def main() -> None:
    seed = seed_everything()
    device = torch.device("cpu")

    if not V2_WEIGHTS.exists():
        raise FileNotFoundError(f"v2 weights missing: {V2_WEIGHTS}")
    v2_model, _ = load_isotope_pinn_checkpoint(V2_WEIGHTS, map_location=device)
    v2_model.eval()

    use_float64 = os.environ.get("PILSTM_FLOAT64", "1").lower() in ("1", "true", "yes")
    pilstm_dtype = torch.float64 if use_float64 else torch.float32

    pilstm_trained = False
    pilstm_config = None
    if PILSTM_WEIGHTS.exists():
        blob = torch.load(PILSTM_WEIGHTS, map_location=device, weights_only=False)
        pilstm_config = blob.get("config") if isinstance(blob, dict) else None
        pilstm = PhysicsInformedLSTM.load(PILSTM_WEIGHTS, map_location=device)
        pilstm_trained = True
    else:
        pilstm = PhysicsInformedLSTM()
    pilstm.to(device=device, dtype=pilstm_dtype).eval()

    scenarios = canonical_heldout_scenarios(22, seed=2024)
    v2_errs = evaluate_endpoints(scenarios, lambda sc: v2_endpoint(v2_model, sc))
    pilstm_errs = None
    if pilstm_trained:
        pilstm_errs = evaluate_endpoints(
            scenarios,
            lambda sc: pilstm_endpoint(pilstm, sc, device=device, dtype=pilstm_dtype, n_steps=N_STEPS),
        )

    # Vanilla LSTM baseline (trained via v3_pilstm/analysis/train_baseline.py).
    baseline_trained = False
    baseline = None
    if BASELINE_WEIGHTS.exists():
        baseline = BaselineLSTM.load(BASELINE_WEIGHTS, map_location=device)
        baseline.to(device=device, dtype=pilstm_dtype).eval()
        baseline_trained = True
    baseline_errs = None
    if baseline_trained:
        baseline_errs = evaluate_endpoints(
            scenarios,
            lambda sc: baseline_endpoint(baseline, sc, device=device, dtype=pilstm_dtype, n_steps=N_STEPS),
        )

    # Full-trajectory Ac-225 for PI-LSTM (diagnostic).
    pilstm_traj_ac225: list[float] = []
    if pilstm_trained:
        from data.trajectory_dataset import integrate_scenario
        for sc in scenarios:
            t_norm, y_norm = integrate_scenario(sc, n_steps=N_STEPS)
            e_feat = float(neutron_energy_ev_to_feature_numpy(sc.energy_ev))
            ic_norm = sc.ic / SCALES
            seq_len = len(t_norm)
            feats = np.zeros((1, seq_len, 8), dtype=np.float32)
            for k in range(seq_len):
                feats[0, k, 0] = t_norm[k]
                feats[0, k, 1] = sc.phi / DEFAULT_PHI_SCALE
                feats[0, k, 2] = e_feat
                feats[0, k, 3:8] = ic_norm
            with torch.no_grad():
                traj = pilstm(torch.from_numpy(feats).to(device=device, dtype=pilstm_dtype)).numpy()[0]
            true_ac = y_norm[:, 2]
            pred_ac = traj[:, 2]
            m = true_ac > 1e-10
            if m.any():
                pilstm_traj_ac225.extend(
                    (np.abs(pred_ac[m] - true_ac[m]) / np.maximum(true_ac[m], 1e-12)).tolist()
                )

    t_v2 = 0.0
    t_pi = 0.0
    t_bl = 0.0
    for sc in scenarios:
        t0 = time.perf_counter()
        v2_endpoint(v2_model, sc)
        t_v2 += time.perf_counter() - t0
        if pilstm_trained:
            t0 = time.perf_counter()
            pilstm_endpoint(pilstm, sc, device=device, dtype=pilstm_dtype, n_steps=N_STEPS)
            t_pi += time.perf_counter() - t0
        if baseline_trained:
            t0 = time.perf_counter()
            baseline_endpoint(baseline, sc, device=device, dtype=pilstm_dtype, n_steps=N_STEPS)
            t_bl += time.perf_counter() - t0

    hf = _high_flux_ra227_scenario()
    ode_hf = ode_endpoint(hf)
    v2_hf = v2_endpoint(v2_model, hf)
    ra227_v2_overshoot = float(max(0.0, (v2_hf[3] - ode_hf[3]) / max(ode_hf[3], 1e-30)))

    ra227_pi_overshoot = None
    if pilstm_trained:
        pi_hf = pilstm_endpoint(pilstm, hf, device=device, dtype=pilstm_dtype, n_steps=N_STEPS)
        ra227_pi_overshoot = float(max(0.0, (pi_hf[3] - ode_hf[3]) / max(ode_hf[3], 1e-30)))

    v2_medians = median_species_errors(v2_errs)
    pi_medians = median_species_errors(pilstm_errs) if pilstm_errs else {}
    bl_medians = median_species_errors(baseline_errs) if baseline_errs else {}

    summary = {
        "seed": seed,
        "pilstm_weights_loaded": pilstm_trained,
        "baseline_weights_loaded": baseline_trained,
        "pilstm_checkpoint_has_config": pilstm_config is not None,
        "pilstm_checkpoint_config": pilstm_config,
        "n_scenarios": len(scenarios),
        "pilstm_ac225_full_traj_median_rel": (
            float(np.median(pilstm_traj_ac225)) if pilstm_traj_ac225 else None
        ),
        "species_median_rel_error": {
            name: {
                "v2": v2_medians[name],
                "pilstm": pi_medians.get(name) if pilstm_trained else None,
                "baseline_lstm": bl_medians.get(name) if baseline_trained else None,
            }
            for name in SPECIES
        },
        "high_flux_ra227_overshoot_fraction": {
            "v2": ra227_v2_overshoot,
            "pilstm": ra227_pi_overshoot,
        },
        "mean_inference_ms": {
            "v2": 1000.0 * t_v2 / len(scenarios),
            "pilstm": 1000.0 * t_pi / len(scenarios) if pilstm_trained else None,
            "baseline_lstm": 1000.0 * t_bl / len(scenarios) if baseline_trained else None,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
