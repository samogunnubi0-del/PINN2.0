"""Honest speed benchmark: v2 PINN vs v3 PI-LSTM (eager) vs Radau ODE.

Measures, on the canonical held-out scenarios:
  (a) single-scenario LATENCY (median ms per scenario, eager, one at a time)
  (b) batched THROUGHPUT (scenarios/s at several batch sizes; the ODE cannot
      batch — reported as sequential reference only)

No cherry-picking: eager per-scenario numbers are always reported, even where
the LSTM loses to v2 or the ODE. Results + machine info -> results/speed_benchmark.json.

Usage (from project root):
    python v3_pilstm/scripts/speed_benchmark.py
    SPEED_N_SCENARIOS=4 SPEED_BATCH_SIZES=1,8 SPEED_REPEATS=2 \
        python v3_pilstm/scripts/speed_benchmark.py   # quick smoke
"""
from __future__ import annotations

import json
import os
import platform
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
    SCALES,
    _features_for_scenario,
    ode_endpoint,
    pilstm_endpoint,
    v2_endpoint,
)
from data.trajectory_dataset import canonical_heldout_scenarios  # noqa: E402
from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402
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
from seed_utils import seed_everything  # noqa: E402

V2_WEIGHTS = PROJECT_ROOT / "weights" / "pinn_best_weights.pth"
PILSTM_WEIGHTS = V3_ROOT / "weights" / "pi_lstm_best.pth"
OUT_JSON = Path(os.environ.get(
    "SPEED_OUT_JSON", str(PROJECT_ROOT / "results" / "speed_benchmark.json")))

N_SCENARIOS = int(os.environ.get("SPEED_N_SCENARIOS", "22"))
N_STEPS = int(os.environ.get("PILSTM_N_STEPS", "64"))
REPEATS = int(os.environ.get("SPEED_REPEATS", "5"))
WARMUP = int(os.environ.get("SPEED_WARMUP", "2"))
BATCH_SIZES = [
    int(x) for x in os.environ.get("SPEED_BATCH_SIZES", "1,8,32,128").split(",") if x.strip()
]


def _machine_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() and os.environ.get("SPEED_CUDA", "0") == "1" else "cpu",
    }


def _time_median_ms(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(times))


def _v2_batch_inputs(scenarios) -> np.ndarray:
    rows = []
    for sc in scenarios:
        rows.append([
            sc.t_end_h / DEFAULT_T_REF_H,
            sc.phi / DEFAULT_PHI_SCALE,
            float(neutron_energy_ev_to_feature_numpy(sc.energy_ev)),
            sc.ic[0] / DEFAULT_N226_SCALE,
            sc.ic[1] / DEFAULT_N225_SCALE,
            sc.ic[2] / DEFAULT_NAC_SCALE,
            sc.ic[3] / DEFAULT_N227_SCALE,
            sc.ic[4] / DEFAULT_NAC227_SCALE,
        ])
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    seed = seed_everything()
    info = _machine_info()
    device = torch.device(info["device"])
    dtype = torch.float32
    scenarios = canonical_heldout_scenarios(N_SCENARIOS, seed=2024)

    if not V2_WEIGHTS.exists():
        raise FileNotFoundError(f"v2 weights missing: {V2_WEIGHTS}")
    v2_model, _ = load_isotope_pinn_checkpoint(V2_WEIGHTS, map_location=device)
    v2_model.eval()

    pilstm = None
    if PILSTM_WEIGHTS.exists():
        pilstm = PhysicsInformedLSTM.load(PILSTM_WEIGHTS, map_location=device)
        pilstm.to(device=device, dtype=dtype).eval()
    else:
        print(f"WARNING: PI-LSTM weights missing at {PILSTM_WEIGHTS}; skipping pilstm rows")

    # ---------------- (a) single-scenario latency --------------------------
    def lat_v2():
        for sc in scenarios:
            v2_endpoint(v2_model, sc)

    def lat_pilstm():
        for sc in scenarios:
            pilstm_endpoint(pilstm, sc, device=device, dtype=dtype, n_steps=N_STEPS)

    def lat_ode():
        for sc in scenarios:
            ode_endpoint(sc)

    for _ in range(WARMUP):
        lat_v2()
        if pilstm is not None:
            lat_pilstm()
    lat = {
        "v2_pinn_ms_per_scenario": _time_median_ms(lat_v2, REPEATS) / len(scenarios),
        "radau_ode_ms_per_scenario": _time_median_ms(lat_ode, max(1, REPEATS // 2)) / len(scenarios),
        "pilstm_eager_ms_per_scenario": (
            _time_median_ms(lat_pilstm, REPEATS) / len(scenarios) if pilstm is not None else None
        ),
    }

    # ---------------- (b) batched throughput -------------------------------
    v2_inputs = torch.from_numpy(_v2_batch_inputs(scenarios)).to(device=device)
    thr: dict[str, dict[str, float | None]] = {"v2_pinn": {}, "pilstm_eager": {}, "radau_ode": {}}
    for bs in BATCH_SIZES:
        reps = max(1, int(np.ceil(bs / len(scenarios))))
        x_v2 = v2_inputs.repeat((reps, 1))[:bs]

        def v2_batched():
            with torch.no_grad():
                v2_model(x_v2)

        for _ in range(WARMUP):
            v2_batched()
        ms = _time_median_ms(v2_batched, REPEATS)
        thr["v2_pinn"][str(bs)] = 1000.0 * bs / ms

        if pilstm is not None:
            mats = [_features_for_scenario(sc, N_STEPS) for sc in scenarios]
            base = torch.from_numpy(np.concatenate(mats, axis=0)).to(device=device, dtype=dtype)
            x_pi = base.repeat((reps, 1, 1))[:bs]

            def pi_batched():
                with torch.no_grad():
                    pilstm(x_pi)

            for _ in range(WARMUP):
                pi_batched()
            ms = _time_median_ms(pi_batched, REPEATS)
            thr["pilstm_eager"][str(bs)] = 1000.0 * bs / ms
        else:
            thr["pilstm_eager"][str(bs)] = None

        # ODE is inherently sequential — same number at every "batch size".
        thr["radau_ode"][str(bs)] = (
            1000.0 / lat["radau_ode_ms_per_scenario"]
            if lat["radau_ode_ms_per_scenario"]
            else None
        )

    summary = {
        "seed": seed,
        "machine": info,
        "n_scenarios": len(scenarios),
        "n_steps_pilstm": N_STEPS,
        "repeats": REPEATS,
        "warmup": WARMUP,
        "batch_sizes": BATCH_SIZES,
        "latency_ms_per_scenario": lat,
        "throughput_scenarios_per_s": thr,
        "speedup_vs_ode": {
            "v2_pinn_eager_latency": (
                lat["radau_ode_ms_per_scenario"] / lat["v2_pinn_ms_per_scenario"]
                if lat["v2_pinn_ms_per_scenario"]
                else None
            ),
            "pilstm_eager_latency": (
                lat["radau_ode_ms_per_scenario"] / lat["pilstm_eager_ms_per_scenario"]
                if pilstm is not None and lat["pilstm_eager_ms_per_scenario"]
                else None
            ),
            "pilstm_best_batch_throughput": (
                (max(v for v in thr["pilstm_eager"].values() if v is not None)
                 / (1000.0 / lat["radau_ode_ms_per_scenario"]))
                if pilstm is not None and any(v is not None for v in thr["pilstm_eager"].values())
                else None
            ),
        },
        "notes": [
            "Eager per-scenario latency is reported even where the LSTM loses; no cherry-picking.",
            "ODE (Radau, rtol=1e-9) is inherently sequential; its throughput is 1/latency at every batch size.",
            "Throughput gains for the surrogates come from batching scenarios into one forward pass.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
