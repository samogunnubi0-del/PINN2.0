"""Deep-ensemble runner for PI-LSTM (K seeds -> endpoint spread UQ).

Trains K PI-LSTM instances with different PI_LSTM_SEED values (sequentially,
as subprocesses, reusing the deterministic seeding machinery), then evaluates
all members on the canonical held-out scenarios and reports the ensemble
spread of endpoint predictions — a classic deep-ensemble UQ complement to the
conformal intervals (Lakshminarayanan, Pritzel & Blundell 2017,
arXiv:1612.01474).

Usage (from project root):
    python v3_pilstm/scripts/run_ensemble.py                 # full K=5 x 6000 epochs
    ENSEMBLE_K=2 ENSEMBLE_EPOCHS=2 ENSEMBLE_N_TRAIN=8 \
    ENSEMBLE_N_STEPS=24 ENSEMBLE_HIDDEN=32 ENSEMBLE_FOURIER=2 ENSEMBLE_TIME_FOURIER=0 \
    ENSEMBLE_WORKDIR=v3_pilstm/results/smoke_tmp/ensemble \
        python v3_pilstm/scripts/run_ensemble.py             # tiny smoke
    ENSEMBLE_SKIP_TRAIN=1 python v3_pilstm/scripts/run_ensemble.py  # re-eval only
"""
from __future__ import annotations

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

from analysis.endpoint_eval import SPECIES, ode_endpoint, pilstm_endpoint  # noqa: E402
from data.trajectory_dataset import canonical_heldout_scenarios  # noqa: E402
from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402
from seed_utils import seed_everything  # noqa: E402

K = int(os.environ.get("ENSEMBLE_K", "5"))
SEED_BASE = int(os.environ.get("ENSEMBLE_SEED_BASE", "1000"))
EPOCHS = os.environ.get("ENSEMBLE_EPOCHS", "6000")
N_TEST = int(os.environ.get("ENSEMBLE_N_TEST", "22"))
N_STEPS = int(os.environ.get("PILSTM_N_STEPS", "64"))
SKIP_TRAIN = os.environ.get("ENSEMBLE_SKIP_TRAIN", "0").strip().lower() in ("1", "true", "yes")
WORKDIR = Path(os.environ.get("ENSEMBLE_WORKDIR", str(V3_ROOT / "weights" / "ensemble")))
OUT_JSON = Path(os.environ.get(
    "ENSEMBLE_OUT_JSON", str(V3_ROOT / "results" / "ensemble_summary.json")))


def _train_member(seed: int, weights: Path, results: Path) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "PI_LSTM_SEED": str(seed),
            "PILSTM_EPOCHS": EPOCHS,
            "PILSTM_WEIGHTS_PATH": str(weights),
            "PILSTM_STATE_PATH": str(weights.with_suffix(".state.pth")),
            "PILSTM_PROGRESS_PATH": str(results.with_suffix(".progress.json")),
            "PILSTM_RESULTS_PATH": str(results),
        }
    )
    # Pass through small-config overrides used by smoke runs.
    for src, dst in [
        ("ENSEMBLE_N_TRAIN", "PILSTM_N_TRAIN"),
        ("ENSEMBLE_N_VAL", "PILSTM_N_VAL"),
        ("ENSEMBLE_N_TEST", "PILSTM_N_TEST"),
        ("ENSEMBLE_N_STEPS", "PILSTM_N_STEPS"),
        ("ENSEMBLE_BATCH", "PILSTM_BATCH"),
        ("ENSEMBLE_HIDDEN", "PILSTM_HIDDEN"),
        ("ENSEMBLE_FOURIER", "PILSTM_FOURIER"),
        ("ENSEMBLE_TIME_FOURIER", "PILSTM_TIME_FOURIER"),
    ]:
        if os.environ.get(src, "").strip():
            env[dst] = os.environ[src].strip()
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(V3_ROOT / "train_pi_lstm.py")],
        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=3600 * 12,
    )
    return {
        "seed": seed,
        "returncode": proc.returncode,
        "elapsed_s": time.time() - t0,
        "stdout_tail": proc.stdout.strip().splitlines()[-3:],
        "stderr_tail": proc.stderr.strip().splitlines()[-3:],
    }


def main() -> None:
    seed_everything()
    WORKDIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dtype = torch.float32
    seeds = [SEED_BASE + i for i in range(K)]

    members = []
    for i, seed in enumerate(seeds):
        weights = WORKDIR / f"pi_lstm_seed{seed}.pth"
        results = WORKDIR / f"pi_lstm_seed{seed}.summary.json"
        train_info = None
        if not SKIP_TRAIN:
            print(f"[ensemble {i + 1}/{K}] training seed={seed} -> {weights}")
            train_info = _train_member(seed, weights, results)
            if train_info["returncode"] != 0:
                print(f"  member {seed} FAILED (rc={train_info['returncode']}); skipping")
                members.append({"seed": seed, "train": train_info, "ok": False})
                continue
        if not weights.exists():
            print(f"  weights missing for seed={seed}; skipping eval")
            members.append({"seed": seed, "train": train_info, "ok": False})
            continue
        members.append({"seed": seed, "train": train_info, "ok": True, "weights": str(weights)})

    ok_members = [m for m in members if m.get("ok")]
    test_sc = canonical_heldout_scenarios(N_TEST, seed=2024)

    preds = []
    per_member_median: list[float | None] = []
    for m in ok_members:
        model = PhysicsInformedLSTM.load(m["weights"], map_location=device)
        model.to(device=device, dtype=dtype).eval()
        ep = np.stack(
            [pilstm_endpoint(model, sc, device=device, dtype=dtype, n_steps=N_STEPS) for sc in test_sc]
        )
        preds.append(ep)
        m["endpoint_preds_shape"] = list(ep.shape)

    summary: dict = {
        "k_requested": K,
        "k_trained": len(ok_members),
        "seeds": seeds,
        "n_test_scenarios": len(test_sc),
        "members": members,
        "note": "Ensemble spread = std across seed members of endpoint predictions (atoms).",
    }

    if preds:
        P = np.stack(preds, axis=0)                      # (K, n_test, 5)
        truths = np.stack([ode_endpoint(sc) for sc in test_sc])  # (n_test, 5)
        mean_pred = P.mean(axis=0)
        std_pred = P.std(axis=0)
        ac = SPECIES.index("Ac-225")
        rel = lambda a: np.abs(a - truths) / np.maximum(np.abs(truths), 1e-30)  # noqa: E731
        for ep in preds:
            per_member_median.append(float(np.median(rel(ep)[:, ac])))
        ens_rel = rel(mean_pred)
        summary["ensemble"] = {
            "ac225_endpoint_spread_rel_median": float(
                np.median(std_pred[:, ac] / np.maximum(np.abs(mean_pred[:, ac]), 1e-30))
            ),
            "ac225_endpoint_spread_rel_p90": float(
                np.percentile(std_pred[:, ac] / np.maximum(np.abs(mean_pred[:, ac]), 1e-30), 90)
            ),
            "per_member_ac225_median_rel_error": per_member_median,
            "ensemble_mean_ac225_median_rel_error": float(np.median(ens_rel[:, ac])),
            "per_species_ensemble_mean_median_rel_error": {
                name: float(np.median(ens_rel[:, i])) for i, name in enumerate(SPECIES)
            },
        }
        print(json.dumps(summary["ensemble"], indent=2))

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
