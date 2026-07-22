"""
Split conformal validation for IsotopePINN v2 (and optional PI-LSTM).

Two modes (CONFORMAL_MODE env):
  * "legacy" (default): exact pre-2026-07-18 behavior — calibration/test split
    from the canonical held-out scenarios (seed 2024), first n_cal -> cal,
    remainder -> test (n=11/11). Keep for reproducing old results.
  * "large": statistically meaningful n — fresh seeded scenario pools drawn
    from the SAME regime distribution but DISJOINT id ranges and seeds:
    >=100 calibration + >=100 test scenarios (CONFORMAL_N_CAL /
    CONFORMAL_N_TEST, CONFORMAL_CAL_SEED / CONFORMAL_TEST_SEED).
  * "jackknife" / "cv+": frozen-checkpoint Jackknife+ (leave-one-out folds) /
    CV+ (K=5 folds) with bootstrapped residual quantiles, per Barber, Candes,
    Ramdas & Tibshirani 2021 (arXiv:1905.02928). With a frozen model the
    interval itself degenerates to split conformal (verified + reported); the
    fold/bootstrap machinery reports interval-WIDTH STABILITY. Scenario pools
    use the same seeded generation as "large". See v3_pilstm/uq/jackknife_plus.py
    for exactly which coverage guarantee applies.

All modes report per-species coverage AND median relative interval width.

Usage:
    python v3_pilstm/analysis/run_conformal_validation.py
    CONFORMAL_MODEL=pilstm python v3_pilstm/analysis/run_conformal_validation.py
    CONFORMAL_MODE=large CONFORMAL_MODEL=pilstm \
        python v3_pilstm/analysis/run_conformal_validation.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3_pilstm"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

from data.trajectory_dataset import _sample_scenarios, canonical_heldout_scenarios  # noqa: E402
from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402
from pinn_model import load_isotope_pinn_checkpoint  # noqa: E402
from physics.conformal import MultiSpeciesConformal, SplitConformalRegressor  # noqa: E402
from analysis.endpoint_eval import (  # noqa: E402
    SPECIES,
    AC225_IDX,
    ode_endpoint,
    pilstm_endpoint,
    v2_endpoint,
)
from seed_utils import seed_everything  # noqa: E402

V2_WEIGHTS = PROJECT_ROOT / "weights" / "pinn_best_weights.pth"
PILSTM_WEIGHTS = V3_ROOT / "weights" / "pi_lstm_best.pth"
MODE = os.environ.get("CONFORMAL_MODE", "legacy").strip().lower()
_DEFAULT_OUT = {
    "legacy": V3_ROOT / "results" / "conformal_validation.json",
    "large": V3_ROOT / "results" / "conformal_validation_large.json",
    "jackknife": V3_ROOT / "results" / "conformal_validation_jackknife.json",
    "cv+": V3_ROOT / "results" / "conformal_validation_cvplus.json",
}.get(MODE, V3_ROOT / "results" / f"conformal_validation_{MODE}.json")
OUT_JSON = Path(os.environ.get("CONFORMAL_OUT_JSON", str(_DEFAULT_OUT)))
N_HELDOUT = int(os.environ.get("CONFORMAL_N_SCENARIOS", "22"))
_POOLED_MODES = ("large", "jackknife", "cv+")
N_CAL = int(os.environ.get("CONFORMAL_N_CAL", "11" if MODE == "legacy" else "100"))
N_TEST = int(os.environ.get("CONFORMAL_N_TEST", "100"))
CAL_SEED = int(os.environ.get("CONFORMAL_CAL_SEED", "77001"))
TEST_SEED = int(os.environ.get("CONFORMAL_TEST_SEED", "88001"))
CV_K_FOLDS = int(os.environ.get("CONFORMAL_CV_FOLDS", "5"))
N_BOOT = int(os.environ.get("CONFORMAL_N_BOOT", "200"))
SCENARIO_VERSION = os.environ.get("SCENARIO_VERSION", "v1").strip().lower()
ALPHA = float(os.environ.get("CONFORMAL_ALPHA", "0.1"))
N_STEPS = int(os.environ.get("PILSTM_N_STEPS", "64"))


def _collect_endpoints(scenarios, predict_fn) -> tuple[np.ndarray, np.ndarray]:
    truths, preds = [], []
    for sc in scenarios:
        truths.append(ode_endpoint(sc))
        preds.append(predict_fn(sc))
    return np.asarray(truths), np.asarray(preds)


def main() -> None:
    seed = seed_everything()
    model_name = os.environ.get("CONFORMAL_MODEL", "v2").strip().lower()
    device = torch.device("cpu")
    use_float64 = os.environ.get("PILSTM_FLOAT64", "0").lower() in ("1", "true", "yes")
    pilstm_dtype = torch.float64 if use_float64 else torch.float32

    if MODE in _POOLED_MODES:
        cal_rng = np.random.default_rng(CAL_SEED)
        test_rng = np.random.default_rng(TEST_SEED)
        cal_sc = _sample_scenarios(
            N_CAL, cal_rng, id_offset=20_000, structured=True, scenario_version=SCENARIO_VERSION
        )
        test_sc = _sample_scenarios(
            N_TEST, test_rng, id_offset=30_000, structured=True, scenario_version=SCENARIO_VERSION
        )
    elif MODE == "legacy":
        all_sc = canonical_heldout_scenarios(N_HELDOUT, seed=2024, scenario_version=SCENARIO_VERSION)
        cal_sc = all_sc[:N_CAL]
        test_sc = all_sc[N_CAL:]
    else:
        raise ValueError(
            f"Unknown CONFORMAL_MODE {MODE!r}; expected 'legacy', 'large', 'jackknife', or 'cv+'"
        )
    if not cal_sc or not test_sc:
        raise ValueError(f"Need at least 2 scenarios; got N_CAL={len(cal_sc)}, N_TEST={len(test_sc)}")

    if model_name == "pilstm":
        if not PILSTM_WEIGHTS.exists():
            raise FileNotFoundError(f"PI-LSTM weights missing: {PILSTM_WEIGHTS}")
        model = PhysicsInformedLSTM.load(PILSTM_WEIGHTS, map_location=device)
        model.to(device=device, dtype=pilstm_dtype).eval()
        predict_fn = lambda sc: pilstm_endpoint(  # noqa: E731
            model, sc, device=device, dtype=pilstm_dtype, n_steps=N_STEPS
        )
    else:
        if not V2_WEIGHTS.exists():
            raise FileNotFoundError(f"v2 weights missing: {V2_WEIGHTS}")
        model, _ = load_isotope_pinn_checkpoint(V2_WEIGHTS, map_location=device)
        model.eval()
        predict_fn = lambda sc: v2_endpoint(model, sc)  # noqa: E731

    y_cal, p_cal = _collect_endpoints(cal_sc, predict_fn)
    y_test, p_test = _collect_endpoints(test_sc, predict_fn)

    if MODE in ("jackknife", "cv+"):
        from uq.jackknife_plus import (
            FrozenCVPlus,
            cv_plus_intervals,
            median_relative_width,
        )

        k_folds = len(cal_sc) if MODE == "jackknife" else CV_K_FOLDS
        per_species: dict[str, dict] = {}
        for i, name in enumerate(SPECIES):
            entry: dict[str, dict] = {}
            for score_mode in ("absolute", "relative"):
                if score_mode == "relative":
                    cal_r_y = y_cal[:, i]
                    # relative residuals via scaling: fit on y, report q/|y| —
                    # implemented by fitting FrozenCVPlus on scaled residuals
                    res_cal = np.abs(p_cal[:, i] - y_cal[:, i]) / np.maximum(np.abs(y_cal[:, i]), 1e-30)
                    f = FrozenCVPlus(alpha=ALPHA, k_folds=k_folds, n_bootstrap=N_BOOT, seed=seed)
                    f.fit(np.zeros_like(res_cal), res_cal)  # residuals only
                    lo = np.maximum(p_test[:, i] * (1.0 - f.q_pooled_), 0.0)
                    hi = p_test[:, i] * (1.0 + f.q_pooled_)
                    cov = float(np.mean((y_test[:, i] >= lo) & (y_test[:, i] <= hi)))
                    width = float(2.0 * f.q_pooled_)
                else:
                    f = FrozenCVPlus(alpha=ALPHA, k_folds=k_folds, n_bootstrap=N_BOOT, seed=seed)
                    f.fit(y_cal[:, i], p_cal[:, i])
                    cov = f.coverage(y_test[:, i], p_test[:, i])
                    width = median_relative_width(p_test[:, i], f.q_pooled_)
                # Degeneracy check: exact CV+ with identical (frozen) fold
                # models must reproduce the same +/-q band (absolute mode).
                degen = None
                if score_mode == "absolute":
                    lo_x, hi_x = cv_plus_intervals(
                        y_cal[:, i], p_cal[:, i],
                        np.tile(p_test[:, i], (len(cal_sc), 1)), alpha=ALPHA,
                    )
                    lo_s, hi_s = f.interval(p_test[:, i])
                    degen = bool(np.allclose(lo_x, lo_s) and np.allclose(hi_x, hi_s))
                entry[score_mode] = {
                    "test_coverage": cov,
                    "median_relative_width": width,
                    "stability": f.stability_report(),
                    "degenerates_to_split_conformal": degen,
                }
            per_species[name] = entry

        report = {
            "model": model_name,
            "mode": MODE,
            "method": (
                "Frozen-checkpoint jackknife+ (leave-one-out folds)" if MODE == "jackknife"
                else f"Frozen-checkpoint CV+ (K={k_folds} folds)"
            ),
            "guarantee_note": (
                "Frozen model => interval degenerates to split conformal (verified per "
                "species). Strict jackknife+ 1-2*alpha distribution-free guarantee "
                "(Barber et al. 2021, arXiv:1905.02928) requires per-fold RETRAINED "
                "models; inherited guarantee here is the standard split-conformal "
                "marginal coverage. Fold/bootstrap stats quantify interval-width stability."
            ),
            "citation": "Barber, Candes, Ramdas, Tibshirani, 'Predictive inference with the jackknife+', Ann. Statist. 49(1), 2021, arXiv:1905.02928",
            "seed": seed,
            "scenario_version": SCENARIO_VERSION,
            "alpha": ALPHA,
            "nominal_coverage": 1.0 - ALPHA,
            "n_calibration": len(cal_sc),
            "n_test": len(test_sc),
            "k_folds": k_folds,
            "n_bootstrap": N_BOOT,
            "calibration_seed": CAL_SEED,
            "test_seed": TEST_SEED,
            "per_species": per_species,
            "test_ac225_median_rel_error": float(
                np.median(
                    np.abs(p_test[:, AC225_IDX] - y_test[:, AC225_IDX])
                    / np.maximum(np.abs(y_test[:, AC225_IDX]), 1e-30)
                )
            ),
        }
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Wrote {OUT_JSON}")
        return

    # Ac-225 headline metric (absolute + relative conformal).
    ac_cal_t, ac_cal_p = y_cal[:, AC225_IDX], p_cal[:, AC225_IDX]
    ac_test_t, ac_test_p = y_test[:, AC225_IDX], p_test[:, AC225_IDX]

    ac_abs = SplitConformalRegressor(alpha=ALPHA, score_mode="absolute")
    ac_abs.fit(ac_cal_t, ac_cal_p)
    ac_rel = SplitConformalRegressor(alpha=ALPHA, score_mode="relative")
    ac_rel.fit(ac_cal_t, ac_cal_p)

    multi_abs = MultiSpeciesConformal(species=SPECIES, alpha=ALPHA, score_mode="absolute")
    multi_abs.fit(y_cal, p_cal)
    multi_rel = MultiSpeciesConformal(species=SPECIES, alpha=ALPHA, score_mode="relative")
    multi_rel.fit(y_cal, p_cal)

    report = {
        "model": model_name,
        "mode": MODE,
        "seed": seed,
        "scenario_version": SCENARIO_VERSION,
        "alpha": ALPHA,
        "nominal_coverage": 1.0 - ALPHA,
        "n_calibration": len(cal_sc),
        "n_test": len(test_sc),
        "calibration_seed": None if MODE == "legacy" else CAL_SEED,
        "test_seed": None if MODE == "legacy" else TEST_SEED,
        "calibration_scenario_ids": [s.scenario_id for s in cal_sc],
        "test_scenario_ids": [s.scenario_id for s in test_sc],
        "Ac-225": {
            "absolute": {
                "q": ac_abs.q,
                "test_coverage": ac_abs.coverage(ac_test_t, ac_test_p),
                "median_relative_width": ac_abs.median_relative_width(ac_test_p),
                "calibration_median_score": float(np.median(ac_abs.calibration_scores)),
            },
            "relative": {
                "q": ac_rel.q,
                "test_coverage": ac_rel.coverage(ac_test_t, ac_test_p),
                "median_relative_width": ac_rel.median_relative_width(ac_test_p),
                "calibration_median_score": float(np.median(ac_rel.calibration_scores)),
            },
        },
        "all_species_absolute_coverage": multi_abs.coverage_report(y_test, p_test),
        "all_species_relative_coverage": multi_rel.coverage_report(y_test, p_test),
        "all_species_absolute_median_relative_width": multi_abs.width_report(p_test),
        "all_species_relative_median_relative_width": multi_rel.width_report(p_test),
        "conformal_models": {
            "absolute": multi_abs.to_dict(),
            "relative": multi_rel.to_dict(),
        },
        "test_ac225_median_rel_error": float(
            np.median(np.abs(ac_test_p - ac_test_t) / np.maximum(np.abs(ac_test_t), 1e-30))
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
