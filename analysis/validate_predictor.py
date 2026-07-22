"""
Held-out ODE validation for the trained PINN.

Run after training:
    python analysis/validate_predictor.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_model import (  # noqa: E402
    DEFAULT_N226_SCALE,
    DEFAULT_N225_SCALE,
    DEFAULT_NAC_SCALE,
    DEFAULT_N227_SCALE,
    DEFAULT_NAC227_SCALE,
    DEFAULT_PHI_SCALE,
    DEFAULT_T_REF_H,
    DEFAULT_LAMBDA_AC_H,
    DEFAULT_LAMBDA_AC7_H,
    load_isotope_pinn_checkpoint,
    neutron_energy_ev_to_feature_numpy,
)
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation  # noqa: E402

WEIGHTS = ROOT / "weights" / "pinn_best_weights.pth"
OUT_DIR = ROOT / "analysis" / "validation"
SPECIES = ["Ra-226", "Ra-225", "Ac-225", "Ra-227", "Ac-227"]
SCALES = np.array(
    [DEFAULT_N226_SCALE, DEFAULT_N225_SCALE, DEFAULT_NAC_SCALE, DEFAULT_N227_SCALE, DEFAULT_NAC227_SCALE],
    dtype=np.float64,
)
IMPURITY_LIMIT = 0.0015
RNG = np.random.default_rng(42)


def _regime(energy_ev: float) -> str:
    if energy_ev < 1.0:
        return "thermal"
    if energy_ev < 1.0e4:
        return "epithermal"
    if energy_ev < 7.0e6:
        return "threshold"
    return "fast14"


def _pinn_predict(model, phi, energy_ev, time_h, ic) -> np.ndarray:
    x = torch.tensor(
        [[
            time_h / DEFAULT_T_REF_H,
            phi / DEFAULT_PHI_SCALE,
            float(neutron_energy_ev_to_feature_numpy(energy_ev)),
            ic[0] / DEFAULT_N226_SCALE,
            ic[1] / DEFAULT_N225_SCALE,
            ic[2] / DEFAULT_NAC_SCALE,
            ic[3] / DEFAULT_N227_SCALE,
            ic[4] / DEFAULT_NAC227_SCALE,
        ]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        pred = model(x).numpy()[0]
    return pred * SCALES


def _ode_final(phi, energy_ev, time_h, ic) -> np.ndarray:
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    _, Y = run_simulation(
        env,
        t_end_h=time_h,
        n_points=max(401, int(time_h * 4) + 1),
        N_ra0=ic[0],
        N_ra225_0=ic[1],
        N_ac0=ic[2],
        N_ra227_0=ic[3],
        N_ac227_0=ic[4],
    )
    return Y[-1]


def _impurity_pct(ac225: float, ac227: float) -> float:
    lam_ac = DEFAULT_LAMBDA_AC_H / 3600.0
    lam_a7 = DEFAULT_LAMBDA_AC7_H / 3600.0
    a225 = ac225 * lam_ac
    a227 = ac227 * lam_a7
    tot = a225 + a227
    return a227 / tot if tot > 0 else 0.0


def _gen_scenarios() -> list[dict]:
    scenarios: list[dict] = []
    for case_type, ic_fn in (
        ("empty", lambda: np.zeros(5)),
        ("virgin", lambda: np.array([RNG.uniform(1e21, 1e23), 0, 0, 0, 0])),
        (
            "recycled_trace_inventory",
            lambda: np.array([
                RNG.uniform(1e18, 1e23),
                RNG.uniform(1e16, 1e19),
                RNG.uniform(1e14, 1e18),
                RNG.uniform(1e10, 1e15),
                RNG.uniform(1e12, 1e17),
            ]),
        ),
    ):
        n = 2 if case_type == "empty" else (11 if case_type == "virgin" else 9)
        for _ in range(n):
            ic = ic_fn()
            log_phi = RNG.uniform(11.0, 15.3)
            energy = {
                "thermal": RNG.uniform(0.015, 0.08),
                "epithermal": RNG.uniform(0.1, 1e4),
                "threshold": RNG.uniform(5.8e6, 7.5e6),
                "fast14": RNG.uniform(6.5e6, 2.0e7),
            }[RNG.choice(["thermal", "epithermal", "threshold", "fast14"])]
            scenarios.append(
                {
                    "case_type": case_type,
                    "phi": 10.0 ** log_phi,
                    "energy_ev": energy,
                    "time_h": float(RNG.uniform(1.0, 400.0)),
                    "ic": ic,
                }
            )
    return scenarios


def main() -> None:
    if not WEIGHTS.is_file():
        print(f"Missing weights: {WEIGHTS}")
        sys.exit(1)

    model, _ = load_isotope_pinn_checkpoint(str(WEIGHTS))
    model.eval()

    rows: list[dict] = []
    for case_id, sc in enumerate(_gen_scenarios()):
        ic = sc["ic"]
        truth = _ode_final(sc["phi"], sc["energy_ev"], sc["time_h"], ic)
        pred = _pinn_predict(model, sc["phi"], sc["energy_ev"], sc["time_h"], ic)
        regime = _regime(sc["energy_ev"])
        for j, sp in enumerate(SPECIES):
            t_val, p_val = float(truth[j]), float(pred[j])
            rel = abs(p_val - t_val) / max(abs(t_val), 1.0)
            rows.append(
                {
                    "regime": regime,
                    "case_type": sc["case_type"],
                    "phi": sc["phi"],
                    "energy_ev": sc["energy_ev"],
                    "time_h": sc["time_h"],
                    "N_Ra226_0": ic[0],
                    "N_Ra225_0": ic[1],
                    "N_Ac225_0": ic[2],
                    "N_Ra227_0": ic[3],
                    "N_Ac227_0": ic[4],
                    "case_id": case_id,
                    "species": sp,
                    "truth": t_val,
                    "prediction": p_val,
                    "abs_error": abs(p_val - t_val),
                    "rel_error": rel,
                }
            )
        t_imp = _impurity_pct(truth[2], truth[4])
        p_imp = _impurity_pct(pred[2], pred[4])
        rows.append(
            {
                "regime": regime,
                "case_type": sc["case_type"],
                "phi": sc["phi"],
                "energy_ev": sc["energy_ev"],
                "time_h": sc["time_h"],
                "N_Ra226_0": ic[0],
                "N_Ra225_0": ic[1],
                "N_Ac225_0": ic[2],
                "N_Ra227_0": ic[3],
                "N_Ac227_0": ic[4],
                "case_id": case_id,
                "species": "Ac-227 activity impurity",
                "truth": t_imp,
                "prediction": p_imp,
                "abs_error": abs(p_imp - t_imp),
                "rel_error": abs(p_imp - t_imp),
                "truth_usable": bool(t_imp <= IMPURITY_LIMIT),
                "prediction_usable": bool(p_imp <= IMPURITY_LIMIT),
                "decision_correct": bool((t_imp <= IMPURITY_LIMIT) == (p_imp <= IMPURITY_LIMIT)),
            }
        )

    details = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details.to_csv(OUT_DIR / "heldout_validation_details.csv", index=False)

    summary_rows: list[dict] = []
    isotope_rows = details[~details["species"].str.contains("impurity")]
    for (regime, case_type, species), grp in isotope_rows.groupby(["regime", "case_type", "species"]):
        summary_rows.append(
            {
                "regime": regime,
                "case_type": case_type,
                "species": species,
                "n": len(grp),
                "median_rel_error": grp["rel_error"].median(),
                "p90_rel_error": grp["rel_error"].quantile(0.9),
                "p95_rel_error": grp["rel_error"].quantile(0.95),
                "max_rel_error": grp["rel_error"].max(),
                "mae_atoms": grp["abs_error"].mean(),
            }
        )
    for species in SPECIES + ["Ac-227 activity impurity"]:
        grp = details[details["species"] == species]
        row = {
            "regime": "all",
            "case_type": "all",
            "species": species,
            "n": len(grp),
            "median_rel_error": grp["rel_error"].median(),
            "p90_rel_error": grp["rel_error"].quantile(0.9),
            "p95_rel_error": grp["rel_error"].quantile(0.95),
            "max_rel_error": grp["rel_error"].max(),
            "mae_atoms": grp["abs_error"].mean(),
        }
        if species == "Ac-227 activity impurity":
            imp = grp.dropna(subset=["decision_correct"])
            row["decision_accuracy"] = imp["decision_correct"].astype(bool).mean() if len(imp) else np.nan
            false = imp[imp["truth_usable"] == True]  # noqa: E712
            row["false_usable_rate"] = (
                (false["prediction_usable"] == False).astype(bool).mean() if len(false) else np.nan  # noqa: E712
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "heldout_validation_summary.csv", index=False)
    
    canary_rows = []

    # 1. fast14_virgin_ac225_zero_collapse
    f14 = details[(details["case_type"] == "virgin") & (details["regime"] == "fast14") & (details["species"] == "Ac-225")]
    n_f14 = len(f14)
    if n_f14 > 0:
        med_f14 = f14["rel_error"].median()
        max_f14 = f14["rel_error"].max()
        z_f14 = float((f14["prediction"] == 0.0).mean())
        mean_t_f14 = f14["truth"].mean()
        mean_p_f14 = f14["prediction"].mean()
        status_f14 = "pass" if med_f14 <= 0.10 and z_f14 < 0.05 else "fail"
    else:
        med_f14, max_f14, z_f14, mean_t_f14, mean_p_f14, status_f14 = np.nan, np.nan, np.nan, np.nan, np.nan, "fail"
    canary_rows.append({
        "canary": "fast14_virgin_ac225_zero_collapse",
        "n": n_f14,
        "median_rel_error": med_f14,
        "max_rel_error": max_f14,
        "zero_prediction_rate": z_f14,
        "mean_truth": mean_t_f14,
        "mean_prediction": mean_p_f14,
        "status": status_f14,
        "decision_accuracy": np.nan,
        "false_usable_rate": np.nan,
        "median_abs_impurity_error": np.nan,
    })

    # 2. threshold_virgin_ac225_zero_collapse
    thresh = details[(details["case_type"] == "virgin") & (details["regime"] == "threshold") & (details["species"] == "Ac-225")]
    n_thresh = len(thresh)
    if n_thresh > 0:
        med_thresh = thresh["rel_error"].median()
        max_thresh = thresh["rel_error"].max()
        z_thresh = float((thresh["prediction"] == 0.0).mean())
        mean_t_thresh = thresh["truth"].mean()
        mean_p_thresh = thresh["prediction"].mean()
        status_thresh = "pass" if med_thresh <= 0.10 and z_thresh < 0.05 else "fail"
    else:
        med_thresh, max_thresh, z_thresh, mean_t_thresh, mean_p_thresh, status_thresh = np.nan, np.nan, np.nan, np.nan, np.nan, "fail"
    canary_rows.append({
        "canary": "threshold_virgin_ac225_zero_collapse",
        "n": n_thresh,
        "median_rel_error": med_thresh,
        "max_rel_error": max_thresh,
        "zero_prediction_rate": z_thresh,
        "mean_truth": mean_t_thresh,
        "mean_prediction": mean_p_thresh,
        "status": status_thresh,
        "decision_accuracy": np.nan,
        "false_usable_rate": np.nan,
        "median_abs_impurity_error": np.nan,
    })

    # 3. empty_phantom_atoms
    empty = details[details["case_type"] == "empty"]
    n_empty = len(empty)
    if n_empty > 0:
        max_empty = empty["abs_error"].max()
        z_empty = float((empty["prediction"] == 0.0).mean())
        status_empty = "pass" if max_empty <= 1e-10 else "fail"
    else:
        max_empty, z_empty, status_empty = np.nan, np.nan, "fail"
    canary_rows.append({
        "canary": "empty_phantom_atoms",
        "n": n_empty,
        "median_rel_error": 0.0 if status_empty == "pass" else np.nan,
        "max_rel_error": 0.0 if status_empty == "pass" else np.nan,
        "zero_prediction_rate": z_empty,
        "mean_truth": 0.0,
        "mean_prediction": empty["prediction"].mean() if n_empty > 0 else np.nan,
        "status": status_empty,
        "decision_accuracy": np.nan,
        "false_usable_rate": np.nan,
        "median_abs_impurity_error": np.nan,
    })

    # 4. recycled_impurity_decision
    rec = details[(details["species"] == "Ac-227 activity impurity") & (details["case_type"] == "recycled_trace_inventory")]
    n_rec = len(rec)
    if n_rec > 0:
        da_rec = rec["decision_correct"].astype(bool).mean()
        fur_rec = (rec["prediction_usable"].astype(bool) & ~rec["truth_usable"].astype(bool)).mean()
        mae_rec = rec["abs_error"].median()
        status_rec = "pass" if da_rec >= 0.90 else "fail"
    else:
        da_rec, fur_rec, mae_rec, status_rec = np.nan, np.nan, np.nan, "fail"
    canary_rows.append({
        "canary": "recycled_impurity_decision",
        "n": n_rec,
        "median_rel_error": np.nan,
        "max_rel_error": np.nan,
        "zero_prediction_rate": np.nan,
        "mean_truth": np.nan,
        "mean_prediction": np.nan,
        "status": status_rec,
        "decision_accuracy": da_rec,
        "false_usable_rate": fur_rec,
        "median_abs_impurity_error": mae_rec,
    })

    canary_df = pd.DataFrame(canary_rows)
    canary_df.to_csv(OUT_DIR / "heldout_canary_report.csv", index=False)

    print(f"Wrote validation to {OUT_DIR}")
    ac = summary[(summary.regime == "all") & (summary.species == "Ac-225")].iloc[0]
    print(f"Ac-225 all median rel error: {ac['median_rel_error']:.4f}")


if __name__ == "__main__":
    main()
