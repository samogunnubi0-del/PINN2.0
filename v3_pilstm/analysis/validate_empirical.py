"""
Compare PI-LSTM predictions to literature benchmark rows (if CSV present).

Usage (from project root):
    python v3_pilstm/analysis/validate_empirical.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3_pilstm"
CSV_PATH = PROJECT_ROOT / "data" / "literature_benchmarks.csv"
OUT_JSON = V3_ROOT / "results" / "empirical_validation.json"
PILSTM_WEIGHTS = V3_ROOT / "weights" / "pi_lstm_best.pth"
V2_WEIGHTS = PROJECT_ROOT / "weights" / "pinn_best_weights.pth"

AC225_HALF_LIFE_S = 9.920 * 24.0 * 3600.0
AC225_LAMBDA = np.log(2.0) / AC225_HALF_LIFE_S
AC227_HALF_LIFE_S = 21.772 * 365.25 * 24.0 * 3600.0
AC227_LAMBDA = np.log(2.0) / AC227_HALF_LIFE_S
DEFAULT_N_RA226 = 2.664e21  # ~1 g Ra-226

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

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
from data.trajectory_dataset import TrajectoryScenario, integrate_scenario  # noqa: E402
from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation  # noqa: E402

SCALES = np.array(
    [DEFAULT_N226_SCALE, DEFAULT_N225_SCALE, DEFAULT_NAC_SCALE, DEFAULT_N227_SCALE, DEFAULT_NAC227_SCALE],
    dtype=np.float64,
)


def _parse_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return float(s)


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ode_ac225(phi: float, energy_ev: float, time_h: float, ic: np.ndarray) -> float:
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    _, y = run_simulation(
        env,
        t_end_h=time_h,
        n_points=401,
        N_ra0=ic[0],
        N_ra225_0=ic[1],
        N_ac0=ic[2],
        N_ra227_0=ic[3],
        N_ac227_0=ic[4],
    )
    return float(y[-1, 2])


def _v2_ac225(model, phi: float, energy_ev: float, time_h: float, ic: np.ndarray) -> float:
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
    return float(pred[2] * DEFAULT_NAC_SCALE)


# Match train/compare default (Results-6 / Run C use 64).
N_STEPS = int(os.environ.get("PILSTM_N_STEPS", "64"))


def _pilstm_endpoint(model, phi: float, energy_ev: float, time_h: float, ic: np.ndarray, device, dtype) -> np.ndarray:
    """Return the full 5-species endpoint (atoms) from PI-LSTM."""
    sc = TrajectoryScenario(phi=phi, energy_ev=energy_ev, t_end_h=time_h, ic=ic, scenario_id=0)
    t_norm, _ = integrate_scenario(sc, n_steps=N_STEPS)
    e_feat = float(neutron_energy_ev_to_feature_numpy(energy_ev))
    ic_norm = ic / SCALES
    seq_len = len(t_norm)
    feats = np.zeros((1, seq_len, 8), dtype=np.float32)
    for k in range(seq_len):
        feats[0, k, 0] = t_norm[k]
        feats[0, k, 1] = phi / DEFAULT_PHI_SCALE
        feats[0, k, 2] = e_feat
        feats[0, k, 3:8] = ic_norm
    with torch.no_grad():
        pred = model(torch.from_numpy(feats).to(device=device, dtype=dtype)).numpy()[0, -1] * SCALES
    return pred


def _pilstm_ac225(model, phi: float, energy_ev: float, time_h: float, ic: np.ndarray, device, dtype) -> float:
    return float(_pilstm_endpoint(model, phi, energy_ev, time_h, ic, device, dtype)[2])


def _ode_endpoint(phi: float, energy_ev: float, time_h: float, ic: np.ndarray) -> np.ndarray:
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    _, y = run_simulation(
        env, t_end_h=time_h, n_points=401,
        N_ra0=ic[0], N_ra225_0=ic[1], N_ac0=ic[2], N_ra227_0=ic[3], N_ac227_0=ic[4],
    )
    return y[-1]


def _v2_endpoint(model, phi: float, energy_ev: float, time_h: float, ic: np.ndarray) -> np.ndarray:
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


def _reference_value(row: dict) -> tuple[float | None, str]:
    n_ac = _parse_float(row.get("N_Ac225"))
    if n_ac is not None:
        return n_ac, "N_Ac225"
    a_bq = _parse_float(row.get("A_Ac225_Bq"))
    if a_bq is not None:
        return a_bq / AC225_LAMBDA, "A_Ac225_Bq"
    return None, ""


def _reference_227ac(row: dict) -> float | None:
    """Return reference 227Ac atom count from A_Ac227_Bq (Lit-17 impurity channel)."""
    a_bq = _parse_float(row.get("A_Ac227_Bq"))
    if a_bq is not None:
        return a_bq / AC227_LAMBDA
    return None


def main() -> None:
    from seed_utils import seed_everything
    seed_everything()  # deterministic eval (PI_LSTM_SEED, default 42)
    rows = _load_rows(CSV_PATH)
    if not rows:
        print(f"No literature CSV at {CSV_PATH} — nothing to validate.")
        return

    use_float64 = os.environ.get("PILSTM_FLOAT64", "0").lower() in ("1", "true", "yes")
    dtype = torch.float64 if use_float64 else torch.float32
    device = torch.device("cpu")

    pilstm_loaded = False
    if PILSTM_WEIGHTS.exists():
        model = PhysicsInformedLSTM.load(PILSTM_WEIGHTS, map_location=device)
        pilstm_loaded = True
    else:
        model = PhysicsInformedLSTM()
    model.to(device=device, dtype=dtype).eval()

    v2_loaded = False
    v2_model = None
    if V2_WEIGHTS.exists():
        v2_model, _ = load_isotope_pinn_checkpoint(V2_WEIGHTS, map_location=device)
        v2_model.eval()
        v2_loaded = True

    results: list[dict] = []
    mape_vals_v2: list[float] = []
    mape_vals_pi: list[float] = []
    mape_227ac_v2: list[float] = []
    mape_227ac_pi: list[float] = []

    print(f"Loaded {len(rows)} literature row(s) from {CSV_PATH}")
    if not v2_loaded:
        print(f"Warning: v2 weights missing at {V2_WEIGHTS} — v2 columns will be null.")
    if not pilstm_loaded:
        print(f"Warning: PI-LSTM weights missing at {PILSTM_WEIGHTS} — PI-LSTM columns will be null.")

    for i, row in enumerate(rows):
        source = row.get("source_citation", f"row_{i}")
        source_type = row.get("source_type", "").strip()
        ref_n, ref_kind = _reference_value(row)

        entry: dict = {
            "source_citation": source,
            "source_type": source_type,
            "reference_kind": ref_kind or None,
            "skipped": False,
            "skip_reason": None,
        }

        # Lit-17: 227Ac impurity-channel validation for thermal (n,gamma) rows
        # (Hogle HFIR, Kuznetsov SM). Validates the Ra-226->Ra-227->Ac-227 leg,
        # which is where v2 is already strong. Uses A_Ac227_Bq if present.
        ref_227 = _reference_227ac(row)
        if ref_227 is not None:
            phi_i = _parse_float(row.get("phi_n_cm2_s"))
            energy_i = _parse_float(row.get("energy_ev"))
            time_i = _parse_float(row.get("time_h"))
            if phi_i is not None and energy_i is not None and time_i is not None:
                n_ra0 = _parse_float(row.get("N_Ra226_0")) or DEFAULT_N_RA226
                ic = np.array([n_ra0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                ode_227 = float(_ode_endpoint(phi_i, energy_i, time_i, ic)[4])
                v2_227 = float(_v2_endpoint(v2_model, phi_i, energy_i, time_i, ic)[4]) if v2_loaded else None
                pi_227 = float(_pilstm_endpoint(model, phi_i, energy_i, time_i, ic, device, dtype)[4]) if pilstm_loaded else None
                rel_ode = abs(ode_227 - ref_227) / max(abs(ref_227), 1e-30)
                rel_v2 = abs(v2_227 - ref_227) / max(abs(ref_227), 1e-30) if v2_227 is not None else None
                rel_pi = abs(pi_227 - ref_227) / max(abs(ref_227), 1e-30) if pi_227 is not None else None
                entry.update({
                    "channel": "227Ac (n,gamma) impurity",
                    "phi_n_cm2_s": phi_i, "energy_ev": energy_i, "time_h": time_i,
                    "N_Ra226_0": n_ra0,
                    "reference_N_Ac227": ref_227,
                    "reference_A_Ac227_Bq": ref_227 * AC227_LAMBDA,
                    "ode_N_Ac227": ode_227, "ode_rel_error_227ac": rel_ode,
                    "v2_N_Ac227": v2_227, "v2_rel_error_227ac": rel_v2,
                    "pilstm_N_Ac227": pi_227, "pilstm_rel_error_227ac": rel_pi,
                    "notes": row.get("notes", ""),
                })
                results.append(entry)
                if rel_v2 is not None:
                    mape_227ac_v2.append(rel_v2)
                if rel_pi is not None:
                    mape_227ac_pi.append(rel_pi)
                v2s = f"v2 rel={100*rel_v2:.1f}% | " if rel_v2 is not None else ""
                pis = f"PI-LSTM rel={100*rel_pi:.1f}%" if rel_pi is not None else ""
                print(f"  [227Ac] ref_A={ref_227*AC227_LAMBDA:.3e} Bq | ODE rel={100*rel_ode:.1f}% | {v2s}{pis}")
                continue

        if ref_n is None:
            entry["skipped"] = True
            entry["skip_reason"] = "no A_Ac225_Bq or N_Ac225"
            entry["notes"] = row.get("notes", "")
            results.append(entry)
            print(f"  [skip] no activity: {source[:60]}...")
            continue

        ref_a = ref_n * AC225_LAMBDA
        phi = _parse_float(row.get("phi_n_cm2_s"))
        energy_ev = _parse_float(row.get("energy_ev"))
        time_h = _parse_float(row.get("time_h"))

        # cross_route rows with Ac-225 activity but no neutron beam conditions:
        # record literature reference; ODE/PI-LSTM (n,2n) comparison not applicable.
        if source_type == "cross_route" and (phi is None or energy_ev is None):
            entry.update(
                {
                    "reference_N_Ac225": ref_n,
                    "reference_A_Ac225_Bq": ref_a,
                    "time_h": time_h,
                    "skipped": True,
                    "skip_reason": "cross_route: activity recorded; neutron flux/energy N/A (different production route)",
                    "notes": row.get("notes", ""),
                }
            )
            results.append(entry)
            print(f"  [cross_route ref] A_Ac225={ref_a:.3e} Bq | {source[:50]}...")
            continue

        # Lit-18: decay-leg validation (229Th->225Ra->225Ac). We cannot model
        # 229Th, but if the row gives an initial 225Ra inventory we validate the
        # 225Ra->225Ac Bateman ingrowth with the neutron flux OFF (phi=0).
        n_ra225_0 = _parse_float(row.get("N_Ra225_0"))
        if source_type == "decay_leg" and n_ra225_0 is not None and time_h is not None:
            ic_dl = np.array([0.0, n_ra225_0, 0.0, 0.0, 0.0], dtype=np.float64)
            ode_n = _ode_ac225(0.0, 0.025, time_h, ic_dl)
            pi_n = _pilstm_ac225(model, 0.0, 0.025, time_h, ic_dl, device, dtype) if pilstm_loaded else None
            rel_ode = abs(ode_n - ref_n) / max(abs(ref_n), 1e-30)
            rel_pi = abs(pi_n - ref_n) / max(abs(ref_n), 1e-30) if pi_n is not None else None
            entry.update({
                "mode": "decay_leg (phi=0, 225Ra->225Ac)",
                "time_h": time_h,
                "N_Ra225_0": n_ra225_0,
                "reference_N_Ac225": ref_n,
                "reference_A_Ac225_Bq": ref_a,
                "ode_N_Ac225": ode_n,
                "ode_rel_error": rel_ode,
                "pilstm_N_Ac225": pi_n,
                "pilstm_rel_error": rel_pi,
                "notes": row.get("notes", ""),
            })
            results.append(entry)
            if rel_pi is not None:
                mape_vals_pi.append(rel_pi)
            print(f"  [decay_leg] ref_A={ref_a:.3e} Bq | ODE rel={100*rel_ode:.1f}%")
            continue

        if phi is None or energy_ev is None or time_h is None:
            entry["skipped"] = True
            entry["skip_reason"] = "missing phi_n_cm2_s, energy_ev, or time_h"
            entry["reference_A_Ac225_Bq"] = ref_a
            results.append(entry)
            print(f"  [skip] incomplete neutron conditions: {source[:60]}...")
            continue

        n_ra0 = _parse_float(row.get("N_Ra226_0")) or DEFAULT_N_RA226
        ic = np.array([n_ra0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        ode_n = _ode_ac225(phi, energy_ev, time_h, ic)
        ode_a = ode_n * AC225_LAMBDA

        v2_n = None
        v2_a = None
        if v2_loaded and v2_model is not None:
            v2_n = _v2_ac225(v2_model, phi, energy_ev, time_h, ic)
            v2_a = v2_n * AC225_LAMBDA

        pi_n = None
        pi_a = None
        if pilstm_loaded:
            pi_n = _pilstm_ac225(model, phi, energy_ev, time_h, ic, device, dtype)
            pi_a = pi_n * AC225_LAMBDA

        rel_ode = abs(ode_n - ref_n) / max(abs(ref_n), 1e-30)
        rel_v2 = abs(v2_n - ref_n) / max(abs(ref_n), 1e-30) if v2_n is not None else None
        rel_pi = abs(pi_n - ref_n) / max(abs(ref_n), 1e-30) if pi_n is not None else None

        entry.update(
            {
                "phi_n_cm2_s": phi,
                "energy_ev": energy_ev,
                "time_h": time_h,
                "N_Ra226_0": n_ra0,
                "reference_N_Ac225": ref_n,
                "reference_A_Ac225_Bq": ref_a,
                "ode_N_Ac225": ode_n,
                "ode_A_Ac225_Bq": ode_a,
                "ode_rel_error": rel_ode,
                "v2_N_Ac225": v2_n,
                "v2_A_Ac225_Bq": v2_a,
                "v2_rel_error": rel_v2,
                "pilstm_N_Ac225": pi_n,
                "pilstm_A_Ac225_Bq": pi_a,
                "pilstm_rel_error": rel_pi,
                "notes": row.get("notes", ""),
            }
        )
        results.append(entry)

        if rel_v2 is not None:
            mape_vals_v2.append(rel_v2)
        if rel_pi is not None:
            mape_vals_pi.append(rel_pi)
        v2_str = f"v2 rel={100*rel_v2:.1f}% | " if rel_v2 is not None else ""
        pi_str = f"PI-LSTM rel={100*rel_pi:.1f}%" if rel_pi is not None else ""
        print(f"  [{source_type}] ref_A={ref_a:.3e} Bq | ODE rel={100*rel_ode:.1f}% | {v2_str}{pi_str}")

    n_activity_rows = sum(1 for r in results if r.get("reference_A_Ac225_Bq") is not None)
    n_cross_route_ref = sum(
        1
        for r in results
        if r.get("source_type") == "cross_route"
        and r.get("reference_A_Ac225_Bq") is not None
        and r.get("skipped")
    )

    summary = {
        "csv_path": str(CSV_PATH),
        "v2_weights_loaded": v2_loaded,
        "pilstm_weights_loaded": pilstm_loaded,
        "n_rows": len(rows),
        "n_activity_rows": n_activity_rows,
        "n_cross_route_activity_reference": n_cross_route_ref,
        "n_compared": len(mape_vals_pi),
        "mape_v2_vs_literature": float(np.mean(mape_vals_v2)) if mape_vals_v2 else None,
        "mape_pilstm_vs_literature": float(np.mean(mape_vals_pi)) if mape_vals_pi else None,
        "n_compared_227ac": max(len(mape_227ac_v2), len(mape_227ac_pi)),
        "mape_v2_227ac_vs_literature": float(np.mean(mape_227ac_v2)) if mape_227ac_v2 else None,
        "mape_pilstm_227ac_vs_literature": float(np.mean(mape_227ac_pi)) if mape_227ac_pi else None,
        "n_steps": N_STEPS,
        "rows": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")
    if mape_vals_v2:
        print(f"v2 MAPE vs literature ({len(mape_vals_v2)} point(s)): {100*np.mean(mape_vals_v2):.2f}%")
    if mape_vals_pi:
        print(f"PI-LSTM MAPE vs literature ({len(mape_vals_pi)} point(s)): {100*np.mean(mape_vals_pi):.2f}%")
    if mape_227ac_v2:
        print(f"v2 227Ac MAPE ({len(mape_227ac_v2)} point(s)): {100*np.mean(mape_227ac_v2):.2f}%")
    if mape_227ac_pi:
        print(f"PI-LSTM 227Ac MAPE ({len(mape_227ac_pi)} point(s)): {100*np.mean(mape_227ac_pi):.2f}%")
    if not mape_vals_v2 and not mape_vals_pi:
        print("No activity rows compared — fill A_Ac225_Bq or N_Ac225 in the CSV.")


if __name__ == "__main__":
    main()
