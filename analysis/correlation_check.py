"""
Post-training sanity check: PINN vs ODE correlation on reference scenarios.

Run after train.py + weights saved:
  python analysis/correlation_check.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_model import (  # noqa: E402
    DEFAULT_N226_SCALE,
    DEFAULT_N225_SCALE,
    DEFAULT_NAC_SCALE,
    DEFAULT_PHI_SCALE,
    DEFAULT_T_REF_H,
    load_isotope_pinn_checkpoint,
    neutron_energy_ev_to_feature_numpy,
)
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation  # noqa: E402

WEIGHTS = ROOT / "weights" / "pinn_best_weights.pth"


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return 1.0 - ss_res / ss_tot


def evaluate_pinn(
    model,
    phi: float,
    energy_ev: float,
    times_h: np.ndarray,
    n226: float,
    n225: float,
    nac: float,
) -> np.ndarray:
    rows = []
    for t_h in times_h:
        rows.append([
            float(t_h) / DEFAULT_T_REF_H,
            phi / DEFAULT_PHI_SCALE,
            float(neutron_energy_ev_to_feature_numpy(energy_ev)),
            n226 / DEFAULT_N226_SCALE,
            n225 / DEFAULT_N225_SCALE,
            nac / DEFAULT_NAC_SCALE,
            0.0,
            0.0,
        ])
    x = torch.tensor(rows, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x).numpy()
    scales = np.array([DEFAULT_N226_SCALE, DEFAULT_N225_SCALE, DEFAULT_NAC_SCALE])
    return pred[:, :3] * scales


def simulate_ode(phi, energy_ev, times_h, n226, n225, nac):
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    t_h, Y = run_simulation(
        env,
        t_end_h=float(times_h[-1]),
        n_points=len(times_h),
        N_ra0=n226,
        N_ra225_0=n225,
        N_ac0=nac,
    )
    return t_h, Y[:, :3]


def main() -> None:
    if not WEIGHTS.is_file():
        print(f"Missing weights: {WEIGHTS}")
        sys.exit(1)

    model, _ = load_isotope_pinn_checkpoint(str(WEIGHTS))
    model.eval()
    times = np.linspace(0.0, 300.0, 61)

    primary = ("reference_Ra226_supply", 1e14, 0.025, DEFAULT_N226_SCALE, 0.0, 0.0)
    name, phi, e_ev, n226, n225, nac = primary
    pred = evaluate_pinn(model, phi, e_ev, times, n226, n225, nac)
    _, truth = simulate_ode(phi, e_ev, times, n226, n225, nac)

    print("=== Correlation / fit vs ODE ===")
    print(f"Scenario: {name}  phi={phi:g}  E={e_ev} eV\n")

    all_ok = True
    for i, lab in enumerate(["Ra-226", "Ra-225", "Ac-225"]):
        r2 = r2_score(truth[:, i], pred[:, i])
        rel = np.abs(pred[:, i] - truth[:, i]) / np.maximum(np.abs(truth[:, i]), 1.0)
        cv = float(np.std(truth[:, i]) / max(np.mean(np.abs(truth[:, i])), 1.0))
        if cv >= 1e-3:
            species_ok = r2 >= 0.85
        else:
            species_ok = float(np.max(rel)) <= 0.01
        all_ok = all_ok and species_ok
        flag = "PASS" if species_ok else "FAIL"
        print(f"  {lab}: R² = {r2:.6f}   max|rel err| = {np.max(rel):.4e}   [{flag}]")

    print()
    if all_ok:
        print("PASS: reference supply scenario tracks ODE well enough.")
    else:
        print("FAIL: R² below 0.85 on at least one species.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
