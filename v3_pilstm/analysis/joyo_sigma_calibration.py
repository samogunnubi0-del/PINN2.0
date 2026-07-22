"""
Lit-16: Joyo (n,2n) cross-section calibration — NON-DESTRUCTIVE.

Both the default ODE and the models predict ~10-25x more Ac-225 than the Joyo
fast-reactor simulations (Sasaki 2023: 15.4 GBq @ 45 d; Iwahashi 2022: 30 GBq
@ 60 d). This script fits a single scalar multiplier on sigma_n2n so the ODE
matches those endpoints, WITHOUT touching pinn_model defaults, the frozen v2
model, or the training ODE. It only reports the recommended scale + residuals
so you can cite a calibrated cross-section on the poster.

Usage:
    python v3_pilstm/analysis/joyo_sigma_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation  # noqa: E402
from pinn_model import DEFAULT_SIGMA_N2N  # noqa: E402

OUT_JSON = PROJECT_ROOT / "v3_pilstm" / "results" / "joyo_sigma_calibration.json"
AC225_LAMBDA = np.log(2.0) / (9.920 * 24.0 * 3600.0)

# Joyo simulation anchors: (label, phi, E_eV, time_h, N_Ra226_0 atoms, target A_Ac225 Bq)
ANCHORS = [
    ("Sasaki 2023", 5.7e15, 1.45e7, 1080.0, 2.664e21, 1.54e10),
    ("Iwahashi 2022", 5.7e15, 1.0e7, 1488.0, 2.664e21, 3.0e10),
]


def _ac225_bq(sigma_n2n: float, phi: float, energy_ev: float, time_h: float, n_ra0: float) -> float:
    env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma_n2n, neutron_energy_ev=energy_ev)
    _, y = run_simulation(env, t_end_h=time_h, n_points=401, N_ra0=n_ra0)
    return float(y[-1, 2]) * AC225_LAMBDA


def _mape(scale: float) -> float:
    errs = []
    for _, phi, e, t, n0, ref in ANCHORS:
        pred = _ac225_bq(DEFAULT_SIGMA_N2N * scale, phi, e, t, n0)
        errs.append(abs(pred - ref) / ref)
    return float(np.mean(errs))


def main() -> None:
    # Coarse-to-fine 1-D search on log10(scale).
    grid = np.logspace(-2.0, 0.5, 400)
    mapes = np.array([_mape(s) for s in grid])
    best_i = int(np.argmin(mapes))
    best_scale = float(grid[best_i])

    rows = []
    for label, phi, e, t, n0, ref in ANCHORS:
        default_pred = _ac225_bq(DEFAULT_SIGMA_N2N, phi, e, t, n0)
        cal_pred = _ac225_bq(DEFAULT_SIGMA_N2N * best_scale, phi, e, t, n0)
        rows.append({
            "anchor": label,
            "reference_A_Ac225_Bq": ref,
            "default_A_Ac225_Bq": default_pred,
            "default_rel_error": abs(default_pred - ref) / ref,
            "calibrated_A_Ac225_Bq": cal_pred,
            "calibrated_rel_error": abs(cal_pred - ref) / ref,
        })

    summary = {
        "default_sigma_n2n_cm2": DEFAULT_SIGMA_N2N,
        "recommended_scale": best_scale,
        "calibrated_sigma_n2n_cm2": DEFAULT_SIGMA_N2N * best_scale,
        "calibrated_sigma_n2n_mb": DEFAULT_SIGMA_N2N * best_scale * 1e27,
        "mape_default": _mape(1.0),
        "mape_calibrated": float(mapes[best_i]),
        "note": (
            "Non-destructive: default ODE/v2 unchanged. Apply this scale only in "
            "a dedicated 'Joyo-calibrated' figure. O'Connor 1960 gives 1.60 b at "
            "14.5 MeV; the default 27 mb is a fast-spectrum average, so a Joyo-"
            "specific effective sigma is physically reasonable."
        ),
        "anchors": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(
        f"Default MAPE {100*summary['mape_default']:.0f}% -> calibrated "
        f"{100*summary['mape_calibrated']:.0f}% at sigma scale x{best_scale:.3f} "
        f"({summary['calibrated_sigma_n2n_mb']:.2f} mb)"
    )


if __name__ == "__main__":
    main()
