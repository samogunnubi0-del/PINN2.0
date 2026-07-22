"""
Spectrum-aware re-validation of the ODE against the Joyo anchors
(ODE_DATA_VERSION=v2, SPECTRUM_MODE=mono|watt|twogroup).

Story so far (results/ode_data_v2_validation_20260718.json): with the correct
evaluated sigma(n,2n), the monoenergetic scenarios overpredict the Joyo anchors
by 370-2153x — because they treat the whole 5.7e15 n/cm2/s core flux as
above-threshold (6.4218 MeV) neutrons. This script folds sigma(E) over
documented parametric spectra and INVERTS the two-group model for the
above-threshold fraction f that reproduces Sano 2024 (15.4 +/- 6.2 GBq).

The inferred f is an EFFECTIVE PARAMETER WITH UNCERTAINTY, not a tuned truth:
the Joyo MK-III measured spectrum is paywalled, so the spectra are labelled
assumptions (Watt fission form: Watt 1952 / ENDF-standard a,b; two-group:
Watt-shaped tail above threshold + thermal slow group).

Output: results/ode_data_v2_spectrum_20260718.json

Usage (from project root):
    python analysis/validate_ode_spectrum.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import ra226_ac225_transmutation as R  # noqa: E402
from validate_ode_v2 import _lambdas_per_h, _milking_harvest_bq, _run_endpoint  # noqa: E402

OUT_JSON = ROOT / "results" / "ode_data_v2_spectrum_20260718.json"

ONE_G_RA226 = 2.664e21
PHI_JOYO = 5.7e15

SANO_BQ, SANO_ERR_BQ = 15.4e9, 6.2e9
IWAHASHI_BQ = 30.0e9
IWAHASHI_MILK_BQ = 15.7e9


def _set(version: str, mode: str, f: float | None = None) -> None:
    os.environ[R.ODE_DATA_VERSION_ENV] = version
    os.environ[R.SPECTRUM_MODE_ENV] = mode
    if f is None:
        os.environ.pop(R.SPECTRUM_FAST_FRACTION_ENV, None)
    else:
        os.environ[R.SPECTRUM_FAST_FRACTION_ENV] = repr(float(f))


def _sano_bq(version: str, mode: str, f: float | None = None) -> float:
    _set(version, mode, f)
    lam = _lambdas_per_h(version)["ac225"]
    end = _run_endpoint(PHI_JOYO, 14.5e6, 1080.0, np.array([ONE_G_RA226, 0, 0, 0, 0], float))
    return lam * end[2] / 3600.0


def _iwahashi_bq(version: str, mode: str, f: float | None = None) -> float:
    _set(version, mode, f)
    lam = _lambdas_per_h(version)["ac225"]
    end = _run_endpoint(PHI_JOYO, 1.0e7, 1488.0, np.array([ONE_G_RA226, 0, 0, 0, 0], float))
    return lam * end[2] / 3600.0


def _iwahashi_milk_bq(version: str, mode: str, f: float | None = None) -> float:
    _set(version, mode, f)
    lam = _lambdas_per_h(version)["ac225"]
    return _milking_harvest_bq(PHI_JOYO, 1.0e7, 60.0 * 24.0, 3, 17.5 * 24.0, lam)


def _solve_f_for(target_bq: float) -> float:
    """Brentq on log10(f): two-group fast fraction reproducing target_bq (Sano scenario)."""
    def resid(logf: float) -> float:
        return _sano_bq("v2", "twogroup", 10.0 ** logf) - target_bq
    return float(10.0 ** brentq(resid, -6.0, 0.0, xtol=1e-10))


def main() -> None:
    anchors: list[dict] = []

    def add(anchor_id: str, source: str, measured: float, err: float | None,
            preds: dict[str, float], note: str = "") -> None:
        entry = {
            "id": anchor_id, "source": source,
            "measured_bq": measured, "measured_err_bq": err,
        }
        for k, v in preds.items():
            entry[f"{k}_bq"] = v
            entry[f"{k}_ratio_pred_over_meas"] = v / measured
        if note:
            entry["note"] = note
        anchors.append(entry)
        pretty = " | ".join(f"{k}={v:.3e} ({v/measured:.2f}x)" for k, v in preds.items())
        print(f"{anchor_id:28s} meas={measured:.3e} | {pretty}")

    # --- inversion: above-threshold fraction for Sano central + band -------
    print("Inverting two-group f for Sano 15.4 +/- 6.2 GBq ...")
    f_star = _solve_f_for(SANO_BQ)
    f_lo = _solve_f_for(SANO_BQ - SANO_ERR_BQ)   # lower measured -> smaller f
    f_hi = _solve_f_for(SANO_BQ + SANO_ERR_BQ)
    sig_star_b, _ = R.spectrum_averaged_sigmas_b("twogroup", f_star)
    print(f"f* = {f_star:.4e} [{f_lo:.4e}, {f_hi:.4e}]  -> <sigma_n2n> = {sig_star_b*1e3:.3f} mb")

    # --- predictions under each regime --------------------------------------
    sano_preds = {
        "v1": _sano_bq("v1", "mono"),
        "v2_mono": _sano_bq("v2", "mono"),
        "v2_watt": _sano_bq("v2", "watt"),
        "v2_twogroup_fstar": _sano_bq("v2", "twogroup", f_star),
    }
    add("sano_2024_joyo_45d", "Sano et al. 2024 Joyo uncertainty analysis, JNST 61:509",
        SANO_BQ, SANO_ERR_BQ, sano_preds)

    iwa_preds = {
        "v1": _iwahashi_bq("v1", "mono"),
        "v2_mono": _iwahashi_bq("v2", "mono"),
        "v2_watt": _iwahashi_bq("v2", "watt"),
        "v2_twogroup_fstar": _iwahashi_bq("v2", "twogroup", f_star),
    }
    add("iwahashi_2022_joyo_60d", "Iwahashi et al. 2022 Joyo ORIGEN, MDPI Processes 10(7)1239",
        IWAHASHI_BQ, None, iwa_preds,
        note="Consistency check: f* was inferred from Sano, NOT refit here.")

    milk_preds = {
        "v1": _iwahashi_milk_bq("v1", "mono"),
        "v2_mono": _iwahashi_milk_bq("v2", "mono"),
        "v2_watt": _iwahashi_milk_bq("v2", "watt"),
        "v2_twogroup_fstar": _iwahashi_milk_bq("v2", "twogroup", f_star),
    }
    add("iwahashi_2022_milking_17p5d",
        "Iwahashi et al. 2022 per data/evaluated/README_retrieval.md (3x milking @17.5 d)",
        IWAHASHI_MILK_BQ, None, milk_preds,
        note="Milking = post-EOB harvests (assumption-laden); f* from Sano, not refit.")

    # --- f sensitivity sweep (Sano scenario) --------------------------------
    sweep_f = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 2e-1]
    sweep = []
    for f in sweep_f:
        b = _sano_bq("v2", "twogroup", f)
        s_b, _ = R.spectrum_averaged_sigmas_b("twogroup", f)
        sweep.append({"f": f, "sigma_avg_mb": s_b * 1e3, "pred_bq": b,
                      "ratio_vs_sano": b / SANO_BQ})
        print(f"  sweep f={f:8.1e}  <sigma>={s_b*1e3:9.3f} mb  pred={b:.3e} ({b/SANO_BQ:8.2f}x)")

    # --- plausibility assessment --------------------------------------------
    _set("v2", "watt")
    watt_frac = R.watt_fraction_above_threshold()
    sig_watt_b, _ = R.spectrum_averaged_sigmas_b("watt")
    plausibility = {
        "inferred_fast_fraction_fstar": f_star,
        "inferred_fast_fraction_band": [f_lo, f_hi],
        "inferred_sigma_avg_mb": sig_star_b * 1e3,
        "band_definition": "f values reproducing Sano 15.4 +/- 6.2 GBq exactly (ODE inversion)",
        "bare_watt_fraction_above_threshold": watt_frac,
        "bare_watt_sigma_avg_mb": sig_watt_b * 1e3,
        "softening_factor_vs_bare_fission": watt_frac / f_star,
        "assessment": (
            f"The inferred above-threshold fraction f* = {f_star:.3e} "
            f"[{f_lo:.3e}, {f_hi:.3e}] is ~{watt_frac/f_star:.0f}x SMALLER than a bare "
            f"U-235 fission (Watt) spectrum's >6.4218 MeV tail ({watt_frac:.2%}). "
            "Order-of-magnitude plausible for a sodium-cooled MOX fast breeder: "
            "elastic/inelastic scattering on Na-23, Fe, O and U-238 strongly "
            "degrades the multi-MeV tail, and Joyo spectral softness is known to "
            "vary strongly with irradiation position (Aoyama et al. 2005, J. Nucl. "
            "Radiochem. Sci. 6(3): MK-III reflector spectrum markedly softer, "
            "spectral index ~0.6-0.7; Iwahashi et al. 2022, MDPI Processes "
            "10(7):1239 Fig. 5: MK-III core spectra). NOT verifiable against the "
            "measured MK-III spectrum table (paywalled, ref [13] in Iwahashi). "
            "Also note Sano (+/-40%) and Iwahashi (30 GBq vs Sano 15.4 GBq for a "
            "longer irradiation) disagree with each other by ~2x, so no single f "
            "can satisfy both exactly."
        ),
        "spectrum_citations": [
            "Watt B.E. 1952, Phys. Rev. 87, 1037 (Watt fission spectrum form; "
            "a=0.988 MeV, b=2.249/MeV as tabulated for U-235 thermal fission in "
            "the ENDF-6 formats manual)",
            "Iwahashi D. et al. 2022, Processes 10(7):1239 (open access; Joyo "
            "MK-III/MK-IV core, 6.4 MeV threshold, spectrum in Fig. 5)",
            "Aoyama T. et al. 2005, J. Nucl. Radiochem. Sci. 6(3) (Joyo MK-III "
            "irradiation environments; softer reflector spectrum, index 0.6-0.7)",
        ],
    }

    # --- headline sanity: watt <sigma> ~= old synthetic 27 mb ----------------
    legacy_note = (
        f"Bare-Watt fold of the evaluated table gives <sigma_n2n> = {sig_watt_b*1e3:.2f} mb, "
        "strikingly close to the legacy synthetic '27 mb spectrum average' — the v1 "
        "constant was effectively a FISSION-spectrum average, while a real reactor "
        "spectrum at the irradiation position is far softer (inferred <sigma> "
        f"= {sig_star_b*1e3:.2f} mb)."
    )
    print("\n" + legacy_note)

    summary = {
        "generated": "2026-07-18",
        "script": "analysis/validate_ode_spectrum.py",
        "description": (
            "Spectrum-aware (SPECTRUM_MODE) re-validation against the Joyo anchors. "
            "v1 = legacy synthetic sigmoid; v2_mono = evaluated pointwise at the CSV "
            "representative energy; v2_watt = evaluated sigma folded over a bare "
            "U-235 Watt fission spectrum (no free parameters); v2_twogroup_fstar = "
            "two-group model with the fast fraction inferred from Sano 2024. "
            "Spectra are PARAMETRIC ASSUMPTIONS (Joyo MK-III spectrum paywalled)."
        ),
        "inversion": {
            "target": "Sano 2024, 15.4 +/- 6.2 GBq, 1 g Ra-226, 45 d, phi=5.7e15",
            "method": "brentq on log10(f), exact ODE evaluation per step",
            **plausibility,
        },
        "legacy_27mb_explanation": legacy_note,
        "anchors": anchors,
        "f_sensitivity_sweep_sano": sweep,
        "out_of_scope_note": (
            "Hogle (thermal) and Snow (phi=0 decay leg) anchors are "
            "spectrum-independent; they stay at v2-mono values from "
            "results/ode_data_v2_validation_20260718.json."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _set("v1", "mono")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
