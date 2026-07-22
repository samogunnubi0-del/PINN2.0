"""
Validate the five-species ODE against literature anchors under BOTH data
versions: ODE_DATA_VERSION=v1 (legacy synthetic sigmoid) and v2 (evaluated
JENDL-5 / ENDF/B-VIII.0 / EXFOR / NuDat data, retrieved 2026-07-18).

Anchors (all traceable to data/literature_benchmarks.csv post-merge, or to
data/evaluated/README_retrieval.md where noted):

  1. Sano et al. 2024 (Joyo):       15.4 ± 6.2 GBq Ac-225 per g Ra-226, 45 d,
                                    core center (simulation endpoint).
  2. Iwahashi et al. 2022 (Joyo):   ~30 GBq after 60 d + 8 d cool (ORIGEN sim).
  3. Iwahashi 2022 milking cycle:   15.7 GBq/cycle with 3x milking at 17.5 d
                                    (scenario per data/evaluated/README_retrieval.md;
                                    milking modelled as post-EOB harvests — ASSUMPTION-LADEN).
  4. Hogle et al. 2016 (HFIR):      Ac-227 (n,gamma) impurity-leg series at
                                    3.01 / 7.00 / 26.09 d (empirical, phi=2.0e15 thermal).
  5. Snow et al. 2025 (INL):        phi=0 ingrowth: 285.2 Bq Ra-225 (EOB) ->
                                    126.8 ± 12.6 Bq Ac-225 at 17 d post-EOB (empirical).

Output: results/ode_data_v2_validation_20260718.json
  Per anchor: measured value, v1 prediction, v2 prediction, ratios
  (prediction / measurement). No tuning is applied — whatever the ODE produces
  is reported.

Usage (from project root):
    python analysis/validate_ode_v2.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ra226_ac225_transmutation as R  # noqa: E402

LIT_CSV = ROOT / "data" / "literature_benchmarks.csv"
OUT_JSON = ROOT / "results" / "ode_data_v2_validation_20260718.json"

VERSIONS = ("v1", "v2")
ONE_G_RA226 = 2.664e21
ONE_UG_RA226 = 2.664e15


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _set_version(v: str) -> None:
    os.environ[R.ODE_DATA_VERSION_ENV] = v


def _lambdas_per_h(v: str) -> dict[str, float]:
    """Decay constants the given data version uses (for activity conversion)."""
    _set_version(v)
    env = R.IsotopeEnvironment(phi=0.0, neutron_energy_ev=1.0e7)
    return {
        "ra225": env.lambda_ra225_per_h,
        "ac225": env.lambda_ac225_per_h,
        "ac227": env.lambda_ac227_per_h,
    }


def _run_endpoint(phi: float, energy_ev: float, time_h: float, ic: np.ndarray,
                  n_points: int = 801) -> np.ndarray:
    env = R.IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    _, y = R.run_simulation(
        env, t_end_h=time_h, n_points=n_points,
        N_ra0=ic[0], N_ra225_0=ic[1], N_ac0=ic[2], N_ra227_0=ic[3], N_ac227_0=ic[4],
    )
    return y[-1]


def _milking_harvest_bq(phi: float, energy_ev: float, t_irr_h: float,
                        n_milk: int, dt_milk_h: float, lam_ac225: float) -> float:
    """
    Irradiate 1 g for t_irr_h, then perform n_milk post-EOB milkings spaced
    dt_milk_h apart, harvesting ALL Ac-225 at each milking (Ra-225 keeps
    decaying into it between harvests). Returns total harvested Ac-225
    activity [Bq], each harvest valued as A = lambda * N at harvest time.
    """
    state = _run_endpoint(phi, energy_ev, t_irr_h, np.array([ONE_G_RA226, 0, 0, 0, 0], float))
    total_bq = 0.0
    for _ in range(n_milk):
        state = _run_endpoint(0.0, energy_ev, dt_milk_h, state)
        total_bq += lam_ac225 * state[2] / 3600.0  # A [Bq] = lambda[/h] / 3600 * N
        state = state.copy()
        state[2] = 0.0  # milk off all Ac-225
    return total_bq


def _lit_rows() -> list[dict]:
    with LIT_CSV.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("source_citation")]


def _find(rows: list[dict], needle: str) -> dict:
    hits = [r for r in rows if needle in r["source_citation"]]
    if not hits:
        raise SystemExit(f"anchor row {needle!r} not found in {LIT_CSV}")
    return hits[0]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    rows = _lit_rows()

    # --- anchor definitions pulled from the merged literature CSV ----------
    sano = _find(rows, "Sano et al. 2024")
    iwa = _find(rows, "Iwahashi et al. 2022")
    hogle_d3 = _find(rows, "HFIR 226Ra->227Ac series")
    hogle_d7 = _find(rows, "ORNL HFIR Ra-226 irradiation")
    hogle_d26 = _find(rows, "HFIR 226Ra->227Ac saturation")
    snow = _find(rows, "OSTI 3028837 full text")

    anchors: list[dict] = [
        {
            "id": "sano_2024_joyo_45d",
            "kind": "ac225_activity",
            "source": sano["source_citation"],
            "phi": float(sano["phi_n_cm2_s"]), "energy_ev": float(sano["energy_ev"]),
            "time_h": float(sano["time_h"]), "n_ra0": float(sano["N_Ra226_0"]),
            "measured_bq": float(sano["A_Ac225_Bq"]), "measured_err_bq": 6.2e9,
        },
        {
            "id": "iwahashi_2022_joyo_60d",
            "kind": "ac225_activity",
            "source": iwa["source_citation"],
            "phi": float(iwa["phi_n_cm2_s"]), "energy_ev": float(iwa["energy_ev"]),
            "time_h": float(iwa["time_h"]), "n_ra0": float(iwa["N_Ra226_0"]),
            "measured_bq": float(iwa["A_Ac225_Bq"]), "measured_err_bq": None,
        },
        {
            "id": "iwahashi_2022_milking_17p5d",
            "kind": "ac225_milking",
            "source": "Iwahashi et al. 2022 per data/evaluated/README_retrieval.md "
                      "(15.7 GBq/cycle, 3x milking at 17.5 d optimum)",
            "phi": 5.7e15, "energy_ev": 1.0e7, "t_irr_h": 60.0 * 24.0,
            "n_milk": 3, "dt_milk_h": 17.5 * 24.0,
            "measured_bq": 15.7e9, "measured_err_bq": None,
            "assumptions": (
                "Milking modelled as post-EOB harvests every 17.5 d (x3) after a "
                "60-d irradiation of 1 g Ra-226 at phi=5.7e15, E=10 MeV; each "
                "harvest removes all Ac-225. The paper's exact in-cycle milking "
                "schedule is not public in the retrieved material — ASSUMPTION-LADEN."
            ),
        },
        {
            "id": "hogle_2016_ac227_3p01d",
            "kind": "ac227_activity",
            "source": hogle_d3["source_citation"],
            "phi": float(hogle_d3["phi_n_cm2_s"]), "energy_ev": float(hogle_d3["energy_ev"]),
            "time_h": float(hogle_d3["time_h"]), "n_ra0": float(hogle_d3["N_Ra226_0"]),
            "measured_bq": float(hogle_d3["A_Ac227_Bq"]), "measured_err_bq": 1100.0,
        },
        {
            "id": "hogle_2016_ac227_7d",
            "kind": "ac227_activity",
            "source": hogle_d7["source_citation"],
            "phi": float(hogle_d7["phi_n_cm2_s"]), "energy_ev": float(hogle_d7["energy_ev"]),
            "time_h": float(hogle_d7["time_h"]), "n_ra0": float(hogle_d7["N_Ra226_0"]),
            "measured_bq": float(hogle_d7["A_Ac227_Bq"]), "measured_err_bq": 2200.0,
        },
        {
            "id": "hogle_2016_ac227_26p09d",
            "kind": "ac227_activity",
            "source": hogle_d26["source_citation"],
            "phi": float(hogle_d26["phi_n_cm2_s"]), "energy_ev": float(hogle_d26["energy_ev"]),
            "time_h": float(hogle_d26["time_h"]), "n_ra0": float(hogle_d26["N_Ra226_0"]),
            "measured_bq": float(hogle_d26["A_Ac227_Bq"]), "measured_err_bq": 7400.0,
        },
        {
            "id": "snow_2025_phi0_ingrowth_17d",
            "kind": "ac225_ingrowth",
            "source": snow["source_citation"],
            # Ra-225 at EOB = 285.2 ± 28.5 Bq (Snow full text, via merged CSV notes);
            # converted to atoms with the NuDat Ra-225 half-life (14.9 d).
            "ra225_eob_bq": 285.2, "ra225_eob_err_bq": 28.5,
            "time_h": float(snow["time_h"]),  # 408 h = 17 d post-EOB
            "measured_bq": float(snow["A_Ac225_Bq"]), "measured_err_bq": 12.6,
        },
    ]

    lam = {v: _lambdas_per_h(v) for v in VERSIONS}

    results: list[dict] = []
    for a in anchors:
        entry: dict = {
            "id": a["id"], "kind": a["kind"], "source": a["source"],
            "measured_bq": a["measured_bq"], "measured_err_bq": a["measured_err_bq"],
        }
        if a.get("assumptions"):
            entry["assumptions"] = a["assumptions"]
        for v in VERSIONS:
            _set_version(v)
            if a["kind"] == "ac225_activity":
                ic = np.array([a["n_ra0"], 0, 0, 0, 0], float)
                end = _run_endpoint(a["phi"], a["energy_ev"], a["time_h"], ic)
                pred = lam[v]["ac225"] * end[2] / 3600.0
            elif a["kind"] == "ac227_activity":
                ic = np.array([a["n_ra0"], 0, 0, 0, 0], float)
                end = _run_endpoint(a["phi"], a["energy_ev"], a["time_h"], ic)
                pred = lam[v]["ac227"] * end[4] / 3600.0
            elif a["kind"] == "ac225_milking":
                pred = _milking_harvest_bq(
                    a["phi"], a["energy_ev"], a["t_irr_h"], a["n_milk"], a["dt_milk_h"],
                    lam[v]["ac225"],
                )
            elif a["kind"] == "ac225_ingrowth":
                # atoms of Ra-225 at EOB from measured activity, using the NuDat
                # Ra-225 half-life (14.9 d) for the conversion in BOTH versions.
                n_ra225 = a["ra225_eob_bq"] / (np.log(2.0) / (14.9 * 24.0)) * 3600.0
                ic = np.array([0, n_ra225, 0, 0, 0], float)
                end = _run_endpoint(0.0, 0.025, a["time_h"], ic)
                pred = lam[v]["ac225"] * end[2] / 3600.0
            else:  # pragma: no cover
                raise ValueError(a["kind"])
            entry[f"{v}_bq"] = pred
            entry[f"{v}_ratio_pred_over_meas"] = pred / a["measured_bq"]
            entry[f"{v}_rel_error"] = abs(pred - a["measured_bq"]) / a["measured_bq"]
        results.append(entry)
        print(
            f"{a['id']:32s} meas={a['measured_bq']:11.4e} Bq | "
            f"v1={entry['v1_bq']:11.4e} ({entry['v1_ratio_pred_over_meas']:8.2f}x) | "
            f"v2={entry['v2_bq']:11.4e} ({entry['v2_ratio_pred_over_meas']:8.2f}x)"
        )

    # --- data-layer cross-checks (all numbers trace to data/evaluated/) ----
    _set_version("v2")
    ev = R.load_evaluated_nuclear_data()
    sigma_check = {
        "jendl5_vs_endfb8_n2n_max_dev_b": ev.n2n_jendl_vs_endfb8_max_dev_b,
        "jendl5_vs_endfb8_ng_max_dev_b": ev.ng_jendl_vs_endfb8_max_dev_b,
        "n2n_threshold_ev": ev.n2n_threshold_ev,
        "eval_n2n_b_at_14MeV": float(R.sigma_n2n_eval_b(14.0e6)),
        "eval_n2n_b_at_14p5MeV_interp": float(R.sigma_n2n_eval_b(14.5e6)),
        "exfor21405_n2n_b_at_14p5MeV": R.EXFOR_N2N_14P5MEV_B,
        "exfor21405_n2n_err_b": R.EXFOR_N2N_14P5MEV_ERR_B,
        "eval_over_exfor_at_14p5MeV": float(R.sigma_n2n_eval_b(14.5e6)) / R.EXFOR_N2N_14P5MEV_B,
        "v1_sigmoid_n2n_b_at_14MeV": 27e-27 * float(R.sigma_scale_threshold_n2n(14.0e6)) / 1e-24,
        "v2_over_v1_n2n_at_14MeV": float(R.sigma_n2n_eval_b(14.0e6)) / (
            27e-27 * float(R.sigma_scale_threshold_n2n(14.0e6)) / 1e-24),
        "v1_ngamma_b_thermal": 12.8,
        "v2_ngamma_b_at_0p0253eV": float(R.sigma_ngamma_eval_b(0.0253)),
        "exfor31760_ngamma_b_thermal": R.THERMAL_NGAMMA_ANCHOR_B,
        "half_lives_v2_hours": {k: v_ / 3600.0 for k, v_ in ev.half_life_s.items()},
        "half_lives_v1_note": "v1: Ra-226 1600 y, Ra-225 14.8 d, Ac-225 9.92 d, "
                              "Ra-227 42.2 min, Ac-227 21.772 y (hard-coded)",
    }

    summary = {
        "generated": "2026-07-18",
        "script": "analysis/validate_ode_v2.py",
        "description": (
            "ODE predictions vs literature anchors under ODE_DATA_VERSION=v1 "
            "(legacy synthetic sigmoid) and v2 (evaluated JENDL-5/EXFOR/NuDat). "
            "No tuning applied; ratios are prediction/measurement. Activities use "
            "each version's own decay constants. The Joyo anchors treat the full "
            "quoted core flux as monoenergetic at the CSV representative energy — "
            "a deliberate apples-to-apples scenario comparison, NOT a spectrum-"
            "folded reactor calculation (see docs/DATA_PROVENANCE.md)."
        ),
        "data_files": [
            "data/evaluated/jendl5_ra226_n2n_sigmaE.csv",
            "data/evaluated/endfb8_ra226_n2n_sigmaE.csv",
            "data/evaluated/jendl5_ra226_ngamma_sigmaE.csv",
            "data/evaluated/endfb8_ra226_ngamma_sigmaE.csv",
            "data/evaluated/exfor_ra226_n2n.csv",
            "data/evaluated/exfor_ra226_ngamma_thermal.csv",
            "data/evaluated/halflives_nndc.csv",
            "data/literature_benchmarks.csv (merged 2026-07-18)",
        ],
        "sigma_and_halflife_checks": sigma_check,
        "anchors": results,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _set_version("v1")  # leave the process in the default state
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
