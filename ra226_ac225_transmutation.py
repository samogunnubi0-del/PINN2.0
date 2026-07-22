"""
Bateman-style simulation for the five-species Ra-226 transmutation chain.

Two competing neutron channels on Ra-226:

  (n,2n)  Ra-226 -> Ra-225 -> Ac-225   (desired product)  [THRESHOLD reaction, ≥6.42 MeV]
  (n,γ)   Ra-226 -> Ra-227 -> Ac-227   (impurity pathway) [1/v reaction, active at thermal]

    dN_Ra226/dt = -lambda_226 * N_226 - k_n2n * N_226 - k_ngamma * N_226
    dN_Ra225/dt = +k_n2n * N_226 - lambda_225 * N_225
    dN_Ac225/dt = +lambda_225 * N_225 - lambda_Ac225 * N_Ac225
    dN_Ra227/dt = +k_ngamma * N_226 - lambda_227 * N_227
    dN_Ac227/dt = +lambda_227 * N_227 - lambda_Ac227 * N_Ac227

PHYSICS NOTES (NNDC / JENDL-5 / ENDF/B-VIII.0 verified):
  sigma_n2n(Ra-226):  threshold ~6.42 MeV; 27 mb spectrum-averaged (fast reactor).
                      ZERO for thermal neutrons. NOT 1/v.
  sigma_ngamma(Ra-226): ~12.8 barns at thermal (0.025 eV). Follows 1/v law.
  Ra-225 T½: 14.8 days (NNDC NuDat3 best value, prev 14.9 was within uncertainty)
  Ra-226 T½: 1600 years | Ac-225 T½: 9.92 days
  Ra-227 T½: 42.2 min  | Ac-227 T½: 21.772 years

Solver upgraded from odeint → solve_ivp with method='Radau' (A-stable stiff solver),
which handles the stiffness ratio λ_Ra227/λ_Ac225 ≈ 338 correctly.

DATA VERSION FLAG (ODE_DATA_VERSION env var):
  v1 (default): legacy synthetic physics — sigmoid (n,2n) saturating at 27 mb,
      12.8 b thermal (n,γ) with 1/v law, hard-coded half-lives. Bit-preserved:
      all existing checkpoints and smoke tests were produced against v1.
  v2: evaluated nuclear data retrieved 2026-07-18 (see data/evaluated/ and
      docs/DATA_PROVENANCE.md): JENDL-5 σ(n,2n)(E) table (ENDF/B-VIII.0 adopted
      the same evaluation), JENDL-5 σ(n,γ)(E) table with a 1/v tail below 1 keV
      anchored at the experimental 13.8 b thermal point (Bagheri 2015, EXFOR
      31760 — the libraries themselves give ZERO thermal capture below 1 keV),
      and NuDat 3 half-lives. The v1 sigmoid is ~28× too small at 14 MeV
      (27 mb vs evaluated 755.7 mb).
"""

from __future__ import annotations

import csv
import os
import pathlib
import sys
from dataclasses import dataclass, field
from enum import Enum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Physical constants  (NNDC NuDat3 / JENDL-5 / ENDF/B-VIII.0)
# ---------------------------------------------------------------------------
LN2 = np.log(2.0)

HALF_LIFE_RA226_H  = 1600.0   * 365.25 * 24.0   # 1600 y
HALF_LIFE_RA225_H  = 14.8     * 24.0             # 14.8 d  (NNDC best value)
HALF_LIFE_AC225_H  = 9.920    * 24.0             # 9.920 d  (NNDC best value)
HALF_LIFE_RA227_H  = 42.2     / 60.0             # 42.2 min
HALF_LIFE_AC227_H  = 21.772   * 365.25 * 24.0    # 21.772 y

# Cross sections (NNDC / JENDL-5)
SIGMA_NGAMMA_THERMAL_CM2 = 12.8e-24   # 12.8 barn thermal (n,γ) — correct, 1/v law applies
SIGMA_N2N_FAST_CM2       = 27e-27     # 27 mb spectrum-averaged fast reactor (JENDL-5)
                                       # ZERO for thermal energies (threshold ~6.42 MeV)

# (n,2n) threshold energy
E_THRESHOLD_N2N_EV = 6.42e6   # 6.42 MeV — Ra-226(n,2n) threshold (ENDF/B-VIII.0)

THERMAL_REFERENCE_EV = 0.025   # thermal energy reference for 1/v scaling
PINN_ENERGY_MIN_EV   = 0.025
PINN_ENERGY_MAX_EV   = 2.0e7   # 20 MeV — covers full fast neutron range


# ---------------------------------------------------------------------------
# Versioned data layer (ODE_DATA_VERSION) — see docs/DATA_PROVENANCE.md
# ---------------------------------------------------------------------------
ODE_DATA_VERSION_ENV = "ODE_DATA_VERSION"
_VALID_DATA_VERSIONS = ("v1", "v2")

_EVALUATED_DIR = pathlib.Path(__file__).resolve().parent / "data" / "evaluated"

# v2 (n,γ) handling: ENDF/B-VIII.0 and JENDL-5 both tabulate ZERO capture below
# 1 keV (no evaluated thermal capture at all). The first NONZERO evaluated
# point is at 1 keV. Below that energy we fall back to a 1/v extrapolation
# anchored at the experimental thermal point 13.8 ± 0.3 b at 0.0253 eV
# (Bagheri et al. 2015, EXFOR entry 31760 — the modern recommended value).
NGAMMA_EVAL_TABLE_MIN_EV   = 1.0e3
THERMAL_NGAMMA_ANCHOR_B    = 13.8
THERMAL_NGAMMA_ANCHOR_EV   = 0.0253
# v2 (n,2n): only ONE experimental point exists (1.60 ± 0.20 b @ 14.5 MeV,
# EXFOR 21405, O'Connor & Perkin 1960); the evaluated shape is theory-based.
EXFOR_N2N_14P5MEV_B        = 1.60
EXFOR_N2N_14P5MEV_ERR_B    = 0.20
EXFOR_N2N_14P5MEV_EV       = 14.5e6


def ode_data_version() -> str:
    """Active ODE data version: 'v1' (default, legacy) or 'v2' (evaluated)."""
    v = os.environ.get(ODE_DATA_VERSION_ENV, "v1").strip().lower()
    if v not in _VALID_DATA_VERSIONS:
        raise ValueError(
            f"{ODE_DATA_VERSION_ENV} must be one of {_VALID_DATA_VERSIONS}, got {v!r}"
        )
    return v


def _read_evaluated_sigma_table(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a data/evaluated σ(E) CSV: comment/header lines start with '#',
    data rows are 'energy_ev,sigma_barn,sigma_mb'. Returns (energy_ev, sigma_b).
    Fails loudly if the file is missing (real data, no silent fallback).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluated data file missing: {path}. These files are required for "
            f"{ODE_DATA_VERSION_ENV}=v2; see data/evaluated/README_retrieval.md "
            "for how to re-fetch them from IAEA-NDS / NNDC."
        )
    energies: list[float] = []
    sigmas: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            energies.append(float(row[0]))
            sigmas.append(float(row[1]))  # barns
    return np.asarray(energies, dtype=np.float64), np.asarray(sigmas, dtype=np.float64)


def _read_evaluated_half_lives(path: pathlib.Path) -> dict[str, float]:
    """Parse halflives_nndc.csv -> {nuclide: half_life_seconds}. Fails loudly."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluated data file missing: {path}. Required for "
            f"{ODE_DATA_VERSION_ENV}=v2; see data/evaluated/README_retrieval.md."
        )
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            out[row[0].strip()] = float(row[3])  # half_life_seconds column
    return out


@dataclass(frozen=True)
class EvaluatedNuclearData:
    """Container for the real evaluated datasets (ODE_DATA_VERSION=v2)."""
    n2n_energy_ev: np.ndarray          # JENDL-5 σ(n,2n) grid [eV]
    n2n_sigma_b: np.ndarray            # JENDL-5 σ(n,2n) [barn]
    ng_energy_ev: np.ndarray           # JENDL-5 σ(n,γ) grid, nonzero part (E ≥ 1 keV)
    ng_sigma_b: np.ndarray             # JENDL-5 σ(n,γ) [barn]
    half_life_s: dict                  # nuclide -> half-life [s] (NuDat 3)
    n2n_threshold_ev: float            # first tabulated energy (= 6.4218 MeV)
    n2n_jendl_vs_endfb8_max_dev_b: float   # cross-library check (barns)
    ng_jendl_vs_endfb8_max_dev_b: float    # cross-library check (barns)


_EVALUATED_CACHE: EvaluatedNuclearData | None = None


def load_evaluated_nuclear_data() -> EvaluatedNuclearData:
    """
    Load (cached) the real evaluated datasets from data/evaluated/.
    Cross-checks the JENDL-5 and ENDF/B-VIII.0 tables against each other and
    records the max deviation (ENDF/B-VIII.0 adopted the JENDL Ra-226
    evaluation, so the tables should agree exactly).
    """
    global _EVALUATED_CACHE
    if _EVALUATED_CACHE is not None:
        return _EVALUATED_CACHE

    e_n2n_j, s_n2n_j = _read_evaluated_sigma_table(_EVALUATED_DIR / "jendl5_ra226_n2n_sigmaE.csv")
    e_n2n_e, s_n2n_e = _read_evaluated_sigma_table(_EVALUATED_DIR / "endfb8_ra226_n2n_sigmaE.csv")
    e_ng_j, s_ng_j = _read_evaluated_sigma_table(_EVALUATED_DIR / "jendl5_ra226_ngamma_sigmaE.csv")
    e_ng_e, s_ng_e = _read_evaluated_sigma_table(_EVALUATED_DIR / "endfb8_ra226_ngamma_sigmaE.csv")

    # Cross-library agreement check (same grid expected; compare on union grid).
    dev_n2n = float(np.max(np.abs(s_n2n_j - s_n2n_e))) if e_n2n_j.shape == e_n2n_e.shape \
        and np.allclose(e_n2n_j, e_n2n_e) else float("nan")
    dev_ng = float(np.max(np.abs(s_ng_j - s_ng_e))) if e_ng_j.shape == e_ng_e.shape \
        and np.allclose(e_ng_j, e_ng_e) else float("nan")

    # (n,γ): keep only the NONZERO evaluated points (E ≥ 1 keV). The zero
    # tabulation below 1 keV is a library gap, not physics — the experimental
    # 1/v tail anchored at 13.8 b (EXFOR 31760) takes over there. Note the raw
    # table repeats E=1 keV twice (σ=0 then σ=3.111 b, an ENDF discontinuity);
    # keeping only nonzero points resolves the duplicate-x ambiguity.
    mask = s_ng_j > 0.0
    e_ng = e_ng_j[mask]
    s_ng = s_ng_j[mask]

    _EVALUATED_CACHE = EvaluatedNuclearData(
        n2n_energy_ev=e_n2n_j,
        n2n_sigma_b=s_n2n_j,
        ng_energy_ev=e_ng,
        ng_sigma_b=s_ng,
        half_life_s=_read_evaluated_half_lives(_EVALUATED_DIR / "halflives_nndc.csv"),
        n2n_threshold_ev=float(e_n2n_j[0]),
        n2n_jendl_vs_endfb8_max_dev_b=dev_n2n,
        ng_jendl_vs_endfb8_max_dev_b=dev_ng,
    )
    return _EVALUATED_CACHE


def sigma_n2n_eval_b(energy_ev: float | np.ndarray) -> float | np.ndarray:
    """
    v2 evaluated σ(n,2n)(E) [barn], JENDL-5 (= ENDF/B-VIII.0) pointwise.
    Zero below the 6.4218 MeV threshold; linear interpolation on the tabulated
    grid (the first tabulated point is σ=0 at threshold, so clamping handles
    sub-threshold energies); held constant above the last tabulated point
    (20 MeV) — documented extrapolation clamp.
    """
    ev = load_evaluated_nuclear_data()
    e = np.asarray(energy_ev, dtype=np.float64)
    return np.interp(e, ev.n2n_energy_ev, ev.n2n_sigma_b,
                     left=0.0, right=float(ev.n2n_sigma_b[-1]))


def sigma_ngamma_eval_b(energy_ev: float | np.ndarray) -> float | np.ndarray:
    """
    v2 σ(n,γ)(E) [barn]: JENDL-5 evaluated table for E ≥ 1 keV (linear
    interpolation, clamped to the last point above 20 MeV); below the table's
    lower bound (1 keV, where BOTH libraries tabulate zero capture) falls back
    to a 1/v extrapolation anchored at the experimental 13.8 b thermal point
    (Bagheri 2015, EXFOR 31760, 0.0253 eV).
    """
    ev = load_evaluated_nuclear_data()
    e = np.asarray(energy_ev, dtype=np.float64)
    e_safe = np.maximum(e, 1e-30)
    tab = np.interp(e_safe, ev.ng_energy_ev, ev.ng_sigma_b,
                    left=float(ev.ng_sigma_b[0]), right=float(ev.ng_sigma_b[-1]))
    one_over_v = THERMAL_NGAMMA_ANCHOR_B * np.sqrt(THERMAL_NGAMMA_ANCHOR_EV / e_safe)
    return np.where(e_safe < NGAMMA_EVAL_TABLE_MIN_EV, one_over_v, tab)


def half_life_hours_eval(nuclide: str) -> float:
    """v2 half-life [hours] from the NuDat 3 retrieval (halflives_nndc.csv)."""
    return load_evaluated_nuclear_data().half_life_s[nuclide] / 3600.0


# ---------------------------------------------------------------------------
# Spectrum folding (ODE_DATA_VERSION=v2 only; SPECTRUM_MODE env var)
# ---------------------------------------------------------------------------
# Why: pointwise (monoenergetic) v2 exposes that treating the whole Joyo core
# flux as >threshold neutrons overpredicts the Joyo anchors by 370-2150x.
# The yield is controlled by the SMALL high-energy tail above the 6.4218 MeV
# (n,2n) threshold, so reactor scenarios must fold sigma(E) over a spectrum:
#     <sigma> = ∫ sigma(E) phi(E) dE / ∫ phi(E) dE
# The measured Joyo MK-III spectrum is paywalled, so the spectra here are
# PARAMETRIC ASSUMPTIONS with citable functional forms:
#   watt:     bare U-235 thermal-fission Watt spectrum (Watt 1952; ENDF
#             standard parameters a=0.988 MeV, b=2.249 1/MeV). No free knobs.
#   twogroup: fraction f of flux in a fast group with the Watt shape above
#             the threshold, rest below threshold (thermal for (n,gamma)).
#             f = SPECTRUM_FAST_FRACTION (default 0.01) is an ADJUSTABLE
#             ASSUMPTION — see results/ode_data_v2_spectrum_20260718.json
#             for the value INFERRED from the Sano 2024 anchor.
SPECTRUM_MODE_ENV = "SPECTRUM_MODE"
SPECTRUM_FAST_FRACTION_ENV = "SPECTRUM_FAST_FRACTION"
_VALID_SPECTRUM_MODES = ("mono", "watt", "twogroup")
DEFAULT_FAST_FRACTION = 0.01

# Standard U-235 thermal-fission Watt parameters (Watt 1952, Phys. Rev. 87,
# 1037; same values tabulated in the ENDF-6 formats manual / ENDF/B-VIII.0).
WATT_A_MEV = 0.988
WATT_B_INV_MEV = 2.249


def spectrum_mode() -> str:
    """Active spectrum mode for v2: 'mono' (default), 'watt', or 'twogroup'."""
    m = os.environ.get(SPECTRUM_MODE_ENV, "mono").strip().lower()
    if m not in _VALID_SPECTRUM_MODES:
        raise ValueError(
            f"{SPECTRUM_MODE_ENV} must be one of {_VALID_SPECTRUM_MODES}, got {m!r}"
        )
    return m


def spectrum_fast_fraction() -> float:
    """Above-threshold fast-group fraction f for twogroup mode (default 0.01)."""
    raw = os.environ.get(SPECTRUM_FAST_FRACTION_ENV, "")
    if not raw.strip():
        return DEFAULT_FAST_FRACTION
    f = float(raw)
    if not (0.0 < f <= 1.0):
        raise ValueError(f"{SPECTRUM_FAST_FRACTION_ENV} must be in (0, 1], got {f}")
    return f


def watt_spectrum_density(energy_ev: float | np.ndarray) -> float | np.ndarray:
    """Unnormalized Watt fission-spectrum density C·exp(-E/a)·sinh(√(bE))."""
    x = np.asarray(energy_ev, dtype=np.float64) / 1.0e6  # MeV
    return np.exp(-x / WATT_A_MEV) * np.sinh(np.sqrt(WATT_B_INV_MEV * x))


_FOLD_GRID_EV: np.ndarray | None = None
_FOLD_CACHE: dict[tuple[str, float], tuple[float, float]] = {}


def _fold_grid() -> np.ndarray:
    """Dense log-spaced quadrature grid for spectrum folding (0.01 eV–20 MeV)."""
    global _FOLD_GRID_EV
    if _FOLD_GRID_EV is None:
        _FOLD_GRID_EV = np.logspace(-2.0, np.log10(2.0e7), 200_000)
    return _FOLD_GRID_EV


def spectrum_averaged_sigmas_b(
    mode: str | None = None,
    fast_fraction: float | None = None,
) -> tuple[float, float]:
    """
    Spectrum-averaged one-group cross sections (<sigma_n2n>, <sigma_ngamma>)
    in barns, folded over the evaluated JENDL-5 sigma(E) tables (v2 data).
    Cached. Only 'watt' and 'twogroup' are foldable; 'mono' raises ValueError
    (monoenergetic scenarios should use sigma_*_eval_b pointwise instead).
    """
    m = (mode or spectrum_mode()).strip().lower()
    if m == "mono":
        raise ValueError("mono mode uses pointwise sigma(E), not spectrum folding")
    if m not in _VALID_SPECTRUM_MODES:
        raise ValueError(f"unknown spectrum mode {m!r}")
    f = spectrum_fast_fraction() if fast_fraction is None else float(fast_fraction)
    key = (m, f)
    if key in _FOLD_CACHE:
        return _FOLD_CACHE[key]

    ev = load_evaluated_nuclear_data()
    E = _fold_grid()
    s_n2n = np.interp(E, ev.n2n_energy_ev, ev.n2n_sigma_b,
                      left=0.0, right=float(ev.n2n_sigma_b[-1]))
    s_ng = np.asarray(sigma_ngamma_eval_b(E), dtype=np.float64)
    w = watt_spectrum_density(E)
    norm = float(np.trapezoid(w, E))

    if m == "watt":
        # Bare normalized Watt spectrum over the full range.
        avg_n2n = float(np.trapezoid(s_n2n * w, E) / norm)
        avg_ng = float(np.trapezoid(s_ng * w, E) / norm)
    else:  # twogroup
        # Fast group: Watt-shaped tail ABOVE the (n,2n) threshold, fraction f.
        above = E >= ev.n2n_threshold_ev
        w_a = w[above]
        norm_a = float(np.trapezoid(w_a, E[above]))
        fast_n2n = float(np.trapezoid(s_n2n[above] * w_a, E[above]) / norm_a)
        fast_ng = float(np.trapezoid(s_ng[above] * w_a, E[above]) / norm_a)
        # Slow group: thermalized at the experimental anchor energy (documented
        # simplification; irrelevant for fast-reactor (n,2n) anchors).
        slow_ng = float(sigma_ngamma_eval_b(THERMAL_NGAMMA_ANCHOR_EV))
        avg_n2n = f * fast_n2n
        avg_ng = (1.0 - f) * slow_ng + f * fast_ng

    _FOLD_CACHE[key] = (avg_n2n, avg_ng)
    return _FOLD_CACHE[key]


def watt_fraction_above_threshold() -> float:
    """Fraction of the bare Watt fission spectrum above the (n,2n) threshold."""
    ev = load_evaluated_nuclear_data()
    E = _fold_grid()
    w = watt_spectrum_density(E)
    above = E >= ev.n2n_threshold_ev
    return float(np.trapezoid(w[above], E[above]) / np.trapezoid(w, E))


def sigma_scale_one_over_v(energy_ev: float | np.ndarray,
                            reference_ev: float = THERMAL_REFERENCE_EV) -> float | np.ndarray:
    """
    1/v cross-section scaling for (n,γ): σ(E) ∝ 1/v ∝ E^{-1/2}.
    Returns dimensionless factor σ(E)/σ(reference).
    Vectorised for numpy arrays.
    """
    E = np.maximum(np.asarray(energy_ev, dtype=float), 1e-30)
    return np.sqrt(float(reference_ev) / E)


def sigma_scale_threshold_n2n(energy_ev: float | np.ndarray,
                               threshold_ev: float = E_THRESHOLD_N2N_EV,
                               width_ev: float = 5e5) -> float | np.ndarray:
    """
    Smooth threshold model for (n,2n) cross section.
    Uses sigmoid so it is differentiable — important so the ODE data is smooth.
    Returns 0 well below threshold, rises to 1 above.
    Peak cross section is SIGMA_N2N_FAST_CM2 * this factor.
    """
    E = np.asarray(energy_ev, dtype=float)
    return 1.0 / (1.0 + np.exp(-(E - threshold_ev) / width_ev))


class NeutronEnergyGroup(str, Enum):
    THERMAL    = "Thermal"
    EPITHERMAL = "Epithermal"
    FAST       = "Fast"


# Energy group → approximate representative energy in eV for threshold calculation
_ENERGY_GROUP_EV: dict[str, float] = {
    NeutronEnergyGroup.THERMAL.value:    0.025,
    NeutronEnergyGroup.EPITHERMAL.value: 1.0,
    NeutronEnergyGroup.FAST.value:       14.0e6,
}


@dataclass
class IsotopeEnvironment:
    """
    Nuclear parameters for the five-species transmutation + decay model.

    Two competing neutron channels on Ra-226:
        (n,2n)  -> Ra-225  [threshold reaction; needs E > 6.42 MeV]
        (n,γ)   -> Ra-227  [1/v reaction; active at thermal energies]
    """
    phi: float = 1.0e14
    sigma_ra226: float = SIGMA_N2N_FAST_CM2       # (n,2n) reference [cm²]  27 mb
    sigma_ngamma: float = SIGMA_NGAMMA_THERMAL_CM2 # (n,γ) reference  [cm²] 12.8 b
    energy_group: str = NeutronEnergyGroup.FAST.value
    neutron_energy_ev: float | None = None
    target_mass_g: float = 1.0
    # v2 only: 'mono' (default = pointwise sigma(E)) | 'watt' | 'twogroup'.
    # None -> read SPECTRUM_MODE env. Scenario-level hook for the future
    # trajectory-dataset change documented in docs/DATA_PROVENANCE.md §6.
    spectrum: str | None = None

    lambda_ra226_per_h: float = field(init=False)
    lambda_ra225_per_h: float = field(init=False)
    lambda_ac225_per_h: float = field(init=False)
    lambda_ra227_per_h: float = field(init=False)
    lambda_ac227_per_h: float = field(init=False)

    # Separate energy scales for the two channels
    _ng_energy_scale:  float = field(init=False)   # 1/v factor for (n,γ)
    _n2n_energy_scale: float = field(init=False)   # threshold factor for (n,2n)

    def __post_init__(self) -> None:
        version = ode_data_version()

        if version == "v2":
            # NuDat 3 half-lives (Ra-226 1603 y, Ra-225 14.9 d, Ac-225 10.0 d,
            # Ra-227 42.2 min, Ac-227 21.772 y) — halflives_nndc.csv.
            self.lambda_ra226_per_h = LN2 / half_life_hours_eval("Ra-226")
            self.lambda_ra225_per_h = LN2 / half_life_hours_eval("Ra-225")
            self.lambda_ac225_per_h = LN2 / half_life_hours_eval("Ac-225")
            self.lambda_ra227_per_h = LN2 / half_life_hours_eval("Ra-227")
            self.lambda_ac227_per_h = LN2 / half_life_hours_eval("Ac-227")
        else:
            # v1 legacy hard-coded half-lives (bit-preserved).
            self.lambda_ra226_per_h = LN2 / HALF_LIFE_RA226_H
            self.lambda_ra225_per_h = LN2 / HALF_LIFE_RA225_H
            self.lambda_ac225_per_h = LN2 / HALF_LIFE_AC225_H
            self.lambda_ra227_per_h = LN2 / HALF_LIFE_RA227_H
            self.lambda_ac227_per_h = LN2 / HALF_LIFE_AC227_H

        if self.neutron_energy_ev is not None:
            e_ev = float(self.neutron_energy_ev)
        else:
            e_ev = _ENERGY_GROUP_EV.get(self.energy_group, 14.0e6)

        if version == "v2":
            # Evaluated σ in cm². effective_sigma_*() multiplies the reference
            # σ by these scales, so the scales absorb the evaluated value and
            # sigma_ra226 / sigma_ngamma are inert under v2.
            mode = (self.spectrum or spectrum_mode()).strip().lower()
            if mode == "mono":
                # Monoenergetic scenario: pointwise evaluated σ(E).
                sig_n2n_cm2 = float(sigma_n2n_eval_b(e_ev)) * 1e-24
                sig_ng_cm2 = float(sigma_ngamma_eval_b(e_ev)) * 1e-24
            else:
                # Reactor scenario: spectrum-averaged one-group σ (the
                # scenario's neutron_energy_ev is then NOT the driver — the
                # parametric spectrum is). Assumption, see DATA_PROVENANCE §6.
                sig_n2n_b, sig_ng_b = spectrum_averaged_sigmas_b(mode)
                sig_n2n_cm2 = sig_n2n_b * 1e-24
                sig_ng_cm2 = sig_ng_b * 1e-24
            self._n2n_energy_scale = sig_n2n_cm2 / float(self.sigma_ra226)
            self._ng_energy_scale = sig_ng_cm2 / float(self.sigma_ngamma)
        else:
            # v1 legacy: (n,γ) 1/v scaling; (n,2n) smooth sigmoid threshold.
            self._ng_energy_scale = float(sigma_scale_one_over_v(e_ev))
            self._n2n_energy_scale = float(sigma_scale_threshold_n2n(e_ev))

    def effective_sigma_n2n(self) -> float:
        """(n,2n) effective cross section [cm²] — zero for thermal neutrons."""
        return float(self.sigma_ra226) * self._n2n_energy_scale

    def effective_sigma_ngamma(self) -> float:
        """(n,γ) effective cross section [cm²] — 1/v scaled."""
        return float(self.sigma_ngamma) * self._ng_energy_scale

    def k_n2n_per_h(self) -> float:
        shielding = float(np.exp(-0.01 * self.target_mass_g))
        return float(self.phi) * shielding * self.effective_sigma_n2n() * 3600.0

    def k_ngamma_per_h(self) -> float:
        shielding = float(np.exp(-0.01 * self.target_mass_g))
        return float(self.phi) * shielding * self.effective_sigma_ngamma() * 3600.0

    # Legacy compatibility
    def effective_sigma_ra226(self) -> float:
        return self.effective_sigma_n2n()

    def transmutation_rate_constant_per_s(self) -> float:
        return float(self.phi) * self.effective_sigma_n2n()

    def transmutation_rate_constant_per_h(self) -> float:
        return self.transmutation_rate_constant_per_s() * 3600.0


def bateman_transmutation_rhs(
    t: float,
    N: np.ndarray,
    k_n2n: float,
    k_ng: float,
    lam226: float,
    lam225: float,
    lam_ac: float,
    lam227: float,
    lam_ac7: float,
) -> np.ndarray:
    """
    RHS for solve_ivp: dN/dt for five-species system.
    Signature is (t, N, *args) — note t comes first (solve_ivp convention).
    """
    N_ra226 = max(float(N[0]), 0.0)
    N_ra225 = max(float(N[1]), 0.0)
    N_ac225 = max(float(N[2]), 0.0)
    N_ra227 = max(float(N[3]), 0.0)
    N_ac227 = max(float(N[4]), 0.0)

    dN_ra226 = -(lam226 + k_n2n + k_ng) * N_ra226
    dN_ra225 =  k_n2n * N_ra226 - lam225 * N_ra225
    dN_ac225 =  lam225 * N_ra225 - lam_ac * N_ac225
    dN_ra227 =  k_ng * N_ra226 - lam227 * N_ra227
    dN_ac227 =  lam227 * N_ra227 - lam_ac7 * N_ac227

    return np.array([dN_ra226, dN_ra225, dN_ac225, dN_ra227, dN_ac227], dtype=float)


def run_simulation(
    env: IsotopeEnvironment,
    t_end_h: float = 200.0,
    n_points: int = 501,
    N_ra0: float = 6.022e23,
    N_ra225_0: float = 0.0,
    N_ac0: float = 0.0,
    N_ra227_0: float = 0.0,
    N_ac227_0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the five-species system from t=0 to t=t_end_h using Radau (stiff solver).

    Returns (t_hours, Y) where Y has shape (n_points, 5):
        col 0: N_Ra226, col 1: N_Ra225, col 2: N_Ac225,
        col 3: N_Ra227, col 4: N_Ac227.
    """
    t_hours = np.linspace(0.0, float(t_end_h), int(n_points))
    y0 = np.array([N_ra0, N_ra225_0, N_ac0, N_ra227_0, N_ac227_0], dtype=float)

    k_n2n  = env.k_n2n_per_h()
    k_ng   = env.k_ngamma_per_h()
    lam226 = env.lambda_ra226_per_h
    lam225 = env.lambda_ra225_per_h
    lam_ac = env.lambda_ac225_per_h
    lam227 = env.lambda_ra227_per_h
    lam_ac7= env.lambda_ac227_per_h

    sol = solve_ivp(
        bateman_transmutation_rhs,
        t_span=(0.0, float(t_end_h)),
        y0=y0,
        method="Radau",          # A-stable stiff solver — handles Ra-227 T½=42min correctly
        t_eval=t_hours,
        args=(k_n2n, k_ng, lam226, lam225, lam_ac, lam227, lam_ac7),
        rtol=1e-9,
        atol=1e-12,
        dense_output=False,
    )

    Y = np.maximum(sol.y.T, 0.0)   # (n_points, 5), clip negatives from solver numerics
    return t_hours, Y


def ac225_peak_metrics(
    t_hours: np.ndarray,
    N_ac225: np.ndarray,
) -> tuple[float, float]:
    i_peak = int(np.argmax(N_ac225))
    return float(N_ac225[i_peak]), float(t_hours[i_peak])


def run_simulation_peak(
    env: IsotopeEnvironment,
    t_end_h: float,
    n_points: int | None = None,
    N_ra0: float = 6.022e23,
    N_ac0: float = 0.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    if n_points is None:
        n_points = max(400, int(np.clip(t_end_h * 8, 400, 15000)))
    t_h, Y = run_simulation(
        env, t_end_h=t_end_h, n_points=n_points, N_ra0=N_ra0, N_ra225_0=0.0, N_ac0=N_ac0
    )
    N_ac = Y[:, 2]
    max_yield, t_peak = ac225_peak_metrics(t_h, N_ac)
    return max_yield, t_peak, t_h, N_ac


def generate_training_runs(
    n_runs: int = 1000,
    sigma_ra226: float = SIGMA_N2N_FAST_CM2,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
    if rng is None:
        rng = np.random.default_rng()
    rows: list[dict[str, float]] = []
    for _ in range(n_runs):
        log_phi = rng.uniform(13.0, 15.0)
        phi = 10.0**log_phi
        total_time = float(rng.uniform(10.0, 500.0))
        env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma_ra226)
        max_ac225, time_to_peak, _, _ = run_simulation_peak(env, total_time, N_ra0=N_ra0)
        rows.append({
            "phi": phi, "sigma": sigma_ra226,
            "total_time": total_time,
            "max_ac225_yield": max_ac225, "time_to_peak": time_to_peak,
        })
    return rows


TARGETED_PHI_LOG_MIN = 11.0
TARGETED_PHI_LT      = 1e13
TARGETED_TIME_MIN_H  = 500.0
TARGETED_TIME_MAX_H  = 8760.0


def generate_targeted_data(
    n_runs: int = 200,
    sigma_ra226: float = SIGMA_N2N_FAST_CM2,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
    if rng is None:
        rng = np.random.default_rng()
    log_phi_hi = np.log10(TARGETED_PHI_LT) - 1e-6
    rows: list[dict[str, float]] = []
    for _ in range(n_runs):
        log_phi = float(rng.uniform(TARGETED_PHI_LOG_MIN, log_phi_hi))
        phi = 10.0**log_phi
        total_time = float(rng.uniform(TARGETED_TIME_MIN_H, TARGETED_TIME_MAX_H))
        env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma_ra226)
        max_ac225, time_to_peak, _, _ = run_simulation_peak(env, total_time, N_ra0=N_ra0)
        rows.append({
            "phi": phi, "sigma": sigma_ra226,
            "total_time": total_time,
            "max_ac225_yield": max_ac225, "time_to_peak": time_to_peak,
        })
    return rows


def append_training_csv(path: str, rows: list[dict[str, float]]) -> None:
    fieldnames = ["phi", "sigma", "total_time", "max_ac225_yield", "time_to_peak"]
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def augment_fail_zone_training(
    csv_path: str = "isotope_training_data.csv",
    n_targeted: int = 200,
    sigma_ra226: float = SIGMA_N2N_FAST_CM2,
    rng: np.random.Generator | None = None,
) -> int:
    rng = rng or np.random.default_rng(2026)
    new_rows = generate_targeted_data(n_runs=n_targeted, sigma_ra226=sigma_ra226, rng=rng)
    append_training_csv(csv_path, new_rows)
    return len(new_rows)


def save_training_csv(path: str, rows: list[dict[str, float]]) -> None:
    fieldnames = ["phi", "sigma", "total_time", "max_ac225_yield", "time_to_peak"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def generate_pinn_training_runs(
    n_runs: int = 1500,
    sigma_ra226: float = SIGMA_N2N_FAST_CM2,
    rng: np.random.Generator | None = None,
    N_ra0: float = 6.022e23,
) -> list[dict[str, float]]:
    """
    PINN-oriented dataset: random flux (log-uniform), neutron energy (eV, log-uniform
    between PINN_ENERGY_MIN_EV and PINN_ENERGY_MAX_EV = 20 MeV), and irradiation time.
    Energy range now covers fast neutrons so (n,2n) threshold fires at correct energies.
    """
    if rng is None:
        rng = np.random.default_rng()
    log_e_min = np.log10(PINN_ENERGY_MIN_EV)
    log_e_max = np.log10(PINN_ENERGY_MAX_EV)
    rows: list[dict[str, float]] = []
    for _ in range(n_runs):
        log_phi   = float(rng.uniform(13.0, 15.0))
        phi       = 10.0**log_phi
        time_h    = float(rng.uniform(10.0, 500.0))
        log_e     = float(rng.uniform(log_e_min, log_e_max))
        energy_ev = float(10.0**log_e)

        env = IsotopeEnvironment(phi=phi, sigma_ra226=sigma_ra226, neutron_energy_ev=energy_ev)
        n_pts = max(400, int(np.clip(time_h * 8, 400, 15000)))
        _, Y  = run_simulation(env, t_end_h=time_h, n_points=n_pts, N_ra0=N_ra0)
        end   = Y[-1]

        rows.append({
            "phi": phi, "energy": energy_ev, "time": time_h,
            "N_Ra226": float(end[0]), "N_Ra225": float(end[1]),
            "N_Ac225": float(end[2]), "N_Ra227": float(end[3]),
            "N_Ac227": float(end[4]),
        })
    return rows


def save_pinn_training_csv(path: str, rows: list[dict[str, float]]) -> None:
    fieldnames = ["phi", "energy", "time", "N_Ra226", "N_Ra225", "N_Ac225", "N_Ra227", "N_Ac227"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def plot_yield_heatmap_flux_time(
    rows: list[dict[str, float]],
    outfile: str = "graphs/ac225_yield_heatmap.png",
    n_bins_flux: int = 24,
    n_bins_time: int = 24,
) -> None:
    phi  = np.array([r["phi"] for r in rows], dtype=float)
    tt   = np.array([r["total_time"] for r in rows], dtype=float)
    yld  = np.array([r["max_ac225_yield"] for r in rows], dtype=float)
    phi_edges  = np.logspace(13.0, 15.0, n_bins_flux + 1)
    time_edges = np.linspace(10.0, 500.0, n_bins_time + 1)
    stat, _, _, _ = stats.binned_statistic_2d(phi, tt, yld, statistic="max",
                                               bins=[phi_edges, time_edges])
    Z = np.ma.masked_invalid(stat.T)
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y_m  = np.meshgrid(phi_edges, time_edges)
    pcm = ax.pcolormesh(X, Y_m, Z, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label=r"max $^{225}$Ac yield (atoms)")
    ax.set_xscale("log")
    ax.set_xlabel(r"Neutron flux $\phi$ (n cm$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("Irradiation time (hours)")
    ax.set_title(r"$^{225}$Ac yield heatmap (max per bin over Monte Carlo runs)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    try:
        import graph_provenance
        graph_provenance.record_graph_write(
            pathlib.Path(__file__).resolve().parent,
            pathlib.Path(outfile).resolve(),
            producer="ra226_ac225_transmutation.py",
            run_id=graph_provenance.new_run_id(),
        )
    except Exception:
        pass


def save_results_csv(path, t_hours, N_ra226, N_ra225, N_ac225):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_h", "N_Ra226", "N_Ra225", "N_Ac225"])
        for i in range(len(t_hours)):
            w.writerow([t_hours[i], N_ra226[i], N_ra225[i], N_ac225[i]])


def plot_ac225_growth(t_hours, N_ac225, outfile="graphs/ac225_growth.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_hours, N_ac225, color="darkgreen", lw=2, label=r"$^{225}$Ac")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(r"$N_{^{225}\mathrm{Ac}}$ (atoms)")
    ax.set_title(r"Growth of $^{225}$Ac from $^{226}$Ra transmutation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
        try:
            import graph_provenance
            graph_provenance.record_graph_write(
                pathlib.Path(__file__).resolve().parent,
                pathlib.Path(outfile).resolve(),
                producer="ra226_ac225_transmutation.py",
                run_id=graph_provenance.new_run_id(),
            )
        except Exception:
            pass
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(42)
    rows = generate_training_runs(n_runs=1000, sigma_ra226=SIGMA_N2N_FAST_CM2, rng=rng)
    save_training_csv("isotope_training_data.csv", rows)
    plot_yield_heatmap_flux_time(rows, outfile="graphs/ac225_yield_heatmap.png")


def main_pinn() -> None:
    rng = np.random.default_rng(43)
    rows = generate_pinn_training_runs(n_runs=1500, sigma_ra226=SIGMA_N2N_FAST_CM2, rng=rng)
    save_pinn_training_csv("pinn_training_data.csv", rows)
    print("Wrote pinn_training_data.csv with", len(rows), "rows")


def demo_single_run() -> None:
    env = IsotopeEnvironment(phi=1.0e14, sigma_ra226=SIGMA_N2N_FAST_CM2,
                             neutron_energy_ev=14e6)  # 14 MeV fast
    t_h, Y = run_simulation(env, t_end_h=200.0, n_points=501)
    save_results_csv("raw_physics_data.csv", t_h, Y[:, 0], Y[:, 1], Y[:, 2])
    plot_ac225_growth(t_h, Y[:, 2], outfile="graphs/ac225_growth.png")


if __name__ == "__main__":
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd == "augment":
        n = augment_fail_zone_training()
        print(f"Appended {n} targeted rows to isotope_training_data.csv")
    elif cmd == "pinn":
        main_pinn()
    else:
        main()
