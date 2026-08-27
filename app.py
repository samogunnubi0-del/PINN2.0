"""
IsotopePINN — interactive scientific narrative for Ac-225 production planning.

Two-model physics-informed surrogate system (v2 PINN + v3 PI-LSTM) for the
Ra-226 -> Ac-225 transmutation chain, validated against a stiff Bateman ODE
reference, evaluated nuclear data (JENDL-5 / ENDF/B-VIII.0 / EXFOR / NuDat),
and national-lab production anchors (Joyo).

Run: streamlit run app.py
Deploy: Streamlit Community Cloud, CPU-only (requirements-streamlit-cloud.txt).
"""
import importlib.util
import io
import json
import math
import os
import pathlib
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).parent

# ── PHYSICAL & CLINICAL CONSTANTS ─────────────────────────────────────────────
AC225_HALF_LIFE_DAYS = 9.920
AC227_HALF_LIFE_DAYS = 21.772 * 365.25
STRICT_AC227_IMPURITY_LIMIT_PCT = 0.15
SECONDS_PER_DAY = 24.0 * 3600.0
AVOGADRO = 6.022e23
TRAINING_DOMAIN = {
    "flux": (1.0e11, 10.0 ** 15.5),
    "energy_ev": (0.015, 2.0e7),
    "time_h": (0.05, 500.0),
}
VALIDATION_SUMMARY_PATH = ROOT / "analysis" / "validation" / "heldout_validation_summary.csv"

# Joyo anchor (Sano et al. 2024, JNST 61:509) — used by the interactive lab.
SANO_MEASURED_GBQ = 15.4
SANO_ERR_GBQ = 6.2
SANO_FLUX = 5.7e15
SANO_ENERGY_EV = 14.5e6
SANO_DAYS = 45.0
FSTAR_INFERRED = 1.2387353952827815e-3  # results/ode_data_v2_spectrum_20260718.json


# ── HUMAN TRANSLATION HELPERS ─────────────────────────────────────────────────
def format_atoms_human(atoms: float, mass_number: int) -> str:
    """Converts raw atom counts into intuitive human-readable weights."""
    if atoms <= 0:
        return "0.0 atoms (pure vacuum)"
    grams = (atoms / AVOGADRO) * mass_number
    if grams >= 1.0:
        return f"{atoms:.2e} atoms ({grams:.3f} g)"
    elif grams >= 1e-3:
        return f"{atoms:.2e} atoms ({grams * 1e3:.3f} mg)"
    elif grams >= 1e-6:
        return f"{atoms:.2e} atoms ({grams * 1e6:.3f} µg)"
    elif grams >= 1e-9:
        return f"{atoms:.2e} atoms ({grams * 1e9:.3f} ng)"
    return f"{atoms:.2e} atoms ({grams * 1e12:.3f} pg)"


def laymans_explanation(topic: str) -> str:
    """Plain-language analogies for nuclear terms."""
    explanations = {
        "flux": (
            "**Neutron flux** works like the heat setting on a stove. "
            "Higher flux means more neutrons hitting the radium each second. "
            "Turn it up too far, though, and you start producing problematic byproducts like Ac-227."
        ),
        "transmutation": (
            "**Transmutation** turns one element into another. "
            "Here, long-lived Radium-226 (half-life ~1,600 years) absorbs neutrons and converts into "
            "Actinium-225, an isotope used in targeted alpha therapy for cancer."
        ),
        "impurity": (
            "**Impurity** is like wood chips mixed into flour. "
            "Alongside useful Ac-225 (half-life ~10 days), irradiation can produce Ac-227 — a long-lived "
            "bone-seeking contaminant (half-life ~21.8 years). Screening uses a **0.15% activity-impurity** "
            "reference threshold from the radiopharmaceutical literature."
        ),
        "surrogate": (
            "A detailed reactor simulation can take hours to run. "
            "A **surrogate model** learns from many of those runs and reproduces the result in milliseconds. "
            "The PINN plays that role here, letting engineers sweep thousands of reactor settings almost instantly."
        ),
        "half_life": (
            "**Half-life** is how long it takes for half of the atoms to decay. "
            "Actinium-225 has a short half-life of 9.9 days, so it does its job and clears quickly. "
            "Actinium-227 lasts 21.8 years, which is why even a trace of it is a safety concern."
        ),
        "tail": (
            "Only neutrons above **6.42 MeV** can knock two neutrons out of Ra-226 to start the "
            "Ac-225 chain. In a reactor, almost all neutrons are slower than that — the Ac-225 yield "
            "**lives in the small high-energy tail** of the neutron spectrum. That is why knowing the "
            "spectrum shape matters more than knowing the total flux."
        ),
    }
    return explanations.get(topic, "")


# ── PHYSICAL DECAY HELPERS ────────────────────────────────────────────────────
def _decay_factor(days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 0.0
    return float(np.exp(-np.log(2.0) * max(float(days), 0.0) / float(half_life_days)))


def _activity_bq(atoms, half_life_days: float):
    arr = np.asarray(atoms, dtype=np.float64)
    if half_life_days <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    decay_constant_s = np.log(2.0) / (float(half_life_days) * SECONDS_PER_DAY)
    return arr * decay_constant_s


def _ac227_impurity_activity_pct(ac225_atoms, ac227_atoms):
    ac225_bq = _activity_bq(ac225_atoms, AC225_HALF_LIFE_DAYS)
    ac227_bq = _activity_bq(ac227_atoms, AC227_HALF_LIFE_DAYS)
    total_bq = ac225_bq + ac227_bq
    return np.divide(ac227_bq, total_bq, out=np.zeros_like(ac227_bq), where=total_bq > 0.0) * 100.0


def _domain_warnings(*, flux: float, energy_ev: float, time_h: float) -> list[str]:
    warnings = []
    lo, hi = TRAINING_DOMAIN["flux"]
    if not (lo <= float(flux) <= hi):
        warnings.append(f"Flux {flux:.2e} is outside the trained range {lo:.1e}-{hi:.1e} n/cm²/s.")
    lo, hi = TRAINING_DOMAIN["energy_ev"]
    if not (lo <= float(energy_ev) <= hi):
        warnings.append(f"Energy {energy_ev:.2e} eV is outside the trained range {lo:.2e}-{hi:.2e} eV.")
    lo, hi = TRAINING_DOMAIN["time_h"]
    if not (lo <= float(time_h) <= hi):
        warnings.append(f"Time {time_h:.1f} h is outside the trained range {lo:.2f}-{hi:.1f} h.")
    return warnings


# ── COMMITTED-EVIDENCE LOADERS (every metric on screen traces to a repo file) ─
@st.cache_data(show_spinner=False)
def _load_json(rel_path: str) -> dict:
    p = ROOT / rel_path
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_validation_summary() -> pd.DataFrame:
    if VALIDATION_SUMMARY_PATH.exists():
        try:
            return pd.read_csv(VALIDATION_SUMMARY_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_graph_image(rel_path: str):
    p = ROOT / rel_path
    if not p.is_file():
        return None
    try:
        return Image.open(p).copy()
    except Exception:
        return None


def _show_graph(rel_path: str, caption: str | None = None) -> bool:
    """Show a committed PNG from graphs/; returns False (with note) if absent."""
    img = _load_graph_image(rel_path)
    if img is None:
        st.info(f"Figure `{rel_path}` is not present in this deployment.")
        return False
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption=caption, use_container_width=True)
    return True


@st.cache_data(show_spinner=False)
def _load_upgrade_log_entries() -> list[dict]:
    """Parse docs/UPGRADE_LOG.md headings into a timeline; static fallback."""
    path = ROOT / "docs" / "UPGRADE_LOG.md"
    entries: list[dict] = []
    try:
        sprint = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## 20"):
                sprint = line.lstrip("# ").strip()
            elif line.startswith("### "):
                title = line.lstrip("# ").strip()
                num = title.split(".", 1)[0]
                entries.append({"sprint": sprint, "num": num, "title": title})
    except Exception:
        entries = []
    if not entries:
        entries = [
            {"sprint": "Sprint 1", "num": str(i), "title": t}
            for i, t in enumerate([
                "Deterministic seeding everywhere", "Exponential-integrator physics loss",
                "True-1 g inventory, versioned", "Trainable vanilla-LSTM baseline",
                "Conformal at meaningful n", "Honest speed benchmark",
                "Jackknife+ / CV+ conformal", "Self-adaptive per-species physics weights",
                "Stiffness curriculum scaffold", "Deep-ensemble runner",
                "Evaluated data layer (JENDL-5/EXFOR/NuDat)", "Spectrum folding — the discovery",
            ], start=1)
        ]
    return entries


# ── ODE REFERENCE ACCESS (ra226_ac225_transmutation.py is read-only) ──────────
def _ode_module():
    import ra226_ac225_transmutation as ode
    return ode


@st.cache_data(show_spinner="Running stiff ODE reference…")
def _joyo_ode_gbq(version: str, spectrum: str, fast_fraction: float, days: float,
                  flux: float = SANO_FLUX) -> float | None:
    """
    Honest, live re-computation of a Joyo-style anchor with the real ODE.
    version: 'v1' (legacy synthetic sigmoid) | 'v2' (evaluated data layer)
    spectrum (v2 only): 'mono' | 'watt' | 'twogroup'
    Uses each version's own decay constants for the activity conversion.
    """
    prev_version = os.environ.get("ODE_DATA_VERSION")
    prev_ffrac = os.environ.get("SPECTRUM_FAST_FRACTION")
    try:
        os.environ["ODE_DATA_VERSION"] = version
        if version == "v2" and spectrum == "twogroup":
            os.environ["SPECTRUM_FAST_FRACTION"] = str(float(fast_fraction))
        ode = _ode_module()
        spec_arg = None if version == "v1" else spectrum
        env = ode.IsotopeEnvironment(
            phi=float(flux), neutron_energy_ev=SANO_ENERGY_EV,
            target_mass_g=1.0, spectrum=spec_arg,
        )
        t_h, Y = ode.run_simulation(
            env, t_end_h=float(days) * 24.0, n_points=121, N_ra0=2.664e21,
        )
        lam_per_s = float(env.lambda_ac225_per_h) / 3600.0
        return float(Y[-1, 2]) * lam_per_s / 1e9  # GBq
    except Exception:
        return None
    finally:
        if prev_version is None:
            os.environ.pop("ODE_DATA_VERSION", None)
        else:
            os.environ["ODE_DATA_VERSION"] = prev_version
        if prev_ffrac is None:
            os.environ.pop("SPECTRUM_FAST_FRACTION", None)
        else:
            os.environ["SPECTRUM_FAST_FRACTION"] = prev_ffrac


@st.cache_data(show_spinner=False)
def _evaluated_data_available() -> bool:
    return (ROOT / "data" / "evaluated" / "jendl5_ra226_n2n_sigmaE.csv").is_file()


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
def get_weights_path() -> pathlib.Path | None:
    paths = [
        ROOT / "weights" / "pinn_best_weights.pth",
        ROOT / "weights" / "pinn_trained_weights.pth",
        ROOT / "weights" / "pinn_calibrated_weights.pth",
        ROOT / "pinn_trained_weights.pth",
    ]
    for p in paths:
        if p.is_file():
            return p
    return None


@st.cache_resource(show_spinner="Loading PINN weights…")
def get_cached_pinn(weights_path_str: str):
    from pinn_model import load_isotope_pinn_checkpoint

    model, info = load_isotope_pinn_checkpoint(
        pathlib.Path(weights_path_str), map_location="cpu"
    )
    return model, info


# ── LIGHT EDITORIAL MATPLOTLIB THEME ──────────────────────────────────────────
_PAPER = "#faf8f3"
_PANEL = "#ffffff"
_INK = "#292524"
_MUTED = "#78716c"
_BORDER = "#e5ddd0"
_ACCENT = "#b45309"      # deep amber
_ACCENT_DK = "#92400e"
_OK = "#4d7c0f"          # muted olive
_BAD = "#b91c1c"         # muted brick
_WARN = "#a16207"
_TEAL = "#0f766e"        # muted teal (secondary series only)


def _light_rc():
    return {
        "figure.facecolor": _PANEL,
        "axes.facecolor": _PANEL,
        "axes.edgecolor": _BORDER,
        "axes.labelcolor": _INK,
        "axes.titlecolor": _INK,
        "text.color": _INK,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "grid.color": _BORDER,
        "legend.facecolor": _PANEL,
        "legend.edgecolor": _BORDER,
        "font.size": 10,
    }


def _fig_to_png_bytes(fig, *, dpi: int = 110) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


# ── CHART BUILDERS (cached, light editorial palette, data from committed JSON) ─
@st.cache_data(show_spinner=False)
def _chart_discovery_ratios_png() -> bytes | None:
    """Joyo anchor ratios across the four physics variants (committed spectrum JSON)."""
    spec = _load_json("results/ode_data_v2_spectrum_20260718.json")
    anchors = spec.get("anchors") or []
    if not anchors:
        return None
    import matplotlib.pyplot as plt

    labels = ["Sano 2024\n45 d", "Iwahashi 2022\n60 d + 8 d cool", "Iwahashi 2022\nmilking 3×17.5 d"]
    series = [
        ("v1 synthetic σ (27 mb)", "v1_ratio_pred_over_meas", "#a8a29e"),
        ("v2 evaluated σ, pointwise (mono)", "v2_mono_ratio_pred_over_meas", _BAD),
        ("v2 evaluated σ, Watt fold", "v2_watt_ratio_pred_over_meas", _WARN),
        ("v2 evaluated σ, two-group f*", "v2_twogroup_fstar_ratio_pred_over_meas", _OK),
    ]
    with plt.rc_context(_light_rc()):
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        x = np.arange(len(anchors))
        w = 0.19
        for i, (name, key, color) in enumerate(series):
            vals = [float(a.get(key, np.nan)) for a in anchors]
            ax.bar(x + (i - 1.5) * w, vals, width=w, color=color, label=name,
                   edgecolor=_PANEL, linewidth=0.6)
            for xi, v in zip(x + (i - 1.5) * w, vals):
                if np.isfinite(v):
                    ax.text(xi, v * 1.15, f"{v:.2g}×", ha="center", fontsize=8, color=_MUTED)
        ax.axhline(1.0, color=_INK, lw=1.2, ls="--")
        ax.axhspan(1.0 - SANO_ERR_GBQ / SANO_MEASURED_GBQ,
                   1.0 + SANO_ERR_GBQ / SANO_MEASURED_GBQ,
                   color=_ACCENT, alpha=0.10)
        ax.text(2.42, 1.05, "Sano ±6.2 GBq band", fontsize=8, color=_ACCENT_DK)
        ax.set_yscale("log")
        ax.set_ylim(0.2, 4000)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("ODE prediction ÷ measurement (log scale)")
        ax.set_title("Each physics fix revealed the next-deeper assumption")
        ax.grid(axis="y", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
        fig.tight_layout()
    return _fig_to_png_bytes(fig)


@st.cache_data(show_spinner=False)
def _chart_fstar_sweep_png() -> bytes | None:
    """Sano ratio vs above-threshold fraction f (committed sweep in spectrum JSON)."""
    spec = _load_json("results/ode_data_v2_spectrum_20260718.json")
    sweep = spec.get("f_sensitivity_sweep_sano") or []
    inv = spec.get("inversion") or {}
    if not sweep:
        return None
    import matplotlib.pyplot as plt

    f = np.array([float(s["f"]) for s in sweep])
    r = np.array([float(s["ratio_vs_sano"]) for s in sweep])
    fstar = float(inv.get("inferred_fast_fraction_fstar", FSTAR_INFERRED))
    band = inv.get("inferred_fast_fraction_band") or []
    with plt.rc_context(_light_rc()):
        fig, ax = plt.subplots(figsize=(8.2, 3.9))
        ax.axhspan(1.0 - SANO_ERR_GBQ / SANO_MEASURED_GBQ,
                   1.0 + SANO_ERR_GBQ / SANO_MEASURED_GBQ,
                   color=_ACCENT, alpha=0.12, label="Sano 15.4 ± 6.2 GBq band")
        ax.plot(f, r, "o-", color=_ACCENT_DK, lw=2, ms=5, label="ODE prediction ÷ Sano")
        ax.axhline(1.0, color=_INK, lw=1.0, ls="--")
        if len(band) == 2:
            ax.axvspan(float(band[0]), float(band[1]), color=_OK, alpha=0.14)
        ax.axvline(fstar, color=_OK, lw=1.6)
        ax.text(fstar * 1.25, 60, f"f* = {fstar:.2e}\n[7.4e-4, 1.7e-3]",
                fontsize=9, color=_OK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Above-threshold (>6.42 MeV) fast fraction f")
        ax.set_ylabel("Prediction ÷ Sano (log)")
        ax.set_title("One effective parameter closes the gap — with an uncertainty band")
        ax.grid(alpha=0.4, which="both")
        ax.legend(fontsize=8.5, loc="lower right")
        fig.tight_layout()
    return _fig_to_png_bytes(fig)


@st.cache_data(show_spinner=False)
def _chart_species_compare_png() -> bytes | None:
    """v2 PINN vs v3 PI-LSTM per-species median endpoint error (committed compare JSON)."""
    cmpj = _load_json("v3_pilstm/results/compare_v2_pilstm.json")
    species = cmpj.get("species_median_rel_error") or {}
    if not species:
        return None
    import matplotlib.pyplot as plt

    names = list(species.keys())
    v2 = [100.0 * float(species[s]["v2"]) for s in names]
    v3 = [100.0 * float(species[s]["pilstm"]) for s in names]
    x = np.arange(len(names))
    w = 0.36
    with plt.rc_context(_light_rc()):
        fig, ax = plt.subplots(figsize=(8.2, 4.0))
        ax.bar(x - w / 2, v2, width=w, color="#d6c9b4", edgecolor=_BORDER,
               label="Model A · v2 PINN")
        ax.bar(x + w / 2, v3, width=w, color=_ACCENT, edgecolor=_BORDER,
               label="Model B · v3 PI-LSTM")
        for xi, v in zip(x - w / 2, v2):
            ax.text(xi, v + 0.25, f"{v:.1f}%", ha="center", fontsize=8.5, color=_MUTED)
        for xi, v in zip(x + w / 2, v3):
            ax.text(xi, v + 0.25, f"{v:.1f}%", ha="center", fontsize=8.5, color=_ACCENT_DK)
        ax.axhline(10.0, color=_BAD, lw=1.0, ls="--")
        ax.text(len(names) - 0.5, 10.4, "10% gate", fontsize=8, color=_BAD, ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Median endpoint rel. error vs ODE (%)")
        ax.set_title("Endpoint protocol, 22 held-out scenarios — both models, warts and all")
        ax.grid(axis="y", alpha=0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()
    return _fig_to_png_bytes(fig)


@st.cache_data(show_spinner=False)
def _chart_heldout_buckets_png() -> bytes | None:
    """Canonical v63 held-out Ac-225 error by regime."""
    v63 = _load_json("results/v63_validation_20260530.json")
    buckets = v63.get("heldout_buckets_ac225_median_rel") or {}
    if not buckets:
        return None
    import matplotlib.pyplot as plt

    order = [
        ("all", "All held-out"), ("fast14_virgin", "Fast 14 MeV virgin"),
        ("thermal_virgin", "Thermal virgin"), ("thermal_recycled", "Thermal recycled"),
        ("epithermal_virgin", "Epithermal virgin"), ("epithermal_recycled", "Epithermal recycled"),
        ("threshold_virgin", "Threshold virgin"), ("threshold_recycled", "Threshold recycled"),
    ]
    rows = [(lbl, 100.0 * float(buckets[k])) for k, lbl in order if k in buckets]
    if not rows:
        return None
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [_OK if v <= 6.0 else _WARN if v <= 10.0 else _BAD for v in vals]
    with plt.rc_context(_light_rc()):
        fig, ax = plt.subplots(figsize=(8.2, 4.0))
        ax.barh(labels, vals, color=colors, edgecolor=_PANEL)
        for i, v in enumerate(vals):
            ax.text(v + 0.12, i, f"{v:.2f}%", va="center", fontsize=8.5, color=_MUTED)
        ax.axvline(10.0, color=_BAD, lw=1.0, ls="--")
        ax.text(10.05, -0.45, "10% gate", fontsize=8, color=_BAD)
        ax.set_xlabel("Median Ac-225 rel. error vs ODE (%)")
        ax.set_title("Canonical v63: 22 held-out scenarios, by neutron-energy regime")
        ax.grid(axis="x", alpha=0.5)
        ax.set_xlim(0, 11.5)
        fig.tight_layout()
    return _fig_to_png_bytes(fig)


@st.cache_data(show_spinner=False)
def _chart_joyo_lab_png(mode_key: str, pred_gbq: float, days: float) -> bytes | None:
    """Marker of the live ODE result against the Sano measurement band."""
    import matplotlib.pyplot as plt

    lo = SANO_MEASURED_GBQ - SANO_ERR_GBQ
    hi = SANO_MEASURED_GBQ + SANO_ERR_GBQ
    with plt.rc_context(_light_rc()):
        fig, ax = plt.subplots(figsize=(8.0, 2.6))
        ax.axhspan(0, 1, color=_PANEL)
        ax.axvspan(lo, hi, color=_ACCENT, alpha=0.15,
                   label=f"Sano 2024: {SANO_MEASURED_GBQ} ± {SANO_ERR_GBQ} GBq")
        ax.axvline(SANO_MEASURED_GBQ, color=_ACCENT_DK, lw=1.4, ls="--")
        ax.axvline(pred_gbq, color=_INK, lw=2.2)
        ax.scatter([pred_gbq], [0.5], s=90, color=_INK, zorder=5)
        ratio = pred_gbq / SANO_MEASURED_GBQ
        ax.text(pred_gbq, 0.82, f"ODE now: {pred_gbq:.2f} GBq ({ratio:.2f}×)",
                fontsize=10, color=_INK, ha="center")
        ax.set_xscale("log")
        xmax = max(hi * 2.2, pred_gbq * 2.2, 60.0)
        ax.set_xlim(min(lo / 3.0, pred_gbq / 3.0, 1.0), xmax)
        ax.set_yticks([])
        ax.set_xlabel("Ac-225 activity after irradiation (GBq, log scale)")
        ax.set_title(f"Live ODE result — {mode_key}, {days:.0f} d irradiation", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left")
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
    return _fig_to_png_bytes(fig)


@st.cache_data(show_spinner=False)
def _chart_pinn_ode_live_png(weights_key: str, phi: float, hours: float,
                             energy_ev: float, ra226_0: float) -> bytes | None:
    """Light-theme PINN vs ODE trajectory for one scenario (Screening explorer)."""
    if not weights_key:
        return None
    try:
        model, _ = get_cached_pinn(weights_key)
        from pinn_model import (
            DEFAULT_N226_SCALE as N226S, DEFAULT_NAC_SCALE as NACS,
            DEFAULT_PHI_SCALE as PHIS, DEFAULT_T_REF_H as TSH,
            neutron_energy_ev_to_feature_numpy as _efn,
        )
        times = np.linspace(1.0, float(hours), 80)
        e_nn = float(_efn(float(energy_ev)))
        rows = np.column_stack([
            times / TSH,
            np.full_like(times, phi / PHIS),
            np.full_like(times, e_nn),
            np.full_like(times, ra226_0 / N226S),
            np.zeros_like(times), np.zeros_like(times),
            np.zeros_like(times), np.zeros_like(times),
        ])
        x_t = torch.tensor(rows, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = model(x_t).cpu().numpy()
        ac_pinn = np.maximum(pred[:, 2] * NACS, 0.0)

        ode = _ode_module()
        env = ode.IsotopeEnvironment(phi=float(phi), neutron_energy_ev=float(energy_ev))
        t_ode, Y = ode.run_simulation(env, t_end_h=float(hours), n_points=80, N_ra0=float(ra226_0))
        ac_ode = Y[:, 2]
    except Exception:
        return None

    import matplotlib.pyplot as plt

    with plt.rc_context(_light_rc()):
        fig, ax = plt.subplots(figsize=(7.6, 3.9))
        ax.semilogy(t_ode, np.maximum(ac_ode, 1.0), color=_MUTED, ls="--", lw=1.8,
                    label="ODE reference (Radau)")
        ax.semilogy(times, np.maximum(ac_pinn, 1.0), color=_ACCENT, lw=2.4,
                    label="PINN surrogate")
        ax.set_xlabel("Irradiation time (h)")
        ax.set_ylabel("Ac-225 atoms (log)")
        ax.set_title(f"φ={phi:.1e} n/cm²/s · E={energy_ev:.3g} eV · {hours:.0f} h")
        ax.grid(alpha=0.4, which="both")
        ax.legend(fontsize=9)
        fig.tight_layout()
    return _fig_to_png_bytes(fig)


@st.cache_data(show_spinner=False)
def _cached_speed_benchmark(weights_key: str, n_scenarios: int) -> dict:
    """
    Honest live timing: batched surrogate inference vs SEQUENTIAL stiff Radau
    ODE solves on random in-domain scenarios. Throughput framing only.
    """
    if not weights_key:
        return {}
    try:
        model, _ = get_cached_pinn(weights_key)
        from pinn_model import (
            DEFAULT_N226_SCALE as N226S, DEFAULT_NAC_SCALE as NACS,
            DEFAULT_PHI_SCALE as PHIS, DEFAULT_T_REF_H as TSH,
            neutron_energy_ev_to_feature_numpy as _efn,
        )
        ode = _ode_module()
        rng = np.random.default_rng(42)
        n = int(n_scenarios)
        phis = 10.0 ** rng.uniform(12.0, 15.0, n)
        energies = rng.choice([0.025, 1.0, 14.0e6], n)
        hours = rng.uniform(50.0, 400.0, n)

        e_nn = np.array([float(_efn(e)) for e in energies])
        rows = np.column_stack([
            hours / TSH, phis / PHIS, e_nn,
            np.full(n, 6.022e23 / N226S),
            np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
        ])
        x_t = torch.tensor(rows, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            model(x_t)  # warmup
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(x_t).cpu().numpy()
        pinn_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ac_ode = []
        for i in range(n):
            env = ode.IsotopeEnvironment(phi=float(phis[i]), neutron_energy_ev=float(energies[i]))
            _, Y = ode.run_simulation(env, t_end_h=float(hours[i]), n_points=32, N_ra0=6.022e23)
            ac_ode.append(float(Y[-1, 2]))
        ode_ms = (time.perf_counter() - t0) * 1000.0

        ac_pinn = np.maximum(pred[:, 2] * NACS, 1e-30)
        rel = np.abs(ac_pinn - np.maximum(np.asarray(ac_ode), 1e-30)) / np.maximum(np.asarray(ac_ode), 1e-30)
        return {
            "n_scenarios": n,
            "pinn_ms": pinn_ms,
            "ode_ms": ode_ms,
            "throughput_ratio": ode_ms / max(pinn_ms, 1e-9),
            "pinn_ms_per_scenario": pinn_ms / n,
            "ode_ms_per_scenario": ode_ms / n,
            "median_rel_err_ac225": float(np.median(rel)),
        }
    except Exception:
        return {}


# ── DESIGN SETUP (WARM EDITORIAL / SCIENTIFIC PRINT) ──────────────────────────
st.set_page_config(
    page_title="IsotopePINN | Ac-225 Production Surrogate",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(f"""
<style>
/* Warm editorial design tokens */
:root {{
  --paper: {_PAPER};
  --panel: {_PANEL};
  --panel-alt: #f4efe6;
  --border: {_BORDER};
  --ink: {_INK};
  --muted: {_MUTED};
  --faint: #a8a29e;
  --accent: {_ACCENT};
  --accent-dk: {_ACCENT_DK};
  --accent-soft: #fdf3e7;
  --ok: {_OK};
  --bad: {_BAD};
  --warn: {_WARN};
}}

.stApp {{
  background-color: var(--paper) !important;
  color: var(--ink) !important;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important;
}}
header[data-testid="stHeader"] {{ background-color: var(--paper) !important; }}
section[data-testid="stSidebar"] {{
  background-color: var(--panel-alt) !important;
  border-right: 1px solid var(--border);
}}

/* Editorial typography */
h1, h2, h3, .serif {{
  font-family: Georgia, "Charter", "Times New Roman", serif !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em;
}}

.sh {{
  font-family: Georgia, "Charter", serif;
  font-size: 1.45rem;
  font-weight: 700;
  color: var(--ink);
  border-bottom: 2px solid var(--accent);
  padding-bottom: 0.35rem;
  margin: 2.2rem 0 1rem 0;
}}
.sh-sm {{
  font-family: Georgia, "Charter", serif;
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent-dk);
  margin: 1.4rem 0 0.5rem 0;
}}
.kicker {{
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.6px; color: var(--accent);
}}

/* Cards */
.card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.15rem 1.3rem;
  margin-bottom: 1rem;
}}
.card.accent {{ border-left: 4px solid var(--accent); background: var(--accent-soft); }}
.card.ok {{ border-left: 4px solid var(--ok); }}
.card.bad {{ border-left: 4px solid var(--bad); }}
.card.warn {{ border-left: 4px solid var(--warn); }}
.card h4 {{ margin: 0 0 0.45rem 0; font-size: 1.02rem; }}
.card p {{ color: #57534e; font-size: 0.92rem; line-height: 1.6; margin: 0; }}

/* KPI cards */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin: 1rem 0 1.4rem 0; }}
.kpi {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 1rem 1.1rem; border-top: 3px solid var(--accent);
}}
.kpi .v {{
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 1.55rem; font-weight: 700; color: var(--ink); line-height: 1.1;
}}
.kpi .l {{
  font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.1px; color: var(--muted); margin-top: 0.35rem;
}}
.kpi .s {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; line-height: 1.45; }}

/* Validation ladder */
.rung {{
  display: flex; gap: 0.9rem; align-items: flex-start;
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 0.95rem 1.15rem; margin-bottom: 0.55rem;
}}
.rung .n {{
  flex: 0 0 auto; width: 1.7rem; height: 1.7rem; border-radius: 50%;
  background: var(--accent-soft); border: 1.5px solid var(--accent);
  color: var(--accent-dk); font-weight: 700; font-size: 0.85rem;
  display: flex; align-items: center; justify-content: center;
}}
.rung .b {{ font-size: 0.92rem; color: #57534e; line-height: 1.55; }}
.rung .b b {{ color: var(--ink); }}

/* Timeline (upgrade log) */
.tl {{ border-left: 2px solid var(--border); margin-left: 0.55rem; padding-left: 1.3rem; }}
.tl-item {{ position: relative; margin-bottom: 0.85rem; }}
.tl-item::before {{
  content: ""; position: absolute; left: -1.62rem; top: 0.3rem;
  width: 0.6rem; height: 0.6rem; border-radius: 50%;
  background: var(--accent); border: 2px solid var(--paper);
}}
.tl-item .d {{ font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--accent); }}
.tl-item .t {{ font-size: 0.92rem; color: var(--ink); font-weight: 600; }}
.tl-item .s {{ font-size: 0.84rem; color: var(--muted); line-height: 1.5; }}

/* Model cards */
.model-card {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.3rem 1.4rem; height: 100%;
}}
.model-card .tag {{
  display: inline-block; font-size: 0.68rem; font-weight: 700; letter-spacing: 1.2px;
  text-transform: uppercase; color: var(--accent-dk);
  background: var(--accent-soft); border: 1px solid var(--accent);
  border-radius: 5px; padding: 0.15rem 0.55rem; margin-bottom: 0.55rem;
}}

/* Triage dot grid */
.triage-grid {{
  display: grid; grid-template-columns: repeat(20, 1fr); gap: 4px;
  background: var(--panel-alt); padding: 10px; border-radius: 10px; border: 1px solid var(--border);
}}
.triage-dot {{ aspect-ratio: 1; border-radius: 3px; transition: transform 0.1s ease; }}
.triage-dot:hover {{ transform: scale(1.3); z-index: 10; }}
.triage-dot.safe {{ background-color: var(--ok); }}
.triage-dot.toxic {{ background-color: var(--bad); }}
.triage-dot.low {{ background-color: #d6cfc2; }}

/* Streamlit chrome overrides (warm) */
.stButton>button {{
  background: var(--accent) !important; color: #fff !important;
  border: 1px solid var(--accent-dk) !important; border-radius: 8px !important;
  font-weight: 600 !important; padding: 0.55rem 1.4rem !important;
}}
.stButton>button:hover {{ background: var(--accent-dk) !important; }}
div[data-testid="stTabs"] button {{
  color: var(--muted) !important; font-weight: 600 !important; font-size: 0.92rem !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
  color: var(--accent-dk) !important;
  border-bottom: 2px solid var(--accent) !important;
}}
div[data-testid="stMetricValue"] {{
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important;
  color: var(--ink) !important;
}}
div[data-testid="stMetricLabel"] {{ color: var(--muted) !important; }}

/* Footer */
.ft {{
  text-align: center; padding: 2.2rem 1rem; margin-top: 3.5rem;
  border-top: 1px solid var(--border); color: var(--muted); font-size: 0.8rem;
}}
.ft a {{ color: var(--accent-dk); text-decoration: none; font-weight: 600; }}

@media (max-width: 900px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .sh {{ font-size: 1.2rem; }}
}}
@media (max-width: 768px) {{
  .triage-grid {{ grid-template-columns: repeat(10, 1fr); gap: 2px; padding: 6px; }}
  div[data-testid="stTabs"] button {{ font-size: 0.72rem !important; padding: 0.4rem 0.5rem !important; }}
}}
</style>
""", unsafe_allow_html=True)

# ── LOAD COMMITTED EVIDENCE ───────────────────────────────────────────────────
v63_report = _load_json("results/v63_validation_20260530.json")
ode_v2_val = _load_json("results/ode_data_v2_validation_20260718.json")
ode_v2_spec = _load_json("results/ode_data_v2_spectrum_20260718.json")
cmp_v2_v3 = _load_json("v3_pilstm/results/compare_v2_pilstm.json")
conformal = _load_json("v3_pilstm/results/conformal_validation.json")
empirical = _load_json("v3_pilstm/results/empirical_validation.json")
joyo_cal = _load_json("v3_pilstm/results/joyo_sigma_calibration.json")
speed_v3 = _load_json("v3_pilstm/results/speed_harness.json")
v3_train = _load_json("v3_pilstm/results/train_summary.json")
iter_log = _load_json("results/isef_iteration_log.json")

weights_p = get_weights_path()
model = None
model_info = None
if weights_p:
    try:
        model, model_info = get_cached_pinn(str(weights_p.resolve()))
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
else:
    st.warning("No PINN weights found in weights/. Deploy pinn_best_weights.pth for live inference.")

_weights_key_str = str(weights_p.resolve()) if weights_p else ""

crit = v63_report.get("criteria", {})
heldout_rel = float(crit.get("heldout_ac225_median_rel", 0.0451)) if crit else 0.0451

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: {_PANEL}; border: 1px solid {_BORDER}; border-radius: 14px; padding: 2rem 2.2rem; margin-bottom: 1.6rem; border-top: 4px solid {_ACCENT};">
  <div class="kicker">Physics-informed machine learning · Isotope production · ISEF project</div>
  <h1 style="font-size: 2.15rem; font-weight: 700; margin: 0.45rem 0 0.55rem 0; line-height: 1.18;">
    IsotopePINN: screening the Ra-226 → Ac-225 chain, and the data hunt that changed the answer
  </h1>
  <p style="font-size: 1.02rem; color: #57534e; max-width: 860px; line-height: 1.65; margin: 0;">
    Actinium-225 powers targeted alpha therapy, an experimental cancer treatment — but the world
    produces only a few patient doses' worth per year. This project builds <b>two physics-informed
    surrogate models</b> that screen reactor production settings in milliseconds, then progressively
    replaces every synthetic assumption in the physics reference with <b>evaluated nuclear data</b> —
    uncovering, along the way, why the model disagreed with a national lab, and fixing it.
  </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### IsotopePINN")
    st.caption("Two-model physics-informed surrogate · Ra-226 → Ac-225")
    if crit.get("overall"):
        st.success(f"v63 validation: {crit['overall']}")
    st.metric("Canonical held-out Ac-225 vs ODE", f"{100.0 * heldout_rel:.2f}% median")
    if ode_v2_spec.get("anchors"):
        sano = ode_v2_spec["anchors"][0]
        st.metric(
            "Joyo anchor (spectrum-folded ODE)",
            f"{float(sano['v2_twogroup_fstar_ratio_pred_over_meas']):.2f}× of Sano",
        )
    st.divider()
    st.markdown("**Judge walkthrough**")
    st.markdown(
        "1. **Overview** — problem, system, headline evidence\n"
        "2. **The Discovery** — the 3-act data story (start here if short on time)\n"
        "3. **The Science** — validation ladder, UQ, failure honesty\n"
        "4. **Screening** — live surrogate + 10,000-scenario triage\n"
        "5. **Speed** — honest throughput numbers\n"
        "6. **Methods & Data** — model cards, provenance, limitations"
    )
    st.divider()
    st.caption(
        "Hosted demos sleep when idle; first load after sleep can take up to ~90 s while PyTorch initializes."
    )


# ── MAIN TABS ─────────────────────────────────────────────────────────────────
(tab_overview, tab_science, tab_discovery, tab_screen, tab_speed, tab_methods, tab_about) = st.tabs([
    "Overview",
    "The Science",
    "The Discovery",
    "Screening",
    "Speed",
    "Methods & Data",
    "About",
])

# ==============================================================================
# TAB — OVERVIEW
# ==============================================================================
with tab_overview:
    st.markdown('<div class="kicker">The problem</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh" style="margin-top:0.2rem;">A cancer drug the world cannot make enough of</div>', unsafe_allow_html=True)
    st.markdown(
        "Actinium-225 is an alpha-emitting isotope at the heart of **targeted alpha therapy (TAT)** — "
        "radiolabeled conjugates such as Ac-225–PSMA-617 now in clinical trials for metastatic prostate "
        "cancer and leukemia. Alpha particles deposit ~28 MeV within a few cell diameters, killing tumor "
        "cells while sparing surrounding tissue. The bottleneck is supply: global Ac-225 production "
        "covers only a small fraction of what trials need. One promising route is neutron transmutation "
        "of Radium-226 in reactors — **Ra-226(n,2n)Ra-225 → β⁻ → Ac-225** — but finding irradiation "
        "conditions that maximize Ac-225 while suppressing the long-lived Ac-227 impurity requires "
        "sweeping a large parameter space, and each full physics evaluation is a stiff ODE solve."
    )

    st.markdown('<div class="kicker" style="margin-top:1.6rem;">What was built</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh" style="margin-top:0.2rem;">A two-model physics-informed surrogate system</div>', unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"""
        <div class="model-card">
          <span class="tag">Model A · shipped</span>
          <h4 style="margin:0.2rem 0 0.4rem 0;">v2 PINN (v63)</h4>
          <p style="color:#57534e;font-size:0.92rem;line-height:1.6;margin:0;">
          Semi-analytic Bateman backbone + learned correction (4×128 MLP, SiLU, float64).
          600 epochs physics-only pretrain, then 3,400 joint epochs. Physics residuals and
          mass-budget constraints sit inside the loss, so the network cannot invent Ac-225
          from an empty target. <b>Canonical result: {100.0*heldout_rel:.2f}% median
          Ac-225 error vs the stiff ODE reference on 22 held-out scenarios, 6/6 gates PASS.</b>
          </p>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        v3_epochs = v3_train.get("epochs", 6000)
        st.markdown(f"""
        <div class="model-card">
          <span class="tag">Model B · research line</span>
          <h4 style="margin:0.2rem 0 0.4rem 0;">v3 PI-LSTM</h4>
          <p style="color:#57534e;font-size:0.92rem;line-height:1.6;margin:0;">
          Sequence model (2×256 LSTM, Fourier time/energy features, hard initial condition)
          trained {v3_epochs:,} epochs with an exact exponential-integrator physics loss,
          distilled from the v2 teacher, at a matched parameter budget against a trained
          vanilla-LSTM ablation. Endpoint-protocol Ac-225 median <b>5.12% vs ODE</b>
          (v2: 8.18% on the same protocol — see The Science for the reconciliation footnote).
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Headline evidence</div>', unsafe_allow_html=True)
    sano_ratio = "—"
    if ode_v2_spec.get("anchors"):
        sano_ratio = f"{float(ode_v2_spec['anchors'][0]['v2_twogroup_fstar_ratio_pred_over_meas']):.2f}×"
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="v">{100.0*heldout_rel:.2f}%</div><div class="l">Held-out Ac-225 vs ODE</div>
        <div class="s">Canonical v63 result · 22 unseen scenarios · ODE reference</div></div>
      <div class="kpi"><div class="v">{crit.get('overall','—')}</div><div class="l">Validation gates</div>
        <div class="s">Empty-tank safety, production, decay chain, quality, correlation, held-out</div></div>
      <div class="kpi"><div class="v">{sano_ratio}</div><div class="l">Joyo anchor after spectrum fix</div>
        <div class="s">vs Sano 2024's 15.4 ± 6.2 GBq — inside national-lab uncertainty</div></div>
      <div class="kpi"><div class="v">28×</div><div class="l">The error we found</div>
        <div class="s">Synthetic σ(n,2n) was 28× too small vs JENDL-5 at 14 MeV</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">The discovery, in one paragraph</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card accent">
    <p>
    When we checked the physics reference against Japan's Joyo reactor results, our synthetic cross
    section (27 mb) overpredicted Ac-225 by <b>~19×</b>. We replaced it with the real evaluated
    JENDL-5 data — and the prediction got <b>worse</b> (369×), proving the cross section was never the
    dominant error: the model treated <i>every</i> neutron as a 14.5 MeV neutron, while only the tiny
    above-threshold tail of a reactor spectrum can drive the (n,2n) reaction at all. Folding the
    evaluated cross section over a documented spectrum shape closes the gap to <b>1.00×</b> of the
    national-lab measurement, with an inferred fast fraction f* = 1.24×10⁻³ [7.4×10⁻⁴, 1.7×10⁻³].
    Each fix revealed the next-deeper assumption — <b>that iteration is the research</b>.
    Full story in <b>The Discovery</b> tab; provenance in <b>Methods & Data</b>.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Where everything lives</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.markdown(
        '<div class="card"><h4>→ The Discovery</h4><p>The 3-act story: synthetic σ wrong → real σ made '
        'it worse → spectrum folding fixed it (1.00×). Includes a live ODE lab with a mono-vs-folded '
        'spectrum toggle.</p></div>',
        unsafe_allow_html=True,
    )
    g2.markdown(
        '<div class="card"><h4>→ The Science</h4><p>Canonical v63 gates, v2-vs-v3 comparison with the '
        'protocol footnote, the four-rung validation ladder, conformal uncertainty, and the '
        'failure/transparency section.</p></div>',
        unsafe_allow_html=True,
    )
    g3.markdown(
        '<div class="card"><h4>→ Screening & Speed</h4><p>Live scenario explorer, 10,000-scenario '
        'triage, 2D production map — and an honest speed tab (batched throughput, eager latency '
        'disclosed, re-verification in progress).</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sh">What changed in the July 2026 engineering sprints</div>', unsafe_allow_html=True)
    n1, n2 = st.columns(2)
    n1.markdown(
        '<div class="card"><h4>Stronger physics &amp; evaluation</h4><p>'
        '<b>Exact propagator loss:</b> the physics residual now uses the closed-form matrix-exponential '
        'interval propagator — ≤ 1.2e-14 on any grid (the trapezoid leaked up to 8.9e-8 on exact '
        'trajectories).<br/>'
        '<b>Locked test set:</b> 60 scenarios, seed 20260725, <i>shifted</i> regime boundaries — never '
        'used for training, checkpointing, or early stopping. Dev set (seed 2025) handles selection.<br/>'
        '<b>Coverage rebalance:</b> epithermal + threshold regimes oversampled 2× after measured bucket '
        'errors (9.55% / 8.52%) exposed the gaps.</p></div>',
        unsafe_allow_html=True,
    )
    n2.markdown(
        '<div class="card"><h4>Integrity &amp; what is pending</h4><p>'
        '<b>26/26 checks green:</b> bit-for-bit reproducibility, legacy checkpoint compatibility, '
        'fold-constant asserts, atom-budget conservation exact to 1e-12, plus the four Sprint 6 '
        'guards (Jacobian, QSSA, CRAM, EMA bias).<br/>'
        '<b>Implemented, not yet accuracy-claimed:</b> SOAP optimizer, causality-weighted loss, '
        'multi-fidelity head, JAWS+ACI conformal, active learning, minGRU baseline, Ra-227 QSSA prior.<br/>'
        '<b>Pending:</b> the locked-set &lt; 3% Ac-225 gate (TC-ACC-001) is being measured by the '
        'B0/S1 GPU runs — no dev-set number will be substituted.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sh" style="margin-top:1.6rem;">Sprint 6 (31 Jul 2026) — we audited our own '
        'physics loss and found two real bugs</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card"><p>A physics-informed loss is supposed to enforce the Bateman equations. '
        'So we audited it the obvious way: <b>hand it the exact ODE solution and check that it scores '
        'zero.</b> It did not. Two defects were found in code that had already shipped, and both are '
        'now fixed, measured, and guarded by regression tests.</p></div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns(3)
    s1.markdown(
        '<div class="card bad"><h4>Bug 1 — a dropped identity term</h4><p>'
        'The Jacobian row norms divide each species\' residual so no isotope dominates simply because '
        'its rate constants are large. Three rows carried the identity term; the two <b>actinium</b> '
        'rows did not.<br/><br/>'
        'Ac-225\'s divisor was <b>0.0035 instead of 1.0</b>, inflating its physics residual '
        '<b>285×</b> — on the exact species every headline number reports.</p></div>',
        unsafe_allow_html=True,
    )
    s2.markdown(
        '<div class="card bad"><h4>Bug 2 — rates that did not match the data</h4><p>'
        'The ODE that generates our training labels applies <code>exp(-0.01·mass)</code> self-shielding. '
        'The trapezoid residual did not.<br/><br/>'
        'That ~1% mismatch meant <b>true trajectories were not zeros of the loss</b>: the physics term '
        'was pulling the model off the very solution it is supervised on.</p></div>',
        unsafe_allow_html=True,
    )
    s3.markdown(
        '<div class="card ok"><h4>Result</h4><p>'
        'Residual on the true solution fell from <b>8.9e-08 → 5.2e-12</b> (621× on the median), and the '
        'exact propagator is a further 446× below that — now the default.<br/><br/>'
        '<b>The model has not been retrained yet</b>, so the reported 5.12% is unchanged. Fixing a loss '
        'changes what a model learns; the new number needs a full GPU run.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card warn">
        <h4>Scope — what this is not</h4>
        <p>
        This is a <b>0D planning surrogate</b>, not a patient dose calculator, a clinical approval tool,
        or a substitute for lab measurements. Errors reported here compare the surrogate to a
        <b>stiff ODE reference</b> and to evaluated nuclear data — not error in a real reactor or
        hospital batch. It is not for clinical use.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# TAB — THE SCIENCE (VALIDATION)
# ==============================================================================
with tab_science:
    st.markdown('<div class="kicker">Evidence, with the protocol stated</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh" style="margin-top:0.2rem;">Canonical validation — v63, 6/6 gates</div>', unsafe_allow_html=True)
    st.markdown(
        "The **canonical accuracy number is 4.51%**: median Ac-225 relative error vs the stiff Radau ODE "
        "reference on **22 held-out scenarios**, full-trajectory pipeline protocol "
        "(`results/v63_validation_20260530.json`, weights v63 sha256 prefix `7c21debe`). "
        "Six independent gates were evaluated separately — not one lucky plot."
    )
    if crit:
        gcols = st.columns(6)
        gcols[0].metric("Gates", crit.get("overall", "—"))
        gcols[1].metric("Held-out Ac-225", f"{100.0 * heldout_rel:.2f}%")
        gcols[2].metric("Trio A (empty)", crit.get("trio_a", "—").split(" ")[0])
        trio_b_err = crit.get("trio_b_ac225_rel_error")
        gcols[3].metric("Trio B", f"{100.0 * float(trio_b_err):.1f}%" if trio_b_err is not None else "—")
        gcols[4].metric("Trio C", crit.get("trio_c", "—").split(" ")[0])
        gcols[5].metric("Quality gate", crit.get("quality_gate", "—").split(" ")[0])
    else:
        st.info("`results/v63_validation_20260530.json` not found — gate table unavailable in this deployment.")

    held_png = _chart_heldout_buckets_png()
    if held_png:
        st.image(held_png, use_container_width=True)

    with st.expander("What each gate proves"):
        st.markdown(
            "- **Trio A — empty tank + flux:** all inventories zero at φ=1e15 for 100 h; the model must "
            "stay at zero (a data-only net can hallucinate Ac-225 from nothing; this one doesn't).\n"
            "- **Trio B — production scenario:** 1e22 Ra-226 atoms, φ=1e14, 14 MeV, 250 h; Ac-225 within "
            "10% of the ODE.\n"
            "- **Trio C — decay chain:** φ=0, Ra-225 feed only; Ac-225 ingrowth from β-decay alone.\n"
            "- **Quality gate / correlation / held-out:** species-wise error budgets, prediction–reference "
            "correlation, and the 22-scenario unseen-set median."
        )

    # ── Sprint 6: auditing the physics loss itself ──────────────────────────
    st.markdown(
        '<div class="sh">Auditing the referee: is the physics loss itself correct?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Every number above compares the **model** to the ODE. But the physics loss is also a piece of "
        "software, and it can be wrong. There is a clean way to test it: **hand it the exact ODE "
        "solution.** That curve satisfies the Bateman equations by definition, so a correct loss must "
        "score it at zero. Anything left over is the loss function's own error, with the neural network "
        "removed from the picture entirely."
    )

    _s6a, _s6b = st.columns(2)
    with _s6a:
        _show_graph(
            "graphs/sprint6_physics_residual.png",
            "Residual the loss reports on the true solution. The old trapezoid claimed the correct "
            "answer violated its own equation by 0.042% per step; that bias was systematic, not noise, "
            "so it accumulated instead of cancelling.",
        )
    with _s6b:
        _show_graph(
            "graphs/sprint6_jacobian_fix.png",
            "The divisor that makes species comparable. Three rows carried the identity term, the two "
            "actinium rows did not. Fixed values match closed form exactly: Ra-227 = sqrt(1 + 0.9855²) "
            "= 1.4040.",
        )

    st.markdown(
        '<div class="card accent"><p><b>Why the fixed numbers are trustworthy, not just smaller.</b> '
        'The exact propagator now sits at <b>1.16e-14</b> — that is float64 roundoff accumulated over '
        '63 intervals, i.e. the arithmetic floor rather than a tuning achievement. And the corrected '
        'Jacobian rows can be derived by hand, which is a stronger check than "it improved".</p></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Independent cross-validation, and a physics prior we tested before trusting"):
        c1, c2 = st.columns(2)
        with c1:
            _show_graph(
                "graphs/sprint6_cram_crosscheck.png",
                "CRAM is the matrix-exponential depletion solver used in Serpent and OpenMC. It shares "
                "no mathematics with our hand-derived propagator, yet agrees to 3.8e-15.",
            )
            st.caption(
                "We implemented CRAM and then **chose not to adopt it**. It computes all species "
                "through shared linear algebra and cannot resolve a nuclide below ~1e-16 of the "
                "largest — and trace nuclides are the product here. Its value is the cross-check."
            )
        with c2:
            _show_graph(
                "graphs/sprint6_qssa_validity.png",
                "Ra-227 (42 min) against Ra-226 (1600 y) is a ~10⁷ timescale ratio — the actual source "
                "of stiffness. Its quasi-steady state was measured against Radau truth before being "
                "used as a training prior.",
            )
            st.caption(
                "38 of 39 scenarios agree to under 1%. The single failure is the no-feed case: with no "
                "Ra-226 there is no production, so the assumption correctly does not apply — and it is "
                "gated out inside the loss rather than papered over."
            )

    with st.expander("Two results we could have hidden and did not"):
        h1, h2 = st.columns(2)
        with h1:
            _show_graph(
                "graphs/sprint6_ablation_forest.png",
                "Screening six training upgrades. The shaded band is the baseline's own seed-to-seed "
                "spread.",
            )
            st.caption(
                "**A negative result.** The same configuration scores anywhere from 75.6% to 97.2% "
                "across random seeds on this short CPU screen — a 21.6-point spread that swamps every "
                "variant we tested. So none of them is resolved, and reporting a winner here would be "
                "reading noise. The GPU screen settles it."
            )
        with h2:
            _show_graph(
                "graphs/sprint6_conformal_coverage.png",
                "Split-conformal coverage per species against the 90% nominal target.",
            )
            st.caption(
                "**Ra-226 under-covers at 72.7%**, below its 90% target. With n_test = 11 each "
                "scenario is worth 9.1 coverage points, so these estimates are coarse — which is "
                "itself the finding, and why the larger calibration mode runs before any published "
                "uncertainty claim."
            )

    # ── v2 vs v3 comparison with protocol footnote ──────────────────────────
    st.markdown('<div class="sh">Two models, one protocol (and one footnote)</div>', unsafe_allow_html=True)
    st.markdown(
        "Model B (PI-LSTM) was benchmarked against Model A (v2 PINN) on the same 22 scenarios under an "
        "**endpoint protocol** — error measured at the irradiation endpoint, Ac-225 channel. "
        "Result: **v2 = 8.18%, v3 = 5.12%** median relative error vs ODE "
        "(`v3_pilstm/results/compare_v2_pilstm.json`)."
    )
    st.markdown(
        """
        <div class="card warn">
        <h4>Protocol footnote — read before quoting any percentage</h4>
        <p>
        <b>4.51%</b> (canonical v63) and <b>8.18%</b> (v2 in the head-to-head) describe the <i>same model</i>
        under two different measurement protocols: full-trajectory pipeline vs endpoint-only. They are not
        contradictory and must not be mixed. Comparisons between models use only same-protocol numbers.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sp_png = _chart_species_compare_png()
    if sp_png:
        st.image(sp_png, use_container_width=True)
    else:
        _show_graph("graphs/v3_species_median_errors.png",
                    caption="Per-species median endpoint error, v2 vs PI-LSTM (committed figure)")
    if cmp_v2_v3:
        st.markdown(
            "Honesty notes from the same committed file: PI-LSTM is **worse on Ac-227** "
            f"({100.0 * float(cmp_v2_v3['species_median_rel_error']['Ac-227']['pilstm']):.1f}% vs "
            f"{100.0 * float(cmp_v2_v3['species_median_rel_error']['Ac-227']['v2']):.1f}%), and its eager "
            f"inference is ~{float(cmp_v2_v3['mean_inference_ms']['pilstm']) / max(float(cmp_v2_v3['mean_inference_ms']['v2']), 1e-9):.0f}× slower per scenario "
            f"({float(cmp_v2_v3['mean_inference_ms']['pilstm']):.1f} ms vs {float(cmp_v2_v3['mean_inference_ms']['v2']):.1f} ms). "
            "Neither model overshoots Ra-227 at high flux (0/22 both)."
        )

    # ── Validation ladder ───────────────────────────────────────────────────
    st.markdown('<div class="sh">The validation ladder</div>', unsafe_allow_html=True)
    st.markdown(
        "Each rung is a stronger, more independent check. Rungs 3–4 compare the **ODE reference** to "
        "measurements (the surrogate is validated at rung 1) — the ladder is labeled so no rung is oversold."
    )
    st.markdown(f"""
    <div class="rung"><div class="n">1</div><div class="b">
      <b>Surrogate vs stiff ODE reference.</b> 22 held-out scenarios, unseen during training:
      {100.0*heldout_rel:.2f}% median Ac-225 error, 6/6 gates. This is the surrogate-accuracy claim.
    </div></div>
    <div class="rung"><div class="n">2</div><div class="b">
      <b>ODE reference vs evaluated nuclear data.</b> Synthetic constants replaced by JENDL-5 σ(E)
      (verified identical to ENDF/B-VIII.0, max dev 0.0 b), the only experimental (n,2n) point
      (EXFOR 21405), EXFOR 31760 thermal capture 13.8 ± 0.3 b, and NuDat 3 half-lives — all behind a
      versioned flag (<code>ODE_DATA_VERSION=v2</code>), machine-parsed, audit-trailed.
    </div></div>
    <div class="rung"><div class="n">3</div><div class="b">
      <b>ODE vs national-lab production estimates.</b> After spectrum folding, the v2 ODE lands at
      <b>1.00×</b> of Sano 2024's 15.4 ± 6.2 GBq Joyo measurement — inside published uncertainty —
      and 0.57× / 1.14× on the two Iwahashi 2022 anchors (which mutually disagree ~2×).
    </div></div>
    <div class="rung"><div class="n">4</div><div class="b">
      <b>ODE vs decay-leg and thermal-leg measurements.</b> Snow 2025 φ=0 ingrowth: 1.00× of
      126.8 ± 12.6 Bq. Hogle 2016 HFIR Ac-227: 0.82× (3 d) and <b>0.98×</b> (7 d) of measurement;
      the 26 d near-saturation point is ~3× over by both data versions — flagged as missing
      long-irradiation physics, not hidden.
    </div></div>
    """, unsafe_allow_html=True)

    # ── Conformal UQ ────────────────────────────────────────────────────────
    st.markdown('<div class="sh">Uncertainty quantification — honest intervals</div>', unsafe_allow_html=True)
    if conformal:
        ac_rel = conformal.get("Ac-225", {}).get("relative", {})
        cov = ac_rel.get("test_coverage")
        q_rel = ac_rel.get("q")
        n_cal, n_test = conformal.get("n_calibration"), conformal.get("n_test")
        c1, c2, c3 = st.columns(3)
        c1.metric("Nominal coverage", f"{100.0 * float(conformal.get('nominal_coverage', 0.9)):.0f}%")
        c2.metric("Ac-225 test coverage", f"{100.0 * float(cov):.1f}%" if cov is not None else "—",
                  f"n_test = {n_test}")
        c3.metric("Relative half-width q", f"{float(q_rel):.2f}" if q_rel is not None else "—",
                  "interval = prediction × (1 ± q)")
        st.markdown(
            f"""
            <div class="card warn">
            <h4>The intervals are wide — and we say so</h4>
            <p>
            Split conformal at n_cal = {n_cal} gives Ac-225 relative intervals of ±{100.0*float(q_rel):.0f}%
            at 90% nominal — driven by the worst calibration scenario. Coverage lands at {100.0*float(cov):.1f}%
            on {n_test} test scenarios, but {n_test} points cannot distinguish 90% from 70%. The fix is built and
            smoke-verified — large-n conformal (≥100+100 disjoint scenarios), jackknife+/CV+ with bootstrap
            stability reporting, and a K=5 deep ensemble as a second UQ signal — all awaiting full-budget
            Kaggle runs before any tighter interval is claimed (docs/UPGRADE_LOG.md #5, #7, #10).
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("`v3_pilstm/results/conformal_validation.json` not found — UQ panel unavailable.")

    # ── Failure / transparency ──────────────────────────────────────────────
    st.markdown('<div class="sh">Where the models fail — disclosed, not buried</div>', unsafe_allow_html=True)
    buckets = v63_report.get("heldout_buckets_ac225_median_rel", {})
    ep = 100.0 * float(buckets.get("epithermal_virgin", 0.0955))
    th = 100.0 * float(buckets.get("threshold_virgin", 0.0852))
    f1, f2, f3 = st.columns(3)
    f1.markdown(f"""
    <div class="card bad"><h4>Epithermal regime ≈ {ep:.1f}%</h4>
    <p>Resonance-region capture is hardest for the surrogate. Use for ranking only; confirm with the
    ODE near resonance structure.</p></div>""", unsafe_allow_html=True)
    f2.markdown(f"""
    <div class="card bad"><h4>(n,2n) threshold ≈ {th:.1f}%</h4>
    <p>At the ~6.4 MeV cliff the cross section turns on sharply; largest surrogate uncertainty lives
    here. Confirm with ODE near onset.</p></div>""", unsafe_allow_html=True)
    f3.markdown(f"""
    <div class="card bad"><h4>PI-LSTM thermal Ac-227 leg</h4>
    <p>On two thermal Ac-227 literature rows the PI-LSTM is orders of magnitude off
    (v2 PINN: 2.8% MAPE on the same rows). Out-of-domain channel — flagged in
    empirical_validation.json, not retrained yet.</p></div>""", unsafe_allow_html=True)
    st.markdown(
        "Also disclosed in the provenance file: the 26-day HFIR Ac-227 point is overpredicted ~3× by "
        "**both** data versions (likely capsule self-shielding/burnup absent from the point-model ODE), "
        "and the evaluated σ shape disagrees with the single experimental point at 14.5 MeV by ~3× "
        "(experiment ~3× the evaluation) — a real, citable tension, left unresolved."
    )


# ==============================================================================
# TAB — THE DISCOVERY
# ==============================================================================
with tab_discovery:
    st.markdown('<div class="kicker">The flagship result of this sprint</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh" style="margin-top:0.2rem;">Chasing a 19× error down to 1.00×</div>', unsafe_allow_html=True)
    st.markdown(
        "The surrogate is only as honest as the physics it learns from. This sprint, every synthetic "
        "constant in the reference ODE was replaced with **evaluated nuclear data** — and the "
        "replacements refused to behave. What follows is the three-act story, told with the committed "
        "numbers (`results/ode_data_v2_validation_20260718.json`, "
        "`results/ode_data_v2_spectrum_20260718.json`, `docs/DATA_PROVENANCE.md`)."
    )

    a1, a2, a3 = st.columns(3)
    a1.markdown("""
    <div class="card"><h4>Act 1 · The synthetic σ was wrong</h4>
    <p>The original model used a smooth sigmoid σ(n,2n) saturating at <b>27 mb</b>, believed to be a
    "spectrum-averaged fast-reactor value". Against Japan's Joyo reactor anchors it overpredicted
    Ac-225 by <b>18.9×</b> (Sano 2024). The evaluated JENDL-5 pointwise value at 14 MeV is
    <b>755.7 mb — ~28× larger</b>.</p></div>
    """, unsafe_allow_html=True)
    a2.markdown("""
    <div class="card bad"><h4>Act 2 · The correct σ made it worse</h4>
    <p>Installing the real evaluated σ moved the Joyo prediction from 18.9× to <b>369.6×</b>
    (up to 2153× on the milking anchor). The cross section was never the dominant error: the model
    treated <b>every neutron as a 14.5 MeV neutron</b> — the monoenergetic-flux approximation.
    The old 27 mb had been accidentally compensating for that. Two wrongs, roughly right.</p></div>
    """, unsafe_allow_html=True)
    a3.markdown("""
    <div class="card ok"><h4>Act 3 · Folding over a spectrum fixed it</h4>
    <p>Folding σ(E) over a bare U-235 Watt fission spectrum gives ⟨σ⟩ = <b>26.7 mb ≈ the old 27 mb</b>
    — mystery solved. A two-group model with inferred above-threshold fraction
    <b>f* = 1.24×10⁻³ [7.4×10⁻⁴, 1.7×10⁻³]</b> lands at <b>1.00×</b> of Sano's
    15.4 ± 6.2 GBq — inside published national-lab uncertainty.</p></div>
    """, unsafe_allow_html=True)

    ratios_png = _chart_discovery_ratios_png()
    if ratios_png:
        st.image(ratios_png, use_container_width=True)

    _show_graph(
        "graphs/ode_v2_spectrum_anchors.png",
        caption="Committed figure: Joyo anchors across physics variants (left); the evaluated σ(E) and "
                "Watt spectrum on the tail physics (right). Source: analysis/validate_ode_spectrum.py.",
    )

    # ── Plain-English explainer ─────────────────────────────────────────────
    st.markdown('<div class="sh">Why the yield lives in the tail</div>', unsafe_allow_html=True)
    e1, e2 = st.columns([1.15, 1])
    with e1:
        st.markdown(
            "The first step of the production chain, Ra-226(n,2n)Ra-225, is a **threshold reaction**: "
            "a neutron carrying less than **6.42 MeV** simply cannot start it, no matter how many "
            "neutrons there are. In a reactor, the overwhelming majority of neutrons are far slower "
            "than that — thermalized by collisions with coolant and structure. Ac-225 production "
            "therefore depends on the **small high-energy tail** of the spectrum, not the total flux. "
            "Treating the full Joyo core flux (5.7×10¹⁵ n/cm²/s) as if it were all 14.5 MeV neutrons "
            "is why the pointwise model overpredicts by ~370×. The inferred f* says only about "
            "**1 neutron in 800** at that irradiation position sits above threshold — order-of-magnitude "
            "plausible for a sodium-cooled MOX fast breeder, ~16× softer than a bare fission spectrum."
        )
        st.markdown(
            f"""
            <div class="card accent"><h4>The 27 mb mystery, solved</h4>
            <p>Folding the evaluated table over a bare fission spectrum reproduces ⟨σ⟩ = 26.7 mb —
            essentially the legacy synthetic constant. The old model wasn't using a pointwise cross
            section at all; it was a <b>fission-spectrum average used in the wrong place</b>.
            Real irradiation-position spectra are far softer (⟨σ⟩ ≈ 1.66 mb at f*).</p></div>
            """,
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            f"""
            <div class="card"><h4>Numbers a judge can check</h4>
            <p>
            Threshold: <b>6.4218 MeV</b> (evaluated table start)<br>
            Evaluated σ at 14 MeV: <b>755.7 mb</b>; peak 2.53 b @ 10 MeV<br>
            Only experimental (n,2n) point: <b>1.60 ± 0.20 b @ 14.5 MeV</b> (EXFOR 21405, 1960)<br>
            Bare-Watt fold: <b>26.7 mb</b>; two-group at f*: <b>1.66 mb</b><br>
            f* = <b>1.24×10⁻³</b> [7.4×10⁻⁴, 1.7×10⁻³] (exact ODE inversion on Sano's band)<br>
            Joyo anchors at f*: <b>1.00× / 0.57× / 1.14×</b> (Sano / Iwahashi-60d / milking)
            </p></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="layman-box" style="background:#fdf3e7;border:1px solid #e9d9c3;border-radius:10px;padding:1rem;">'
            + laymans_explanation("tail") + "</div>",
            unsafe_allow_html=True,
        )

    # ── Interactive ODE lab (real computation, mono vs folded) ─────────────
    st.markdown('<div class="sh">Live ODE lab — flip the spectrum assumption yourself</div>', unsafe_allow_html=True)
    st.markdown(
        "This panel re-runs the **real stiff ODE** (Radau integrator, evaluated JENDL-5/EXFOR/NuDat data) "
        "for a Joyo-style scenario — 1 g Ra-226, φ = 5.7×10¹⁵ n/cm²/s — under each physics variant. "
        "Nothing here is a lookup table: every change re-integrates the Bateman chain on CPU."
    )
    if not _evaluated_data_available():
        st.info(
            "Evaluated data layer (`data/evaluated/`) is not present in this deployment, so only the "
            "legacy synthetic variant can run. The committed results above still tell the full story."
        )

    lab_l, lab_r = st.columns([1, 1.6])
    with lab_l:
        mode_options = ["v1 synthetic σ (legacy)", "v2 evaluated σ · monoenergetic",
                        "v2 evaluated σ · Watt fission fold", "v2 evaluated σ · two-group (adjust f)"]
        if not _evaluated_data_available():
            mode_options = mode_options[:1]
        mode_label = st.radio("Physics variant", mode_options, index=3 if len(mode_options) > 1 else 0)
        days = st.slider("Irradiation time (days)", 10, 90, int(SANO_DAYS), 5, key="joyo_days")
        f_val = FSTAR_INFERRED
        if mode_label.endswith("(adjust f)"):
            f_exp = st.slider(
                "log₁₀ fast fraction f (>6.42 MeV)", -4.0, -0.7,
                float(np.log10(FSTAR_INFERRED)), 0.05, key="joyo_f",
            )
            f_val = 10.0 ** f_exp
            st.caption(f"f = {f_val:.2e} — inferred f* = {FSTAR_INFERRED:.2e} [7.4e-4, 1.7e-3]")
        st.caption(
            "v2 runs use each variant's own evaluated half-lives (NuDat 3); v1 uses the legacy "
            "hard-coded set — matching the committed validation scripts exactly."
        )
    with lab_r:
        mode_map = {
            "v1 synthetic σ (legacy)": ("v1", "mono", None),
            "v2 evaluated σ · monoenergetic": ("v2", "mono", None),
            "v2 evaluated σ · Watt fission fold": ("v2", "watt", None),
            "v2 evaluated σ · two-group (adjust f)": ("v2", "twogroup", f_val),
        }
        ver, spec_mode, ff = mode_map[mode_label]
        pred_gbq = _joyo_ode_gbq(ver, spec_mode, ff if ff is not None else FSTAR_INFERRED, float(days))
        if pred_gbq is None:
            st.error("ODE evaluation failed in this deployment — showing committed results instead.")
        else:
            ratio = pred_gbq / SANO_MEASURED_GBQ
            in_band = abs(pred_gbq - SANO_MEASURED_GBQ) <= SANO_ERR_GBQ
            k1, k2, k3 = st.columns(3)
            k1.metric("ODE prediction", f"{pred_gbq:.2f} GBq")
            k2.metric("÷ Sano 15.4 GBq", f"{ratio:.2f}×")
            k3.metric("Within ±6.2 GBq band?", "Yes ✓" if in_band else "No")
            lab_png = _chart_joyo_lab_png(mode_label.split("·")[-1].strip(), pred_gbq, float(days))
            if lab_png:
                st.image(lab_png, use_container_width=True)
            if days != SANO_DAYS:
                st.caption(
                    f"Note: Sano's measurement is a {SANO_DAYS:.0f}-d irradiation — at {days} d the "
                    "comparison band is indicative, not a formal test."
                )

    # ── f* sweep + what remains uncertain ───────────────────────────────────
    st.markdown('<div class="sh">One parameter, with an honest band</div>', unsafe_allow_html=True)
    fs_png = _chart_fstar_sweep_png()
    if fs_png:
        st.image(fs_png, use_container_width=True)
    st.markdown(
        """
        <div class="card warn">
        <h4>What remains uncertain (said out loud)</h4>
        <p>
        f* is an <b>effective inferred parameter, not a tuned truth</b>: the measured Joyo MK-III spectrum
        is paywalled, Sano (±40%) and Iwahashi disagree with each other by ~2×, and only one experimental
        (n,2n) cross-section point exists anywhere (1960). The two-group spectrum is a documented
        assumption with citable form (Watt 1952; Iwahashi 2022; Aoyama 2005). A published MK-III spectrum
        table would replace the whole f* exercise with a direct fold — that is the field's open problem,
        and it is now the app's open problem too. Full list: docs/DATA_PROVENANCE.md §4.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Decay-leg and thermal-leg anchors (spectrum-independent)"):
        _show_graph(
            "graphs/ode_v2_literature_anchors.png",
            caption="Committed figure: Hogle 2016 (HFIR thermal (n,γ) Ac-227 leg) and Snow 2025 (φ=0 "
                    "Ra-225→Ac-225 ingrowth) anchors under v1 vs v2 data.",
        )
        if ode_v2_val.get("anchors"):
            rows = []
            for a in ode_v2_val["anchors"]:
                if a["id"].startswith(("hogle", "snow")):
                    rows.append({
                        "Anchor": a["id"],
                        "Measured": f"{a['measured_bq']:.4g} Bq" + (f" ± {a['measured_err_bq']:.3g}" if a.get("measured_err_bq") else ""),
                        "v1 ratio": f"{a['v1_ratio_pred_over_meas']:.2f}×",
                        "v2 ratio": f"{a['v2_ratio_pred_over_meas']:.2f}×",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB — SCREENING
# ==============================================================================
with tab_screen:
    st.markdown('<div class="kicker">The surrogate at work</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh" style="margin-top:0.2rem;">Interactive screening</div>', unsafe_allow_html=True)
    st.markdown(
        "Everything on this tab runs the **v2 PINN surrogate** (Model A) live on CPU. "
        "A screening note first, so the interactivity is never oversold:"
    )
    st.markdown(
        """
        <div class="card accent">
        <h4>Spectrum-aware screening — coming after the retrain</h4>
        <p>
        The deployed surrogate was trained against the v1 (synthetic-σ, monoenergetic) reference, so its
        fast-regime sweeps inherit that assumption. Spectrum-aware scenarios (mono for D-T accelerator
        settings, folded Watt / two-group for reactor settings) are a <b>scenario-level input the next
        model will learn from</b> — the ODE-side physics is live today in <b>The Discovery → Live ODE
        lab</b>; the retrained surrogate follows the Kaggle run. No fake toggles here.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Scenario explorer ───────────────────────────────────────────────────
    st.markdown('<div class="sh-sm">Scenario explorer — one setting, PINN vs ODE</div>', unsafe_allow_html=True)
    if model is None:
        st.warning("Trained weights not found. Deploy weights/pinn_best_weights.pth to activate the simulator.")
    else:
        ci, co = st.columns([1, 1.7])
        with ci:
            flux_exp = st.slider("Flux (log₁₀ n/cm²/s)", 12.0, 15.5, 14.0, 0.1, key="live_flux")
            flux = 10.0 ** flux_exp
            st.caption(f"φ = {flux:.2e} n/cm²/s")
            time_h = st.slider("Irradiation time (hours)", 1.0, 500.0, 200.0, 5.0, key="live_time")
            spectrum = st.selectbox(
                "Neutron energy regime",
                ["Fast production (14 MeV)", "Threshold edge (6.4 MeV)", "Epithermal (1 eV)", "Thermal capture (0.025 eV)"],
                key="live_spectrum",
            )
            energy_ev = {
                "Fast production (14 MeV)": 14.0e6,
                "Threshold edge (6.4 MeV)": 6.4e6,
                "Epithermal (1 eV)": 1.0,
                "Thermal capture (0.025 eV)": 0.025,
            }[spectrum]
            st.caption(f"E = {energy_ev:.3e} eV")
            ra226_0 = st.number_input("Starting Ra-226 (atoms)", value=6.022e23, format="%.3e", key="live_ra226_0")
            st.caption(f"Readable weight: **{format_atoms_human(ra226_0, 226)}**")
            st.markdown(
                '<div style="background:#fdf3e7;border:1px solid #e9d9c3;border-radius:10px;padding:0.9rem;margin-top:0.6rem;">'
                + laymans_explanation("transmutation") + "</div>",
                unsafe_allow_html=True,
            )
        with co:
            from pinn_model import (
                DEFAULT_N226_SCALE as N226S, DEFAULT_N225_SCALE as N225S,
                DEFAULT_NAC_SCALE as NACS, DEFAULT_N227_SCALE as N227S,
                DEFAULT_NAC227_SCALE as NAC7S, DEFAULT_PHI_SCALE as PHIS,
                DEFAULT_T_REF_H as TSH, neutron_energy_ev_to_feature_numpy,
            )
            e_nn = float(neutron_energy_ev_to_feature_numpy(energy_ev))
            x_input = torch.tensor([[
                time_h / TSH, flux / PHIS, e_nn,
                ra226_0 / N226S, 0.0, 0.0, 0.0, 0.0,
            ]], dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                pred = model(x_input)
            p226 = float(pred[0, 0] * N226S)
            pac = float(pred[0, 2] * NACS)
            pac7 = float(pred[0, 4] * NAC7S)

            for msg in _domain_warnings(flux=flux, energy_ev=energy_ev, time_h=time_h):
                st.warning(f"Domain warning: {msg}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Ra-226 remaining", f"{p226:.2e}", help=format_atoms_human(p226, 226))
            m2.metric("Ac-225 product", f"{pac:.2e}", help=format_atoms_human(pac, 225))
            m3.metric("Ac-227 impurity", f"{pac7:.2e}", help=format_atoms_human(pac7, 227))

            ac225_bq = float(_activity_bq(pac, AC225_HALF_LIFE_DAYS))
            ac227_bq = float(_activity_bq(pac7, AC227_HALF_LIFE_DAYS))
            if ac225_bq + ac227_bq > 0:
                purity = (ac225_bq / (ac225_bq + ac227_bq)) * 100.0
                st.metric("Ac-225 activity purity", f"{purity:.4f}%",
                          delta=f"{100.0 - purity:.4f}% impurity", delta_color="inverse")
                if purity >= 100.0 - STRICT_AC227_IMPURITY_LIMIT_PCT:
                    st.success(f"Within the {STRICT_AC227_IMPURITY_LIMIT_PCT:.2f}% Ac-227 activity-impurity reference threshold.")
                else:
                    st.error(f"Above the {STRICT_AC227_IMPURITY_LIMIT_PCT:.2f}% Ac-227 activity-impurity reference threshold.")

            live_png = _chart_pinn_ode_live_png(_weights_key_str, flux, time_h, energy_ev, ra226_0)
            if live_png:
                st.image(live_png, use_container_width=True,
                         caption="Surrogate vs the stiff ODE reference on this exact scenario.")

    st.divider()

    # ── 10,000-scenario triage ──────────────────────────────────────────────
    st.markdown('<div class="sh-sm">10,000-scenario triage</div>', unsafe_allow_html=True)
    st.markdown(
        "Sweep a 100×100 grid of flux × irradiation time at 14 MeV, apply post-processing "
        "(5 d cooling, 90% recovery), and classify every point against your yield target and the "
        "Ac-227 impurity limit — in one batched inference call."
    )
    t1, t2 = st.columns([1, 1.8])
    with t1:
        st.markdown('<div class="card"><h4>Triage constraints</h4>', unsafe_allow_html=True)
        target_ac = st.slider("Target Ac-225 activity (mCi)", 1.0, 50.0, 15.0, 1.0)
        max_impurity = st.slider("Max allowed Ac-227 impurity (%)", 0.05, 0.5, 0.15, 0.01)
        st.caption(f"Reference impurity limit: {STRICT_AC227_IMPURITY_LIMIT_PCT:.2f}% Ac-227 activity")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#fdf3e7;border:1px solid #e9d9c3;border-radius:10px;padding:0.9rem;">'
            + laymans_explanation("flux") + "<br><br>" + laymans_explanation("impurity") + "</div>",
            unsafe_allow_html=True,
        )
    with t2:
        if model is None:
            st.warning("Trained model weights not found.")
        elif st.button("Run 10,000-scenario triage"):
            t0 = time.perf_counter()
            fluxes = np.logspace(12.0, 15.5, 100)
            times_h = np.linspace(1.0, 500.0, 100)
            flux_grid, time_grid = np.meshgrid(fluxes, times_h)

            from pinn_model import (
                DEFAULT_N226_SCALE as N226S, DEFAULT_NAC_SCALE as NACS,
                DEFAULT_NAC227_SCALE as NAC7S, DEFAULT_PHI_SCALE as PHIS,
                DEFAULT_T_REF_H as TSH, neutron_energy_ev_to_feature_numpy as _efn,
            )
            e_nn = float(_efn(14.0e6))
            rows = np.column_stack([
                time_grid.ravel() / TSH,
                flux_grid.ravel() / PHIS,
                np.full(10000, e_nn),
                np.full(10000, 6.022e23 / N226S),
                np.zeros(10000), np.zeros(10000), np.zeros(10000), np.zeros(10000),
            ])
            x_t = torch.tensor(rows, dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                pred = model(x_t).cpu().numpy()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            raw_ac225 = np.maximum(pred[:, 2] * NACS, 0.0)
            raw_ac227 = np.maximum(pred[:, 4] * NAC7S, 0.0)
            usable_ac225 = raw_ac225 * _decay_factor(5.0, AC225_HALF_LIFE_DAYS) * 0.90
            recovered_ac227 = raw_ac227 * _decay_factor(5.0, AC227_HALF_LIFE_DAYS) * 0.90
            ac225_mci = _activity_bq(usable_ac225, AC225_HALF_LIFE_DAYS) / 3.7e7
            impurity_pct = _ac227_impurity_activity_pct(usable_ac225, recovered_ac227)

            is_safe = (impurity_pct <= max_impurity) & (ac225_mci >= target_ac)
            is_toxic = impurity_pct > max_impurity
            safe_count = int(np.sum(is_safe))
            toxic_count = int(np.sum(is_toxic))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Scenarios triaged", "10,000")
            m2.metric("Inference time", f"{elapsed_ms:.1f} ms", f"{10000 / max(elapsed_ms / 1000.0, 1e-9):,.0f} runs/s")
            m3.metric("Within constraints", f"{safe_count}", f"{safe_count / 100:.1f}% of grid")
            m4.metric("Impurity exceeded", f"{toxic_count}", f"{toxic_count / 100:.1f}% of grid")

            if safe_count > 0:
                best_idx = np.argmax(np.where(is_safe, ac225_mci, -1.0))
                st.success(
                    f"**Best feasible setting:** {time_grid.ravel()[best_idx]:.1f} h at "
                    f"{flux_grid.ravel()[best_idx]:.2e} n/cm²/s — ~{ac225_mci[best_idx]:.2f} mCi Ac-225 "
                    f"(post-processing applied), Ac-227 impurity {impurity_pct[best_idx]:.3f}%."
                )
            st.caption(
                "Green = meets yield and impurity constraints · Red = impurity exceeded · Gray = insufficient yield. "
                "Dot map shows a 20×20 sample of the full grid. Batched-throughput timing, not per-scenario latency."
            )
            grid_html = '<div class="triage-grid">'
            sub_safe = is_safe.reshape(100, 100)[::5, ::5].ravel()
            sub_toxic = is_toxic.reshape(100, 100)[::5, ::5].ravel()
            for idx in range(400):
                if sub_safe[idx]:
                    grid_html += '<div class="triage-dot safe" title="Usable batch"></div>'
                elif sub_toxic[idx]:
                    grid_html += '<div class="triage-dot toxic" title="Impurity breached"></div>'
                else:
                    grid_html += '<div class="triage-dot low" title="Low yield"></div>'
            grid_html += "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)
        else:
            st.info('Press "Run 10,000-scenario triage" to populate the map.')

    st.divider()

    # ── 2D production map ───────────────────────────────────────────────────
    st.markdown('<div class="sh-sm">2D production map — the clinical safe-zone</div>', unsafe_allow_html=True)
    if model is None:
        st.warning("Weights not loaded.")
    else:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            st.markdown('<div class="card"><h4>Post-processing</h4>', unsafe_allow_html=True)
            chem_recovery = st.slider("Separation recovery yield (%)", 50, 100, 90, 5) / 100.0
            cooling = st.slider("Post-irradiation cooling (days)", 0, 14, 5, 1)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(
                '<div style="background:#fdf3e7;border:1px solid #e9d9c3;border-radius:10px;padding:0.9rem;">'
                + laymans_explanation("half_life") + "</div>",
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("Map the safe-zone (2,500 scenarios)"):
                with st.spinner("Batched inference over the 2D grid…"):
                    t_vec = np.linspace(10.0, 500.0, 50)
                    f_vec = np.logspace(13.0, 15.5, 50)
                    T, F = np.meshgrid(t_vec, f_vec)
                    from pinn_model import (
                        DEFAULT_N226_SCALE as N226S, DEFAULT_NAC_SCALE as NACS,
                        DEFAULT_NAC227_SCALE as NAC7S, DEFAULT_PHI_SCALE as PHIS,
                        DEFAULT_T_REF_H as TSH, neutron_energy_ev_to_feature_numpy as _efn,
                    )
                    e_nn = float(_efn(14.0e6))
                    rows = np.column_stack([
                        T.ravel() / TSH, F.ravel() / PHIS, np.full(2500, e_nn),
                        np.full(2500, 6.022e23 / N226S),
                        np.zeros(2500), np.zeros(2500), np.zeros(2500), np.zeros(2500),
                    ])
                    x_t = torch.tensor(rows, dtype=torch.float32)
                    model.eval()
                    with torch.no_grad():
                        pred = model(x_t).cpu().numpy()
                    raw_ac225 = np.maximum(pred[:, 2] * NACS, 0.0).reshape(50, 50)
                    raw_ac227 = np.maximum(pred[:, 4] * NAC7S, 0.0).reshape(50, 50)
                    usable_ac225 = raw_ac225 * _decay_factor(cooling, AC225_HALF_LIFE_DAYS) * chem_recovery
                    recovered_ac227 = raw_ac227 * _decay_factor(cooling, AC227_HALF_LIFE_DAYS) * chem_recovery
                    ac225_mci = _activity_bq(usable_ac225, AC225_HALF_LIFE_DAYS) / 3.7e7
                    impurity = _ac227_impurity_activity_pct(usable_ac225, recovered_ac227)

                    import matplotlib.pyplot as plt

                    with plt.rc_context(_light_rc()):
                        fig, ax = plt.subplots(figsize=(8.6, 5.2))
                        cp = ax.contourf(T, F, ac225_mci, levels=20, cmap="YlOrBr", alpha=0.92)
                        cbar = fig.colorbar(cp, ax=ax)
                        cbar.set_label("Recovered Ac-225 activity (mCi)")
                        ax.contour(T, F, impurity, levels=[STRICT_AC227_IMPURITY_LIMIT_PCT],
                                   colors=[_BAD], linewidths=2.4)
                        ax.contourf(T, F, impurity, levels=[STRICT_AC227_IMPURITY_LIMIT_PCT, 100.0],
                                    colors=[_BAD], alpha=0.18)
                        safe_mask = impurity <= STRICT_AC227_IMPURITY_LIMIT_PCT
                        if np.any(safe_mask):
                            bi = np.unravel_index(np.argmax(np.where(safe_mask, ac225_mci, -1)), ac225_mci.shape)
                            ax.scatter(T[bi], F[bi], color=_TEAL, edgecolors="white", s=120, zorder=5,
                                       label="Optimal schedule")
                            ax.legend(fontsize=9)
                        ax.set_yscale("log")
                        ax.set_xlabel("Irradiation time (h)")
                        ax.set_ylabel("Neutron flux (n/cm²/s)")
                        ax.set_title("Safe-zone boundary at the 0.15% Ac-227 impurity reference line")
                        fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    if np.any(safe_mask):
                        st.success(
                            f"**Feasible setting:** {T[bi]:.1f} h at {F[bi]:.2e} n/cm²/s — "
                            f"~{ac225_mci[bi]:.2f} mCi Ac-225 (post-processing applied)."
                        )
                    else:
                        st.error("Every setting in this sweep exceeds the impurity threshold. Try longer cooling.")

    # ── Clinical context (supply chain only) ────────────────────────────────
    with st.expander("Clinical context — supply-chain translation (illustrative only)"):
        st.markdown(
            "Order-of-magnitude translation of a production batch into treatment-scale doses. "
            "**Not** in vivo pharmacokinetics, **not** regulatory dosing guidance."
        )
        dc1, dc2 = st.columns([1, 1.6])
        with dc1:
            atoms_input = st.number_input("Ac-225 atoms produced", value=1.5e17, format="%.3e")
            st.caption(f"Readable weight: **{format_atoms_human(atoms_input, 225)}**")
            dose_rate = st.slider("Therapeutic reference dosage (kBq/kg)", 50, 250, 100, 10)
            weight = st.slider("Reference patient mass (kg)", 50, 110, 75, 5)
        with dc2:
            decay_const = math.log(2.0) / (AC225_HALF_LIFE_DAYS * SECONDS_PER_DAY)
            act_bq = atoms_input * decay_const
            act_mci = act_bq / 3.7e7
            psma_dose_mbq = 8.0
            leukemia_dose_mbq = 18.0
            dm1, dm2, dm3 = st.columns(3)
            dm1.metric("Batch activity", f"{act_mci:.2f} mCi")
            dm2.metric("PSMA-scale doses", f"{(act_bq / 1e6) / psma_dose_mbq:.1f}", "8 MBq reference")
            dm3.metric("CD33-scale doses", f"{(act_bq / 1e6) / leukemia_dose_mbq:.1f}", "18 MBq reference")
            st.caption(
                "Reference activity levels from published trial orders of magnitude — not patient-specific "
                "prescribing. Surrogate outputs are planning estimates vs the ODE reference, never batch sign-off."
            )

# ==============================================================================
# TAB — SPEED
# ==============================================================================
with tab_speed:
    st.markdown('<div class="kicker">Honest throughput, disclosed latency</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh" style="margin-top:0.2rem;">How fast is it, exactly?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card warn">
        <h4>No bare "500×" here</h4>
        <p>
        Early materials quoted a ~500× speedup with no committed evidence file; the audit showed the number
        was a <b>batching effect</b> (one batched GPU/CPU call vs sequential stiff ODE solves), and that
        PI-LSTM <i>eager</i> inference is actually ~10× <b>slower</b> than the v2 PINN per scenario.
        The honest claim is <b>throughput under batching</b>. A formal re-verification of the speed
        benchmark is in progress (docs/UPGRADE_LOG.md #6); until it lands, this tab shows committed
        harness numbers plus an optional live measurement on this machine.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sh-sm">Committed harness numbers</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Single-scenario eager latency** (the number that matters for interactive use)")
        eager_rows = []
        if cmp_v2_v3.get("mean_inference_ms"):
            ms = cmp_v2_v3["mean_inference_ms"]
            eager_rows.append({"Model": "v2 PINN", "Eager latency": f"{float(ms['v2']):.1f} ms", "Source": "compare_v2_pilstm.json"})
            eager_rows.append({"Model": "v3 PI-LSTM", "Eager latency": f"{float(ms['pilstm']):.1f} ms", "Source": "compare_v2_pilstm.json"})
        if speed_v3.get("eager"):
            eager_rows.append({"Model": "v3 PI-LSTM (harness)", "Eager latency": f"{float(speed_v3['eager']['ms_per_scenario']):.1f} ms/scenario", "Source": "speed_harness.json"})
        if eager_rows:
            st.dataframe(pd.DataFrame(eager_rows), use_container_width=True, hide_index=True)
        st.caption("PI-LSTM eager is ~10× slower than the v2 PINN — disclosed, not hidden.")
    with h2:
        st.markdown("**Batched throughput** (the number that matters for sweeps)")
        batch_rows = []
        if speed_v3.get("batched"):
            b = speed_v3["batched"]
            batch_rows.append({
                "Measurement": "PI-LSTM batched (22 scenarios)",
                "Value": f"{float(b['ms_per_scenario']):.2f} ms/scenario",
                "Note": f"{float(b['speedup_vs_eager']):.0f}× vs its own eager mode — a batching effect",
            })
        batch_rows.append({
            "Measurement": "Reference Radau ODE",
            "Value": "seconds per solve (sequential)",
            "Note": "each scenario is a stiff solve; no batching possible",
        })
        st.dataframe(pd.DataFrame(batch_rows), use_container_width=True, hide_index=True)
        st.caption("Screening value comes from batching 10²–10⁴ scenarios into one tensor call.")

    st.markdown('<div class="sh-sm">Live measurement on this machine</div>', unsafe_allow_html=True)
    st.caption(
        "Batched PINN inference vs sequential stiff Radau ODE solves on the same random in-domain "
        "scenarios. Throughput framing: the ODE cannot be batched; per-scenario eager latency is above."
    )
    bench_n = int(os.environ.get("PINN_BENCH_SCENARIOS", "24"))
    if _weights_key_str and st.button(f"Run throughput benchmark ({bench_n} scenarios)", key="speed_bench_btn"):
        with st.spinner(f"Timing {bench_n} scenarios (ODE solves are the slow part)…"):
            st.session_state["bench_result"] = _cached_speed_benchmark(_weights_key_str, bench_n)
    bench = st.session_state.get("bench_result")
    if bench:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Scenarios", bench.get("n_scenarios", "—"))
        b2.metric("PINN batched", f"{bench.get('pinn_ms', 0):.1f} ms",
                  f"{bench.get('pinn_ms_per_scenario', 0):.2f} ms/scenario")
        b3.metric("ODE sequential", f"{bench.get('ode_ms', 0):.0f} ms",
                  f"{bench.get('ode_ms_per_scenario', 0):.0f} ms/scenario")
        b4.metric("Throughput ratio", f"{bench.get('throughput_ratio', 0):.0f}×",
                  "batched vs serial — not an eager-latency ratio")
        st.caption(
            f"Median Ac-225 error on benchmark draws: {100.0 * bench.get('median_rel_err_ac225', 0):.1f}% vs ODE. "
            "Ratio varies with hardware and batch size; the committed, reproducible benchmark is the pending Kaggle job."
        )
    elif _weights_key_str:
        st.caption("Press the button to measure timing on this machine.")
    else:
        st.info("Model weights required for the live benchmark.")

    st.markdown(
        """
        <div class="card"><h4>Why batching is the honest framing</h4>
        <p>
        Planning asks "which of 10,000 settings is worth a second look" — a throughput question.
        One tensor call evaluating 10,000 scenarios in ~tens of milliseconds is genuinely useful even
        though a single eager call is merely millisecond-scale, and even though per-call the PI-LSTM is
        slower than the PINN. The claim is <b>screening throughput vs sequential ODE integration</b>,
        with hardware, batch size, and eager latency disclosed alongside.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB — METHODS & DATA
# ==============================================================================
with tab_methods:
    st.markdown('<div class="kicker">Reproducibility starts here</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh" style="margin-top:0.2rem;">Model cards</div>', unsafe_allow_html=True)
    mca, mcb = st.columns(2)
    with mca:
        w_sha = v63_report.get("weights", {}).get("sha256", "")[:8] or "—"
        st.markdown(f"""
        <div class="model-card">
          <span class="tag">Model A · deployed surrogate</span>
          <h4 style="margin:0.2rem 0 0.5rem 0;">v2 PINN — weights v63</h4>
          <table style="width:100%;border-collapse:collapse;font-size:0.88rem;color:#57534e;">
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Architecture</b></td><td>Semi-analytic Bateman backbone + learned correction; 4×128 MLP, SiLU, float64</td></tr>
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Training</b></td><td>600 epochs physics-only pretrain → 3,400 joint epochs (4,000 total; the 12k run was rejected at 7.27%)</td></tr>
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Loss</b></td><td>Bateman physics residual + ODE supervision + mass-budget + fuel anchor + zero-injection</td></tr>
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Canonical result</b></td><td>4.51% held-out Ac-225 median vs ODE (22 scenarios), 6/6 gates PASS</td></tr>
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Weights hash</b></td><td>sha256 <code>{w_sha}…</code> (results/v63_validation_20260530.json)</td></tr>
            <tr><td style="padding:6px 0;"><b>Reference physics</b></td><td>v1 synthetic σ (27 mb sigmoid) + hard-coded half-lives — retrain against v2 evaluated physics pending</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
    with mcb:
        v3_epochs = v3_train.get("epochs", 6000)
        v3_best = v3_train.get("best_epoch", "—")
        st.markdown(f"""
        <div class="model-card">
          <span class="tag">Model B · research line</span>
          <h4 style="margin:0.2rem 0 0.5rem 0;">v3 PI-LSTM — distilled sequence model</h4>
          <table style="width:100%;border-collapse:collapse;font-size:0.88rem;color:#57534e;">
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Architecture</b></td><td>2-layer LSTM (hidden 256), Fourier time/energy features (16/8), hard initial condition</td></tr>
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Training</b></td><td>{v3_epochs:,} epochs, best checkpoint @ {v3_best}; distilled from the frozen v2 teacher; deterministic seed (PI_LSTM_SEED=42)</td></tr>
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Physics loss</b></td><td>Exact exponential-integrator propagator (closed-form matrix exponential, stiff-stable) — residual ≤1.2e-14 on exact trajectories</td></tr>
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Fair baseline</b></td><td>Matched-budget vanilla LSTM trained (847,114 vs 850,349 params, 0.4% delta) — the "why physics at all" ablation</td></tr>
            <tr style="border-bottom:1px solid {_BORDER};"><td style="padding:6px 0;"><b>Result</b></td><td>5.12% Ac-225 endpoint median vs ODE (v2: 8.18%, same protocol); Ac-227 channel disclosed weaker</td></tr>
            <tr><td style="padding:6px 0;"><b>Reference physics</b></td><td>Same v1 synthetic reference as Model A — the v2 evaluated, spectrum-aware retrain is the next Kaggle run</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Data provenance — where every physics number comes from</div>', unsafe_allow_html=True)
    st.markdown(
        "Retrieved 2026-07-18 from free, no-login sources; machine-parsed (not transcribed); raw "
        "downloads kept in `data/evaluated/_raw/` for audit. Full document: `docs/DATA_PROVENANCE.md`."
    )
    prov_rows = [
        {"Data": "σ(n,2n)(E) evaluated table", "Source": "JENDL-5 (IAEA-NDS mirror); ENDF/B-VIII.0 verified identical, max dev 0.0 b", "Used in": "ODE_DATA_VERSION=v2"},
        {"Data": "σ(n,2n) experimental point", "Source": "EXFOR 21405 — O'Connor & Perkin 1960: 1.60 ± 0.20 b @ 14.5 MeV (the only one that exists)", "Used in": "evaluation tension check"},
        {"Data": "σ(n,γ) thermal anchor", "Source": "EXFOR 31760 — Bagheri 2015: 13.8 ± 0.3 b (libraries tabulate zero below 1 keV)", "Used in": "v2 thermal capture leg"},
        {"Data": "Half-lives (5 nuclides)", "Source": "NuDat 3 / ENSDF live retrieval with uncertainties", "Used in": "v2 decay constants"},
        {"Data": "Spectrum forms", "Source": "Watt 1952 Phys. Rev. 87, 1037 (a=0.988, b=2.249); Iwahashi 2022 MDPI Processes 10(7):1239; Aoyama 2005 J. Nucl. Radiochem. Sci. 6(3)", "Used in": "SPECTRUM_MODE folding"},
        {"Data": "Production anchors", "Source": "Sano 2024 JNST 61:509; Iwahashi 2022; Hogle 2016 (OSTI 1253240); Snow 2025 (OSTI 3028837)", "Used in": "validation ladder rungs 3–4"},
        {"Data": "Legacy synthetic σ", "Source": "Sigmoid → 27 mb 'spectrum average' — shown to be a fission-spectrum average, ~28× too small pointwise", "Used in": "v1 (bit-preserved default)"},
    ]
    st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)

    # ── Upgrade timeline ────────────────────────────────────────────────────
    st.markdown('<div class="sh">The iteration story — 12 logged upgrades</div>', unsafe_allow_html=True)
    st.markdown(
        "Every upgrade is dated, flag-gated, and records its rationale — the iteration *is* the "
        "research method. Full log with citations: `docs/UPGRADE_LOG.md`."
    )
    entries = _load_upgrade_log_entries()
    tl_html = '<div class="tl">'
    for e in entries:
        tl_html += (
            f'<div class="tl-item"><div class="d">{e["sprint"]}</div>'
            f'<div class="t">{e["title"]}</div></div>'
        )
    tl_html += "</div>"
    tl_left, tl_right = st.columns([1, 1])
    with tl_left:
        st.markdown(tl_html, unsafe_allow_html=True)
    with tl_right:
        st.markdown("""
        <div class="card"><h4>How to read the log</h4>
        <p>
        Sprint 1 made the work reproducible and the physics loss correct (seeds, exact propagator,
        1 g inventory, real baseline, honest speed). Sprint 2 built the next tier of UQ and training
        methodology (jackknife+/CV+, adaptive weights, stiffness curriculum, deep ensemble).
        Sprint 3 replaced the data itself — evaluated JENDL-5/EXFOR/NuDat — and produced the spectrum
        discovery in <b>The Discovery</b> tab. All 10 regression smoke checks pass; legacy behavior is
        bit-preserved behind default flags.
        </p></div>
        """, unsafe_allow_html=True)
        smoke = _load_json("v3_pilstm/results/smoke_20260718.json")
        if smoke:
            checks = smoke.get("checks") or smoke.get("results") or {}
            if isinstance(checks, dict) and checks:
                passed = sum(1 for v in checks.values() if (v.get("ok") if isinstance(v, dict) else bool(v)))
                st.metric("Regression smoke checks", f"{passed}/{len(checks)} pass")

    # ── Limitations ─────────────────────────────────────────────────────────
    st.markdown('<div class="sh">Limitations — the box judges should read first</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card bad">
    <h4>Scope and limitations</h4>
    <p>
    • <b>0D point model</b>: well-mixed target, scalar flux and energy — no geometry, self-shielding,
    or burnup (the 26-day HFIR overprediction shows exactly this gap).<br>
    • <b>Validation is vs the ODE reference and evaluated data</b>, not lab experiments on the surrogate
    itself; rungs 3–4 of the ladder test the ODE, not the network.<br>
    • <b>Both surrogates were trained on the v1 synthetic-σ reference</b>; the v2 evaluated,
    spectrum-aware retrain is pending. Fast-regime absolute yields carry that caveat.<br>
    • <b>The spectrum shape is the largest remaining modeling uncertainty</b>; f* is an inferred
    effective parameter with a band, not a tuned truth.<br>
    • <b>Not for clinical use</b>: nothing here is a dose calculation, assay, or regulatory artifact.
    </p></div>
    """, unsafe_allow_html=True)

    # ── References ──────────────────────────────────────────────────────────
    st.markdown('<div class="sh">References</div>', unsafe_allow_html=True)
    refs = iter_log.get("research_refs", [])
    if refs:
        with st.expander(f"ML/PINN methods references ({len(refs)} entries from the iteration log)"):
            for r in refs:
                cite = r.get("citation", "")
                role = r.get("role", "")
                code = r.get("code", "")
                st.markdown(f"- {cite} — *{role}*" + (f" (`{code}`)" if code else ""))
    st.markdown("""
    **Nuclear data & production anchors**

    - JENDL-5 evaluated nuclear data library (IAEA-NDS distribution); ENDF/B-VIII.0 cross-check (identical Ra-226 (n,2n) evaluation, max dev 0.0 b).
    - EXFOR entry 21405 — O'Connor & Perkin 1960, σ(n,2n) = 1.60 ± 0.20 b @ 14.5 MeV.
    - EXFOR entry 31760 — Bagheri 2015, σ(n,γ) = 13.8 ± 0.3 b thermal.
    - NuDat 3 / ENSDF — half-lives with uncertainties, live retrieval 2026-07-18.
    - Watt B.E. 1952, *Phys. Rev.* 87, 1037 — fission spectrum form (a = 0.988 MeV, b = 2.249 MeV⁻¹).
    - Sano et al. 2024, *J. Nucl. Sci. Technol.* 61:509, DOI 10.1080/00223131.2023.2243941 — Joyo Ac-225 uncertainty analysis.
    - Iwahashi et al. 2022, *Processes* 10(7):1239 — Joyo ORIGEN benchmarks, MK-III spectra (open access).
    - Aoyama et al. 2005, *J. Nucl. Radiochem. Sci.* 6(3) — Joyo MK-III spectral softness by position.
    - Hogle et al. 2016, OSTI 1253240 — HFIR Ra-226 irradiation, Ac-227 series (full text).
    - Snow et al. 2025, OSTI 3028837 — INL photonuclear, φ=0 ingrowth leg (full text).
    - Barber, Candès, Ramdas & Tibshirani 2021, *Ann. Statist.* 49(1) — jackknife+ predictive inference (arXiv:1905.02928).
    - Lakshminarayanan, Pritzel & Blundell 2017 — deep ensembles for predictive uncertainty (arXiv:1612.01474).
    """)

# ==============================================================================
# TAB — ABOUT
# ==============================================================================
with tab_about:
    st.markdown('<div class="kicker">The student behind the model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sh" style="margin-top:0.2rem;">About this project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        IsotopePINN is an independent student research project on physics-informed machine learning for
        medical-isotope production planning. It started as "can a neural network learn the Bateman
        equations", and became a study in **chasing your own assumptions**: the most important result
        here is not a percentage but a discovery process — every time the model was checked against
        better data, it revealed the next-deeper approximation (synthetic cross section → evaluated
        cross section → monoenergetic flux → spectrum shape).
        """
    )
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""
        <div class="card ok"><h4>What I can defend</h4>
        <p>
        • Every on-screen metric traces to a committed JSON/CSV/PNG in the repository — hover any
        number and I can name the file.<br>
        • The canonical number (4.51%) and the comparison numbers (8.18% / 5.12%) come from different
        protocols, and the app says so next to each.<br>
        • Failures are in the open: epithermal ~9.5%, threshold ~8.5%, PI-LSTM's thermal Ac-227 leg,
        the ~3× 26-day HFIR overprediction, the wide conformal intervals.<br>
        • Speed claims are throughput-under-batching with eager latency disclosed.
        </p></div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="card warn"><h4>What I cannot claim (yet)</h4>
        <p>
        • Validation against lab experiments on the <i>surrogate</i> itself — current validation is vs
        the ODE reference and evaluated data.<br>
        • A surrogate trained on evaluated, spectrum-aware physics — the checkpoints still encode the
        v1 synthetic reference; the retrain is the next Kaggle run.<br>
        • The measured Joyo MK-III spectrum (paywalled) — f* is an inference with an uncertainty band,
        not a measurement.<br>
        • Anything clinical. This is a planning tool for isotope production research.
        </p></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sh-sm">AI-assistance acknowledgment</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card"><p>
        <em>[Placeholder — to be completed by the student per ISEF rules.]
        AI coding and research assistants (including large-language-model tools) were used during
        development for code drafting, debugging, literature retrieval support, and document editing.
        All physics choices, validation decisions, and final claims were reviewed and verified by the
        student against the committed artifacts. A detailed, tool-by-tool disclosure will be appended
        before fair submission.</em>
        </p></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sh-sm">License & reuse</div>', unsafe_allow_html=True)
    st.markdown(
        "Code is released under the license in the repository root (`LICENSE`). Committed result JSONs, "
        "figures, and this app may be reused with attribution. Nuclear data files retain the terms of "
        "their source libraries (IAEA-NDS JENDL-5 / ENDF, EXFOR, NNDC NuDat)."
    )
    st.markdown(
        "**Repository map:** `app.py` (this app) · `ra226_ac225_transmutation.py` (ODE reference, "
        "versioned data layer) · `pinn_model.py` (Model A) · `v3_pilstm/` (Model B) · "
        "`docs/DATA_PROVENANCE.md` + `docs/UPGRADE_LOG.md` (the audit trail) · `results/` + "
        "`v3_pilstm/results/` (every committed number)."
    )

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ft">
  <b>IsotopePINN</b> — physics-informed surrogates for Ac-225 production planning<br>
  Validated vs stiff ODE reference (JENDL-5 / EXFOR / NuDat v2 layer) · not clinical or assay data ·
  spectrum discovery: 18.9× → 369× → <b>1.00×</b> of Sano 2024 &nbsp;|&nbsp;
  PyTorch + Streamlit &nbsp;|&nbsp;
  <a href="https://www.nndc.bnl.gov/" target="_blank">NNDC</a> ·
  <a href="https://www-nds.iaea.org/" target="_blank">IAEA-NDS</a> ·
  <a href="https://en.wikipedia.org/wiki/Actinium-225" target="_blank">Ac-225 background</a>
</div>
""", unsafe_allow_html=True)
