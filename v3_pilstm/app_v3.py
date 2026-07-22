"""
PI-LSTM Results-6 Streamlit demo (flagship Ac-225 surrogate).

Run from project root:
    streamlit run v3_pilstm/app_v3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v3_pilstm"))

from pinn_model import (  # noqa: E402
    DEFAULT_N226_SCALE,
    DEFAULT_N225_SCALE,
    DEFAULT_NAC_SCALE,
    DEFAULT_N227_SCALE,
    DEFAULT_NAC227_SCALE,
    DEFAULT_PHI_SCALE,
    DEFAULT_T_REF_H,
    neutron_energy_ev_to_feature_numpy,
)
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation  # noqa: E402
from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402

SCALES = np.array(
    [
        DEFAULT_N226_SCALE,
        DEFAULT_N225_SCALE,
        DEFAULT_NAC_SCALE,
        DEFAULT_N227_SCALE,
        DEFAULT_NAC227_SCALE,
    ],
    dtype=np.float64,
)
WEIGHTS = ROOT / "v3_pilstm" / "weights" / "pi_lstm_best.pth"
COMPARE_JSON = ROOT / "v3_pilstm" / "results" / "compare_v2_pilstm.json"
SPEED_JSON = ROOT / "v3_pilstm" / "results" / "speed_harness.json"
CONFORMAL_JSON = ROOT / "v3_pilstm" / "results" / "conformal_validation.json"
BOARD = ROOT / "v3_pilstm" / "results" / "ISEF_BOARD_PACK.md"
SPECIES = ["Ra-226", "Ra-225", "Ac-225", "Ra-227", "Ac-227"]
GRAPH_DIR = ROOT / "graphs"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_resource(show_spinner="Loading PI-LSTM Results-6…")
def load_model():
    if not WEIGHTS.exists():
        return None
    model = PhysicsInformedLSTM.load(WEIGHTS, map_location="cpu")
    model.eval()
    return model


def build_features(phi, energy_ev, t_end_h, ic, n_steps=64):
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    t_h, y = run_simulation(
        env,
        t_end_h=t_end_h,
        n_points=n_steps,
        N_ra0=ic[0],
        N_ra225_0=ic[1],
        N_ac0=ic[2],
        N_ra227_0=ic[3],
        N_ac227_0=ic[4],
    )
    t_norm = t_h / DEFAULT_T_REF_H
    e_feat = float(neutron_energy_ev_to_feature_numpy(energy_ev))
    ic_norm = ic / SCALES
    feats = np.zeros((1, n_steps, 8), dtype=np.float32)
    for k in range(n_steps):
        feats[0, k, 0] = t_norm[k]
        feats[0, k, 1] = phi / DEFAULT_PHI_SCALE
        feats[0, k, 2] = e_feat
        feats[0, k, 3:8] = ic_norm
    return feats, t_h, y


st.set_page_config(
    page_title="PI-LSTM Results-6 | Ac-225",
    page_icon="⚛️",
    layout="wide",
)

st.title("PI-LSTM Results-6 — Ac-225 production surrogate")
st.caption(
    "Physics-informed LSTM with exact/`expmix` loss · mentored review with Jaden Palmer (NCSU ARTISANS) · "
    "errors vs stiff ODE teacher, not reactor assay"
)

compare = _load_json(COMPARE_JSON)
speed = _load_json(SPEED_JSON)
conformal = _load_json(CONFORMAL_JSON)
species_err = compare.get("species_median_rel_error", {})
ac_v2 = 100.0 * float(species_err.get("Ac-225", {}).get("v2", 0.0818))
ac_pi = 100.0 * float(species_err.get("Ac-225", {}).get("pilstm", 0.0512))
batched_ms = float(speed.get("batched", {}).get("ms_per_scenario", 1.65))
cov = conformal.get("ac225_relative_coverage") or conformal.get("coverage_ac225_rel")
if cov is None and isinstance(conformal.get("test"), dict):
    cov = conformal["test"].get("relative_coverage")
if cov is None and isinstance(conformal.get("Ac-225"), dict):
    cov = conformal["Ac-225"].get("relative", {}).get("test_coverage")
cov_pct = 100.0 * float(cov) if cov is not None else 90.9

m1, m2, m3, m4 = st.columns(4)
m1.metric("PI-LSTM Ac-225 endpoint", f"{ac_pi:.2f}%", "vs ODE (22 held-out)")
m2.metric("Frozen v2 (same protocol)", f"{ac_v2:.2f}%")
m3.metric("Batched inference", f"{batched_ms:.2f} ms/sc")
m4.metric("Conformal coverage", f"{cov_pct:.1f}%", "target ~90%")

st.info(
    "Plain English: this model learns the isotope chain from a trusted physics simulator, "
    "then answers “what if we change flux / time / energy?” in milliseconds — without inventing mass. "
    "It is a planning tool, not a clinical dose calculator."
)

tab_live, tab_bench, tab_figs = st.tabs(["Live trajectory", "Benchmark tables", "Figures"])

with tab_live:
    if not WEIGHTS.exists():
        st.warning("Missing `v3_pilstm/weights/pi_lstm_best.pth`.")
    c1, c2, c3 = st.columns(3)
    with c1:
        phi = st.number_input("Flux φ (n/cm²/s)", value=1e14, format="%.3e")
    with c2:
        energy_ev = st.number_input("Neutron energy (eV)", value=14e6, format="%.3e")
    with c3:
        t_end = st.slider("Irradiation time (h)", 10.0, 500.0, 250.0)
    n226 = st.number_input("Initial Ra-226 (atoms)", value=1e22, format="%.3e")

    if st.button("Run PI-LSTM vs ODE", type="primary"):
        model = load_model()
        if model is None:
            st.error("Could not load PI-LSTM weights.")
        else:
            ic = np.array([n226, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            feats, t_h, y_ode = build_features(phi, energy_ev, t_end, ic)
            with torch.no_grad():
                pred = model(torch.from_numpy(feats)).numpy()[0] * SCALES

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            ax = axes[0]
            for i, name in enumerate(SPECIES):
                ax.plot(t_h, y_ode[:, i], "--", alpha=0.7, label=f"ODE {name}")
                ax.plot(t_h, pred[:, i], "-", label=f"PI-LSTM {name}")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Atoms")
            ax.set_yscale("log")
            ax.set_title("Full chain")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            ax2.plot(t_h, y_ode[:, 2], "--", label="ODE Ac-225", color="#64748b")
            ax2.plot(t_h, pred[:, 2], "-", label="PI-LSTM Ac-225", color="#0ea5e9")
            ax2.set_xlabel("Time (h)")
            ax2.set_ylabel("Ac-225 atoms")
            ax2.set_title("Ac-225 focus")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig)

            end_rel = abs(pred[-1, 2] - y_ode[-1, 2]) / max(abs(y_ode[-1, 2]), 1.0)
            st.metric("This scenario · Ac-225 endpoint |rel err| vs ODE", f"{100.0 * end_rel:.2f}%")

with tab_bench:
    if species_err:
        rows = []
        for sp, vals in species_err.items():
            rows.append(
                {
                    "Species": sp,
                    "v2 median rel %": f"{100.0 * float(vals.get('v2', float('nan'))):.2f}",
                    "PI-LSTM median rel %": f"{100.0 * float(vals.get('pilstm', float('nan'))):.2f}",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption("Source: `v3_pilstm/results/compare_v2_pilstm.json` (22 held-out, seed 2024).")
    if speed:
        st.subheader("Speed harness (CPU)")
        st.json(
            {
                "eager_ms_per_scenario": speed.get("eager", {}).get("ms_per_scenario"),
                "batched_ms_per_scenario": speed.get("batched", {}).get("ms_per_scenario"),
                "speedup_vs_eager": speed.get("batched", {}).get("speedup_vs_eager"),
                "ac225_gate_pass": speed.get("batched", {}).get("gate_vs_eager_ac225"),
            }
        )
    if BOARD.is_file():
        with st.expander("ISEF board pack (markdown)"):
            st.markdown(BOARD.read_text(encoding="utf-8"))

with tab_figs:
    shown = 0
    for name, caption in [
        ("v3_v2_vs_pilstm_ac225.png", "v2 vs PI-LSTM Ac-225"),
        ("v3_species_median_errors.png", "Species median errors"),
        ("v3_trajectory_example.png", "Example trajectory"),
        ("v3_literature_anchors.png", "Literature anchors"),
        ("v3_joyo_sigma_calibration.png", "Joyo σ calibration (sensitivity)"),
    ]:
        p = GRAPH_DIR / name
        if p.is_file():
            st.image(str(p), caption=caption, use_container_width=True)
            shown += 1
    if shown == 0:
        st.info("No graphs found under `graphs/`.")

st.markdown(
    "Repo: [github.com/samogunnubi0-del/PINN2.0](https://github.com/samogunnubi0-del/PINN2.0) · "
    "Investigator: Sam Ogunnubi · Mentor: Jaden Palmer (NCSU)"
)
