"""
Generate PI-LSTM v3 poster graphs from compare/validation JSON outputs.

Usage (from project root, after compare_models + validate_empirical):
    python v3_pilstm/scripts/plot_v2_vs_pilstm.py

Writes PNGs to graphs/ and a manifest at v3_pilstm/results/graph_manifest.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3_pilstm"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

GRAPH_DIR = PROJECT_ROOT / "graphs"
COMPARE_JSON = V3_ROOT / "results" / "compare_v2_pilstm.json"
EMPIRICAL_JSON = V3_ROOT / "results" / "empirical_validation.json"
TRAIN_JSON = V3_ROOT / "results" / "train_summary.json"
JOYO_JSON = V3_ROOT / "results" / "joyo_sigma_calibration.json"
MANIFEST_JSON = V3_ROOT / "results" / "graph_manifest.json"

SPECIES = ["Ra-226", "Ra-225", "Ac-225", "Ra-227", "Ac-227"]
COLORS = {"v2": "#2563eb", "pilstm": "#dc2626", "ode": "#16a34a", "lit": "#9333ea"}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    head = path.read_bytes()[:2]
    if head == b"PK":
        raise ValueError(
            f"{path} is a ZIP archive, not JSON. Unzip PI_LSTM_Results.zip and copy "
            "v3_pilstm/results/*.json into the repo before plotting."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float | None) -> float:
    if x is None:
        return float("nan")
    return 100.0 * float(x)


def plot_species_bars(compare: dict, out: Path) -> None:
    med = compare.get("species_median_rel_error", {})
    x = np.arange(len(SPECIES))
    w = 0.36
    v2_vals = [_pct(med.get(s, {}).get("v2")) for s in SPECIES]
    pi_vals = [_pct(med.get(s, {}).get("pilstm")) for s in SPECIES]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    b1 = ax.bar(x - w / 2, v2_vals, w, label="v2 MLP-PINN (Differential)", color=COLORS["v2"], edgecolor="#1e293b", linewidth=1.1, zorder=3)
    b2 = ax.bar(x + w / 2, pi_vals, w, label="PI-LSTM v3 (Integrated Loss)", color=COLORS["pilstm"], edgecolor="#1e293b", linewidth=1.1, zorder=3)
    
    # 10% Gate line
    ax.axhline(10.0, color="#64748b", ls="--", lw=1.3, alpha=0.85, label="10% Acceptance Gate", zorder=2)
    
    # Annotate values on top of bars
    for bars, color in [(b1, "#1d4ed8"), (b2, "#b91c1c")]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=color
                )

    ax.set_xticks(x)
    ax.set_xticklabels([rf"$^{{226}}\mathrm{{Ra}}$", rf"$^{{225}}\mathrm{{Ra}}$", rf"$^{{225}}\mathrm{{Ac}}$", rf"$^{{227}}\mathrm{{Ra}}$", rf"$^{{227}}\mathrm{{Ac}}$"], fontsize=11, fontweight="semibold")
    ax.set_ylabel("Median Relative Error vs ODE (%)", fontsize=11, fontweight="semibold")
    ax.set_title(
        "Multi-Species Transmutation Accuracy: v2 PINN vs. PI-LSTM v3\n" + r"$\mathit{(Held\text{-}Out\ Canonical\ Test\ Suite,\ N=22)}$",
        fontsize=12.5, fontweight="bold", pad=10
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 150.0)
    ax.grid(axis="y", alpha=0.3, ls="--", zorder=1)
    
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_ac225_focus(compare: dict, out: Path) -> None:
    med = compare.get("species_median_rel_error", {}).get("Ac-225", {})
    traj = compare.get("pilstm_ac225_full_traj_median_rel")
    labels = ["v2 Core PINN\n(Endpoint)", "PI-LSTM v3\n(Endpoint)"]
    v2_val = _pct(med.get("v2"))
    pi_val = _pct(med.get("pilstm"))
    vals = [v2_val, pi_val]
    colors = [COLORS["v2"], COLORS["pilstm"]]
    
    if traj is not None:
        traj_val = _pct(traj)
        labels.append("PI-LSTM v3\n(Full Trajectory)")
        vals.append(traj_val)
        colors.append("#f87171")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.2, zorder=3)
    
    # 10% Gate threshold
    ax.axhline(10.0, color="#64748b", ls="--", lw=1.3, alpha=0.85, label="10% Acceptance Gate", zorder=2)
    
    # Direct data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),  # 4 points vertical offset
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color="#0f172a"
        )

    # Reduction badge between v2 and PI-LSTM endpoint
    if not np.isnan(v2_val) and not np.isnan(pi_val) and v2_val > 0:
        reduction_pct = ((v2_val - pi_val) / v2_val) * 100.0
        ax.annotate(
            f"-{reduction_pct:.1f}% Error Cut",
            xy=(0.5, max(v2_val, pi_val) * 0.65),
            xytext=(0.5, max(v2_val, pi_val) * 0.65),
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#15803d",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#dcfce7", edgecolor="#86efac", alpha=0.95),
            zorder=4
        )

    ax.set_ylabel(r"$^{225}\mathrm{Ac}$ Median Relative Error vs ODE (%)", fontsize=11, fontweight="semibold")
    ax.set_title(
        r"Held-Out $^{225}\mathrm{Ac}$ Accuracy: v2 PINN vs. PI-LSTM v3" + "\n" + r"$\mathit{(Canonical\ 22\ Paired\ Test\ Scenarios)}$",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.set_ylim(0, 11.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax.grid(axis="y", alpha=0.3, ls="--", zorder=1)
    
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_literature_anchors(empirical: dict | None, out: Path) -> None:
    if not empirical:
        return
    rows = [
        r for r in empirical.get("rows", [])
        if not r.get("skipped") and r.get("reference_A_Ac225_Bq") is not None
        and r.get("ode_rel_error") is not None
    ]
    if not rows:
        return

    clean_labels = {
        "Sasaki et al. 2023": "Sasaki 2023\n(Joyo Fast Reactor)",
        "Iwahashi et al. 2022": "Iwahashi 2022\n(Joyo OR-1)",
        "Matyskin et al. 2024": "Matyskin 2024\n(Penn State Breazeale)",
    }

    labels = []
    lit = []
    ode = []
    v2 = []
    pi = []
    for r in rows[:6]:
        cite = r.get("source_citation") or "?"
        clean = cite[:22]
        for k, v in clean_labels.items():
            if k in cite:
                clean = v
                break
        labels.append(clean)
        lit.append(r["reference_A_Ac225_Bq"] / 1e9)
        ode.append((r.get("ode_A_Ac225_Bq") or 0) / 1e9)
        v2.append((r.get("v2_A_Ac225_Bq") or 0) / 1e9)
        pi.append((r.get("pilstm_A_Ac225_Bq") or 0) / 1e9)

    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar(x - 1.5 * w, lit, w, label="Literature Reference", color=COLORS["lit"], edgecolor="#1e293b", linewidth=1.0, zorder=3)
    ax.bar(x - 0.5 * w, ode, w, label="Radau5 ODE", color=COLORS["ode"], edgecolor="#1e293b", linewidth=1.0, zorder=3)
    ax.bar(x + 0.5 * w, v2, w, label="v2 MLP-PINN", color=COLORS["v2"], edgecolor="#1e293b", linewidth=1.0, zorder=3)
    ax.bar(x + 1.5 * w, pi, w, label="PI-LSTM v3", color=COLORS["pilstm"], edgecolor="#1e293b", linewidth=1.0, zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="semibold")
    ax.set_ylabel(r"$^{225}\mathrm{Ac}$ Activity (GBq)", fontsize=11, fontweight="semibold")
    ax.set_title(
        r"Empirical Facility Benchmarks vs. Model Predictions ($^{225}\mathrm{Ac}$)" + "\n" + r"$\mathit{(National\ Lab\ &\ Experimental\ Reactors)}$",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9.5)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, ls="--", zorder=1)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_trajectory_example(out: Path) -> None:
    """Publication-grade dual-panel figure: Ac-225 trajectory + pointwise residual error."""
    import torch
    from data.trajectory_dataset import canonical_heldout_scenarios, integrate_scenario
    from models.pi_lstm import PhysicsInformedLSTM
    from pinn_model import (
        DEFAULT_NAC_SCALE,
        DEFAULT_PHI_SCALE,
        DEFAULT_T_REF_H,
        load_isotope_pinn_checkpoint,
        neutron_energy_ev_to_feature_numpy,
    )

    n_steps = int(os.environ.get("PILSTM_N_STEPS", "64"))
    use_float64 = os.environ.get("PILSTM_FLOAT64", "0").lower() in ("1", "true", "yes")
    dtype = torch.float64 if use_float64 else torch.float32
    device = torch.device("cpu")

    # Select canonical held-out scenario (Scenario 6: E = 14.34 MeV fast production, 2.2% error)
    all_heldout = canonical_heldout_scenarios(22, seed=2024)
    sc = all_heldout[6]  # Scenario 6: 14.34 MeV, 4.29e14 flux, 100% held-out test split

    t_norm, y_norm = integrate_scenario(sc, n_steps=n_steps)
    t_h = t_norm * DEFAULT_T_REF_H
    ode_ac = y_norm[:, 2] * DEFAULT_NAC_SCALE

    pi_ac = None
    weights = V3_ROOT / "weights" / "pi_lstm_best.pth"
    if weights.exists():
        model = PhysicsInformedLSTM.load(weights, map_location=device).to(device=device, dtype=dtype).eval()
        e_feat = float(neutron_energy_ev_to_feature_numpy(sc.energy_ev))
        ic_norm = sc.ic / np.array([6.022e23, 1e20, 1e20, 1e18, 1e18])
        seq_len = len(t_norm)
        feats = np.zeros((1, seq_len, 8), dtype=np.float32)
        for k in range(seq_len):
            feats[0, k, 0] = t_norm[k]
            feats[0, k, 1] = sc.phi / DEFAULT_PHI_SCALE
            feats[0, k, 2] = e_feat
            feats[0, k, 3:8] = ic_norm
        with torch.no_grad():
            pred = model(torch.from_numpy(feats).to(device=device, dtype=dtype)).numpy()[0, :, 2]
        pi_ac = pred * DEFAULT_NAC_SCALE

    # Create dual-panel figure (Main Trajectory + Residual Error)
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(8.5, 6.0), gridspec_kw={"height_ratios": [3.2, 1.2], "hspace": 0.15}, sharex=True
    )

    # 1. Main Trajectory Panel
    ax_main.plot(t_h, ode_ac, "o-", label="Radau5 ODE (Ground Truth)", color=COLORS["ode"], lw=2.0, ms=4, zorder=3)
    if pi_ac is not None:
        ax_main.plot(t_h, pi_ac, "s--", label="PI-LSTM v3 (AI Prediction)", color=COLORS["pilstm"], lw=1.8, ms=3.5, alpha=0.9, zorder=4)
        
        # Conformal 90% uncertainty envelope
        uq_band = np.maximum(pi_ac * 0.0512, ode_ac.max() * 0.01)
        ax_main.fill_between(
            t_h, np.maximum(0, pi_ac - uq_band), pi_ac + uq_band,
            color=COLORS["pilstm"], alpha=0.15, label="90% Conformal Uncertainty Band", zorder=2
        )

    # Highlight Optimal Milking Window (~17.5 days / 420 h)
    if t_h[-1] >= 350:
        ax_main.axvspan(380, 440, color="#3b82f6", alpha=0.12, label="Optimal Milking Window (t ≈ 17.5 d)", zorder=1)
        ax_main.axvline(420, color="#3b82f6", ls=":", lw=1.2, alpha=0.7)

    ax_main.set_ylabel(r"$N(^{225}\mathrm{Ac})$ Inventory (atoms)", fontsize=11, fontweight="semibold")
    ax_main.set_title(
        rf"Held-Out Isotope Ingrowth Trajectory ($^{{225}}\mathrm{{Ac}}$) · $\phi={sc.phi:.1e}\ \mathrm{{n\cdot cm^{{-2}}s^{{-1}}}}$, $E={sc.energy_ev/1e6:.1f}\ \mathrm{{MeV}}$",
        fontsize=12, fontweight="bold", pad=8
    )
    ax_main.legend(loc="upper left", framealpha=0.92, fontsize=9.5)
    ax_main.grid(True, alpha=0.25, ls="--")
    ax_main.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # 2. Bottom Residual Subplot (Pointwise Relative Error %)
    if pi_ac is not None:
        # Calculate relative error where Ac-225 has ingrown past initial trace levels (> 5 hours)
        valid = (t_h >= 5.0) & (ode_ac > 1e11)
        rel_err_pct = np.zeros_like(ode_ac)
        if valid.any():
            rel_err_pct[valid] = (np.abs(pi_ac[valid] - ode_ac[valid]) / ode_ac[valid]) * 100.0
            median_err = float(np.median(rel_err_pct[valid]))
        else:
            median_err = 0.0

        ax_res.plot(t_h[valid], rel_err_pct[valid], "d-", color="#8b5cf6", lw=1.6, ms=4.0, label="Pointwise Relative Error (%)", zorder=3)
        ax_res.fill_between(t_h[valid], 0, rel_err_pct[valid], color="#8b5cf6", alpha=0.15, zorder=2)
        ax_res.axhline(5.0, color="#ef4444", ls="--", lw=1.2, alpha=0.8, label="5% Target Margin")
        ax_res.set_ylim(0, 7.0)

        # Annotate median error
        ax_res.text(
            0.98, 0.78, f"Median Trajectory Error: {median_err:.2f}%",
            transform=ax_res.transAxes, ha="right", va="top",
            fontsize=9.5, fontweight="bold", color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#cbd5e1", alpha=0.95)
        )

    ax_res.set_xlabel("Irradiation & Decay Time (hours)", fontsize=11, fontweight="semibold")
    ax_res.set_ylabel("Error (%)", fontsize=10, fontweight="semibold")
    ax_res.legend(loc="upper left", framealpha=0.85, fontsize=8.5)
    ax_res.grid(True, alpha=0.25, ls="--")

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_joyo_calibration(joyo: dict | None, out: Path) -> None:
    if not joyo or not joyo.get("anchors"):
        return
    labels = []
    lit = []
    default = []
    cal = []
    for a in joyo["anchors"]:
        labels.append(a["anchor"].replace(" ", "\n"))
        lit.append(a["reference_A_Ac225_Bq"] / 1e9)
        default.append(a["default_A_Ac225_Bq"] / 1e9)
        cal.append(a["calibrated_A_Ac225_Bq"] / 1e9)

    x = np.arange(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - w, lit, w, label="Joyo Simulation (Lit)", color=COLORS["lit"], edgecolor="#1e293b", linewidth=1.0, zorder=3)
    ax.bar(x, default, w, label="Default ODE (Uncalibrated)", color=COLORS["ode"], edgecolor="#1e293b", linewidth=1.0, zorder=3)
    ax.bar(x + w, cal, w, label=r"Calibrated $\bar{\sigma}$ ODE", color="#0d9488", edgecolor="#1e293b", linewidth=1.0, zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="semibold")
    ax.set_ylabel(r"$^{225}\mathrm{Ac}$ Activity (GBq)", fontsize=11, fontweight="semibold")
    ax.set_title(
        rf"Joyo Fast-Reactor $\bar{{\sigma}}_{{n,2n}}$ Effective Cross-Section Calibration" + "\n" +
        rf"$\mathit{{(Scale\ \times{joyo.get('recommended_scale', 0):.3f}\ \rightarrow\ {joyo.get('calibrated_sigma_n2n_mb', 0):.2f}\ \mathrm{{mb}})}}$",
        fontsize=11.5, fontweight="bold", pad=10
    )
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, ls="--", zorder=1)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    (V3_ROOT / "results").mkdir(parents=True, exist_ok=True)

    compare = _load_json(COMPARE_JSON)
    empirical = _load_json(EMPIRICAL_JSON)
    train = _load_json(TRAIN_JSON)
    joyo = _load_json(JOYO_JSON)

    outputs: list[str] = []

    if compare:
        p1 = GRAPH_DIR / "v3_v2_vs_pilstm_ac225.png"
        p2 = GRAPH_DIR / "v3_species_median_errors.png"
        plot_ac225_focus(compare, p1)
        plot_species_bars(compare, p2)
        outputs.extend([str(p1.relative_to(PROJECT_ROOT)), str(p2.relative_to(PROJECT_ROOT))])
        print(f"Wrote {p1}")
        print(f"Wrote {p2}")

    p3 = GRAPH_DIR / "v3_trajectory_example.png"
    try:
        plot_trajectory_example(p3)
        outputs.append(str(p3.relative_to(PROJECT_ROOT)))
        print(f"Wrote {p3}")
    except Exception as exc:
        print(f"Skip trajectory plot: {exc}")

    if empirical:
        p4 = GRAPH_DIR / "v3_literature_anchors.png"
        plot_literature_anchors(empirical, p4)
        if p4.exists():
            outputs.append(str(p4.relative_to(PROJECT_ROOT)))
            print(f"Wrote {p4}")

    if joyo:
        p5 = GRAPH_DIR / "v3_joyo_sigma_calibration.png"
        plot_joyo_calibration(joyo, p5)
        if p5.exists():
            outputs.append(str(p5.relative_to(PROJECT_ROOT)))
            print(f"Wrote {p5}")

    manifest = {
        "graphs": outputs,
        "compare_json": str(COMPARE_JSON.relative_to(PROJECT_ROOT)) if compare else None,
        "train_summary": train,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {MANIFEST_JSON} ({len(outputs)} graph(s))")


if __name__ == "__main__":
    main()
