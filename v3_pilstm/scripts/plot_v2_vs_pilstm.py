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
    w = 0.35
    v2_vals = [_pct(med.get(s, {}).get("v2")) for s in SPECIES]
    pi_vals = [_pct(med.get(s, {}).get("pilstm")) for s in SPECIES]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, v2_vals, w, label="v2 MLP-PINN", color=COLORS["v2"])
    ax.bar(x + w / 2, pi_vals, w, label="PI-LSTM v3", color=COLORS["pilstm"])
    ax.axhline(10.0, color="gray", ls="--", lw=1, alpha=0.7, label="10% gate")
    ax.set_xticks(x)
    ax.set_xticklabels(SPECIES, rotation=20, ha="right")
    ax.set_ylabel("Median relative error vs ODE (%)")
    ax.set_title("Held-out scenarios: v2 vs PI-LSTM (endpoint)")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ac225_focus(compare: dict, out: Path) -> None:
    med = compare.get("species_median_rel_error", {}).get("Ac-225", {})
    traj = compare.get("pilstm_ac225_full_traj_median_rel")
    labels = ["v2 endpoint", "PI-LSTM endpoint"]
    vals = [_pct(med.get("v2")), _pct(med.get("pilstm"))]
    if traj is not None:
        labels.append("PI-LSTM full traj")
        vals.append(_pct(traj))

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=[COLORS["v2"], COLORS["pilstm"], COLORS["pilstm"]][: len(vals)])
    bars[-1].set_alpha(0.65 if len(vals) > 2 else 1.0)
    ax.axhline(10.0, color="gray", ls="--", lw=1, alpha=0.7, label="10% gate")
    ax.set_ylabel("Ac-225 median rel error vs ODE (%)")
    ax.set_title("Ac-225 accuracy (22 held-out scenarios)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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

    labels = []
    lit = []
    ode = []
    v2 = []
    pi = []
    for r in rows[:6]:
        cite = (r.get("source_citation") or "?")[:28]
        labels.append(cite + "…")
        lit.append(r["reference_A_Ac225_Bq"] / 1e9)
        ode.append((r.get("ode_A_Ac225_Bq") or 0) / 1e9)
        v2.append((r.get("v2_A_Ac225_Bq") or 0) / 1e9)
        pi.append((r.get("pilstm_A_Ac225_Bq") or 0) / 1e9)

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, lit, w, label="Literature", color=COLORS["lit"])
    ax.bar(x - 0.5 * w, ode, w, label="ODE", color=COLORS["ode"])
    ax.bar(x + 0.5 * w, v2, w, label="v2", color=COLORS["v2"])
    ax.bar(x + 1.5 * w, pi, w, label="PI-LSTM", color=COLORS["pilstm"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Ac-225 activity (GBq)")
    ax.set_title("Literature anchors vs model predictions (Ac-225)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_trajectory_example(out: Path) -> None:
    """One canonical held-out scenario: Ac-225 trajectory ODE vs PI-LSTM."""
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

    sc = canonical_heldout_scenarios(1, seed=2024)[0]
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_h, ode_ac, "o-", label="Radau5 ODE", color=COLORS["ode"], ms=3)
    if pi_ac is not None:
        ax.plot(t_h, pi_ac, "s-", label="PI-LSTM v3", color=COLORS["pilstm"], ms=3, alpha=0.85)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("N(Ac-225) atoms")
    ax.set_title("Example held-out trajectory (Ac-225 ingrowth)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w, lit, w, label="Joyo sim (lit)", color=COLORS["lit"])
    ax.bar(x, default, w, label="Default ODE", color=COLORS["ode"])
    ax.bar(x + w, cal, w, label="Calibrated σ ODE", color="#0d9488")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Ac-225 activity (GBq)")
    ax.set_title(
        f"Joyo σ calibration (scale ×{joyo.get('recommended_scale', 0):.3f} → "
        f"{joyo.get('calibrated_sigma_n2n_mb', 0):.2f} mb)"
    )
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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
