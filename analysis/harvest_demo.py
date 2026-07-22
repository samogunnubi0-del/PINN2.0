"""
Harvest Timing Demo: Show how PINN can help decide when to harvest Ac-225.
Demonstrates practical application: "At what time is Ac-225 concentration maximized?"
"""
from __future__ import annotations

import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pinn_model import (
    IsotopePINN,
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
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _weights_candidates() -> tuple[pathlib.Path, ...]:
    return (
        ROOT / "weights" / "pinn_best_weights.pth",
        ROOT / "weights" / "pinn_trained_weights.pth",
        ROOT / "pinn_trained_weights.pth",
    )


WEIGHTS_PATH = next((p for p in _weights_candidates() if p.is_file()), _weights_candidates()[-1])
DEMO_DIR = pathlib.Path(__file__).resolve().parent / "demo_outputs"


def load_model(device: torch.device = torch.device("cpu")) -> IsotopePINN:
    if WEIGHTS_PATH.exists():
        try:
            model, _ = load_isotope_pinn_checkpoint(str(WEIGHTS_PATH), map_location=device)
            return model
        except Exception:
            print("Warning: failed to load weights via checkpoint loader.")
    model = IsotopePINN()
    model.to(device)
    model.eval()
    return model


def predict_trajectory(
    model: IsotopePINN,
    phi: float,
    energy_ev: float,
    time_range: tuple[float, float],
    n_points: int = 100,
    n226_0: float = 1e20,
    n225_0: float = 1e18,
    nac_0: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, [N226, N225, Nac])."""
    times = np.linspace(time_range[0], time_range[1], n_points)
    t_nn = times / float(DEFAULT_T_REF_H)
    phi_nn = float(phi) / float(DEFAULT_PHI_SCALE)
    energy = neutron_energy_ev_to_feature_numpy(
        np.full_like(times, float(energy_ev), dtype=float)
    )
    n0_226_nn = float(n226_0) / float(DEFAULT_N226_SCALE)
    n0_225_nn = float(n225_0) / float(DEFAULT_N225_SCALE)
    n0_ac_nn = float(nac_0) / float(DEFAULT_NAC_SCALE)
    n0_227_nn = 0.0 / float(DEFAULT_N227_SCALE)
    n0_ac227_nn = 0.0 / float(DEFAULT_NAC227_SCALE)

    inputs = np.vstack(
        [
            t_nn,
            np.full_like(times, phi_nn),
            energy,
            np.full_like(times, n0_226_nn),
            np.full_like(times, n0_225_nn),
            np.full_like(times, n0_ac_nn),
            np.full_like(times, n0_227_nn),
            np.full_like(times, n0_ac227_nn),
        ]
    ).T

    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_nn = model(x)
    pred = pred_nn.cpu().numpy()
    pred_atoms = np.vstack(
        [pred[:, 0] * DEFAULT_N226_SCALE, pred[:, 1] * DEFAULT_N225_SCALE, pred[:, 2] * DEFAULT_NAC_SCALE]
    ).T
    return times, pred_atoms


def run_harvest_demo():
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model = load_model(device)

    # Scenario: Recycler goal is to maximize Ac-225 yield.
    # Start with medium Ra-225 (from prior batch), run for optimal harvest window.
    print("=" * 60)
    print("HARVEST TIMING DEMO: When to extract Ac-225?")
    print("=" * 60)

    phi = 1e12  # Moderate flux
    energy = 0.025
    initial_ra225 = 1e19  # Moderate starting inventory
    initial_ra226 = 1e17  # Small Ra226
    initial_ac = 0.0

    # Window must extend past the Ac-225 half-life (9.9 d) and Ra-225 half-life
    # (14.9 d); a 200 h window ends before the ingrowth peak can resolve.
    times, pred = predict_trajectory(
        model,
        phi,
        energy,
        (0, 500),
        n_points=150,
        n226_0=initial_ra226,
        n225_0=initial_ra225,
        nac_0=initial_ac,
        device=device,
    )

    # Find peak Ac-225
    ac_pred = pred[:, 2]
    peak_idx = np.argmax(ac_pred)
    peak_time = times[peak_idx]
    peak_ac = ac_pred[peak_idx]

    print(f"\nInitial conditions:")
    print(f"  Ra-226: {initial_ra226:.2e} atoms")
    print(f"  Ra-225: {initial_ra225:.2e} atoms")
    print(f"  Flux: {phi:.2e} n/cm²/s")
    print(f"\nPredicted Ac-225 Peak:")
    print(f"  Time: {peak_time:.1f} hours ({peak_time/24:.1f} days)")
    print(f"  Amount: {peak_ac:.2e} atoms")
    if 0 < peak_idx < len(times) - 1:
        rate = (ac_pred[peak_idx+1] - ac_pred[peak_idx-1]) / (times[peak_idx+1] - times[peak_idx-1])
        print(f"  Rate of change near peak: {rate:.2e} atoms/hour")
    else:
        print(f"  Rate of change: (peak at boundary; use caution in extrapolation)")

    # Plot trajectory with peak overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, pred[:, 0], "o-", label="Ra-226", markersize=3, alpha=0.7)
    ax.plot(times, pred[:, 1], "s-", label="Ra-225", markersize=3, alpha=0.7)
    ac_line = ax.plot(times, pred[:, 2], "^-", label="Ac-225", markersize=3, alpha=0.7, color="C2")[0]
    ax.axvline(peak_time, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"Peak Ac-225 @ t={peak_time:.0f}h")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Inventory (atoms)")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"Harvest Timing: Ac-225 Production Curve\n(Recycler scenario: Ra-225-mediated decay)")
    fig.tight_layout()
    fig.savefig(DEMO_DIR / "harvest_timing_curve.png", dpi=150)
    plt.close(fig)
    try:
        import graph_provenance
        graph_provenance.record_graph_write(
            ROOT,
            (DEMO_DIR / "harvest_timing_curve.png").resolve(),
            producer="harvest_demo.py",
            run_id=graph_provenance.new_run_id(),
        )
    except Exception:
        pass
    print(f"\n✓ Saved plot: {DEMO_DIR / 'harvest_timing_curve.png'}")

    # Multi-scenario comparison
    scenarios = [
        ("low_flux", 1e11, initial_ra225),
        ("medium_flux", 1e12, initial_ra225),
        ("high_flux", 1e13, initial_ra225),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["C0", "C1", "C2"]
    peak_times_list = []
    for (label, phi_val, ra225_val), color in zip(scenarios, colors):
        _, pred_traj = predict_trajectory(
            model, phi_val, energy, (0, 500), n_points=150,
            n226_0=initial_ra226, n225_0=ra225_val, nac_0=initial_ac, device=device
        )
        # Compute times for this trajectory
        times_scenario = np.linspace(0, 500, 150)
        ax.plot(times_scenario, pred_traj[:, 2], "-", label=f"{label} (φ={phi_val:.0e})", color=color, linewidth=2)
        peak_idx = np.argmax(pred_traj[:, 2])
        peak_times_list.append(times_scenario[peak_idx])
        ax.scatter(times_scenario[peak_idx], pred_traj[peak_idx, 2], s=100, color=color, zorder=5, marker="*")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Ac-225 (atoms)")
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Flux Impact on Ac-225 Peak Timing")
    fig.tight_layout()
    fig.savefig(DEMO_DIR / "harvest_flux_comparison.png", dpi=150)
    plt.close(fig)
    try:
        import graph_provenance
        graph_provenance.record_graph_write(
            ROOT,
            (DEMO_DIR / "harvest_flux_comparison.png").resolve(),
            producer="harvest_demo.py",
            run_id=graph_provenance.new_run_id(),
        )
    except Exception:
        pass
    print(f"✓ Saved plot: {DEMO_DIR / 'harvest_flux_comparison.png'}")

    # Write summary
    peak_at_boundary = peak_idx == 0 or peak_idx == len(times) - 1
    boundary_note = (
        "\n**Caveat:** the peak sits at the edge of the simulated window, so this is a "
        "lower/upper bound on the harvest window, not a resolved interior optimum.\n"
        if peak_at_boundary else ""
    )
    # Only claim a flux->peak-time trend if the computed peaks actually show one.
    flux_trend_line = "- **Flux sensitivity**: no consistent flux→peak-time trend resolved in this window\n"
    if len(peak_times_list) == len(scenarios) and all(
        later < earlier for earlier, later in zip(peak_times_list, peak_times_list[1:])
    ):
        flux_trend_line = "- **Flux sensitivity**: Higher flux means earlier peak (see flux comparison plot)\n"
    summary_lines = [
        "# Harvest Timing Demo\n\n",
        "## Objective\n",
        "Use PINN to predict optimal harvest time for Ac-225 in a recycler scenario.\n\n",
        "## Key Results\n",
        f"- **Optimal harvest time**: ~{peak_time:.0f} hours ({peak_time/24:.1f} days)\n",
        f"- **Peak Ac-225 yield**: {peak_ac:.2e} atoms\n",
        f"{flux_trend_line}",
        f"{boundary_note}",
        "\n## Application\n",
        "- Operators can use PINN to predict harvest windows **without solving ODEs** at query time\n",
        "- Enables real-time optimization of irradiation schedules\n",
        "- Fast inference (milliseconds) vs. minutes for numerical ODE solve\n\n",
    ]

    with open(DEMO_DIR / "harvest_summary.md", "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print(f"✓ Saved summary: {DEMO_DIR / 'harvest_summary.md'}")


if __name__ == "__main__":
    run_harvest_demo()
