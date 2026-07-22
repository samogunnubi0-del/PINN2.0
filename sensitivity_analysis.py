"""
ISEF flux sensitivity: PINN vs integrated ODE with matched phi, energy feature, and time.

Energy column 2 must use the same sqrt(E_ref/E) mapping as training (train.prepare_training_tensors).
"""
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import graph_provenance

# Physical scenario — must match training scalings (train.py)
from train import (
    N226_SCALE,
    N225_SCALE,
    N227_SCALE,
    NAC227_SCALE,
    NAC_SCALE,
    PHI_SCALE,
    TIME_SCALE_H,
    TRAIN_INIT_AC225,
    TRAIN_INIT_RA226,
    TRAIN_INIT_RA225,
    TRAIN_INIT_RA227,
    TRAIN_INIT_AC227,
)
from pinn_model import load_isotope_pinn_checkpoint, neutron_energy_ev_to_feature_torch
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

# Thermal baseline for this figure (state explicitly on the plot; training spans multiple E)
SENSITIVITY_NEUTRON_ENERGY_EV = 0.025


def run_sensitivity():
    print("Starting ISEF Sensitivity Analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(os.path.dirname(__file__), "weights", "pinn_best_weights.pth")
    if not os.path.exists(checkpoint_path):
        print("ERROR: 'pinn_best_weights.pth' not found in current directory.")
        return

    model, _ = load_isotope_pinn_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    print("Successfully loaded pre-trained weights.")

    t_span = torch.linspace(0, 1, 100, device=device).view(-1, 1)
    hours = t_span.detach().cpu().numpy().ravel() * TIME_SCALE_H

    e_ev = torch.tensor(SENSITIVITY_NEUTRON_ENERGY_EV, dtype=torch.float32, device=device)
    energy_feat = neutron_energy_ev_to_feature_torch(e_ev).view(1, 1)

    def create_inputs(time_tensor: torch.Tensor, flux_multiplier: float) -> torch.Tensor:
        batch = time_tensor.shape[0]
        inp = torch.zeros((batch, 8), device=device, dtype=torch.float32)
        inp[:, 0:1] = time_tensor
        inp[:, 1:2] = float(flux_multiplier)
        inp[:, 2:3] = energy_feat.expand(batch, 1)
        # Virgin full Ra-226 load (normalized like prepare_training_tensors)
        inp[:, 3:4] = TRAIN_INIT_RA226 / N226_SCALE
        inp[:, 4:5] = TRAIN_INIT_RA225 / N225_SCALE
        inp[:, 5:6] = TRAIN_INIT_AC225 / NAC_SCALE
        inp[:, 6:7] = TRAIN_INIT_RA227 / N227_SCALE
        inp[:, 7:8] = TRAIN_INIT_AC227 / NAC227_SCALE
        return inp

    flux_variations = [0.5, 0.75, 1.0, 1.25, 1.5]
    plt.figure(figsize=(13, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(flux_variations)))

    ode_deferred_rows: list[dict[str, float]] = []

    print("Running PINN + ODE for different flux levels...")
    for i, fmult in enumerate(flux_variations):
        phi_phys = fmult * PHI_SCALE
        env = IsotopeEnvironment(phi=float(phi_phys), neutron_energy_ev=float(SENSITIVITY_NEUTRON_ENERGY_EV))
        t_ode, Y = run_simulation(
            env,
            t_end_h=float(TIME_SCALE_H),
            n_points=100,
            N_ra0=float(TRAIN_INIT_RA226),
            N_ra225_0=float(TRAIN_INIT_RA225),
            N_ac0=float(TRAIN_INIT_AC225),
            N_ra227_0=float(TRAIN_INIT_RA227),
            N_ac227_0=float(TRAIN_INIT_AC227),
        )
        ac225_ode = Y[:, 2]

        with torch.no_grad():
            inp = create_inputs(t_span, fmult)
            preds = model(inp)
            ac225_pinn_atoms = (preds[:, 2] * NAC_SCALE).detach().cpu().numpy()

        rel = np.abs(ac225_pinn_atoms - ac225_ode) / np.maximum(ac225_ode, 1e-300)
        ode_deferred_rows.append({
            "flux_multiplier": fmult,
            "phi_phys": phi_phys,
            "max_rel_err": float(np.nanmax(rel)),
            "median_rel_err": float(np.nanmedian(rel)),
        })

        label_base = f"Flux {fmult:g}× ({phi_phys:.2e} n/cm²/s)"
        plt.plot(hours, ac225_ode, color=colors[i], linewidth=2.5, linestyle="-", label=f"ODE {label_base}")
        plt.plot(
            hours,
            ac225_pinn_atoms,
            color=colors[i],
            linewidth=2.0,
            linestyle="--",
            alpha=0.85,
            label=f"PINN {label_base}",
        )

    plt.xlabel("Time (hours)", fontsize=14)
    plt.ylabel(r"$^{225}$Ac atoms", fontsize=14)
    plt.title(
        "ISEF: Flux sensitivity — ODE (solid) vs PINN (dashed)\n"
        f"E_n = {SENSITIVITY_NEUTRON_ENERGY_EV:g} eV (thermal); virgin Ra-226 IC; "
        f"φ baseline = {PHI_SCALE:.0e} n/cm²/s × multiplier",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.6)

    root = pathlib.Path(__file__).resolve().parent
    summ_path = root / "results" / "isef_sensitivity_ode_deferred.csv"
    summ_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ode_deferred_rows).to_csv(summ_path, index=False)
    print(f"Wrote relative-error summary for ODE comparison: {summ_path}")

    plt.tight_layout()
    out_path = root / "graphs" / "isef_sensitivity_flux.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Success! Chart saved to '{out_path.relative_to(root)}'.")
    graph_provenance.record_graph_write(
        root,
        out_path,
        producer="sensitivity_analysis.py",
        run_id=graph_provenance.new_run_id(),
        extra={"neutron_energy_ev": SENSITIVITY_NEUTRON_ENERGY_EV, "ode_deferred_csv": str(summ_path)},
    )


if __name__ == "__main__":
    run_sensitivity()
