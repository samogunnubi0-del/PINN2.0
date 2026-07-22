"""
Failure Case Analysis: Identify where PINN underperforms vs ODE.
Generates a markdown report and diagnostic plots.
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

try:
    import train as train_cfg
except Exception:  # pragma: no cover
    train_cfg = None

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
REPORT_PATH = pathlib.Path(__file__).resolve().parent / "FAILURE_CASE_ANALYSIS.md"
FIG_DIR = pathlib.Path(__file__).resolve().parent / "figs"


def load_model(device: torch.device = torch.device("cpu")) -> IsotopePINN:
    if WEIGHTS_PATH.exists():
        try:
            model, _ = load_isotope_pinn_checkpoint(str(WEIGHTS_PATH), map_location=device)
            return model
        except Exception:
            print("Warning: failed to load weights via checkpoint loader; using untrained model.")
    model = IsotopePINN()
    model.to(device)
    model.eval()
    return model


def evaluate_pinn(
    model: IsotopePINN,
    phi: float,
    energy_ev: float,
    times: np.ndarray,
    n226_0: float,
    n225_0: float,
    nac_0: float,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    t_nn = times / float(DEFAULT_T_REF_H)
    phi_nn = float(phi) / float(DEFAULT_PHI_SCALE)
    energy = neutron_energy_ev_to_feature_numpy(
        np.full_like(times, float(energy_ev), dtype=float)
    )
    n0_226_nn = float(n226_0) / float(DEFAULT_N226_SCALE)
    n0_225_nn = float(n225_0) / float(DEFAULT_N225_SCALE)
    n0_ac_nn = float(nac_0) / float(DEFAULT_NAC_SCALE)

    inputs = np.vstack(
        [
            t_nn,
            np.full_like(times, phi_nn),
            energy,
            np.full_like(times, n0_226_nn),
            np.full_like(times, n0_225_nn),
            np.full_like(times, n0_ac_nn),
            np.zeros_like(times),  # Ra-227 initial (zero)
            np.zeros_like(times),  # Ac-227 initial (zero)
        ]
    ).T

    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_nn = model(x)
    pred = pred_nn.cpu().numpy()
    pred_atoms = np.stack(
        [pred[:, 0] * DEFAULT_N226_SCALE, pred[:, 1] * DEFAULT_N225_SCALE, pred[:, 2] * DEFAULT_NAC_SCALE],
        axis=1,
    )
    return pred_atoms


def simulate_ode(phi: float, energy_ev: float, times: np.ndarray, n226_0: float, n225_0: float, nac_0: float) -> np.ndarray:
    t_max = float(np.max(times))
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    n_points = max(200, int(t_max * 5) + 10)
    t_h, Y = run_simulation(env, t_end_h=t_max, n_points=n_points, N_ra0=n226_0, N_ra225_0=n225_0, N_ac0=nac_0)
    times_arr = np.asarray(times, dtype=float)
    Y_interp = np.vstack([np.interp(times_arr, t_h, Y[:, i]) for i in range(3)]).T
    return Y_interp


def compute_errors(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAPE, and max error for each species."""
    errors = {}
    for i, species in enumerate(["Ra-226", "Ra-225", "Ac-225"]):
        p = pred[:, i]
        t = true[:, i]
        # Avoid division by zero
        t_safe = np.where(t > 0, t, 1.0)
        mape = np.mean(np.abs((p - t) / t_safe)) * 100
        rmse = np.sqrt(np.mean((p - t) ** 2))
        max_err = np.max(np.abs(p - t))
        errors[f"{species}_mape"] = mape
        errors[f"{species}_rmse"] = rmse
        errors[f"{species}_max_err"] = max_err
    return errors


def _single_supply_mode() -> bool:
    if train_cfg is None:
        return False
    return bool(getattr(train_cfg, "SINGLE_SUPPLY_MODE", False))


def run_failure_analysis():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    model = load_model(device)
    times = np.linspace(0.0, 300.0, 51)

    single = _single_supply_mode()
    if single:
        scenarios_in = [
            ("in_dist_ref_Ra226_supply", 1e14, 0.025, 6.022e23, 0.0, 0.0),
            ("in_dist_lower_flux", 1e13, 0.025, 6.022e23, 0.0, 0.0),
        ]
        scenarios_ood = [
            ("OOD_Ra225_only_decay", 0.0, 0.025, 0.0, 1e18, 0.0),
            ("OOD_mixed_IC_low_flux", 1e12, 0.04, 1e20, 1e18, 1e17),
        ]
        scenarios = scenarios_in + scenarios_ood
        in_names = {s[0] for s in scenarios_in}
    else:
        scenarios = [
            ("ra225_dom_pure_decay", 0.0, 0.025, 0.0, 1e18, 0.0),
            ("ra226_dom_normal", 1e14, 0.025, 6.022e23, 0.0, 0.0),
            ("mixed_low_flux", 1e12, 0.04, 1e20, 1e18, 1e17),
        ]
        in_names = set()

    results: dict = {}
    for scenario_name, phi, energy, n226_0, n225_0, nac_0 in scenarios:
        pred = evaluate_pinn(model, phi, energy, times, n226_0, n225_0, nac_0, device)
        true = simulate_ode(phi, energy, times, n226_0, n225_0, nac_0)
        errors = compute_errors(pred, true)
        results[scenario_name] = {"pred": pred, "true": true, "errors": errors}

    # Build report
    if single:
        report_lines = [
            "# PINN vs ODE diagnostic\n",
            "## Overview\n",
            "`train.py` has **SINGLE_SUPPLY_MODE**: training emphasizes **virgin Ra-226 fuel + CSV flux/energy**.\n",
            "Below, **in_dist_** scenarios match that story; **OOD_** scenarios are **not** trained for — expect worse MAPE there.\n",
            "Do **not** cite OOD MAPE as evidence for the main claim.\n\n",
            "## In-distribution scenarios\n",
        ]
        for scenario_name, data in results.items():
            if scenario_name not in in_names:
                continue
            report_lines.append(f"### {scenario_name}\n")
            errors = data["errors"]
            report_lines.append("| Species | MAPE (%) | RMSE | Max Error |\n")
            report_lines.append("|---------|----------|------|--------|\n")
            for spec in ["Ra-226", "Ra-225", "Ac-225"]:
                mape = errors.get(f"{spec}_mape", 0)
                rmse = errors.get(f"{spec}_rmse", 0)
                max_e = errors.get(f"{spec}_max_err", 0)
                report_lines.append(f"| {spec} | {mape:.2f}% | {rmse:.2e} | {max_e:.2e} |\n")
            report_lines.append("\n")
        report_lines.append("## Out-of-distribution probes (informational only)\n")
        for scenario_name, data in results.items():
            if scenario_name in in_names:
                continue
            report_lines.append(f"### {scenario_name}\n")
            errors = data["errors"]
            report_lines.append("| Species | MAPE (%) | RMSE | Max Error |\n")
            report_lines.append("|---------|----------|------|--------|\n")
            for spec in ["Ra-226", "Ra-225", "Ac-225"]:
                mape = errors.get(f"{spec}_mape", 0)
                rmse = errors.get(f"{spec}_rmse", 0)
                max_e = errors.get(f"{spec}_max_err", 0)
                report_lines.append(f"| {spec} | {mape:.2f}% | {rmse:.2e} | {max_e:.2e} |\n")
            report_lines.append("\n")
    else:
        report_lines = [
            "# PINN vs ODE diagnostic\n",
            "## Overview\n",
            "Mean absolute percentage error (MAPE) by species for three reference scenarios.\n",
            "Training **up-weights Ra-225 and Ac-225** in the data loss and fits the **same capped**\n",
            "forward pass used at inference, so numbers here should track deployable accuracy.\n\n",
            "## Scenario Results\n",
        ]
        for scenario_name, data in results.items():
            report_lines.append(f"### {scenario_name}\n")
            errors = data["errors"]
            report_lines.append("| Species | MAPE (%) | RMSE | Max Error |\n")
            report_lines.append("|---------|----------|------|--------|\n")
            for spec in ["Ra-226", "Ra-225", "Ac-225"]:
                mape = errors.get(f"{spec}_mape", 0)
                rmse = errors.get(f"{spec}_rmse", 0)
                max_e = errors.get(f"{spec}_max_err", 0)
                report_lines.append(f"| {spec} | {mape:.2f}% | {rmse:.2e} | {max_e:.2e} |\n")
            report_lines.append("\n")

    report_lines.extend([
        "## How to read this\n",
        "- **MAPE** can look large when the true inventory is tiny; check RMSE and the prediction plots.\n",
        "- For single-supply training, improve headline accuracy with more **CSV** coverage in the Ra-226 path, not OOD probes.\n\n",
        "## Summary\n",
        "- **No net alchemy**: hard budget cap + mass loss during training.\n",
        "- **Goal**: PINN tracks the ODE within tolerance on **in-distribution** scenarios you care about.\n",
        "- **If error is still too high**: run longer training, tune `DATA_WEIGHT` / `PHYSICS_WEIGHT` in `train.py`, or enrich data.\n",
    ])

    report_text = "".join(report_lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] Saved report: {REPORT_PATH}")

    # Plot errors per scenario
    n_sc = len(results)
    fig, axes = plt.subplots(1, n_sc, figsize=(4.5 * n_sc, 4))
    if n_sc == 1:
        axes = [axes]
    for (scenario_name, data), ax in zip(results.items(), axes):
        errors = data["errors"]
        species_names = ["Ra-226", "Ra-225", "Ac-225"]
        mape_vals = [errors[f"{s}_mape"] for s in species_names]
        ax.bar(species_names, mape_vals, color=["C0", "C1", "C2"])
        ax.set_ylabel("MAPE (%)")
        ax.set_title(scenario_name)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle("PINN Error by Scenario and Species")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "errors_by_scenario.png", dpi=150)
    plt.close(fig)
    try:
        import graph_provenance
        graph_provenance.record_graph_write(
            ROOT,
            (FIG_DIR / "errors_by_scenario.png").resolve(),
            producer="failure_analysis.py",
            run_id=graph_provenance.new_run_id(),
        )
    except Exception:
        pass
    print(f"[OK] Saved plot: {FIG_DIR / 'errors_by_scenario.png'}")

    return results


if __name__ == "__main__":
    run_failure_analysis()
