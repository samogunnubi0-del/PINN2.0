"""Shared endpoint evaluation vs Radau ODE (compare_models / checkpoint selection)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3_pilstm"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

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
from data.trajectory_dataset import TrajectoryScenario, integrate_scenario  # noqa: E402
from physics.endpoint_project import project_endpoint_trap  # noqa: E402

SCALES = np.array(
    [DEFAULT_N226_SCALE, DEFAULT_N225_SCALE, DEFAULT_NAC_SCALE, DEFAULT_N227_SCALE, DEFAULT_NAC227_SCALE],
    dtype=np.float64,
)
SPECIES = ["Ra-226", "Ra-225", "Ac-225", "Ra-227", "Ac-227"]
AC225_IDX = 2


def ode_endpoint(sc: TrajectoryScenario) -> np.ndarray:
    env = IsotopeEnvironment(phi=sc.phi, neutron_energy_ev=sc.energy_ev)
    _, y = run_simulation(
        env,
        t_end_h=sc.t_end_h,
        n_points=401,
        N_ra0=sc.ic[0],
        N_ra225_0=sc.ic[1],
        N_ac0=sc.ic[2],
        N_ra227_0=sc.ic[3],
        N_ac227_0=sc.ic[4],
    )
    return y[-1]


def v2_endpoint(v2_model, sc: TrajectoryScenario) -> np.ndarray:
    device = next(v2_model.parameters()).device
    x = torch.tensor(
        [[
            sc.t_end_h / DEFAULT_T_REF_H,
            sc.phi / DEFAULT_PHI_SCALE,
            float(neutron_energy_ev_to_feature_numpy(sc.energy_ev)),
            sc.ic[0] / DEFAULT_N226_SCALE,
            sc.ic[1] / DEFAULT_N225_SCALE,
            sc.ic[2] / DEFAULT_NAC_SCALE,
            sc.ic[3] / DEFAULT_N227_SCALE,
            sc.ic[4] / DEFAULT_NAC227_SCALE,
        ]],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        pred = v2_model(x).detach().cpu().numpy()[0]
    return pred * SCALES


def _features_for_scenario(sc: TrajectoryScenario, n_steps: int) -> np.ndarray:
    """Shared (1, n_steps, 8) feature matrix for sequence models."""
    t_norm, _ = integrate_scenario(sc, n_steps=n_steps)
    e_feat = float(neutron_energy_ev_to_feature_numpy(sc.energy_ev))
    ic_norm = sc.ic / SCALES
    seq_len = len(t_norm)
    feats = np.zeros((1, seq_len, 8), dtype=np.float32)
    for k in range(seq_len):
        feats[0, k, 0] = t_norm[k]
        feats[0, k, 1] = sc.phi / DEFAULT_PHI_SCALE
        feats[0, k, 2] = e_feat
        feats[0, k, 3:8] = ic_norm
    return feats


def baseline_endpoint(
    baseline,
    sc: TrajectoryScenario,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_steps: int = 64,
) -> np.ndarray:
    """Physical-scale endpoint (5,) from the vanilla BaselineLSTM (no physics)."""
    feats = _features_for_scenario(sc, n_steps)
    with torch.no_grad():
        feat_t = torch.from_numpy(feats).to(device=device, dtype=dtype)
        traj = baseline(feat_t)
        out = traj.detach().cpu().numpy()[0, -1]
    return out * SCALES


def pilstm_endpoint(
    pilstm,
    sc: TrajectoryScenario,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_steps: int = 64,
    project_trap: bool | None = None,
) -> np.ndarray:
    t_norm, _ = integrate_scenario(sc, n_steps=n_steps)
    e_feat = float(neutron_energy_ev_to_feature_numpy(sc.energy_ev))
    ic_norm = sc.ic / SCALES
    seq_len = len(t_norm)
    feats = np.zeros((1, seq_len, 8), dtype=np.float32)
    for k in range(seq_len):
        feats[0, k, 0] = t_norm[k]
        feats[0, k, 1] = sc.phi / DEFAULT_PHI_SCALE
        feats[0, k, 2] = e_feat
        feats[0, k, 3:8] = ic_norm
    if project_trap is None:
        project_trap = os.environ.get("PILSTM_ENDPOINT_PROJECT", "").strip().lower() in (
            "1", "true", "yes",
        )
    with torch.no_grad():
        feat_t = torch.from_numpy(feats).to(device=device, dtype=dtype)
        traj = pilstm(feat_t)
        if project_trap:
            t_t = torch.from_numpy(t_norm).to(device=device, dtype=dtype).unsqueeze(0)
            phi_t = torch.tensor([sc.phi / DEFAULT_PHI_SCALE], device=device, dtype=dtype)
            e_t = torch.tensor([e_feat], device=device, dtype=dtype)
            endpoint = project_endpoint_trap(
                traj, t_t, phi_t, e_t, t_ref_h=DEFAULT_T_REF_H
            )
            out = endpoint.detach().cpu().numpy()[0]
        else:
            out = traj.detach().cpu().numpy()[0, -1]
    return out * SCALES


def rel_err(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-30)


def evaluate_endpoints(
    scenarios: list[TrajectoryScenario],
    predict_fn,
) -> dict[str, list[float]]:
    """predict_fn(sc) -> physical-scale endpoint (5,). Returns per-species rel errors."""
    out: dict[str, list[float]] = {s: [] for s in SPECIES}
    for sc in scenarios:
        truth = ode_endpoint(sc)
        pred = predict_fn(sc)
        errs = rel_err(pred, truth)
        for i, name in enumerate(SPECIES):
            out[name].append(float(errs[i]))
    return out


def median_species_errors(errs: dict[str, list[float]]) -> dict[str, float]:
    return {name: float(np.median(vals)) if vals else float("nan") for name, vals in errs.items()}


def ac225_endpoint_median(errs: dict[str, list[float]]) -> float:
    vals = errs.get("Ac-225", [])
    return float(np.median(vals)) if vals else float("inf")
