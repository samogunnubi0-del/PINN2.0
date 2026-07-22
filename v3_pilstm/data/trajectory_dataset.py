"""ODE-generated trajectory sequences for PI-LSTM (scenario-level splits).

INVENTORY NORMALIZATION NOTE (fixed 2026-07-18, ISEF readiness P1):
    Scenario regimes named "virgin_*" historically used an Ra-226 inventory of
    6.022e23 atoms (= Avogadro's number = **226 g**, one mole) while calling it
    "virgin_1g". A true 1 g Ra-226 target holds N_A/226.0254 = 2.664e21 atoms
    (see v3_pilstm/analysis/validate_empirical.py, which always used the
    correct value). The mismatch is a 226x inventory error in scenario
    *generation* (training + held-out), not in the empirical validation.

    The fix is versioned so existing checkpoints remain reproducible:
      - ``SCENARIO_VERSION=v1`` (DEFAULT, legacy): inventory_scale =
        ``legacy_226g`` = 6.022e23 atoms. Reproduces every checkpoint trained
        before this fix. The name is intentionally honest about what it was.
      - ``SCENARIO_VERSION=v2``: inventory_scale = ``true_1g`` = 2.664e21
        atoms. Use this for all NEW training runs.
    Every sampled scenario records the scale actually used in
    ``TrajectoryScenario.inventory_scale`` (atoms), and the version string in
    ``TrajectoryScenario.scenario_version``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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

SCALES = np.array(
    [DEFAULT_N226_SCALE, DEFAULT_N225_SCALE, DEFAULT_NAC_SCALE, DEFAULT_N227_SCALE, DEFAULT_NAC227_SCALE],
    dtype=np.float64,
)


@dataclass
class TrajectoryScenario:
    phi: float
    energy_ev: float
    t_end_h: float
    ic: np.ndarray  # (5,) atoms
    scenario_id: int
    # Inventory bookkeeping (post 2026-07-18 fix). None means the scenario was
    # created by a code path that predates inventory-scale tracking.
    inventory_scale: float | None = None      # Ra-226 atoms used for "full target" ICs
    scenario_version: str | None = None       # "v1" (legacy_226g) | "v2" (true_1g)


# --- Inventory scale versioning (see module docstring) ----------------------
ATOMS_PER_GRAM_RA226 = 2.664e21          # N_A / 226.0254 g/mol  (true 1 g)
LEGACY_226G_ATOMS = 6.022e23             # N_A — historically mislabeled "1 g"
INVENTORY_SCALES = {
    "legacy_226g": LEGACY_226G_ATOMS,
    "true_1g": ATOMS_PER_GRAM_RA226,
}
SCENARIO_VERSIONS = {"v1": "legacy_226g", "v2": "true_1g"}


def resolve_inventory_scale(version: str | None = None) -> tuple[str, str, float]:
    """Return (version, label, atoms) for scenario generation.

    Precedence: explicit ``version`` arg > SCENARIO_VERSION env var > "v1".
    Default is intentionally the LEGACY scale so that re-running old training
    commands reproduces existing checkpoints bit-for-bit; new runs should set
    ``SCENARIO_VERSION=v2`` for true 1 g targets.
    """
    v = (version or os.environ.get("SCENARIO_VERSION", "v1")).strip().lower()
    if v not in SCENARIO_VERSIONS:
        raise ValueError(f"Unknown SCENARIO_VERSION {v!r}; expected one of {sorted(SCENARIO_VERSIONS)}")
    label = SCENARIO_VERSIONS[v]
    return v, label, INVENTORY_SCALES[label]


# P1-8: structured scenario mix matching v2 regimes (+ recycled / empty-feed).
# Weights control how often each production regime is sampled.
_REGIME_WEIGHTS = {
    "virgin_fast": 0.14,       # 1 g Ra-226, fast (n,2n) — headline product path
    "virgin_epithermal": 0.08,
    "threshold": 0.20,           # near the 6.42 MeV (n,2n) threshold (stiff, error-prone)
    "high_flux": 0.22,           # high phi — stresses Ra-227 overshoot
    "small_target": 0.06,       # 1e22 atoms virgin
    "ra225_seed": 0.06,        # decay-leg style: start from Ra-225
    "thermal_impurity": 0.04,  # thermal, drives (n,gamma) impurity channel
    "recycled_trace": 0.12,    # non-zero daughter ICs from recycled inventory
    "empty_feed": 0.08,        # phi>0 but IC=0 or near-zero
}


def _sample_t_end(rng: np.random.Generator, t_lo: float = 24.0, t_hi: float = 500.0) -> float:
    """Hybrid t_end: 70% log-uniform, 30% linear on [t_lo, t_hi] hours."""
    if rng.uniform() < 0.70:
        return float(np.exp(rng.uniform(np.log(t_lo), np.log(t_hi))))
    return float(rng.uniform(t_lo, t_hi))


def _sample_one(
    kind: str,
    rng: np.random.Generator,
    *,
    virgin_ra226_atoms: float = LEGACY_226G_ATOMS,
) -> tuple[float, float, float, np.ndarray]:
    """Return (phi, energy_ev, t_end_h, ic) for a named regime.

    ``virgin_ra226_atoms`` is the inventory used for "full virgin target" ICs.
    v1/legacy: 6.022e23 atoms (226 g — historical mislabel). v2: 2.664e21 (true 1 g).
    """
    virgin_1g = np.array([virgin_ra226_atoms, 0.0, 0.0, 0.0, 0.0])
    if kind == "virgin_fast":
        phi = 10.0 ** rng.uniform(12.0, 15.0)
        energy_ev = 10.0 ** rng.uniform(np.log10(6.5e6), np.log10(2.0e7))
        t_end_h = _sample_t_end(rng)
        ic = virgin_1g
    elif kind == "virgin_epithermal":
        phi = 10.0 ** rng.uniform(12.0, 15.0)
        energy_ev = 10.0 ** rng.uniform(0.0, 5.0)
        t_end_h = _sample_t_end(rng)
        ic = virgin_1g
    elif kind == "threshold":
        phi = 10.0 ** rng.uniform(12.5, 15.0)
        energy_ev = 10.0 ** rng.uniform(np.log10(5.5e6), np.log10(8.0e6))
        t_end_h = _sample_t_end(rng)
        ic = virgin_1g
    elif kind == "high_flux":
        phi = 10.0 ** rng.uniform(14.5, 15.0)
        energy_ev = 10.0 ** rng.uniform(np.log10(1.0e6), np.log10(2.0e7))
        t_end_h = _sample_t_end(rng, t_lo=100.0)
        ic = np.array([1.0e22, 0.0, 0.0, 0.0, 0.0])
    elif kind == "small_target":
        phi = 10.0 ** rng.uniform(11.0, 14.0)
        energy_ev = 10.0 ** rng.uniform(np.log10(0.025), np.log10(2.0e7))
        t_end_h = _sample_t_end(rng)
        ic = np.array([1.0e22, 0.0, 0.0, 0.0, 0.0])
    elif kind == "ra225_seed":
        phi = 10.0 ** rng.uniform(11.0, 13.0)
        energy_ev = 10.0 ** rng.uniform(0.0, 5.0)
        t_end_h = _sample_t_end(rng, t_hi=400.0)
        ic = np.array([0.0, 1.0e18, 0.0, 0.0, 0.0])
    elif kind == "recycled_trace":
        phi = 10.0 ** rng.uniform(12.0, 15.0)
        energy_ev = 10.0 ** rng.uniform(np.log10(1.0e6), np.log10(2.0e7))
        t_end_h = _sample_t_end(rng)
        ic = np.array([
            virgin_ra226_atoms * rng.uniform(0.25, 0.85),
            10.0 ** rng.uniform(14.0, 17.0),
            10.0 ** rng.uniform(12.0, 15.0),
            10.0 ** rng.uniform(13.0, 16.0),
            10.0 ** rng.uniform(11.0, 14.0),
        ])
    elif kind == "empty_feed":
        phi = 10.0 ** rng.uniform(13.0, 15.0)
        energy_ev = 10.0 ** rng.uniform(np.log10(6.5e6), np.log10(2.0e7))
        t_end_h = _sample_t_end(rng)
        ic = 10.0 ** rng.uniform(3.0, 9.0, size=5)  # near-zero inventories
    else:  # thermal_impurity
        phi = 10.0 ** rng.uniform(13.0, 15.0)
        energy_ev = float(rng.uniform(0.02, 0.2))
        t_end_h = _sample_t_end(rng)
        ic = virgin_1g
    return phi, energy_ev, t_end_h, ic


def _sample_scenarios(
    n: int,
    rng: np.random.Generator,
    *,
    id_offset: int = 0,
    structured: bool = True,
    scenario_version: str | None = None,
) -> list[TrajectoryScenario]:
    """Sample n scenarios. structured=True uses the v2-matched regime mix.

    scenario_version selects the Ra-226 inventory scale for virgin/recycled
    targets ("v1"=legacy_226g, "v2"=true_1g; see resolve_inventory_scale).
    """
    version, _label, virgin_atoms = resolve_inventory_scale(scenario_version)
    scenarios: list[TrajectoryScenario] = []
    if not structured:
        for i in range(n):
            phi = 10.0 ** rng.uniform(11.0, 15.0)
            energy_ev = 10.0 ** rng.uniform(np.log10(0.025), np.log10(2.0e7))
            t_end_h = float(rng.uniform(24.0, 500.0))
            ic_type = rng.integers(0, 3)
            if ic_type == 0:
                ic = np.array([virgin_atoms, 0.0, 0.0, 0.0, 0.0])
            elif ic_type == 1:
                ic = np.array([1.0e22, 0.0, 0.0, 0.0, 0.0])
            else:
                ic = np.array([0.0, 1.0e18, 0.0, 0.0, 0.0])
            scenarios.append(
                TrajectoryScenario(
                    phi=phi, energy_ev=energy_ev, t_end_h=t_end_h, ic=ic,
                    scenario_id=id_offset + i,
                    inventory_scale=virgin_atoms, scenario_version=version,
                )
            )
        return scenarios

    kinds = list(_REGIME_WEIGHTS.keys())
    probs = np.array([_REGIME_WEIGHTS[k] for k in kinds], dtype=np.float64)
    probs /= probs.sum()
    for i in range(n):
        kind = kinds[int(rng.choice(len(kinds), p=probs))]
        phi, energy_ev, t_end_h, ic = _sample_one(kind, rng, virgin_ra226_atoms=virgin_atoms)
        scenarios.append(
            TrajectoryScenario(
                phi=phi, energy_ev=energy_ev, t_end_h=t_end_h, ic=ic,
                scenario_id=id_offset + i,
                inventory_scale=virgin_atoms, scenario_version=version,
            )
        )
    return scenarios


def canonical_heldout_scenarios(
    n: int = 22,
    seed: int = 2024,
    scenario_version: str | None = None,
) -> list[TrajectoryScenario]:
    """Deterministic held-out set shared by training eval AND compare_models.

    Fixing this set makes PI-LSTM's reported error directly comparable to v2's
    held-out gate (P0-2). Never reseed/change without regenerating v2 comparison.
    The default inventory scale is the legacy one (v1) for exactly that reason;
    pass scenario_version="v2" (or SCENARIO_VERSION=v2) for true-1g evaluation.
    """
    rng = np.random.default_rng(seed)
    return _sample_scenarios(n, rng, id_offset=10_000, structured=True, scenario_version=scenario_version)


def _time_grid(t_end_h: float, n_steps: int, log_spaced: bool) -> np.ndarray:
    """Time grid in hours. Log spacing densely samples early stiff dynamics
    (Ra-227 T½=42 min, Ra-225/Ac-225 ingrowth) while still hitting t=0 and t_end."""
    if not log_spaced or n_steps < 3:
        return np.linspace(0.0, float(t_end_h), int(n_steps))
    t_min = max(float(t_end_h) * 2.0e-3, 0.02)  # ~1.2 min floor, captures Ra-227
    tail = np.logspace(np.log10(t_min), np.log10(float(t_end_h)), int(n_steps) - 1)
    return np.concatenate([[0.0], tail])


def integrate_scenario(
    sc: TrajectoryScenario,
    n_steps: int = 64,
    log_spaced: bool = True,
    dense_steps: int | None = None,
    rate_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_norm seq, normalized Y seq) from Radau ODE on a (log-spaced) grid.

    ``rate_scale`` > 1 DE-STIFFENS the ODE: every rate constant (reaction
    rates AND decay lambdas) is divided by rate_scale, uniformly slowing all
    time constants while preserving equilibrium ratios. Used only by the
    stiffness-curriculum scaffold (PI_LSTM_CURRICULUM); default 1.0 reproduces
    legacy data exactly.
    """
    env = IsotopeEnvironment(phi=sc.phi, neutron_energy_ev=sc.energy_ev)
    if dense_steps is not None and dense_steps > n_steps:
        t_dense = _time_grid(sc.t_end_h, dense_steps, log_spaced)
        t_h, y = _run_on_grid(env, t_dense, sc.ic, rate_scale=rate_scale)
        idx = np.linspace(0, len(t_dense) - 1, int(n_steps), dtype=int)
        t_h = t_h[idx]
        y = y[idx]
    else:
        t_grid = _time_grid(sc.t_end_h, n_steps, log_spaced)
        t_h, y = _run_on_grid(env, t_grid, sc.ic, rate_scale=rate_scale)
    t_norm = t_h / DEFAULT_T_REF_H
    y_norm = y / SCALES
    return t_norm.astype(np.float32), y_norm.astype(np.float32)


def _run_on_grid(
    env: "IsotopeEnvironment",
    t_grid: np.ndarray,
    ic: np.ndarray,
    rate_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the ODE and sample exactly at t_grid (hours)."""
    from scipy.integrate import solve_ivp
    from ra226_ac225_transmutation import bateman_transmutation_rhs

    t_end = float(t_grid[-1]) if len(t_grid) else 0.0
    y0 = np.array(ic, dtype=float)
    inv = 1.0 / float(rate_scale)
    k_n2n = env.k_n2n_per_h() * inv
    k_ng = env.k_ngamma_per_h() * inv
    args = (
        k_n2n, k_ng,
        env.lambda_ra226_per_h * inv, env.lambda_ra225_per_h * inv, env.lambda_ac225_per_h * inv,
        env.lambda_ra227_per_h * inv, env.lambda_ac227_per_h * inv,
    )
    sol = solve_ivp(
        bateman_transmutation_rhs,
        t_span=(0.0, t_end),
        y0=y0,
        method="Radau",
        t_eval=t_grid,
        args=args,
        rtol=1e-9,
        atol=1e-12,
    )
    return t_grid, np.maximum(sol.y.T, 0.0)


class TrajectoryDataset(Dataset):
    """One item = full trajectory for a single (phi, E, IC) scenario."""

    def __init__(
        self,
        scenarios: list[TrajectoryScenario],
        n_steps: int = 64,
        log_spaced: bool = True,
        dense_steps: int | None = None,
        rate_scale: float = 1.0,
    ):
        self.scenarios = scenarios
        self.n_steps = n_steps
        self._cache: list[tuple[np.ndarray, np.ndarray, TrajectoryScenario]] = []
        for sc in scenarios:
            t_norm, y_norm = integrate_scenario(
                sc, n_steps=n_steps, log_spaced=log_spaced, dense_steps=dense_steps,
                rate_scale=rate_scale,
            )
            self._cache.append((t_norm, y_norm, sc))

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t_norm, y_norm, sc = self._cache[idx]
        e_feat = float(neutron_energy_ev_to_feature_numpy(sc.energy_ev))
        ic_norm = sc.ic / SCALES
        phi_norm = sc.phi / DEFAULT_PHI_SCALE

        seq_len = len(t_norm)
        features = np.zeros((seq_len, 8), dtype=np.float32)
        for k in range(seq_len):
            features[k, 0] = t_norm[k]
            features[k, 1] = phi_norm
            features[k, 2] = e_feat
            features[k, 3:8] = ic_norm

        return {
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(y_norm),
            "t_norm": torch.from_numpy(t_norm),
            "phi_norm": torch.tensor(phi_norm, dtype=torch.float32),
            "energy_feature": torch.tensor(e_feat, dtype=torch.float32),
            "ic_norm": torch.from_numpy(ic_norm.astype(np.float32)),
        }


def build_dataloaders(
    n_train: int = 1400,
    n_val: int = 22,
    n_test: int = 22,
    n_steps: int = 64,
    batch_size: int = 16,
    seed: int = 42,
    log_spaced: bool = True,
    dense_steps: int | None = None,
    scenario_version: str | None = None,
    loader_seed: int | None = None,
    train_rate_scale: float = 1.0,
) -> tuple:
    """Training pool is structured/random; val + test use the CANONICAL held-out
    set (P0-2) so PI-LSTM error is directly comparable to v2 and to compare_models.

    ``loader_seed`` seeds the DataLoader shuffle generator explicitly (P1
    reproducibility): same seed + same data => identical batch order every run.
    ``scenario_version`` forwards to the samplers ("v1" legacy_226g / "v2" true_1g).
    ``train_rate_scale`` > 1 de-stiffens the TRAINING ODE data (curriculum
    scaffold); val/test are always generated at full stiffness (scale 1.0).
    """
    from torch.utils.data import DataLoader

    rng = np.random.default_rng(seed)
    train_sc = _sample_scenarios(n_train, rng, id_offset=0, structured=True, scenario_version=scenario_version)
    # Deterministic held-out sets (disjoint id ranges); test == compare_models set.
    val_sc = canonical_heldout_scenarios(n_val, seed=2025, scenario_version=scenario_version)
    test_sc = canonical_heldout_scenarios(n_test, seed=2024, scenario_version=scenario_version)

    def _collate(batch):
        return {
            "features": torch.stack([b["features"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]),
            "t_norm": torch.stack([b["t_norm"] for b in batch]),
            "phi_norm": torch.stack([b["phi_norm"] for b in batch]),
            "energy_feature": torch.stack([b["energy_feature"] for b in batch]),
            "ic_norm": torch.stack([b["ic_norm"] for b in batch]),
        }

    train_ds = TrajectoryDataset(train_sc, n_steps=n_steps, log_spaced=log_spaced, dense_steps=dense_steps,
                                 rate_scale=train_rate_scale)
    val_ds = TrajectoryDataset(val_sc, n_steps=n_steps, log_spaced=log_spaced, dense_steps=dense_steps)
    test_ds = TrajectoryDataset(test_sc, n_steps=n_steps, log_spaced=log_spaced, dense_steps=dense_steps)

    train_gen = None
    if loader_seed is not None:
        train_gen = torch.Generator()
        train_gen.manual_seed(int(loader_seed))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, generator=train_gen
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    return train_loader, val_loader, test_loader, test_sc
