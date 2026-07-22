"""
Check that every training row is consistent with the ODE: same phi, E, IC, time
must reproduce target inventories. Run before/after train.py when debugging correlation.

Usage: python analysis/verify_train_rows_vs_ode.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

DATA_PATH = ROOT / "data" / "pinn_training_data.csv"


def ode_final_state(
    phi: float,
    energy_ev: float,
    time_h: float,
    n226_0: float,
    n225_0: float,
    nac_0: float,
) -> np.ndarray:
    env = IsotopeEnvironment(phi=phi, neutron_energy_ev=energy_ev)
    n_pts = max(400, int(np.clip(time_h * 8, 400, 15000)))
    _t, Y = run_simulation(
        env,
        t_end_h=time_h,
        n_points=n_pts,
        N_ra0=n226_0,
        N_ra225_0=n225_0,
        N_ac0=nac_0,
    )
    return Y[-1, :3].astype(np.float64)


def main() -> None:
    import train as T

    df_raw = pd.read_csv(
        DATA_PATH,
        usecols=["phi", "energy", "time", "N_Ra226", "N_Ra225", "N_Ac225"],
        dtype="float64",
        engine="python",
    )
    df_raw["energy"] = df_raw["energy"].replace([np.inf, -np.inf], np.nan)
    df_raw["energy"] = df_raw["energy"].fillna(T.THERMAL_REFERENCE_EV)
    df_raw["energy"] = df_raw["energy"].clip(lower=1e-6, upper=1e8)

    # Quick check: first 120 CSV rows (full file can take many minutes sequentially).
    max_csv = int(__import__("os").environ.get("VERIFY_CSV_ROWS", "120"))
    if max_csv > 0 and len(df_raw) > max_csv:
        df_raw = df_raw.iloc[:max_csv].copy()
        print(f"VERIFY_CSV_ROWS={max_csv}: using subset of CSV for speed.")

    rng = np.random.default_rng(42)
    # Single-process cache: avoids Windows worker spawn + heavy scipy imports per child.
    triples = T._collect_unique_reference_triples(df_raw)
    ref_cache: dict = {}
    for args in triples:
        key, val = T._reference_traj_worker(args)
        ref_cache[key] = val
    train_dat = T.augment_rows_time_shift(
        df_raw,
        rng,
        augment_per_row=T.AUGMENT_PER_ROW,
        include_unshifted_base=True,
        initial_cache=ref_cache,
    )
    if not getattr(T, "SINGLE_SUPPLY_MODE", False):
        inv = T.augment_inverted_ic_scenarios(rng, n_extra=T.INVERTED_IC_N_EXTRA)
        div = T.augment_diverse_ic_scenarios(rng, n_extra=T.DIVERSE_IC_N_EXTRA)
        train_dat = pd.concat([train_dat, inv, div], ignore_index=True)

    print(f"Rows checked: {len(train_dat)} (sample up to 400)")
    idx = np.random.default_rng(0).choice(len(train_dat), size=min(400, len(train_dat)), replace=False)
    worst = (0.0, -1, "")
    for i in idx:
        r = train_dat.iloc[i]
        y_ode = ode_final_state(
            float(r["phi"]),
            float(r["energy"]),
            float(r["time"]),
            float(r["init_N226"]),
            float(r["init_N225"]),
            float(r["init_NAc"]),
        )
        tgt = np.array([r["N_Ra226"], r["N_Ra225"], r["N_Ac225"]], dtype=np.float64)
        den = np.maximum(np.abs(tgt), 1.0)
        rel = np.max(np.abs(y_ode - tgt) / den)
        if rel > worst[0]:
            worst = (rel, int(i), f"ODE {y_ode} vs tgt {tgt}")
        if rel > 0.05:
            print(f"  row {i} rel_err={rel:.4e} phi={r['phi']:.3e} t={r['time']:.2f}")

    print(f"\nWorst sampled max-relative-error (vs ODE replay): {worst[0]:.4e} at row {worst[1]}")
    print(f"  {worst[2]}")
    if worst[0] < 1e-3:
        print("OK: training targets match ODE replay (augmentation is self-consistent).")
    elif worst[0] < 0.02:
        print("OK-ish: small mismatch (interp / n_points).")
    else:
        print("PROBLEM: rows do not match ODE — fix augmentation or integrator settings.")


if __name__ == "__main__":
    main()
