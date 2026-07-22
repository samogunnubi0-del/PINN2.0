"""
Sanity checks for trained IsotopePINN vs the reference ODE integrator.

**Trio test** (default): three scenarios Gemini-style “audit” for BMAH-style physics.

  A — Empty tank + high flux: all inventories 0, phi high → no spontaneous atoms.
  B — Ra-226 feedstock + flux: N_Ra226(0) = 1e22 → Ac-225 can grow vs ODE.
  C — Pure decay: phi = 0, Ra-225 only → chain decay vs ODE (original single test).

Run:

  python test_single.py              # all three
  python test_single.py --legacy     # only scenario C (old behavior)

Exit code 1 if any scenario prints FAIL (PINN) or FAIL (ODE).
TRIO lines labelled CHECK are advisory and do not affect the exit code.

Original scenario was: phi = 0, Ra-225 = 1e18, Ra-226 = Ac-225 = 0.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import torch

from pinn_model import IsotopePINN, load_isotope_pinn_checkpoint, neutron_energy_ev_to_feature_numpy
from ra226_ac225_transmutation import IsotopeEnvironment, run_simulation

HERE = pathlib.Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "weights" / "pinn_best_weights.pth"

# Must match train.py / pinn_model.py scaling
N226_SCALE = 6.022e23
N225_SCALE = 1e20
NAC_SCALE = 1e20
PHI_SCALE = 1e15       # must match pinn_model.py / train.py
TIME_SCALE_H = 500.0   # must match pinn_model.py / train.py

LN2 = math.log(2.0)
LAMBDA_AC225_PER_H = LN2 / (9.92 * 24.0)


def physics_baseline(
    time_h: float,
    flux_phi: float,
    energy_ev: float,
    n226_0: float,
    n225_0: float,
    nac_0: float,
) -> tuple[float, float, float]:
    env = IsotopeEnvironment(phi=flux_phi, neutron_energy_ev=energy_ev)
    t_h, Y = run_simulation(
        env,
        t_end_h=time_h,
        n_points=max(401, int(time_h * 8) + 1),
        N_ra0=n226_0,
        N_ra225_0=n225_0,
        N_ac0=nac_0,
    )
    return float(Y[-1, 0]), float(Y[-1, 1]), float(Y[-1, 2])


def pinn_predict(
    model: IsotopePINN,
    time_h: float,
    flux_phi: float,
    energy_ev: float,
    n226_0: float,
    n225_0: float,
    nac_0: float,
) -> tuple[float, float, float]:
    t_norm = time_h / TIME_SCALE_H
    phi_nn = flux_phi / PHI_SCALE
    init226 = n226_0 / N226_SCALE
    init225 = n225_0 / N225_SCALE
    init_ac = nac_0 / NAC_SCALE
    e_nn = float(neutron_energy_ev_to_feature_numpy(energy_ev))
    x = torch.tensor(
        [[t_norm, phi_nn, e_nn, init226, init225, init_ac, 0.0, 0.0]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        pred = model(x)
    n226 = float(pred[0, 0].item() * N226_SCALE)
    n225 = float(pred[0, 1].item() * N225_SCALE)
    n_ac = float(pred[0, 2].item() * NAC_SCALE)
    return n226, n225, n_ac


def emit_verdict(
    *,
    title: str,
    time_h: float,
    flux_phi: float,
    energy_ev: float,
    initial_ra226: float,
    initial_ra225: float,
    initial_ac225: float,
    n226: float,
    n225: float,
    n_ac: float,
    ode226: float,
    ode225: float,
    ode_ac: float,
    pinn_ok: bool,
) -> bool:
    total_0 = initial_ra226 + initial_ra225 + initial_ac225
    atol = max(1.0, 1e-7 * max(total_0, 1.0))
    # Large inventories: allow tiny float/cap slack so summed species ~= budget does not flag alchemy.
    budget_slack = max(atol, 1e-3 * max(total_0, 1.0))

    print()
    print("Atom budget (N_Ra226 + N_Ra225 + N_Ac225) -- no net creation:")
    print(f"  initial total = {total_0:.6e}")

    n_ac_theory = initial_ac225 * math.exp(-LAMBDA_AC225_PER_H * time_h)
    if initial_ac225 > 0.0 and initial_ra225 == 0.0 and initial_ra226 == 0.0:
        print("Pure-decay closed form (only Ac-225 at t=0):")
        print(f"  N_Ac225 theory ~ {n_ac_theory:.6e}")
    else:
        print("(Closed-form Ac-225 line skipped -- not an Ac-only initial state.)")

    print()
    pinn_hard_fail = False
    if pinn_ok and not math.isnan(n_ac):
        total_p = n226 + n225 + n_ac
        print(f"  PINN final total = {total_p:.6e}")
        ode226_tol = max(1e3, 1e-9 * max(total_0, 1.0))
        spurious226 = (
            flux_phi == 0.0
            and initial_ra226 <= atol
            and ode226 <= ode226_tol
            and n226 > ode226_tol * 1e3
        )
        if spurious226:
            print(
                "FAIL (PINN): Ra-226 ingrowth at zero flux with no Ra-226 initially "
                f"(PINN N_Ra226={n226:.3e} vs ODE ~{ode226:.3e})."
            )
        alchemy_pinn = total_p > total_0 + budget_slack
        fuel_ok = True
        if flux_phi > 0.0 and initial_ra226 > 1e18 and ode226 > 0.0:
            if n226 < 0.2 * ode226:
                fuel_ok = False

        if spurious226:
            pinn_hard_fail = True
        elif alchemy_pinn:
            print(
                "FAIL (PINN): Net atom gain (alchemy) -- final total exceeds initial total "
                f"(limit ~{total_0 + budget_slack:.6e})."
            )
            pinn_hard_fail = True
        elif not fuel_ok:
            print(
                "FAIL (PINN): Fuel thief -- Ra-226 depleted far below ODE baseline under active flux / feedstock."
            )
            pinn_hard_fail = True
        elif (
            flux_phi == 0.0
            and n_ac > initial_ac225 + atol
            and initial_ra225 <= 0.0
            and initial_ra226 <= 0.0
        ):
            print("FAIL (PINN): Ac-225 grew at zero flux with no parent nuclides -- non-physical.")
            pinn_hard_fail = True
        elif flux_phi > 0.0 and n_ac > initial_ac225 + atol and initial_ra226 <= 0.0 and total_0 <= atol:
            print("FAIL (PINN): Ac-225 grew at high flux with empty feedstock (no Ra-226).")
            pinn_hard_fail = True
        elif flux_phi > 0.0 and n_ac > initial_ac225 + atol and initial_ra226 > 0.0:
            print("PASS (PINN): Ac-225 increased with flux and Ra-226 feedstock; atom budget OK.")
        elif flux_phi == 0.0 and initial_ra225 > 0.0 and n_ac > initial_ac225 + atol and not alchemy_pinn:
            print("PASS (PINN): Ac-225 ingrowth from Ra-225 decay at zero flux; atom budget OK.")
        else:
            print("PASS (PINN): No alchemy or flagged pathologies for this setup.")
    else:
        print("PINN verdict skipped (model load or forward failed).")
        if pinn_ok:
            pinn_hard_fail = True

    ode_total = ode226 + ode225 + ode_ac
    print()
    print(f"  ODE final total  = {ode_total:.6e}")
    alchemy_ode = ode_total > total_0 + budget_slack
    ode_hard_fail = False
    if alchemy_ode:
        print(
            "FAIL (ODE): Net atom gain in reference integrator (unexpected) -- check simulation / tolerances."
        )
        ode_hard_fail = True
    elif flux_phi == 0.0 and ode_ac > initial_ac225 + atol and initial_ra225 > 0.0:
        print(
            "PASS (ODE): Ac-225 increased with flux off because Ra-225 feeds the chain (beta decay / Bateman path)."
        )
    elif flux_phi == 0.0:
        print("PASS (ODE): Zero-flux scenario consistent with decay-only evolution.")
    elif initial_ra226 > 0.0 and ode_ac > initial_ac225 + atol:
        print("PASS (ODE): Ac-225 grew under flux with Ra-226 present.")
    elif initial_ra226 <= 0.0 and total_0 <= atol and ode_ac > initial_ac225 + atol:
        print("FAIL (ODE): Ac-225 grew from empty initial inventories (simulator bug or misconfiguration).")
        ode_hard_fail = True
    else:
        print("PASS (ODE): Reference trajectory OK for this setup.")

    # Trio-specific hints (ODE is ground truth for expected physics)
    if title.startswith("A "):
        empty_tol = max(1e3, 1e-6 * max(ode_total, 1.0))
        if pinn_ok and not math.isnan(n_ac):
            pin_tot = n226 + n225 + n_ac
            if pin_tot <= empty_tol and ode_total <= empty_tol:
                print()
                print("TRIO A: PASS -- empty start stays effectively empty (PINN & ODE).")
            elif pin_tot > empty_tol and total_0 <= atol:
                print()
                print("TRIO A: CHECK -- PINN shows matter from empty start; compare PINN total to ODE above.")
    if title.startswith("B ") and pinn_ok and not math.isnan(n_ac):
        if ode_ac > initial_ac225 + atol:
            rel = abs(n_ac - ode_ac) / max(ode_ac, 1.0)
            print()
            if rel < 0.10:
                print("TRIO B: PASS -- Ac-225 vs ODE within 10% (full-tank transmutation track).")
            elif rel < 0.25:
                print("TRIO B: CHECK -- Ac-225 vs ODE within ~25% (needs <10% for strict ISEF).")
            else:
                print(
                    f"TRIO B: CHECK -- Ac-225 PINN vs ODE differ (~{100.0 * rel:.1f}%); may still be usable."
                )

    return not pinn_hard_fail and not ode_hard_fail


def run_scenario(
    model: IsotopePINN | None,
    *,
    title: str,
    time_h: float,
    flux_phi: float,
    energy_ev: float,
    n226_0: float,
    n225_0: float,
    nac_0: float,
    pinn_ok: bool,
) -> bool:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"Time (h):            {time_h}")
    print(f"Flux phi (n/cm^2/s): {flux_phi:.6e}")
    print(f"Energy E (eV):       {energy_ev}")
    print(
        "Initial inventories (atoms): "
        f"Ra-226={n226_0:.6e}, Ra-225={n225_0:.6e}, Ac-225={nac_0:.6e}"
    )
    print()

    n226 = n225 = n_ac = float("nan")
    if pinn_ok and model is not None:
        try:
            n226, n225, n_ac = pinn_predict(model, time_h, flux_phi, energy_ev, n226_0, n225_0, nac_0)
            print(f"PINN prediction (atoms) after {time_h} h:")
            print(f"  N_Ra226  = {n226:.6e}")
            print(f"  N_Ra225  = {n225:.6e}")
            print(f"  N_Ac225  = {n_ac:.6e}")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("PINN: checkpoint incompatible with current IsotopePINN.")
                print("      Retrain:  python train.py")
                n226 = n225 = n_ac = float("nan")
            else:
                raise
    elif not pinn_ok:
        print("PINN skipped (no weights).")

    ode226, ode225, ode_ac = physics_baseline(
        time_h, flux_phi, energy_ev, n226_0, n225_0, nac_0
    )
    print()
    print(f"ODE baseline (atoms) after {time_h} h:")
    print(f"  N_Ra226  = {ode226:.6e}")
    print(f"  N_Ra225  = {ode225:.6e}")
    print(f"  N_Ac225  = {ode_ac:.6e}")

    return emit_verdict(
        title=title,
        time_h=time_h,
        flux_phi=flux_phi,
        energy_ev=energy_ev,
        initial_ra226=n226_0,
        initial_ra225=n225_0,
        initial_ac225=nac_0,
        n226=n226,
        n225=n225,
        n_ac=n_ac,
        ode226=ode226,
        ode225=ode225,
        ode_ac=ode_ac,
        pinn_ok=pinn_ok,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PINN vs ODE sanity checks (trio or legacy).")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Only scenario C (original single decay test).",
    )
    args = parser.parse_args()

    pinn_ok = WEIGHTS_PATH.is_file()
    if not pinn_ok:
        print(f"Warning: missing weights ({WEIGHTS_PATH}); ODE baselines only.\n")

    model: IsotopePINN | None = None
    if pinn_ok:
        try:
            model, model_info = load_isotope_pinn_checkpoint(WEIGHTS_PATH, map_location="cpu")
            if model_info["n_fourier_freqs"] == 0:
                print("Loaded legacy-compatible checkpoint (no Fourier time encoder).\n")
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("Could not load checkpoint (architecture mismatch). ODE only.\n")
                model = None
                pinn_ok = False
            else:
                raise

    # C — same as historical test_single (pure decay, Ra-225 feed)
    scenario_c = dict(
        title="C - Pure decay (Ra-225 chain, phi = 0)",
        time_h=48.0,
        flux_phi=0.0,
        energy_ev=0.025,
        n226_0=0.0,
        n225_0=1.0e18,
        nac_0=0.0,
    )

    if args.legacy:
        ok = run_scenario(model, pinn_ok=pinn_ok, **scenario_c)
        return 0 if ok else 1

    print(
        "Trio test: empty tank + flux, Ra-226 feed + flux, pure decay. "
        f"Weights: {'yes' if pinn_ok else 'no'}."
    )

    ok = True
    ok &= run_scenario(
        model,
        title="A - Empty tank + high flux (no feedstock)",
        time_h=100.0,
        flux_phi=1.0e15,
        energy_ev=0.025,
        n226_0=0.0,
        n225_0=0.0,
        nac_0=0.0,
        pinn_ok=pinn_ok,
    )
    ok &= run_scenario(
        model,
        title="B - Full tank: Ra-226 = 1e22 + fast reactor flux",
        time_h=250.0,
        flux_phi=1.0e14,
        energy_ev=14.0e6,
        n226_0=1.0e22,
        n225_0=0.0,
        nac_0=0.0,
        pinn_ok=pinn_ok,
    )
    ok &= run_scenario(model, pinn_ok=pinn_ok, **scenario_c)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
