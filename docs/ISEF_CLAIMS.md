# ISEF Claims & Evidence Requirements

**Last updated:** 2026-05-30 (v63 weights sha256 `7c21debe…`; Iter 12 ISEF ship)

> **How I got here:** [`PINN_ARCHITECTURE_FIX.md`](PINN_ARCHITECTURE_FIX.md) — old system flaws, new design, and honest limitations.

## Supported claims (use on poster)

| Claim | Evidence | Status |
|-------|----------|--------|
| Physics safety — no alchemy from empty tank | `test_single.py` Trio A PASS | **Supported** |
| Production Ac-225 within 10% of ODE (Trio B) | `test_single.py` Trio B PASS | **Supported** |
| Decay-chain ingrowth — Ac-225 from Ra-225 (Trio C) | `test_single.py` Trio C PASS | **Supported** |
| Ra-226 inventory tracking | median rel error **0.014%** (`pinn_validation_summary.csv`) | **Supported** |
| Held-out Ac-225 accuracy | median **4.5%** (`heldout_validation_summary.csv`, v63 weights) | **Supported** |
| Predictor quality gate (all species + impurity) | `evaluate_quality_gate.py` **overall PASS** | **Supported** |
| PINN vs ODE correlation | `correlation_check.py` PASS (Ra-226/225/Ac-225) | **Supported** |
| 500×+ throughput under batched inference vs serial stiff-ODE solves | Streamlit speed benchmark tab; no committed benchmark JSON in `results/` yet — **benchmark re-verification in progress** | **Under re-verification** |
| Real-world problem (Ac-225 shortage for TAT) | Literature + NNDC data in `ra226_ac225_transmutation.py` | **Supported** |
| Transparent failure analysis | `analysis/FAILURE_CASE_ANALYSIS.md`, iteration log | **Supported** |

## Claims NOT yet supported (do not use)

| Claim | Why |
|-------|-----|
| "99.8% accuracy" blanket statement | Use species-specific medians (Ac-225 4.51%, Ra-226 0.001%) |
| Optimal harvest window at 342 h | Needs dedicated harvest demo validation on poster |

## Evidence checklist (must all pass for full ISEF claim)

| # | Criterion | Script | Pass condition | Current |
|---|-----------|--------|----------------|---------|
| 1 | Empty-tank safety | `test_single.py` Trio A | PASS | **PASS** |
| 2 | Production scenario | `test_single.py` Trio B | Ac-225 within **10%** of ODE | **PASS** |
| 3 | Decay chain | `test_single.py` Trio C | Ac-225 **nonzero**, budget OK | **PASS** |
| 4 | Held-out quality | `analysis/evaluate_quality_gate.py` | **overall PASS** | **PASS** |
| 5 | Correlation | `analysis/correlation_check.py` | R² > **0.85** all species | **PASS** |
| 6 | Ac-225 held-out | `heldout_validation_summary.csv` | median < **10%** | **PASS (4.5%)** |

## Graphs for poster

| Graph | File | Status |
|-------|------|--------|
| Isotope evolution + harvest | `graphs/isef_isotope_evolution.png` | Regenerated v63+Iter 7–8 |
| Loss + 12k projection | `graphs/isef_loss_trajectory_12k.png` | Full 4k joint; projection guarded |
| Mass conservation | `graphs/isef_mass_conservation.png` | 5-species budget (Iter 8 fix) |
| Ac-225 parity | `graphs/isef_parity_restyled.png` | **4.51% median** on 22 held-out scenarios (v63) |
| Loss components | `graphs/loss_components.png` | Deduped CSV |
| Sensitivity (PINN) | `graphs/isef_sensitivity_flux.png` | PINN sweep — use this, not RF |

### Do NOT put on poster (stale or wrong model)

| Graph | File | Why |
|-------|------|-----|
| Error histogram | `archive/error_histogram.png` | Legacy **RandomForest**, not PINN |
| Flux sensitivity | `archive/flux_sensitivity.png` | Legacy RF pipeline |
| Ac-225 growth | `archive/ac225_growth.png` | **ODE-only** |
| Yield heatmap | `archive/ac225_yield_heatmap.png` | **ODE-only** |

See full audit: [`ISEF_ITERATION_LOG.md`](ISEF_ITERATION_LOG.md) Iterations 7–8.

## Mandatory Kaggle env (4000-epoch official runs)

```python
PINN_MEDIUM_TRAIN=1
PINN_FLOAT64=1
PINN_RESUME=0
PINN_SAFETY_RATIO=0
PINN_DATA_WEIGHT=25
PINN_LOG_DATA_WEIGHT=5
PINN_GRAD_BALANCE=1
PINN_BATEMAN_BACKBONE=1
PINN_COLLOCATION_POINTS=500
PINN_JOINT_CHUNK=0
PINN_CURRIC_RAMP=1200
PINN_COS_T0=800
PINN_LOG_EVERY=50
```

## Honest elevator pitch

> I built a physics-informed neural network with semi-analytic Bateman backbones for the Ra-226 → Ac-225 medical isotope chain. It prevents alchemy (Trio A verified), predicts Ac-225 within 10% of stiff ODE integrators on production and decay scenarios, passes a held-out quality gate at 4.51% median Ac-225 error, and runs hundreds of times faster than classical solvers under batched inference (benchmark re-verification in progress).
