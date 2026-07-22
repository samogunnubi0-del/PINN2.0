# PINN vs ODE diagnostic

> ## ⚠️ HISTORICAL — pre-v5 architecture (superseded)
>
> This report describes the **pre-fix model** (Iterations 1–4, integral-only daughters — before the Iteration 5 semi-analytic Bateman backbone). The huge MAPE values below (e.g. Ac-225 MAPE 13,102%) **do not describe the current model**.
>
> **Current results:** canonical **v63** weights — held-out Ac-225 median **4.51%** vs ODE, 6/6 validation gates PASS (`results/v63_validation_20260530.json`).
>
> Retained deliberately for the iteration/failure-transparency story. See `docs/ISEF_ITERATION_LOG.md` and `docs/PINN_ARCHITECTURE_FIX.md` for what changed and why.

## Overview
Mean absolute percentage error (MAPE) by species for three reference scenarios.
Training **up-weights Ra-225 and Ac-225** in the data loss and fits the **same capped**
forward pass used at inference, so numbers here should track deployable accuracy.

## Scenario Results
### ra225_dom_pure_decay
| Species | MAPE (%) | RMSE | Max Error |
|---------|----------|------|--------|
| Ra-226 | 0.00% | 0.00e+00 | 0.00e+00 |
| Ra-225 | 349.54% | 2.70e+18 | 4.43e+18 |
| Ac-225 | 13102.45% | 3.17e+19 | 5.47e+19 |

### ra226_dom_normal
| Species | MAPE (%) | RMSE | Max Error |
|---------|----------|------|--------|
| Ra-226 | 14.26% | 9.93e+22 | 1.66e+23 |
| Ra-225 | 231641286.05% | 5.62e+18 | 9.55e+18 |
| Ac-225 | 15887173573.28% | 3.10e+19 | 5.34e+19 |

### mixed_low_flux
| Species | MAPE (%) | RMSE | Max Error |
|---------|----------|------|--------|
| Ra-226 | 15.01% | 1.74e+19 | 2.90e+19 |
| Ra-225 | 344.74% | 2.66e+18 | 4.37e+18 |
| Ac-225 | 9691.06% | 3.17e+19 | 5.46e+19 |

## How to read this
- **MAPE** can look large when the true inventory is tiny; check RMSE and the prediction plots.
- For single-supply training, improve headline accuracy with more **CSV** coverage in the Ra-226 path, not OOD probes.

## Summary
- **No net alchemy**: hard budget cap + mass loss during training.
- **Goal**: PINN tracks the ODE within tolerance on **in-distribution** scenarios you care about.
- **If error is still too high**: run longer training, tune `DATA_WEIGHT` / `PHYSICS_WEIGHT` in `train.py`, or enrich data.
