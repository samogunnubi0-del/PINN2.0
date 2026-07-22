# IsotopePINN — Physics-Informed Surrogate for Ac-225 Production Planning

**Computational pharmacology · Physics-informed ML · Nuclear supply planning**

Samuel Ogunnubi · Dual-Enrollment Student, Anne Arundel Community College · sam.ogunnubi0@gmail.com · May 30, 2026

## The Problem

Ac-225 is a scarce alpha-emitting radiopharmaceutical crucial for targeted alpha therapy (TAT). Planning irradiation scenarios—balancing flux, neutron energy, and time—requires solving a stiff five-isotope chain many times. Classical ODE integrators are accurate but too slow for large-scale parameter sweeps. This surrogate enables rapid screening of thousands of flux/energy/time combinations to rank irradiation settings for Ac-225 yield.

## The Approach

- **Reference:** 0D Bateman ODE with NNDC/JENDL half-lives and cross sections (Radau solver).
- **Surrogate:** Physics-informed neural network (600-epoch physics pretrain + 3,400-epoch joint training; mass conservation in loss).
- **Validation:** Six independent checks vs ODE reference (no lab or hospital data).
- **Speed:** 10,000-point parameter sweep in seconds vs hours for sequential ODE (benchmark in Validation tab).

## Validation Results

**Overall: 6/6 PASS** (independent gates)

| Validation Check | Result |
|------------------|--------|
| Empty-target safety | PASS |
| Production scenario (14 MeV, full Ra-226) | PASS (9.9% Ac-225 vs ODE) |
| Decay-chain ingrowth | PASS |
| Species quality gate | PASS |
| PINN vs ODE correlation | PASS |
| Held-out Ac-225 (22 scenarios) | **4.51% median error** |

Strongest regimes: thermal / 14 MeV (~4–5%). Weakest: epithermal (~9.5%), threshold ~6.4 MeV (~8.5%).

**Figure:** `graphs/isef_parity_restyled.png` — held-out Ac-225 parity (22 scenarios; 20 with Ac-225 > 0 plotted on log axes).

## Limitations & Scope

- **Validation boundary:** Ground truth is ODE reference only, not reactor or clinical assay data.
- **Data & reproducibility:** NNDC/ENSDF half-lives; JENDL cross sections; `results/v63_validation_20260530.json`, weights v63.
- **Dimensionality:** 0D lumped model — no MCNP/OpenMC geometry or patient PK.
- **Assumptions:** Impurity/recovery sliders are post-processed, not PINN training targets.
- **Future work:** Transport coupling, expanded held-out regimes, published benchmark comparison.

## Links

- **Interactive demo:** https://lhyjrhmwzxqfpuuwsux7zh.streamlit.app
- **Code & validation:** https://github.com/samogunnubi0-del/PINN
- **Weights:** `weights/pinn_best_weights.pth` · SHA-256 prefix `7c21debe` · v63 · 6/6 PASS

First load after idle may take ~1 minute.
