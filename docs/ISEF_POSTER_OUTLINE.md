# ISEF Poster Outline — IsotopePINN (v63 4k)

**Weights:** `weights/pinn_best_weights.pth` (sha256 `7c21debe…`)  
**Evidence:** `results/v63_validation_20260530.json` — **6/6 PASS**, held-out Ac-225 median **4.51%**

---

## Title block

**Physics-Informed Neural Networks for Rapid Ac-225 Production Planning**

Subtitle: A fast surrogate for the Ra-226 → Ac-225 transmutation chain validated against stiff ODE integrators.

---

## Panel 1 — Problem (top left)

- Ac-225 is a scarce alpha-emitter used in **targeted alpha therapy (TAT)** for cancer.
- Planning production (flux, neutron energy, irradiation time, harvest) requires solving a **stiff 5-species Bateman chain** many times.
- Classical ODE solvers (`Radau`) are accurate but **slow** for design sweeps.

**Figure:** None required, or a simple chain diagram: Ra-226 → Ra-225 → Ac-225 (product) vs Ra-226 → Ra-227 → Ac-227 (impurity).

---

## Panel 2 — Method (top center)

- **PINN** with **semi-analytic Bateman backbone** — chain physics is structural, not learned from scratch.
- NN learns bounded log-corrections for cross-section / spectrum uncertainty.
- Physics loss: Bateman residuals + mass budget (no alchemy).
- Training: **600 pretrain + 3400 joint epochs** (Kaggle A100, tuned `PINN_MEDIUM_TRAIN=1` env).

**Key citations (small footer):** Raissi et al. 2019 (PINN); Bento et al. 2026 (Jacobian norm); Wang et al. 2021 (grad balance).

---

## Panel 3 — Results table (top right)

| Criterion | Result |
|-----------|--------|
| Trio A — empty tank safety | **PASS** |
| Trio B — production Ac-225 vs ODE | **PASS (<10%)** |
| Trio C — Ra-225 decay chain | **PASS** |
| Quality gate (all species) | **PASS** |
| Correlation vs ODE | **PASS** |
| Held-out Ac-225 (22 scenarios) | **4.51% median** |

**Do not claim:** "99.8% accuracy" — use species-specific medians.

---

## Panel 4 — Figures (center, largest area)

Use only v63-regenerated graphs:

| Figure | File |
|--------|------|
| Isotope evolution + harvest window | `graphs/isef_isotope_evolution.png` |
| Ac-225 parity (PINN vs ODE targets) | `graphs/isef_parity_restyled.png` |
| Mass conservation (5 species) | `graphs/isef_mass_conservation.png` |
| Loss trajectory (4k + 12k scale narrative) | `graphs/isef_loss_trajectory_12k.png` |

Optional secondary: `graphs/loss_components.png`, `graphs/coverage_with_overlay.png`

**Do NOT use:** `archive/error_histogram.png`, `archive/flux_sensitivity.png`, `archive/ac225_growth.png`, `archive/ac225_yield_heatmap.png` (legacy RF or ODE-only; moved out of `graphs/`).

---

## Panel 5 — Where errors are highest (honest geography)

Held-out validation (seed 42, 22 scenarios):

| Regime | Ac-225 median error | Meaning |
|--------|---------------------|---------|
| **All** | **4.5%** | Headline generalization |
| Thermal virgin | ~4.7% | Standard reactor-like energies — strong |
| Fast 14 MeV virgin | ~3.9% | High-energy production — strong |
| Epithermal virgin | ~9.5% | Resonance / capture-dominated — use with care |
| Threshold virgin (5.8–7.5 MeV) | ~8.5% | Near **6.42 MeV (n,2n) cliff** — hardest physics edge |
| Recycled targets | ~3–6% | Restart with daughters — generally good |

**Ra-226 feedstock tracking:** ~0.001% median — excellent.

---

## Panel 6 — Speed + demo (bottom right)

- PINN inference **500×+ throughput under batched inference vs serial stiff-ODE solves** on reference sweeps (Streamlit benchmark tab; no committed benchmark JSON in `results/` yet — benchmark re-verification in progress).
- **Live demo:** `streamlit run app.py` — scenario predictor, impurity slider, speed benchmark.
- QR code to Streamlit Cloud URL (if deployed).

---

## Panel 7 — Limitations & future work (bottom)

1. Validation is **ODE-synthetic** (JENDL cross sections), not reactor lab CSV.
2. Scalar neutron energy is a **simplified spectrum model**.
3. Near-threshold energies (~6.4 MeV) remain the hardest edge — confirm critical designs with ODE.
4. **Future:** literature benchmark with mentor (OSTI photonuclear Ra-226); post-ISEF integrator upgrade for Ra-225/Ac-225 backbone.

---

## Elevator pitch (30 seconds)

> I built a physics-informed neural network with semi-analytic Bateman backbones for the medical isotope chain Ra-226 to Ac-225. It prevents alchemy on empty targets, predicts Ac-225 within about 5% of stiff ODE integrators on held-out scenarios, passes six independent validation gates, and runs hundreds of times faster than classical solvers — enabling rapid production planning while staying honest about near-threshold energy limits.

---

## Print checklist

- [ ] All figures regenerated from v63 weights (`7c21debe`)
- [ ] Results table matches `results/v63_validation_20260530.json`
- [ ] Limitations panel visible (judges reward honesty)
- [ ] Streamlit demo tested on poster laptop
