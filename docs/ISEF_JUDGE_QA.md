# ISEF Judge Q&A — IsotopePINN (v63)

Pre-written answers. Numbers from local re-validation 2026-05-30 (`results/v63_validation_20260530.json`).

---

## "What does your project do?"

I train a **physics-informed neural network** to predict how five linked isotopes (Ra-226, Ra-225, Ac-225, Ra-227, Ac-227) evolve inside a neutron-irradiated target. Ac-225 is the medically valuable product. The PINN replaces thousands of slow ODE solves with one fast forward pass for production planning.

---

## "How accurate is it?"

**Species-specific, not one blanket number:**

| Metric | Value |
|--------|-------|
| Held-out Ac-225 median (22 scenarios) | **4.51%** vs ODE |
| Ra-226 (feedstock) median | **~0.001%** |
| Trio B production scenario | **PASS — Ac-225 within 10%** of ODE |
| Training-set Ac-225 median | **~4.5%** |

We do **not** claim 99.8% overall accuracy. Daughter intermediates and impurity species can be noisier; the product isotope Ac-225 is what we optimize and report.

---

## "Did you validate on real reactor data?"

**No lab CSV yet.** All validation compares the PINN to a **reference ODE** using JENDL Ra-226 cross sections and NNDC half-lives. That proves the surrogate tracks established nuclear physics code — not that we reproduced a specific reactor run.

**Next step (with mentor):** compare against published photonuclear / transmutation literature (e.g. OSTI Ra-226 benchmarks).

**Why this is still valid for ISEF:** We show (1) physics safety, (2) agreement with a trusted integrator on held-out scenarios, (3) transparent failure geography, and (4) a path to experimental comparison.

---

## "Could your model create isotopes from nothing?"

**No.** Trio A tests an empty target with high neutron flux — both PINN and ODE stay at zero inventory. Training includes a **mass budget loss** and the forward pass uses Bateman structure with bounded corrections so net atom creation from vacuum is blocked.

---

## "What scenario represents real production?"

**Trio B:** 250 hours irradiation, flux 1×10¹⁴ n/cm²/s, **14 MeV** neutrons, starting with 1×10²² Ra-226 atoms (full feedstock). Ac-225 PINN vs ODE: **~9.9% relative error — PASS** under our 10% gate.

This is a **screening** scenario for "full tank under fast flux," not a claim about a specific national lab setup.

---

## "Where does it fail or struggle?"

Honest answer — errors are **regime-dependent**:

| Regime | Ac-225 median | Real-life impact |
|--------|---------------|------------------|
| Thermal / fast14 virgin | ~4–5% | Good for typical planning sweeps |
| Epithermal virgin | ~9.5% | Spectrum/resonance simplification hurts |
| Threshold virgin (5.8–7.5 MeV) | ~8.5% | **(n,2n) turns on at 6.422 MeV** — small energy errors swing yield |
| Recycled targets | ~3–6% | Restarted targets with daughters — generally OK |

**Real-life meaning:** A 5% Ac-225 error shifts estimated harvest activity and timing — fine for ranking scenarios, not for regulatory release without chemistry and assay.

For **beam energy optimization near 6.4 MeV**, we tell users to confirm with the ODE reference, not PINN alone.

---

## "Why 4,000 epochs and not 12,000?"

We ran 12k on Colab; held-out Ac-225 **worsened** (7.27%) and Trio B failed strict 10%. The tuned **4k v63** recipe (curriculum ramp, cosine schedule, grad balance) is our **best evidence**. Longer training without matching schedule and checkpoint policy can overfit training loss while hurting generalization.

---

## "How is this different from a normal neural network?"

Three things:

1. **Bateman backbone** — analytic chain structure for Ra-225/Ac-225 and substepped Ra-227/Ac-227.
2. **Physics loss** — Bateman residuals at collocation points, not only CSV fit.
3. **Safety tests** — Trio A/B/C plus held-out ODE scenarios, not random train/test split alone.

---

## "What is a PINN? Has this been done before?"

PINNs (Raissi et al., 2019) embed differential equations in the loss. Applying them to **medical transmutation chains with Bateman backbones for Ac-225 planning** is our contribution angle — not claiming to be the first PINN for Bateman equations globally.

---

## "What would you do with more time?"

1. Upgrade Ra-225/Ac-225 forward integrator from Euler to analytic substeps (like Ra-227).
2. Denser training near the 6.42 MeV threshold.
3. Checkpoint on **held-out** Ac-225, not training loss only.
4. Literature / mentor-led benchmark against published production data.

---

## "Show me it works" (demo script)

```bash
cd "New folder"
streamlit run app.py
```

Tabs: scenario predictor, ODE speed benchmark, harvest / impurity post-processing (sliders — not in PINN loss).

Validation scripts (reproducible):

```bash
python test_single.py
python analysis/validate_predictor.py
python analysis/evaluate_quality_gate.py
python analysis/correlation_check.py
```

Expected: **6/6 PASS**, held-out Ac-225 median **~0.045**.
