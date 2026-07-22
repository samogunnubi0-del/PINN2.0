# 🧪 The IsotopePINN Logs: A Comprehensive Research History

*This document is the official development log for Samuel Ogunnubi’s research project. It tracks the evolution of IsotopePINN from an initial, flawed neural network into a state-of-the-art Physics-Informed LSTM aimed at solving the Actinium-225 supply crisis.*

---

## 📅 Spring 2026: Discovering the Actinium-225 Crisis
**The Spark:** Targeted Alpha Therapy (TAT) represents the future of cancer treatment. Actinium-225 releases alpha particles that obliterate cancer cells while sparing healthy tissue. The clinical trials (especially for prostate cancer) are miraculous. 
**The Problem:** The world only produces ~2 Curies of Ac-225 per year—enough for just 1,500 patients. The global demand is projected at 50,000+ patients. 
**The Goal:** Optimize Ac-225 production in nuclear reactors by irradiating Radium-226. However, simulating the reactor irradiation requires solving the **Bateman Equations** for a 5-species decay chain over thousands of hours. Traditional numerical solvers (like implicit Runge-Kutta/Radau5) are too slow for real-time optimization or digital twin applications. I set out to build an AI surrogate that could instantly predict the isotope yields.

---

## 📅 April – Early May 2026: The V1 "Vanilla" Network & The Wall of Stiffness
**Architecture:** Standard Multi-Layer Perceptron (MLP)
**Objective:** Feed time ($t$), flux ($\phi$), and energy ($E$) into a basic deep neural network and map it to 5 isotope concentrations.

> **Log Entry:** *Total failure.* 
> The standard neural network was completely incapable of learning the physics. The reason became mathematically obvious: **Stiffness**.
> 
> In the 5-species chain, we have two competing channels:
> 1. The fast neutron channel: Ra-226 → Ra-225 → **Ac-225** (What we want)
> 2. The thermal channel: Ra-226 → Ra-227 → Ac-227 (The toxic impurity)
> 
> The stiffness ratio is absurd. Ra-226 has a half-life of 1,600 years. Ra-227 has a half-life of 42.2 minutes. That is a stiffness ratio of over **20,000,000 to 1**. The neural network naturally averaged out the fast dynamics, missing the Ra-227 spike entirely. Worse, it hallucinated atoms—predicting negative mass and creating matter out of thin air because it had no concept of physics.

---

## 📅 Late May – Early June 2026: The SciML Breakthrough (PINN v2.0)
**Architecture:** Physics-Informed Neural Network (Differential Form)
**Objective:** Force the network to obey the Bateman ODEs using custom loss functions. 

> **Log Entry:** *The SciML Literature Deep Dive.*
> I spent weeks reading cutting-edge Scientific Machine Learning papers to fix the V1 failures. I implemented an arsenal of techniques to build IsotopePINN v2.0:
> 
> - **Fourier Features (Tancik et al.):** The (n,2n) reaction only happens above a 6.42 MeV energy threshold. Neural networks suffer from "spectral bias" and can't learn sharp cutoffs. Passing the energy input through sine/cosine functions fixed this instantly.
> - **Adaptive Activations (Jagtap et al.):** To handle the stiffness, I replaced standard `tanh` with `tanh(a*x)` where `a` is a learnable parameter. This allowed the early layers to tune their sensitivity to catch the 42-minute Ra-227 spike.
> - **Jacobian Normalization (Bento et al.):** *This was a massive breakthrough.* Because Ac-225 exists in trace amounts compared to the massive Ra-227 decay rate, the loss function was ignoring Ac-225. By dividing each physics residual by the row-norm of the analytical Jacobian matrix, I gave the trace species equal weighting in the loss landscape.
> - **MC Dropout (Gal & Ghahramani):** Added dropout during inference to generate confidence intervals (Uncertainty Quantification).
> 
> **Results:** We hit ~4.5% mean relative error against the `scipy` Radau5 numerical solver, achieving a 1,000x speedup. I deployed the model to a live Streamlit dashboard. It worked. But it still wasn't perfect—extreme neutron flux caused Ra-227 to overshoot its bounds.

---

## 📅 June 20, 2026 (1:50 PM - 3:00 PM): The NCSU ARTISANS Lab Pitch
**Event:** Technical Review meeting with Jaden Palmer (PhD Researcher, NC State University)
**Objective:** Secure domain-expert feedback and an ISEF "Qualified Scientist" signature (Form 2).

> **Log Entry:** *The Expert Stress Test.*
> I pitched IsotopePINN to Jaden. He validated the core concept but pointed out critical blind spots in the methodology—exactly what I needed for ISEF.
> 
> **Jaden's Critiques:**
> 1. **MC Dropout is fundamentally flawed for physics.** He noted that randomly dropping neurons destroys the physical consistency of the PINN and is no longer considered credible for epistemic uncertainty. 
> 2. **The Differential Loss is causing the overshoots.** I was penalizing the derivative ($dN/dt$). Jaden explained that derivatives amplify noise and stiffness. He suggested an **Integrated Loss Function**.
> 3. **The Baseline:** I need to prove the PINN is better than standard time-series AI. He suggested comparing it to an LSTM.
> 4. **Validation:** He pushed for real empirical data or analytical benchmarks rather than just comparing to Radau5.
> 
> *Result:* Jaden agreed to be my Qualified Scientist for ISEF. He sent follow-up literature from IBM and ScienceDirect on LSTMs and integrated physics losses.

---

## 📅 June 21, 2026: The Pivot to PI-LSTM (v3.0)
**Architecture:** Physics-Informed Long Short-Term Memory (PI-LSTM)
**Objective:** Integrate Jaden's feedback to build a publication-ready, ISEF-winning model.

> **Log Entry:** *Rebuilding the Engine.*
> Based on yesterday's meeting, the entire architecture is pivoting. 
> 
> **The New Plan:**
> - **The Backbone:** We are ripping out the standard Multi-Layer Perceptron and replacing it with an **LSTM**. LSTMs are natively designed for temporal forecasting. A Physics-Informed LSTM will combine the memory cells of an LSTM with the mass-conservation constraints of a PINN.
> - **The Loss Function:** We are abandoning the differential form. I will rewrite the loss function using the **Trapezoidal Rule**. Instead of penalizing instantaneous slope errors, the model will enforce that $Mass(t) = Mass(0) + \int(Rates)$. This mathematical smoothing will kill the stiffness overshoots entirely.
> - **Uncertainty Quantification:** Replacing MC Dropout with **Conformal Prediction** to generate mathematically guaranteed confidence intervals.
> - **Empirical Validation:** The hunt begins for real-world Ac-225 reactor yield data.
> 
> *Next step: Build the PI-LSTM architecture in code.*

---

## 📅 June 30, 2026: PI-LSTM Integrated (v3 Fork — v2 Frozen)

**Architecture:** Physics-Informed LSTM with integrated trapezoidal physics loss only  
**Location:** `New folder/v3_pilstm/` (v2 `pinn_model.py` / `train.py` untouched)

> **Log Entry:** *Parallel track deployed.*
> 
> - **v2 frozen** as production baseline (6/6 validation, ~4.5% Ac-225 held-out). See `docs/V2_FROZEN.md`.
> - **PI-LSTM v3** implements Jaden's integrated-loss recommendation using **Nasiri & Dargazany (2022) Reduced-PINN** ([arXiv:2208.12045](https://arxiv.org/abs/2208.12045)) — trapezoidal integral constraints instead of `dN/dt` residuals.
> - **Carried forward:** Fourier energy encoding (Tancik), Jacobian normalization (Bento), shared Bateman RHS from `ra226_ac225_transmutation.py`.
> - **Training:** `v3_pilstm/train_pi_lstm.py` + Colab notebook `v3_pilstm/PI_LSTM_Colab_Run.ipynb` (4000-epoch GPU recipe).
> - **Comparison:** `v3_pilstm/analysis/compare_models.py` — v2 MLP-PINN vs PI-LSTM on 22 held-out ODE scenarios.
> 
> **Local smoke train (80 epochs, CPU):** infrastructure verified; full GPU training required for ISEF-grade metrics.
> 
> | Metric | v2 (frozen) | PI-LSTM (80-epoch local) |
> |--------|-------------|----------------------------|
> | Ac-225 median rel error | ~4.5% | ~48% (improves with Colab 4k) |
> | 6/6 validation | PASS | N/A (separate harness) |
> | Inference | ~11 ms/scenario | ~2 ms/scenario |
> 
> *Next step: Run full `PILSTM_EPOCHS=4000` on Colab; update comparison JSON for poster.*

---

## 📅 July 1–3, 2026: External Validation, Literature Benchmarks & v3.1 Pipeline

**Objective:** Answer Jaden’s push for empirical validation; build an honest external benchmark catalog; upgrade PI-LSTM training so held-out metrics match v2.

> **Log Entry:** *The literature hunt.*
> 
> We searched peer-reviewed sources for real `(φ, E, t) → Ac-225` reactor time-series to train on. **Finding:** Real Ac-225 data exists widely (cyclotrons, linacs, Th-229 generators, HFIR thermal reactor) but **no public multi-point fast-reactor 226Ra(n,2n) trajectories** exist in open literature. Joyo fast-reactor experimental campaigns are planned post-2026.
> 
> **Built `data/literature_benchmarks.csv` (17 rows)** with tiered `source_type` labels:
> 
> | Tier | Examples | Use |
> |------|----------|-----|
> | T1 empirical | Hogle HFIR 2016, Kuznetsov SM 2014, Matyskin BNR 2024 | Real measurements (often wrong channel: n,γ or γ,n) |
> | T2 simulation | Sasaki 2023 Joyo, Iwahashi 2022 ORIGEN | Order-of-magnitude fast-reactor checks |
> | T3 cross-route | Morgenstern cyclotron, Melville linac, O’Connor 1960 σ | Context only — not (n,2n) ODE validation |
> | T4 decay-leg | McDevitt ITU, ORNL 229Th generator | Bateman ingrowth sanity check |
> 
> **Honest ISEF framing:** Train on physics-consistent Radau5 ODE trajectories; validate externally against published endpoints where they exist. Only **3 of 17 rows** are neutron-comparable for Ac-225 scoring; **14 are structurally non-comparable** (wrong reaction, cross-route, or no activity).
> 
> **Supporting artifacts:**
> - `data/literature_benchmarks_README.md` — tier guide and unit conversions
> - `v3_pilstm/analysis/validate_empirical.py` — scores v2 + PI-LSTM vs CSV
> - `analysis/build_benchmark_progress.py` → unified `analysis/benchmark_progress.md`
> - `v3_pilstm/analysis/joyo_sigma_calibration.py` — non-destructive σ(n,2n) scale fit to Joyo sims (default ODE ~1429% MAPE → calibrated ~20% at ~1.44 mb effective σ; **does not change v2 or training ODE**)

---

## 📅 July 3, 2026: PI-LSTM v3.1 Training Overhaul

**Architecture:** Same PI-LSTM + integrated trapezoidal loss; **training recipe** rebuilt to match v2 rigor and fix measurement bugs.

> **Log Entry:** *Why the first Colab run looked worse than it was.*
> 
> The original v3 Colab recipe (4000 epochs, 200 scenarios, endpoint-only checkpoint) produced confusing metrics: **val Ac-225 ~0.8%** but **compare_models ~46%** on held-out. Root cause: best weights were saved on **last-timestep Ac-225 only**, while comparison scored full trajectories on a **different random held-out set**.
> 
> **v3.1 upgrades implemented** (`v3_pilstm/train_pi_lstm.py`, `data/trajectory_dataset.py`, `physics/integrated_loss.py`, `physics/distill.py`, `models/pi_lstm.py`):
> 
> | Change | Purpose |
> |--------|---------|
> | Full-trajectory eval + checkpoint | Save best weights on all-timestep Ac-225 median (fixes val vs compare gap) |
> | Canonical held-out scenarios | Fixed 22-scenario val/test shared with `compare_models.py` |
> | Ra-227 overshoot penalty | Physics-based cap on impurity channel (old run: **583×** overshoot vs v2 **0×**) |
> | 800 structured training scenarios | Virgin / threshold / high-flux mix + log-spaced **50-step** grid |
> | Hard initial condition ansatz | Force N(t=0) = IC exactly (Lagaris trial-solution trick) |
> | v2 distillation teacher | Frozen v2 MLP-PINN guides trajectory shape (`PILSTM_DISTILL=1`) |
> | Stronger physics/mass weights | 25 / 10 (was 10 / 5) |
> | Causal time weighting | Early-time physics emphasis (Wang et al.) |
> | Two-phase pretrain | 15% physics-first epochs |
> | Bigger model | 256 hidden, 8 Fourier bands (config saved in checkpoint) |
> | L-BFGS polish | 60-step fine-tune after best Adam checkpoint |
> | Auto poster graphs | `scripts/plot_v2_vs_pilstm.py` → `graphs/v3_*.png`; Colab Cell 8 zips them |
> | 227Ac impurity rows | Hogle + Kuznetsov in CSV; validate (n,γ) channel separately |
> | Legacy weight loader | Old 128/4 Colab `.pth` files still load for comparison |
> 
> **Colab notebook updated:** `PI_LSTM_Colab_Run.ipynb` — Cell 5 v3.1 env vars, Cell 7 validation + graphs, Cell 8 auto-download zip.

---

## 📅 July 3, 2026: Colab Training Runs

### Run A — v3.0 recipe (4000 epochs, pre-v3.1 code)

| Metric | Value |
|--------|-------|
| Best epoch | 1424 |
| Val Ac-225 median | 0.79% |
| Test Ac-225 median | **19.7%** |
| Training time | ~10 min (T4 GPU) |
| Weights restored locally | `v3_pilstm/weights/pi_lstm_best.pth` |

**Held-out compare (22 scenarios, old weights + old compare script):** PI-LSTM **46%** Ac-225 vs v2 **4.3%**; Ra-227 overshoot **583×** vs v2 **0×**. Inference **~4 ms** vs v2 **~12 ms**.

> *Verdict:* Infrastructure worked; metrics not poster-ready. Motivated v3.1 overhaul.

### Run B — v3.1 recipe (6000 epochs, in progress on Colab)

**Env:** 800 train scenarios, 50 log steps, 256/8 model, distillation, hard IC, overshoot penalty, float64.

| Epoch | Val Ac-225 (full traj) | Best saved |
|-------|------------------------|------------|
| 1 | 710,503% | — |
| 300 | 6.4% | **0.95%** @266 |
| 600 | 12.4% | **0.63%** @599 |
| 900 | 2.1% | **0.53%** @840 |

Loss at epoch 900: total **82.6**, data **3.26**, physics **~0.05** (Bateman constraints satisfied).

> *Preliminary verdict:* Validation Ac-225 **best ~0.53%** — promising vs v2 **4.51%**. **Final poster numbers pending** Cell 6 `compare_models.py` on test split + L-BFGS polish at end of 6000 epochs.
> 
> *Update this row when Run B completes.*

---

## 📅 July 3, 2026 (Evening): Checkpoint Crisis & v3 Run C Overhaul

**Trigger:** Local restore of `PI_LSTM_Results.zip` after Colab Run B  
**Objective:** Reconcile zip JSON (~1.75% test Ac-225) with local `compare_models.py` (~74% endpoint error)

> **Log Entry:** *The zip lied — but the weights were honest.*
> 
> The downloaded results archive reported strong test metrics, yet loading `pi_lstm_best.pth` locally and running the held-out harness produced catastrophic endpoint errors. Investigation traced three compounding failures — not a single bad hyperparameter.

### Mistakes / Lessons Learned (Checkpoint Crisis)

| Issue | What happened | Lesson |
|-------|---------------|--------|
| **Legacy weights in zip** | `PI_LSTM_Results.zip` claimed **~1.75% test Ac-225**, but the bundled `.pth` was a **legacy 128/4 / no hard IC / bare state_dict** checkpoint — not the 256/8 v3.1 model the JSON described | Always verify **architecture + config** match before trusting summary JSON; bare `state_dict` files are ambiguous |
| **Val overlap with training** | Checkpoint validation scenarios **overlapped training seeds** → val Ac-225 looked **optimistically low** (~0.5–1.8%) while true held-out test stayed high | Val split must be **seed-disjoint** from train; never tune on scenarios the model has seen |
| **Architecture mismatch** | Loading legacy 128/4 weights into the 256/8 + hard-IC model caused **~74% endpoint error locally** vs **~19.6%** in the zip JSON (which was scored under a different eval path) | Mismatch between checkpoint arch and eval script silently destroys trajectory fidelity |
| **Trajectory vs endpoint mismatch** | Best weights were sometimes selected on **endpoint-only** or **val-overlapping** metrics while `compare_models.py` scores **full trajectories** on a **fixed 22-scenario held-out set** | One number (endpoint median) ≠ another (trajectory median); align checkpoint metric, val split, and compare harness |

> *Verdict:* Run B numbers were **not poster-ready** until checkpoint selection, seed splits, and config serialization were fixed. Legacy Colab weights are **incompatible** with the current codebase until a full retrain on Run C.

---

### v3 Run C Improvements (July 2026)

**Architecture:** PI-LSTM + integrated trapezoidal loss; **training recipe Run C** — time Fourier, disjoint seeds, endpoint checkpoint, distill taper

> **Log Entry:** *Rebuild the save contract.*
> 
> Run C closes every gap exposed by the checkpoint crisis. Code changes span model, data, training, and Colab packaging.

**Model (`models/pi_lstm.py`):**
- **Time Fourier encoder** — `PILSTM_TIME_FOURIER=16` sinusoidal bands on normalized time (mirrors energy Fourier; helps sharp ingrowth transients)

**Data (`data/trajectory_dataset.py`):**
- Scenario mix: **recycled_trace**, **empty_feed**, **high_flux** with explicit weighting
- **Hybrid `t_end`** sampling (short + long irradiations)
- **`n_train=1400`**, **`n_steps=64`** log-spaced grid (denser temporal resolution)

**Training (`train_pi_lstm.py`):**
- **Disjoint seeds:** val seed **2025**, test seed **2024** (no train overlap)
- Checkpoint metric: **`endpoint_ac225`** median on val (aligned with poster headline number)
- **Distill schedule:** teacher weight ramps to **0** over epochs **60–100%** (late training physics-native)
- **Epoch budget:** **6000** default (not 10k) — see plateau evidence below; quick pipeline test **3000** (~1 hr T4)
- Saves **`config` + `state_dict`** in every checkpoint (hidden size, Fourier bands, hard IC flag — no more ambiguous `.pth`)

**Physics (`physics/endpoint_project.py`):**
- Optional **Newton projection** onto Bateman endpoint constraints at inference

**Colab / docs:**
- `PI_LSTM_Colab_Run.ipynb` — **Run C recipe** (env vars, Cells 5–8)
- `README.md` — Run C env reference and success criteria
- `build_colab_zip.py` — packages config-aware weights + results JSON

> **Compatibility note:** All pre–Run C Colab weights (128/4, bare state_dict) are **legacy-incompatible** until a fresh Colab retrain completes.

---

### Why Run C Works (Design Rationale)

- **Endpoint checkpoint** — optimizes the metric we report on held-out scenarios (Ac-225 endpoint median), not a proxy that diverges from `compare_models.py`
- **Seed-disjoint val/test** — eliminates optimistic val from train-scenario leakage
- **Time Fourier** — reduces spectral bias on fast Ra-227 / Ac-225 transients across 64-step grids
- **Scenario mix + weighting** — recycled_trace / empty_feed / high_flux coverage matches real reactor operating modes
- **Distill taper (60→100% epochs)** — v2 teacher shapes early trajectory; physics + data loss dominate convergence
- **6000-epoch budget (not 10k)** — no early stopping; best checkpoint saved by endpoint Ac-225. Prior runs plateaued well before max: Run A best @ **1424**/4000; Run B val best @ **~840**/6000; completed 6k Colab run best @ **4886**/6000 (`graph_manifest.json`, ~54 min lighter config). Extra epochs only burn GPU after plateau; 10k adds ~67% runtime with diminishing returns. Distill taper math at 6000: full weight until epoch **3600**, linear ramp to **0** at epoch 6000 (2400 taper epochs). At 3000 quick-test: taper epochs 1800–3000 still reaches zero.
- **Hard IC ansatz** — exact `N(t=0)` removes spurious mass at t=0 that poisoned integrated loss
- **Config in checkpoint** — loader instantiates the correct arch; no silent 128/4 vs 256/8 mismatch

---

## 📊 Current Model Comparison (v2 frozen vs PI-LSTM)

| Metric | v2 MLP-PINN (frozen) | PI-LSTM v3 Run A (4k) | PI-LSTM v3.1 Run B (@ ep 900) |
|--------|----------------------|------------------------|-------------------------------|
| **6/6 validation** | **PASS** | N/A | TBD |
| **Ac-225 vs ODE** | **4.51%** median | 19.7% test / 46% compare | **~0.53% val best** (prelim) |
| **Ra-227 overshoot** | **0×** | 583× | TBD (penalty added) |
| **Inference** | ~11 ms | ~4 ms | ~4 ms (est.) |
| **Literature Joyo sim** | ~2400% fail | closer on endpoints | TBD |
| **Poster-ready today** | **Yes** | No | After Run B + Cell 6–8 |

---

## 📊 Literature External Validation (summary)

| Category | Count | v2 | PI-LSTM (Run A) |
|----------|------:|----|-----------------|
| Neutron-comparable Ac-225 rows | 3 | All fail (~1400% MAPE) | All fail (Joyo sim gap in **ODE**, not just PINN) |
| Reference-only (cross-route, decay) | 7 | Listed, not scored | Listed |
| Skipped (σ-only, wrong isotope, no activity) | 7 | — | — |

**Key insight:** Both models track the same Bateman ODE. Joyo simulation endpoints (15–30 GBq) sit **~10–25× below** our default ODE at Joyo conditions — a **cross-section / neutronics normalization gap**, not a neural-network regression alone. O’Connor 1960 anchors σ(n,2n) = 1.60 b @ 14.5 MeV; Joyo-calibrated effective σ ≈ 1.44 mb explains the discrepancy for poster discussion.

**Poster sentence:**
> *We validate self-consistency against Radau5 on held-out scenarios, then compare endpoint activities to peer-reviewed reactor simulations, cross-route accelerators, and decay-chain data — labeled honestly by production route.*

---

## 📁 Key File Locations (July 2026)

```
New folder/
  ISEF_Planning/project_history.md          ← this log
  docs/V2_FROZEN.md                         ← v2 freeze contract
  data/literature_benchmarks.csv            ← 17 external anchors
  analysis/benchmark_progress.md            ← unified v2 vs v3 vs literature
  graphs/v3_*.png                           ← auto-generated poster figures
  v3_pilstm/
    train_pi_lstm.py                        ← v3.1 training
    PI_LSTM_Colab_Run.ipynb                 ← GPU recipe
    weights/pi_lstm_best.pth                ← latest weights (Run A restored; Run B overwrites)
    results/train_summary.json
    results/compare_v2_pilstm.json
    results/empirical_validation.json
    results/joyo_sigma_calibration.json
    scripts/plot_v2_vs_pilstm.py
```

---

## 🔜 Next Steps

1. **Colab Run C (v3 Run C recipe):**
   - Upload fresh **`IsotopePINN_Project.zip`** to Colab (includes Run C code)
   - Set Run C env vars from `PI_LSTM_Colab_Run.ipynb` Cell 5 (`PILSTM_TIME_FOURIER=16`, val seed 2025, test seed 2024, `n_train=1400`, `n_steps=64`, distill taper, etc.)
   - **Quick test:** `PILSTM_EPOCHS=2500–3000` (~1 hr on T4) to verify pipeline before full 6k run
   - **Success criterion:** held-out **endpoint Ac-225 median < 4.5%** on test split (Cell 6 `compare_models.py`)
   - Run Cells 6–8; download config-aware `PI_LSTM_Results.zip`
2. **Update this log** with Run C final test Ac-225, `compare_v2_pilstm.json`, and Ra-227 overshoot.
3. **Poster:** Lead with v2 (6/6, 4.5%); PI-LSTM Run C as expert-feedback iteration (integrated loss + distillation + time Fourier); literature table as honest external context.
4. **Watch Joyo 2026** for first published fast-reactor Ra-226 → Ac-225 experimental data.

