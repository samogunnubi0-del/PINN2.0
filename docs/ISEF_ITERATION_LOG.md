# ISEF Improvement Iteration Log

> **Architecture narrative:** For a full explanation of why the old integral-only PINN failed, what changed in Iterations 5–6, and why it works, see [`PINN_ARCHITECTURE_FIX.md`](PINN_ARCHITECTURE_FIX.md).  
> **Claims checklist:** [`ISEF_CLAIMS.md`](ISEF_CLAIMS.md) · **Machine-readable log:** [`results/isef_iteration_log.json`](../results/isef_iteration_log.json)

---

## Research literature used (complete bibliography)

These are **published methods we adapted** — not the project’s core novelty claim. Our novelty is the **Ra-226 → Ac-225 five-species application**, **semi-analytic Bateman backbone**, and **parametric surrogate** (flux, energy, time, ICs), validated against stiff ODE reference.

| Rank | Reference | What we used it for | Where in code |
|------|-----------|---------------------|---------------|
| 1 | **Raissi, Perdikaris & Karniadakis (2019)** J. Comput. Phys. 378, 686–707 | Core PINN: physics + data loss | `compute_physics_loss`, `train.py` |
| 2 | **Project design (Iterations 5–6)** — see [`PINN_ARCHITECTURE_FIX.md`](PINN_ARCHITECTURE_FIX.md) | Semi-analytic Bateman backbone + bounded correction | `pinn_model.py` `forward_raw` |
| 3 | **Bento et al. (2026)** [arXiv:2602.21988](https://arxiv.org/abs/2602.21988) | Jacobian normalization of Bateman residuals (trace-daughter zero-collapse fix) | `compute_physics_loss` (comment + `fn / jn`) |
| 4 | **Wang, Teng & Perdikaris (2021)** SIAM J. Sci. Comput. 43(5) A3055–A3081 | Supervised vs unsupervised gradient balancing | `PINN_GRAD_BALANCE=1` in `train.py` |
| 5 | **Krishnapriyan et al. (2021)** NeurIPS, arXiv:2109.01050 | Time-bin curriculum along `t_nn` | `PINN_CURRIC_BINS`, `compose_physics_point_weights` |
| 6 | **Wang et al. (2022)** arXiv:2203.07404 | Causal / early-time physics emphasis | `PINN_CAUSAL_PHYS` |
| 7 | **McClenny & Braga-Neto (2023)** J. Comput. Phys. 474, 111722 | Self-adaptive focal physics weights | `PINN_SA_PHYSICS` |
| 8 | **Tancik et al. (2020)** NeurIPS | Fourier features on log-energy (6.4 MeV threshold) | `PINN_ENERGY_FOURIER` |
| 9 | **Jagtap & Karniadakis (2020)** Commun. Comput. Phys. 28(5), 2002–2041 | XPINN-style expert agreement at threshold | Regime-gated dual heads |
| 10 | **Lagaris et al. (1998)** IEEE Trans. Neural Networks 9(5), 987–1000 | Exponential ansatz for Ra-226 | `exp(−∫k dt)` branch |
| 11 | **He et al. (2016)** CVPR | Residual MLP blocks | `res_blocks` in `pinn_model.py` |
| 12 | **Jagtap, Kawaguchi & Karniadakis (2020)** J. Comput. Phys. 404, 109136 | Learnable activation slopes | `tanh(a·x)` layers |
| 13 | **Loshchilov & Hutter (2017)** ICLR | Cosine LR warm restarts | `CosineAnnealingWarmRestarts` |
| 14 | **Kendall, Gal & Cipolla (2018)** CVPR | Optional uncertainty loss weighting | `UncertaintyWeighter` |
| 15 | **Gal & Ghahramani (2016)** ICML | MC dropout uncertainty | `predict_mcd` (**v2 only**; deprecated in v3) |
| 16 | **Jordan et al. (2024)** — Muon optimizer | Optional alternative to Adam | `MuonOptimizer` (`PINN_MUON=1`) |
| 17 | **Nasiri & Dargazany (2022)** arXiv:2208.12045 | Integrated trapezoidal physics loss (Reduced-PINN) | `v3_pilstm/physics/integrated_loss.py` |

**Non-ML references:** NNDC half-lives/cross sections; classical Bateman/transmutation theory; scipy **`Radau`** stiff ODE as training reference (`ra226_ac225_transmutation.py`).

**Related prior art (not our code — cite for honesty on poster):**

| Work | Overlap | How we differ |
|------|---------|---------------|
| [Pompe/Pacifico — PINN nuclear decay equation](https://github.com/mihapompe/PINN-Nuclear-decay-equation) | PINN + math-informed architecture for `dN/dt = AN` | We target **5-species Ra→Ac clinical chain** with **(flux, E, t, IC)** inputs, not time-only decay matrices |
| [Boiger et al., PINN-PAD Padova 2024](https://pinn-pad.dicea.unipd.it/presentations/boiger.pdf) | Bateman PINN vs CRAM on ²⁴¹Pu | We build a **fast parametric surrogate** for Ac-225 planning, not a single-isotope decay benchmark |
| Ac-225 production papers (photonuclear, reactor (n,2n)) | Same physics problem | Traditional ODE/MC — **no published PINN triage dashboard** found for this exact use case |

**Novelty framing for judges:** synthesis of published PINN tools + **project-specific Bateman backbone** + **medical isotope application** — not “first PINN in nuclear physics.”

## Epoch budget reference

| Profile | Pretrain | Joint | Total | vs original 12k |
|---------|----------|-------|-------|-----------------|
| **Original (Max-Fix v2)** | 2000 | 10000 | **12000** | 100% |
| **Kaggle CPU fallback** | 500 | 2000 | **2500** | 21% (auto if CPU, no PINN_EPOCHS) |
| **PINN_QUICK_TRAIN** | 120 | 400 | **520** | debug only |
| **PINN_MEDIUM_TRAIN (ISEF)** | 600 | 3400 | **4000** | **33%** — recommended |
| Last completed Kaggle run | ~2000 | ~2036 | **~4036** | interrupted / partial |

Activate short profile: `PINN_MEDIUM_TRAIN=1` (sets 600+3400 unless overridden).

---

## Iteration 1 — species_drift + daughter scales (2026-05-23)

**Diagnosis:** Ra-225/Ac-225 predicted ~0. Root causes:
1. `species_drift_loss` penalized virgin IC daughters toward zero (mass_weight × 0.1).
2. Learnable `daughter_rate_log_scales` suppressed Ra-225 (mult ≈ 0.21).
3. Negative Ac-225 integral clamped to 0 at inference.

**Changes (`pinn_model.py`):**
- Mask `species_drift_loss` to nonzero-IC channels only.
- Clamp daughter multipliers to ≥ 1.0 (no suppression).
- Raise Ra-225 dyn_scale floor to 1e-2.

**Pre-metrics (old weights):**
| Criterion | Result |
|-----------|--------|
| Trio A | PASS |
| Trio B Ac-225 vs ODE | ~100% error |
| Trio C Ac-225 | 0 |
| Ac-225 median rel error | 0.37 |
| Quality gate | FAIL |

**Kaggle:** Pushed kernel v59 with fixes; run in progress.

---

## Iteration 2 — PINN_MEDIUM_TRAIN short-epoch profile (2026-05-24)

**Goal:** Reach ISEF quality in **4000 epochs** (not 12k).

**Changes (`train.py` + notebook):**
- `PINN_MEDIUM_TRAIN=1`: 600 pretrain + 3400 joint.
- Scale warmups/curriculum to budget (warmup ~333, virgin ~667, curric ramp ~1200).
- Boost daughter losses: DATA ×1.5, log Ac-225 weight 20, VIRGIN_AC225 ×2.
- Trace-balanced batches: 40% trace rows per chunk.
- Collocation 500 pts/epoch on GPU.
- Oversample virgin fast14 (+900 scenarios); trim low-signal aug.
- Notebook: `PINN_JOINT_CHUNK=0`, `PINN_COS_T0=800`.

**Status:** v60 pushed 2026-05-24; Kaggle **RUNNING** with full MEDIUM profile + iteration 3 virgin ingrowth fix.

---

## Iteration 3 — virgin ingrowth forward fix + empirical sync (2026-05-24)

**Diagnosis:** Signed-only `tanh` derivatives integrate negative for virgin IC → zero daughters after clamp.

**Changes:**
- `pinn_model.py`: virgin feedstock + flux → `softplus(tanh(raw))` for positive daughter ingrowth
- `train.py`: `apply_empirical_flux_jitter()` via `PINN_FLUX_JITTER_SIGMA`
- `scripts/empirical_sync.py` + `data/empirical/*` templates
- `scripts/isef_figures.py` — ISEF publication plots
- `kaggle_sync.py`: empirical push files, pull encoding tolerance, promote step

**v59 sync metrics (weights trained with old forward; new forward over-predicts until v60):**
| Criterion | Result |
|-----------|--------|
| Trio A | PASS |
| Trio B Ac-225 | nonzero but ~8600% vs ODE (needs v60 retrain) |
| Quality gate | FAIL |
| Kaggle pull | 9 files synced, 19 promoted |

**Kaggle:** v60 pushed; RUNNING.

---

## Iteration 4 — blended ingrowth + SAFETY HALT tuning (2026-05-24)

**Diagnosis (v60 COMPLETE, run `e572f3fe`):**
- Training halted early at joint epoch ~1524 via `[SAFETY HALT]` (physics/data > 100:1).
- Full softplus virgin ingrowth caused ~8179% Trio B Ac-225 over-prediction.
- Trio C Ac-225 still 0 — pure Ra-225 decay path lacked positive ingrowth mask.

**Changes:**
- `pinn_model.py`: per-channel blended ingrowth (35% softplus + 65% signed); Ac-225 positive mask when Ra-225 parent present and Ac IC ≈ 0.
- `train.py`: configurable `PINN_SAFETY_RATIO` (default 100; notebook sets 250); `PINN_DATA_WEIGHT` env.
- Notebook: `PINN_SAFETY_RATIO=250`, `PINN_DATA_WEIGHT=15`, `PINN_RESUME=0`.

**v60 COMPLETE metrics (run `e572f3fe-9263-4e67-9116-0996d56dd5d3`, 2026-05-24T00:46 UTC):**
| Criterion | Result |
|-----------|--------|
| Trio A | PASS |
| Trio B Ac-225 vs ODE | ~8179% (over-predict) |
| Trio C Ac-225 | 0 (FAIL) |
| Ac-225 median rel error | 0.37 |
| Quality gate | FAIL |
| Correlation R² | FAIL (Ra-225, Ac-225) |
| Kaggle pull | 10 synced, 19 promoted |
| Joint epochs completed | ~1524 / 3400 (SAFETY HALT) |

**Graphs regenerated locally (2026-05-24):** `isef_*`, `loss_components`, `production_curves`, `analysis/figs/pred_vs_true_*` — provenance in `results/graph_manifest.json`.

**Kaggle:** v61 push with iteration 4 fixes.

---

## Iteration 5 — Bateman backbone + log loss rebalance (2026-05-24)

**Diagnosis:** Integral tanh forward cannot encode parent→daughter Bateman chain; quality gate stuck at 37%.

**Changes:**
- `pinn_model.py`: `_integrate_bateman_ra225_ac225()` semi-analytic backbone + bounded NN correction (`PINN_BATEMAN_BACKBONE=1`)
- `train.py`: Ac-225 DATA weight 500, LOG weight 5, 50% trace batches, `PINN_SAFETY_RATIO=0`
- `scripts/isef_figures.py`: CSV dedupe, real parity scatter, guarded 12k projection
- Notebook v62 env: DATA_WEIGHT=25, GRAD_BALANCE=1, SAFETY=0

**v61 baseline (full 4k, run `9ca9c34c`, joint epoch 4000):**
| Criterion | Result |
|-----------|--------|
| Trio A | PASS |
| Trio B (integral forward) | ~2169% |
| Quality gate | FAIL (37%) |

**Iteration 5 forward-only (old weights + Bateman backbone, no retrain):**
| Criterion | Result |
|-----------|--------|
| Trio A | PASS |
| Trio B Ac-225 vs ODE | **PASS (~25%)** |
| Trio C Ac-225 | **PASS (nonzero)** |
| Ac-225 R² | **0.995** |
| Quality gate | FAIL (37% — needs v62 retrain) |

**Kaggle:** v62 pushed for full 4k retrain with Iteration 5 code.

---

## Iteration 6 — Mass budget + stiff Ra-227 Bateman (2026-05-24)

**Diagnosis:** Independent NN corrections caused Trio C alchemy (~4% atom gain); Ra-227 Euler blew up (42 min half-life); ngamma used thermal σ at fast energies.

**Changes:**
- Shared log-correction for Ra-225/Ac-225; zero-flux budget clamp
- Substepped analytic Ra-227/Ac-227 Bateman with 1/v ngamma scaling
- `correlation_check.py`: near-constant Ra-226 uses max rel err
- Parity plot `energy` column alias

**v62 full 4k + Iteration 6 forward (no new epochs):**
| Criterion | Result |
|-----------|--------|
| Trio A/B/C | **PASS** |
| Quality gate | **PASS** |
| Correlation | **PASS** |
| Held-out Ac-225 | **3.7% median PASS** |
| **Overall** | **6/6** |

---

## Iteration v63 — Full retrain with Iteration 5+6 architecture (2026-05-24)

**Diagnosis:** Iteration 6 forward guards achieved 6/6 on v62 weights without new epochs; v63 confirms the NN learns corrections under the final architecture end-to-end.

**Changes:**
- Fresh `PINN_RESUME=0` Kaggle run with Bateman backbone + 227 substep + 1/v ngamma
- Official weights: `weights/pinn_best_weights.pth` (`sha256=7c21debe…`)

**v63 metrics (600+3400, joint epoch 4000):**
| Criterion | Result |
|-----------|--------|
| Trio A/B/C | **PASS** |
| Quality gate | **PASS** |
| Correlation | **PASS** |
| Held-out Ac-225 median | **~4.5% PASS** |
| **Overall** | **6/6 retrained** |

**Kaggle:** v63 COMPLETE — baseline for poster claims until optional tuned 12k beats it.

---

## Iteration 7 — Parity graph bug (184% vs 6/6 contradiction) (2026-05-29)

**Symptom:** Poster graph `isef_parity_restyled.png` showed **~184% median** Ac-225 error while validation scripts reported **6/6 PASS** and held-out **~4.5%**.

**Root cause:** `scripts/isef_figures.py` → `plot_parity_restyled()` built model inputs using CSV **target** columns (`N_Ra226`, `N_Ra225`, `N_Ac225` at time *t*) as if they were **initial conditions**. The PINN was evaluated on the wrong 8-feature vector — apples-to-oranges vs `prepare_training_tensors()` used everywhere else.

**Fix:**
- Added `_ensure_parity_init_columns()` with `init_N*` defaults from `TRAIN_INIT_*`
- Parity path now calls `prepare_training_tensors(sub, …)` — same contract as training/validation
- Median parity after fix: **~5.5%**, aligned with v63 held-out

**Lesson:** Graph scripts must use the **same input API** as `train.py` / `validate_predictor.py`. A passing test suite does not protect against a broken plotting path.

---

## Iteration 8 — Full graph audit + stale figure purge (2026-05-29)

**Goal:** Every poster figure must match current v63 weights and PINN (not ODE-only or legacy ML).

### Bad / misleading graphs (what went wrong)

| Graph / script | Problem | Poster action |
|----------------|---------|---------------|
| `isef_parity_restyled.png` | Wrong IC columns (Iter 7) | **Fixed** — keep on poster |
| `error_histogram.png` | From legacy **RandomForest** `model_trainer.py`, not PINN | **Remove** from poster |
| `flux_sensitivity.png` | Same stale RF pipeline | **Remove** from poster |
| `ac225_growth.png` | ODE-only curve, not PINN | **Remove** (or label “ODE reference only”) |
| `ac225_yield_heatmap.png` | ODE-only grid | **Remove** (or label “ODE reference only”) |
| `isef_mass_conservation.png` | Summed wrong species / incomplete 5-species budget | **Fixed** — 5-species sum |
| `analysis/plot_predictions.py` figures | Log-scale on zero/negative → blank or misleading panels | **Fixed** — log only when positive signal |
| `loss_history.csv` concatenated runs | Fake loss spikes, bad 12k projection | **Fixed** earlier (dedupe); re-validate before plot |
| `graphs/*` from partial / wrong weights | Repo-root 207 KB **3-species** `pinn_best_weights.pth` vs 5-species v63 | **Exclude** duplicate root weights from zip; use `weights/pinn_best_weights.pth` only |

### Code fixes applied

| File | Change |
|------|--------|
| `scripts/isef_figures.py` | Parity IC fix; mass conservation 5-species; loss CSV dedupe |
| `scripts/extra_plots.py` | Inlined coverage plot (removed broken `model_trainer` import); trajectory N227 scale fix |
| `analysis/plot_predictions.py` | Conditional log-scale |
| `train.py` | Truncate/replace `loss_history.csv` policy documented |

**Regenerate after any weight change:** `python scripts/isef_figures.py` + Colab Cell 6 pipeline.

---

## Iteration 9 — Local 12k failure diagnosis + training guardrails (2026-05-29)

**Symptom:** Local untuned `python train.py` (12k, default env) produced **bad weights** (~184% parity, failed Trio tests) while Kaggle **v63 4k** passed 6/6.

**Root causes (documented, not single bug):**

| Mistake | Effect |
|---------|--------|
| Default env vs **`PINN_MEDIUM_TRAIN=1`** tuned recipe | Wrong warmup/loss balance |
| **`PINN_RESUME=1`** + corrupt/truncated `pinn_training_resume.pt` | Resumed from broken state; overwrote good `pinn_best_weights.pth` |
| **float32** on CPU vs **`PINN_FLOAT64=1`** on GPU | Trace inventory numerical drift |
| Rogue **207 KB** 3-species weights at repo root | Graphs/loads picked stale architecture |
| **`PINN_SAFETY_RATIO=100`** default on long runs | Early halt (seen in v60 at ~1524 joint epochs) |

**Fixes:**
- `train.py`: `_resume_checkpoint_is_usable()`, `_backup_existing_checkpoint()`, `best_epoch` / `best_joint_epoch` tracking
- `IsotopePINN_Colab_Run.ipynb`: tuned 12k recipe + graph regen + 6/6 validation cells; commented 4k v63 fallback
- `build_colab_zip.py`: excludes corrupt resume, archive, stale root weights, `.venv`

**Acceptance rule for 12k Colab run:** Keep new weights only if Cell 7 passes **6/6** AND held-out Ac-225 median **beats ~4.5%**; else keep v63 4k weights.

---

## Iteration 10 — Literature review + Bento (2026) documentation (2026-05-30)

**Action:** Online prior-art search + full codebase citation audit.

**Findings recorded in this log (Research literature section above):**
- PINN for Bateman/decay chains **exists** (Pompe, Padova 2024) — we are **not** first globally
- **No match found** for published PINN **Ra-226 → Ac-225 parametric surrogate** with clinical triage app
- **Bento et al. arXiv:2602.21988 (Feb 2026)** was already in `compute_physics_loss` comments but **missing** from `pinn_model.py` header bibliography — now listed here and in JSON `research_refs`

**Poster honesty:** Cite Bento 2026 for Jacobian normalization; cite Raissi 2019 for PINN; cite Pompe/Padova as related work; claim **application + architecture synthesis**, not invention of PINNs.

---

## Graph & Data Log Sheet (expanded)

| Iteration | Graph / artifact | Symptom (bad) | Root cause | Fix | Poster? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| v1.0 | `isef_parity_*` | Flat at zero | Daughters untrained | drift mask, scale clamp | — |
| v2.0 | `isef_loss_trajectory_12k` | N/A (new) | — | loss trajectory module | **Keep** |
| v3.0 | `isef_isotope_evolution` | Wrong daughter track | Integral forward | virgin ingrowth patch (later superseded) | **Keep** (regen) |
| v4.0 | all `isef_*` | Metrics vs graphs mismatch | v60 partial run + wrong forward | regen from run `e572f3fe` | regen |
| v4.1 | `loss_components.png` | Fake spikes | Concatenated `loss_history.csv` | dedupe in `prepare_loss_history_df_for_plot` | **Keep** |
| v5.0 | parity scatter | Still wrong inputs in some paths | Used `energy` vs `energy_ev` alias | column alias + Bateman retrain | regen |
| v6.0 | all poster graphs | Trio pass but some panels ugly | Forward guards only, no v63 yet | v63 full 4k retrain | **Keep** |
| **v7.0** | **`isef_parity_restyled.png`** | **~184% median** vs **~4.5% held-out** | **Targets used as ICs** | **`prepare_training_tensors` + `init_N*`** | **Keep (fixed)** |
| **v8.0** | `error_histogram.png` | Shows RF errors | Legacy `model_trainer` | Do not regenerate for PINN poster | **Remove** |
| **v8.0** | `flux_sensitivity.png` | RF sensitivity | Legacy pipeline | — | **Remove** |
| **v8.0** | `ac225_growth.png`, `ac225_yield_heatmap.png` | Implies PINN | ODE-only scripts | Label ODE-only or remove | **Remove** |
| **v8.0** | `isef_mass_conservation.png` | Wrong budget | Incomplete species sum | 5-species conservation | **Keep** |
| **v8.0** | `pred_vs_true_*.png` | Blank log panels | log(0) | positive-only log scale | optional appendix |
| **v9.0** | any graph from local 12k untuned | Failed Trio / high error | resume + untuned env + float32 | Colab tuned recipe + guardrails | use v63 until 12k passes |

### Graphs approved for poster (v63 + Iter 7–8 fixes)

| Graph | File | Notes |
|-------|------|-------|
| Ac-225 parity | `graphs/isef_parity_restyled.png` | ~5.5% median after IC fix |
| Isotope evolution | `graphs/isef_isotope_evolution.png` | PINN vs ODE overlay |
| Loss trajectory | `graphs/isef_loss_trajectory_12k.png` | Deduped CSV; projection guarded |
| Mass conservation | `graphs/isef_mass_conservation.png` | 5-species budget |
| Loss components | `graphs/loss_components.png` | Data vs physics split |
| Residual histogram | `graphs/pinn_ac225_rel_residual_hist.png` | Ac-225-only residual hist; the generic `isef_residual_histogram.png` was never regenerated from v63 — use this file or regenerate |
| Sensitivity | `graphs/isef_sensitivity_flux.png` | PINN sweep, not RF |

---

## Mistakes & lessons register (all iterations)

| # | Mistake | When | Impact | Fix / prevention |
|---|---------|------|--------|------------------|
| 1 | `species_drift_loss` on virgin daughters | Iter 1 | Ra-225/Ac-225 → 0 | Mask to nonzero IC only |
| 2 | Daughter scale multipliers < 1 | Iter 1 | Suppressed ingrowth | Clamp ≥ 1 |
| 3 | Signed tanh integral, zero IC | Iter 2–4 | Zero-attractor daughters | Bateman backbone (Iter 5) |
| 4 | Full softplus virgin ingrowth | Iter 3 | ~8600% Trio B | Replaced by Bateman |
| 5 | `PINN_SAFETY_RATIO=100` halt | Iter 4 | Training stopped ~1524/3400 | `SAFETY_RATIO=0` for full runs |
| 6 | Integral-only forward (no chain prior) | Iter 1–4 | Quality gate ~37% | Semi-analytic Bateman (Iter 5) |
| 7 | Independent NN corrections | Iter 6 | Trio C alchemy ~4% | Shared log-correction + budget clamp |
| 8 | Thermal σ_γ at fast neutron E | Iter 6 | Ra-227 10⁴× error | 1/v energy scaling in forward |
| 9 | Ra-227 Euler with 9 h steps | Iter 6 | NaN inventories | Substepped analytic 227 chain |
| 10 | Parity plot used `N_*` as IC | Iter 7 | 184% false graph | `init_N*` + `prepare_training_tensors` |
| 11 | Stale RF graphs on poster | Iter 8 | Misrepresents model | Remove from poster list |
| 12 | `PINN_RESUME` + corrupt checkpoint | Iter 9 | Destroyed good weights | Resume guardrails + backup |
| 13 | Untuned 12k local train | Iter 9 | Bad weights vs v63 | Colab tuned env block |
| 14 | Claiming “99.8% accuracy” | ongoing | Overclaim | Species-specific medians only |
| 15 | Missing Bento 2026 in bibliography | Iter 10 | Incomplete citations | Added to this log + JSON |

---

## Iteration 11 — Colab A100 12k sync (2026-05-30)

**Source:** `IsotopePINN_Results (2).zip` from Google Drive Colab run (~1h14m, epochs 12000, best epoch **11380**).

**Synced into repo:** 24 files — `weights/`, `graphs/`, `results/`, `data/`, `analysis/validation/`, `analysis/figs/`.  
**Backup before overwrite:** `archive/sync_backup_20260530_105656/`  
**Sync report:** `results/colab_sync_20260530.json`

**12k metrics (from zip + local re-check on synced weights):**

| Criterion | Result |
|-----------|--------|
| Quality gate | **PASS** |
| Held-out Ac-225 median | **7.27%** |
| Trio A | PASS |
| Trio B Ac-225 vs ODE | **CHECK ~25%** (strict wants &lt;10%) |
| Trio C | PASS |
| Correlation | PASS |

**Decision (acceptance rule):** **Keep v63 as active `pinn_best_weights.pth`** — 12k held-out **7.27%** is worse than v63 **~4.5%**.  
12k weights saved as `weights/pinn_best_weights_12k_colab_20260530.pth` (sha256 `352c4a56…`).  
v63 copy: `weights/pinn_best_weights_v63.pth` (sha256 `77e01981…`).

**Use 12k run for:** loss curves (`loss_history.csv`, `isef_loss_trajectory_12k.png`, `pinn_loss_history.png`) — proves full 12k training completed.  
**Use v63 for:** poster/app inference and held-out Ac-225 headline number.


## Iteration 12 — ISEF ship package (2026-05-30)

**Action:** Local ISEF delivery on true Kaggle v63 weights (`sha256=7c21debe…` from `kaggle_results/weights/`).

**Problem found:** Active copy `pinn_best_weights_v63.pth` (`77e01981…`) failed 6/6 when re-tested — not the documented v63 checkpoint.

**Re-validation (7c21debe weights):**

| Criterion | Result |
|-----------|--------|
| Trio A/B/C | **PASS** (Trio B Ac-225 ~9.9% vs ODE) |
| Quality gate | **PASS** |
| Correlation | **PASS** |
| Held-out Ac-225 median | **4.51% PASS** |
| **Overall** | **6/6** |

**Deliverables:** `docs/ISEF_POSTER_OUTLINE.md`, `docs/ISEF_JUDGE_QA.md`, `results/v63_validation_20260530.json`, `app.py`, rebuilt `IsotopePINN_Project.zip` (excludes 12k ablation weights).

**Verdict:** **ISEF ship-ready on v63 4k.**

---

## Iteration v3 — PI-LSTM integrated fork (2026-06-30)

**Goal:** Implement Jaden Palmer feedback (integrated loss + LSTM backbone) without modifying v2.

**Changes (all under `v3_pilstm/`):**

| File | Role |
|------|------|
| `models/pi_lstm.py` | 2-layer LSTM + Fourier energy encoder |
| `physics/integrated_loss.py` | Trapezoidal Reduced-PINN loss + Jacobian norm |
| `physics/bateman_rhs.py` | Torch Bateman RHS (shared constants with v2) |
| `data/trajectory_dataset.py` | ODE trajectory sequences, scenario-level splits |
| `train_pi_lstm.py` | Training loop (`PILSTM_EPOCHS`, Colab env vars) |
| `analysis/compare_models.py` | v2 frozen vs PI-LSTM held-out comparison |
| `PI_LSTM_Colab_Run.ipynb` | GPU 4000-epoch recipe |
| `app_v3.py` | Streamlit preview (optional) |

**v2 freeze:** `docs/V2_FROZEN.md`, `results/v2_frozen_baseline.json`

**Paper PDF:** `docs/papers/Nasiri_Dargazany_2022_Reduced_PINN.pdf`

**Status:** Code complete; local 80-epoch smoke train verifies pipeline. **Full Colab 4k train** required before claiming PI-LSTM beats v2 on poster.


## ISEF Compliance Checklist (updated 2026-05-30)

- [x] **Strict Physics Enforcement** — Trio A/B/C PASS; Bateman residuals in loss
- [x] **12K Scalability** — `budget_scale` in train.py; `isef_loss_trajectory_12k.png`
- [x] **Data Integrity** — ODE-synthetic + empirical manifest; held-out validation PASS
- [x] **Visual Clarity** — Parity IC fix (Iter 7); stale RF/ODE graphs removed (Iter 8)
- [x] **Full ISEF criteria** — **6/6 PASS** (v63 retrained weights, ~4.5% held-out Ac-225)
- [x] **Transparent failure log** — Mistakes register + graph audit in this file and `results/isef_iteration_log.json`
- [x] **Research citations** — Full bibliography including Bento et al. (2026) arXiv:2602.21988


```python
os.environ["PINN_MEDIUM_TRAIN"] = "1"
os.environ["PINN_FLOAT64"] = "1"
os.environ["PINN_RESUME"] = "0"
os.environ["PINN_COLLOCATION_POINTS"] = "500"
os.environ["PINN_JOINT_CHUNK"] = "0"
os.environ["PINN_CURRIC_RAMP"] = "1200"
os.environ["PINN_COS_T0"] = "800"
os.environ["PINN_SAFETY_RATIO"] = "0"
os.environ["PINN_DATA_WEIGHT"] = "25"
os.environ["PINN_LOG_DATA_WEIGHT"] = "5"
os.environ["PINN_GRAD_BALANCE"] = "1"
os.environ["PINN_BATEMAN_BACKBONE"] = "1"
os.environ["PINN_LOG_EVERY"] = "50"
# Do NOT use PINN_AMP with float64
```
