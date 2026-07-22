# PI-LSTM Upgrade Log — "Why We Did It" Record

Chronological, dated rationale for every engineering/science upgrade to the
v3 PI-LSTM surrogate (Ra-226 → Ac-225 Bateman transmutation chain). Each
entry states **what** changed, **why** (the problem it solves / the judge
question it answers), and the citation where applicable.

Scope note: all changes are flag-gated — legacy behavior (checkpoints,
numbers, scripts) is preserved behind default settings. No accuracy claim
appears anywhere without a measured, committed artifact behind it.

---

## 2026-07-18 — Sprint 1: reproducibility & physics correctness

### 1. Deterministic seeding everywhere (`PI_LSTM_SEED`, default 42)
- **What:** `v3_pilstm/seed_utils.py` seeds python/numpy/torch, enables
  `torch.use_deterministic_algorithms(True, warn_only=True)`, seeds the
  DataLoader shuffle generator, and records the seed in every
  training/eval summary. Wired into the trainer and all eval scripts.
- **Why:** the readiness report (§2.6) found *no* `torch.manual_seed` anywhere
  in the project — a judge asking "re-run this and get the same number" could
  not be answered. Now every artifact carries its seed. Bit-for-bit
  reproducibility of init + one training step is smoke-verified.

### 2. Exponential-integrator physics loss (`PI_LSTM_LOSS=trap|expmix`)
- **What:** the stiff physics residual is now the *exact* constant-rate
  propagator (closed-form matrix exponential of the two Bateman chains,
  float64, sinhc-stable φ₁/φ₂ kernels) instead of the trapezoidal collocation.
  Per-species scaling uses relative per-interval change, not the Jacobian row
  norm. Legacy trapezoid preserved as `PI_LSTM_LOSS=trap` (default).
- **Why:** on log-spaced grids the final intervals reach ~50 h while Ra-227
  has T½ = 42 min (λΔt ≈ 50) — far outside the trapezoid's stability region —
  and Bento-style Jacobian normalization divided fast-species residuals by
  ~λ, suppressing exactly the stiff channel the loss exists to police.
  Judge question answered: "your physics loss is inconsistent on your own
  grid — how do you know the constraint means anything?" Measured: expmix
  residual on exact Radau trajectories ≤ 1.2e-14 (vs trapezoid up to 8.9e-8);
  propagator matches `scipy.linalg.expm` to 1.1e-13.
- **Citations:** Cox & Matthews 2002 (exponential time differencing);
  Pusa & Leppänen 2010 (CRAM — exact matrix-exponential depletion);
  Nasiri & Dargazany 2022 (integrated/weak-form loss lineage).

### 3. True-1 g inventory, versioned (`SCENARIO_VERSION=v1|v2`)
- **What:** scenario regimes named "virgin" used 6.022e23 atoms (= 226 g,
  one mole) while labeled "1 g". Correct 1 g Ra-226 = 2.664e21 atoms
  (already used in `validate_empirical.py`). Now an explicit, documented
  `inventory_scale`: `v1` = `legacy_226g` (default; reproduces every existing
  checkpoint), `v2` = `true_1g`. Every scenario records the scale used.
- **Why:** readiness §2.6 flagged the 226× mass-unit inconsistency — a
  poster-killing unit error if a judge diffs scenario generation against the
  empirical validation. New runs should use `SCENARIO_VERSION=v2`.

### 4. Trainable vanilla-LSTM baseline (`v3_pilstm/analysis/train_baseline.py`)
- **What:** `baseline_lstm.py` went from a 30-line untrained stub to a real
  baseline: matched parameter budget (measured 847,114 PI-LSTM vs 850,349
  baseline = 0.38% delta, within the required ±10%), same dataset/seed/epoch
  budget, same checkpoint-selection metric, NO physics loss / hard IC /
  distillation. Wired into `compare_models.py`.
- **Why:** readiness §2.4 (P0): "why physics at all?" cannot be answered
  without the matched-budget ablation trained and reported. This is the
  fair-baseline evidence judges demand.

### 5. Conformal at meaningful n (`CONFORMAL_MODE=large`)
- **What:** split conformal now supports ≥100 calibration + ≥100 test
  scenarios from fresh seeded pools (disjoint ids), reporting per-species
  coverage AND median relative interval width. Legacy n=11 mode preserved.
- **Why:** with n_cal = n_test = 11, coverage statements like "90% intervals"
  were vacuous (readiness §2.6) — 11 points cannot distinguish 90% from 70%.

### 6. Honest speed benchmark (`v3_pilstm/scripts/speed_benchmark.py`)
- **What:** measures (a) single-scenario eager latency and (b) batched
  throughput for v2 PINN, v3 PI-LSTM eager, and the Radau ODE reference;
  writes `results/speed_benchmark.json` with machine info and batch sizes.
- **Why:** the README/poster "500× speedup" had no committed evidence
  (readiness §2.5); the existing harness showed 142× as a batching effect
  and PI-LSTM eager *losing* to v2 per scenario. This benchmark reports eager
  numbers even where the LSTM loses — the claim becomes "throughput under
  batching", honestly measured.

---

## 2026-07-18 — Sprint 2: next-tier UQ & training methodology

### 7. Jackknife+ / CV+ conformal (`CONFORMAL_MODE=jackknife|cv+`)
- **What:** new `v3_pilstm/uq/jackknife_plus.py`: exact CV+ formula
  (`cv_plus_intervals`, for future per-fold retrained models) plus the
  frozen-checkpoint variant (`FrozenCVPlus`, K-fold or leave-one-out folds
  with bootstrapped residual quantiles) reporting coverage, median width, and
  interval-width *stability* (fold spread + bootstrap q distribution). Writes
  `conformal_validation_jackknife.json` / `conformal_validation_cvplus.json`.
- **Why:** judges ask "how robust is your uncertainty band to the choice of
  calibration scenarios?" The bootstrap/fold spread answers it directly. The
  module documents exactly which guarantee applies: with a frozen model the
  interval degenerates to split conformal (numerically verified, reported as
  `degenerates_to_split_conformal: true`); the strict 1−2α distribution-free
  jackknife+ guarantee requires per-fold retraining (a Kaggle-scale job).
- **Citation:** Barber, Candès, Ramdas, Tibshirani 2021, "Predictive
  inference with the jackknife+", Ann. Statist. 49(1), arXiv:1905.02928.

### 8. Self-adaptive per-species physics weights (`PI_LSTM_ADAPTIVE_WEIGHTS=1`)
- **What:** `v3_pilstm/physics/weights.py` — EMA-smoothed grad-norm-ratio
  per-species multipliers on the physics loss, refreshed every
  `PI_LSTM_ADAPTIVE_EVERY` epochs (default 5), mean-normalized so the global
  loss scale is unchanged. Weight history recorded in the training summary.
  Default OFF (legacy uniform pooling).
- **Why:** even with the exact propagator, pooled per-species residuals have
  disparate gradient shares; the stiff Ra-227 channel can end up
  under-weighted — the same suppression pathology, one level up. Balancing by
  gradient-norm ratio is the standard remedy. Smoke-verified: weights update
  and are recorded; 2-epoch training runs clean.
- **Citations:** McClenny & Braga-Neto 2023, "Self-Adaptive PINNs" (soft
  attention weighting), arXiv:2009.04544; Wang, Yu & Perdikaris 2022, "When
  and why PINNs fail to train" (NTK/gradient pathologies), arXiv:2007.14527.

### 9. Stiffness curriculum scaffold (`PI_LSTM_CURRICULUM=1` or explicit ladder)
- **What:** λ-rescale ladder (default 100,10,1): stage 1 trains on the
  de-stiffened ODE (all rate constants — reactions and decays — uniformly
  slowed by 1/scale, preserving equilibrium ratios), annealing to full
  stiffness; training set rebuilt per stage, physics loss evaluated at the
  matching rate scale (`rate_scale` threaded through data generation,
  trapezoid, and the exact propagator). Val/test always full-stiffness.
- **Why:** stiff transfer learning — fitting the slow backbone first and
  annealing stiffness in — is a published route to train on systems whose
  fast transients otherwise stall PINN-style optimization. Scaffold +
  smoke-verified (de-stiffened data differs, loss exact at scale, stage
  switch at the right epoch); the full curriculum-vs-baseline comparison is a
  Kaggle run.
- **Citation:** Seiler et al. 2025 (stiff transfer learning for PINNs),
  arXiv:2501.17281.

### 10. Deep-ensemble runner (`v3_pilstm/scripts/run_ensemble.py`)
- **What:** trains K=5 members with different `PI_LSTM_SEED` values
  (sequential subprocesses, temp redirectable paths), evaluates all members
  on the canonical held-out scenarios, and reports endpoint ensemble spread
  (relative std across members, median + p90) plus ensemble-mean error into
  `v3_pilstm/results/ensemble_summary.json`.
- **Why:** a second, independent UQ signal to cross-check the conformal
  intervals — judges trust uncertainty claims more when two different
  mechanisms agree. Script + tiny-K smoke only (K=2 × 2 epochs verified);
  the full K=5 × 6000-epoch ensemble is a Kaggle run.
- **Citation:** Lakshminarayanan, Pritzel & Blundell 2017, "Simple and
  scalable predictive uncertainty estimation using deep ensembles",
  arXiv:1612.01474.

---

## 2026-07-18 — Sprint 3: real nuclear data + the spectrum discovery

### 11. Evaluated data layer (`ODE_DATA_VERSION=v1|v2`, default v1)
- **What:** real fetched nuclear data (JENDL-5 σ(n,2n)(E) & σ(n,γ)(E) tables —
  ENDF/B-VIII.0 verified identical, max dev 0.0 b; EXFOR 21405 (n,2n) point;
  EXFOR 31760 13.8±0.3 b thermal (n,γ); NuDat 3 half-lives) behind a version
  flag with cached, fail-loud loaders (`data/evaluated/`). Legacy synthetic
  sigmoid preserved bit-identically as v1.
- **Why:** the synthetic sigmoid saturating at 27 mb was **~28× too small** at
  14 MeV vs the evaluated 755.7 mb — every physics-based number downstream
  inherited that. Judge question answered: "where do your cross sections
  actually come from?"
- **Measured:** installing the *correct* σ made the Joyo anchors **worse**
  (369–2153× vs 12–24× overprediction) — revealing the monoenergetic-flux
  approximation, not σ, as the dominant error
  (`results/ode_data_v2_validation_20260718.json`).

### 12. Spectrum folding (`SPECTRUM_MODE=mono|watt|twogroup`, default mono)
- **What:** one-group ⟨σ⟩ = ∫σ(E)φ(E)dE/∫φ(E)dE for reactor scenarios,
  folding the evaluated tables over documented parametric spectra: bare U-235
  Watt fission spectrum (no free parameters) and a two-group model
  (Watt-shaped tail above the 6.4218 MeV threshold + thermal slow group,
  fraction `SPECTRUM_FAST_FRACTION`). `IsotopeEnvironment` also accepts a
  per-scenario `spectrum=` argument (future dataset hook:
  `v3_pilstm/data/trajectory_dataset.py:273`; design documented in
  `docs/DATA_PROVENANCE.md` §5, not implemented).
- **Why:** the (n,2n) yield lives entirely in the small above-threshold tail;
  a pointwise 14.5 MeV scenario treats 100% of the flux as tail neutrons.
  The measured Joyo MK-III spectrum is paywalled, hence parametric spectra
  labelled as assumptions with citable forms.
- **Measured:** bare-Watt fold of the evaluated table gives ⟨σ⟩ = 26.7 mb ≈
  **the old synthetic 27 mb** — the legacy constant was effectively a fission
  spectrum average; the real irradiation-position spectrum is far softer.
  Exact ODE inversion on Sano 2024 (15.4±6.2 GBq) gives an inferred
  above-threshold fraction **f\* = 1.24×10⁻³ [7.4×10⁻⁴, 1.7×10⁻³]**
  (⟨σ⟩ = 1.66 mb; ~16× softer than bare fission — plausible for a
  sodium-cooled MOX core; effective parameter with uncertainty, not tuned
  truth). Two-group f\* lands 1.00× / 0.57× / 1.14× on Sano / Iwahashi-60d /
  Iwahashi-milking (the two literature anchors mutually differ ~2×).
  (`results/ode_data_v2_spectrum_20260718.json`,
  `graphs/ode_v2_spectrum_anchors.png`.)
- **Citations:** Watt 1952 Phys. Rev. 87, 1037 (spectrum form; a=0.988,
  b=2.249 per ENDF-6 formats manual); Iwahashi et al. 2022, MDPI Processes
  10(7):1239 (open access; Joyo core, threshold, spectrum figure);
  Aoyama et al. 2005, J. Nucl. Radiochem. Sci. 6(3) (Joyo MK-III spectral
  softness by position).
- **ISEF narrative:** each fix revealed the next-deeper assumption — wrong σ
  → real σ → mono-flux error → spectrum model. **This IS the research.**
  Retrain guidance: the surrogate should see spectrum mode as a scenario-level
  input (mono for D-T accelerator scenarios, folded for reactor scenarios) so
  one model learns both flux regimes.

---

## Verification standing

- `v3_pilstm/results/smoke_20260718.json`: **all 10 smoke checks pass**
  (loss exactness, bit-for-bit reproducibility, legacy checkpoint compat,
  expmix training path, baseline training, speed benchmark, jackknife/CV+,
  adaptive weights, curriculum, ensemble).
- All legacy paths (`trap` loss, n=11 conformal, v1 inventory, existing
  checkpoints) verified unchanged.
- No accuracy improvement is claimed for any new feature; every number that
  would go on a poster awaits the full-budget Kaggle runs (commands are in
  the engineering changelog / commit history).
