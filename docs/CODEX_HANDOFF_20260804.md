# Codex / agent handoff — IsotopePINN PI-LSTM (as of 2026-08-04)

Paste this whole file into a fresh Codex/Cursor chat that has **not** seen this repo recently. It is the authoritative state dump.

**Do not revise poster accuracy claims.** The only measured headline number still on the board is **Results-6**: test endpoint Ac-225 median relative error **5.124%** (val selection 3.479% @ epoch 3425). That run used the **old buggy** physics loss. Post–Sprint-6 retrains have **not** produced a finished better number.

---

## 0. Where the live code is (easy to get wrong)

There are **two** nested trees under Downloads:

| Path | Role |
|------|------|
| `C:\Users\ogunn\Downloads\New folder\New folder\` | **LIVE Colab zip source.** `build_colab_zip.py` packs **this** tree. Edit here. |
| `C:\Users\ogunn\Downloads\New folder\` (outer) | Workspace root; often **stale** copies of `v3_pilstm/`. Do not assume outer == nested. |

Zip builder: `C:\Users\ogunn\Downloads\New folder\build_colab_zip.py`  
→ writes `C:\Users\ogunn\Downloads\IsotopePINN_Project.zip`  
→ user uploads that to Google Drive root as `IsotopePINN_Project.zip`.

Colab notebooks live in:
- `New folder\colab_runs\RUN_*_ISEF.ipynb`
- copies also under `C:\Users\ogunn\Downloads\` for easy upload

Drive run artifacts: `/content/drive/MyDrive/IsotopePINN_runs/{A,A2,A2b,...}/`

Mentor context: Jayden Palmer (NCSU). Goal: ISEF-ready accuracy + **honest** claims.

---

## 1. What the project is

**IsotopePINN / v3 PI-LSTM** = physics-informed LSTM surrogate for the **Ra-226 → Ac-225** Bateman transmutation chain under neutron flux (accelerator / spectrum scenarios).

### Species (5 outputs, normalized)
Index order everywhere:

| idx | Nuclide | Role |
|-----|---------|------|
| 0 | Ra-226 | long-lived target feedstock |
| 1 | Ra-225 | intermediate |
| 2 | **Ac-225** | **product / poster species** |
| 3 | Ra-227 | **stiffness source** (T½ ≈ 42 min) |
| 4 | Ac-227 | impurity branch |

### Physics
- Linear Bateman ODEs with (n,2n) and (n,γ) reaction rates depending on flux φ and energy.
- Truth generator: Radau via `IsotopeEnvironment` / `run_simulation`.
- Stiffness: Ra-227 vs Ra-226 timescale ratio ~ **10⁷**.
- Training grid: typically **64 log-spaced** time steps to `t_end` (early stiff dynamics + late endpoint).

### Model (`v3_pilstm/models/pi_lstm.py`)
- LSTM backbone, hidden **256**, Fourier energy + time features.
- Inputs per step: `[t_norm, phi_norm, energy_feat, ic_norm×5]` (8 cols; v3 can add spectrum extras).
- **Hard IC ansatz**: prediction constrained to match initial condition at t=0.
- Outputs: `(batch, seq, 5)` **normalized** non-negative inventories.
- Optional: EMA eval, conserve projection, multifidelity head, exp correction (flags; mostly off in current Colab recipes).

### Poster / checkpoint metric (CRITICAL)
```text
rel_i = |pred_Ac225(t_end) - truth_Ac225(t_end)| / max(|truth|, 1e-30)
best  = median(rel_i) over selection scenarios
```
- Code: `analysis/endpoint_eval.py` (`rel_err`, `ac225_endpoint_median`)
- Trainer: `PILSTM_CKPT_METRIC=endpoint_ac225` (default)
- This is **NOT** full-trajectory Smooth-L1. Endpoint is ~1/64 of a uniform traj mean.

### Results-6 (legacy measured claim — still the board number)
- Test endpoint Ac-225: **0.05124 (5.124%)**
- Val best: **0.03479 (3.479%) @ epoch 3425**
- Trained under **buggy** physics residual (see Sprint 6), strong data supervision, seed 42-ish recipe.
- Artifact: `v3_pilstm/results/train_summary.json` (and nested copy).

---

## 2. What is RIGHT (do not undo)

### Sprint 6 physics correctness (2026-07-31) — verified
Documented in `docs/UPGRADE_LOG.md` and smoke checks.

1. **Jacobian identity on Ac rows** (`physics/bateman_rhs.py`)  
   Ac-225 / Ac-227 row norms were missing the `1.0` identity term → Ac-225 residual inflated **~285×**. Fixed. Guard: `smoke_checks.py --parts jacobian`.

2. **Trap self-shielding** (`physics/integrated_loss.py` `_trap_shielding`)  
   ODE generator applies `exp(-0.01*mass)`; trap residual used unshielded rates → truth was not a zero of the residual. Fixed. Flag `PI_LSTM_TRAP_SHIELDING=0` restores old bug for control Run B.

3. **Default physics loss = `expmix`** (`PI_LSTM_LOSS`)  
   Exact piecewise-exponential propagator residual (float64 hours grid). On exact Radau traj, residual ~1e-14 vs trap much worse on log grids / Ra-227.

4. **EMA bias correction**, **n_val 22→88**, **late eval every 5**, **QSSA Ra-227 prior** (flagged), **CRAM cross-check (not adopted as train residual)**, ablation harness noise-floor honesty.

### Engineering that is intentional
- Flag-gated everything; defaults preserve legacy A/B unless preset overrides.
- FP64 training (`PILSTM_FLOAT64=1`) — keep; under-precision can freeze stiff PINN optimizers.
- Distillation from frozen v2 PINN teacher when weights exist.
- Hard IC on.
- Species weights in data loss: `[1, 2, 5, 2, 3]` — Ac-225 already heaviest linear weight.
- Colab checkpoints on Drive (`PILSTM_WEIGHTS_PATH` etc.) so disconnects don’t wipe runs.
- Zip staleness gates fail if Sprint 6 / endpoint symbols missing.

### Documentation already written
- `docs/ENDPOINT_COLLAPSE_HISTORY_20260731.md` — full A→A2→A2b failure write-up
- `docs/UPGRADE_LOG.md` items 29–36 (Sprint 6) and endpoint items 32–33 (note: numbering collided with Sprint 6 “32”; both sections exist — read by date/title)
- Smoke part `endpoint_focus` exists and passed when added

---

## 3. CURRENT ISSUE (what is going wrong RIGHT NOW)

### One-sentence version
**After fixing the physics residual, Colab retrains still fail the poster metric because (a) training did not optimize endpoint relative error, then (b) when we added that term naively it exploded and forced Ac-225 endpoint zero-collapse (`best ≈ 1.0`).**

### Status of runs (Drive)

| Run | Intent | Outcome | Use as headline? |
|-----|--------|---------|------------------|
| **A** | Sprint 6 fixes, old traj objective | Traj Ac-225 got better; `best` stuck **≈0.9996** for ~thousands of epochs | No — ablation |
| **A2** | + endpoint rel loss + late data + QSSA + **curriculum ON** | `ep~1e27`; `best→1.0` by ep25; train stiffness ≠ score stiffness | No — partial |
| **A2b** | A2 but **curriculum OFF** | Same collapse fingerprint through **ep≥225**: `best=1.0000` frozen since ep25 | No — stopped |
| **A2c** | **NEXT:** robust endpoint *train* loss | **Not implemented / not run yet** | Intended path |
| **B** | Pre–Sprint-6 trap control | **Still must be run** for honest before/after | Control, required |

### Failure mechanisms (all three matter for history)

#### Failure A — train ≠ score
- Optimized: full-traj Smooth-L1 (+ physics, mass, overshoot, distill).
- Scored: median **endpoint** Ac-225 relative error.
- Physics causality weights **early** times (Wang); endpoint failure needs **late** emphasis.
- Linear Smooth-L1 under-penalizes near-zero **trace** Ac-225 vs huge Ra-226.
- Symptom: traj median ↓ while `best → 1.0` (zero endpoint ≈ relative error 1).

#### Failure A2 — curriculum mismatch
- `PI_LSTM_CURRICULUM=1` → train ODEs at `rate_scale=100,10,1` (destiffened labels + physics).
- `best` **always** full-stiffness Radau endpoints.
- Faithful fit to stage-1 data ⇒ systematic Ac-225 underprediction vs score.
- Lesson: curriculum off for endpoint-focused runs until train/score stiffness match.

#### Failure A2b — bare mean relative endpoint loss (**active blocker**)
Training term shipped as:
```python
ep = mean_batch(|pred - truth| / max(|truth|, 1e-30))  # last step, Ac-225 only
loss += PILSTM_ENDPOINT_WEIGHT * ep   # A2b uses 25
```
This **matches eval algebra** but is a terrible early SGD surrogate:

1. Random/early preds ≫ tiny Ac-225 ⇒ ratios `1e16`–`1e27`.
2. Weight 25 ⇒ total `loss ~ 1e28`, endpoint dominates everything.
3. Easy win: drive preds → **0** ⇒ relative error → **~1**.
4. `best` is a **median**; training uses a **mean** of unclipped ratios ⇒ mean can fall while median stays glued at 1.0 (exactly what A2b logs showed: `ep` 1e27→8e15, `best` stuck).
5. Once in the near-zero basin, traj Smooth-L1 + late/log boosts are too weak to escape.

**A2b log fingerprint (curriculum=off, correct flags):**
```text
endpoint_w=25 late_power=2 ac225_log_late=4 qssa implied via preset
epoch 1:    ep=1.26e27  best=6.45@1
epoch 25:   ep=6.27e16  best=1.0000@25
epoch 225:  ep=8.38e15  best=1.0000@25   # unchanged since 25 → STOP
```

### What we are trying to fix NOW (A2c)
1. **Keep** eval/checkpoint metric = median endpoint relative error (poster definition unchanged).
2. **Change** the **training** endpoint term to a **robust** surrogate, e.g.:
   - `smooth_l1` / Huber on `(p-t)/clamp(|t|, floor)`, and/or
   - `|log(p+ε)-log(t+ε)|` on endpoint Ac-225 only  
   so outliers contribute O(1)–O(10), not `1e27`.
3. Keep A2b recipe otherwise:
   - `PI_LSTM_CURRICULUM=0`
   - `PILSTM_ENDPOINT_WEIGHT` nontrivial (may retune after robustification)
   - `PILSTM_DATA_LATE_POWER=2`
   - `PILSTM_AC225_LOG_LATE=4`
   - `PILSTM_QSSA_WEIGHT=20`
   - `PILSTM_FLOAT64=1`, `PI_LSTM_LOSS=expmix`, `SCENARIO_VERSION=v2`, distill on
4. New Drive folder **A2c**, fresh start (**never resume** A/A2/A2b into it).
5. Rebuild zip + staleness gate for new loss symbol.
6. Still run **B** as control before claiming Sprint 6 / endpoint fixes helped.
7. Only then update HTML/deck/poster numbers from `train_summary.json`.

### What NOT to do
- Do not claim &lt;5% or “Sprint 6 fixed accuracy” without a finished measured run.
- Do not resume stuck checkpoints into a new objective.
- Do not turn curriculum back on for endpoint runs unless score uses same `rate_scale`.
- Do not “fix” Run B into the new losses — B is the pre-fix control.
- Do not adopt CRAM as the training residual (trace nuclides; see UPGRADE_LOG).
- Do not quote CPU ablation deltas as evidence (noise floor ~21 points across seeds).

---

## 4. Model / training stack details (small things that still matter)

### Loss cocktail (`train_one_epoch`)
```text
total =
  data_w * data_trajectory_loss          # Smooth-L1 + log; optional late_power, ac225_log_late
+ endpoint_w * ac225_endpoint_rel_loss   # CURRENTLY UNCLIPPED MEAN REL — broken as train term
+ phys_w * integrated_physics_loss       # expmix or trap; causal early time weights
+ mass_w * mass_conservation_loss
+ overshoot_w * impurity_overshoot_loss  # one-sided Ra-227-ish overshoot
+ distill_w * distillation_loss          # vs v2 teacher traj
+ rollout_w * semigroup_rollout_loss     # usually 0
+ qssa_w * qssa_ra227_loss               # two-sided QSSA, default 0
```
Typical Colab BASE weights: data 35, phys 20, mass 10, overshoot 20, distill 5, log_weight 2.0, causal_eps 2.5 (physics only).

### Data loss extras (already coded, flag-gated)
- `PILSTM_DATA_LATE_POWER` — `w_t ∝ (t/t_max)^p`, mean-normalized; **upweights late** (opposite of physics causality).
- `PILSTM_AC225_LOG_LATE` — Ac-225 log channel × `(1 + boost * frac_t)`.

### Physics notes
- `expmix` needs float64 hours grid (`t_h64`) for dt parity; `PI_LSTM_TGRID64=0` forces legacy.
- Curriculum scales **training data generation** and physics `rate_scale` together; val/test endpoint score does **not** use curriculum scale.
- QSSA validated offline (`analysis/qssa_ra227.py`): 38/39 scenarios; gated when assumption invalid.

### Checkpoint / eval plumbing
- Selection pool seed **2025**, test seed **2024**, disjoint.
- `PILSTM_N_VAL=88` default (was 22; Results-6 overfit selection).
- Endpoint scoring caches features+ODE truths (224× faster than re-solving every eval).
- L-BFGS polish is data(+endpoint)-only; can worsen endpoint — trainer rejects polish if score doesn’t improve.
- Pretrain fraction ~0.20: early epochs downweight data vs physics.

### Diagnostics
- `v3_pilstm/scripts/diagnose_ac225_collapse.py` — prints median `pred/truth` at endpoints. ≈0 + rel≈1 ⇒ zero-collapse confirmed.
- Local `pi_lstm_best.pth` diagnose was **PARTIAL** (old Results-era weights), not the stuck Colab A ckpt.

### Colab notebooks
| File | RUN | Notes |
|------|-----|-------|
| `RUN_A_champion_ISEF.ipynb` | A | Sprint 6 defaults only |
| `RUN_A2_endpoint_ISEF.ipynb` | A2 | endpoint+late+QSSA+**curriculum on** — superseded |
| `RUN_A2b_endpoint_no_curriculum_ISEF.ipynb` | A2b | curriculum off — stopped at collapse |
| `RUN_B_control_ISEF.ipynb` | B | trap, unshielded, n_val=22 — **keep** |
| `RUN_C_qssa_ISEF.ipynb` | C | QSSA-only ablation |
| Generator: `scripts/build_isef_notebook.py` → `PI_LSTM_ISEF_2026.ipynb` |

### Streamlit
Nested `app.py` may be what’s running on port 8501; outer app was stale earlier — check which tree the process cwd uses.

---

## 5. What “done” looks like for the next agent task

**Primary coding task:** implement robust Ac-225 endpoint **training** loss (A2c), wire env flag (keep old bare relative behind a flag or replace with robust default when `endpoint_w>0`), smoke test, A2c notebook, rebuild zip gates, short UPGRADE_LOG entry.

**Success criteria for a training run (later):**
- Early `ep` (or robust endpoint term) is O(1)–O(100), not `1e16+`.
- `best` moves below ~0.9 within a few hundred epochs and keeps improving (not glued at 1.0000).
- Finished `train_summary.json` with test endpoint Ac-225; compare to Results-6 5.124% **honestly**.
- Run B completed under matched control recipe for the Sprint-6 story.

**Non-goals for the immediate patch:** rewriting the whole loss, enabling SOAP/EMA/rollout by default, changing the poster metric definition, deleting A/A2/A2b Drive logs.

---

## 6. Suggested first commands for Codex

```text
1. Work only under: C:\Users\ogunn\Downloads\New folder\New folder\
2. Read: docs/ENDPOINT_COLLAPSE_HISTORY_20260731.md
3. Read: docs/UPGRADE_LOG.md (Sprint 6 + endpoint sections)
4. Read: v3_pilstm/physics/integrated_loss.py  (ac225_endpoint_rel_loss, data_trajectory_loss)
5. Read: v3_pilstm/train_pi_lstm.py           (endpoint_w wiring)
6. Implement robust endpoint train loss + A2c notebook + smoke + zip gates
7. python build_colab_zip.py from the OUTER folder that contains build_colab_zip.py
```

---

## 7. Honest narrative (mentor / judge)

> We found and fixed two real bugs in the physics residual (Jacobian identity on actinium rows; trap self-shielding). That made the physics loss consistent with Radau truth. Retraining under the old data objective still failed the **endpoint** Ac-225 metric we put on the poster (zero-collapse, `best≈1`). We then trained the metric explicitly, but an unclipped mean relative loss is an unstable surrogate and recreated collapse; stiffness curriculum also mismatched train labels vs full-stiffness scoring. Next step is a robust endpoint training loss with curriculum off, control Run B retained, and no accuracy claim until measured.

---

## 8. Key file index

```text
New folder/New folder/
  v3_pilstm/train_pi_lstm.py
  v3_pilstm/physics/integrated_loss.py
  v3_pilstm/physics/bateman_rhs.py
  v3_pilstm/models/pi_lstm.py
  v3_pilstm/data/trajectory_dataset.py
  v3_pilstm/analysis/endpoint_eval.py
  v3_pilstm/scripts/diagnose_ac225_collapse.py
  v3_pilstm/scripts/smoke_checks.py
  v3_pilstm/scripts/build_isef_notebook.py
  colab_runs/RUN_A*_*.ipynb
  colab_runs/RUN_B_control_ISEF.ipynb
  docs/ENDPOINT_COLLAPSE_HISTORY_20260731.md
  docs/UPGRADE_LOG.md
../build_colab_zip.py   # one level up from nested live root
```

**End of handoff.**
