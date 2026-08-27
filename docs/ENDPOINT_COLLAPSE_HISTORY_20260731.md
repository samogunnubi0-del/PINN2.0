# Endpoint collapse history — what we are fixing (2026-07-31)

Living lab note for the ISEF accuracy path. **No poster claim is revised here.**
The measured headline number is still Results-6 (pre–Sprint-6 physics bugs):
test endpoint Ac-225 **5.124%**. Everything below is diagnosis of post-fix
retrains that failed to beat that on the metric we actually score.

---

## The metric that matters

Checkpoint selection and the poster number use **endpoint Ac-225 relative error**:

```text
rel = |pred_Ac225(t_end) - truth_Ac225(t_end)| / max(|truth|, 1e-30)
best = median(rel) over the val/selection scenarios
```

Implemented in `analysis/endpoint_eval.rel_err` and scored in
`train_pi_lstm.py` when `PILSTM_CKPT_METRIC=endpoint_ac225` (default).

That is **not** the same as full-trajectory Smooth-L1 over 64 log-spaced
steps. The last step is only ~1/64 of a uniform mean. A model can look
“good” on traj Ac-225 median while the endpoint median is catastrophic.

---

## Run timeline (Colab, seed 42, float64, expmix, v2 inventory)

| Run | Recipe intent | What happened | Keep? |
|-----|---------------|---------------|-------|
| **A** | Sprint 6 physics fixes, old data objective | Traj improved; `best` fell then **stuck ≈ 0.9996** for thousands of epochs | Ablation / story only |
| **A2** | A + explicit endpoint rel loss + late data + QSSA + **curriculum on** | `ep` exploded (`~1e27`); `best→1.0` by epoch 25; **train/eval stiffness mismatch** | Partial log; documents curriculum bug |
| **A2b** | A2 but **curriculum off** | Same early fingerprint as A2 through epoch ≥225: `best=1.0000@25` frozen | Stopped; proves curriculum was not the only killer |
| **A2c** | *(next)* A2b + **robust** endpoint loss | Not run yet | Intended fix path |
| **B** | Pre–Sprint-6 trap control | Still required | Control |

Drive folders: `IsotopePINN_runs/{A,A2,A2b,...}/`. Do not resume A/A2 into
later recipes (incompatible objectives).

---

## Failure 1 — Run A: train ≠ score

### What we optimized
Full-trajectory `data_trajectory_loss` (Smooth-L1 linear + log), plus physics
(`expmix`), mass, overshoot, distill. Physics time weights
(`causal_time_weights`) emphasize **early** intervals (Wang-style causality).

### What we scored
Median **endpoint** Ac-225 relative error on a held-out selection pool.

### Observed behavior
- Full-traj `val Ac225 med` improved (order ~1 → ~0.1).
- Checkpoint `best` (endpoint) improved briefly then **locked near 1.0**.

### Interpretation
Relative error ≈ 1 with positive truth is exactly what you get if
`pred_Ac225(t_end) ≈ 0` (zero-collapse):

```text
|0 - truth| / truth = 1
```

Trajectory Smooth-L1 under-penalizes near-zero trace product (Ac-225 is tiny
vs Ra-226). Early-heavy physics causality pushes the wrong time region for an
**endpoint** failure. Fixing the Jacobian / trap shielding bugs (Sprint 6)
changed the physics landscape but did **not** automatically align training
with the poster metric — so A failed after the bugs were fixed.

---

## Failure 2 — Run A2: curriculum vs checkpoint mismatch

### What `PI_LSTM_CURRICULUM=1` does
Stiffness curriculum (Seiler et al. 2025 style): stage ladder
`rate_scale = 100 → 10 → 1` over training. Every training ODE rate
(reactions **and** decays) is divided by `rate_scale`, so stage 1 is a
slowed, easier chain. Physics residuals use the same scale. Training
targets are regenerated at that scale.

### What `best` always does
Scores predictions against **full-stiffness** Radau endpoints
(`rate_scale = 1`). Never destiffened.

### Why that hurts endpoint Ac-225
At `rate_scale=100`, Ac-225 production/ingrowth in the **training labels** is
far weaker than in the real chain. A network that faithfully fits stage-1
data systematically **under-predicts** Ac-225 relative to the full-stiffness
truth used by `best`. For roughly the first ~4000/6000 epochs, train and
score answer different questions.

### Observed
A2 log: `Curriculum stage 1/3: rate_scale=100`, then
`ep~1e27`, `best=1.0000@25` within minutes.

### Lesson
Curriculum is not “always bad.” It is bad when **train stiffness ≠ score
stiffness**. A2b turned curriculum off for that reason.

---

## Failure 3 — Run A2b: bare relative endpoint loss (what we are fixing now)

### Config (correct intent)
```text
curriculum=off
endpoint_w=25.0  late_power=2.0  ac225_log_late=4.0
qssa_w=20  float64  expmix  scenario_version=v2
```

A2b removed the curriculum mismatch. Early logs still matched A2’s
explosion fingerprint almost bit-for-bit on `ep` / `best`.

### Observed through epoch 225
```text
epoch 1:   ep=1.26e27   best=6.45@1
epoch 25:  ep=6.27e16   best=1.0000@25
epoch 225: ep=8.38e15   best=1.0000@25   (unchanged since 25)
```

`ep` falling by orders of magnitude only means **outlier** relative errors
are shrinking. The **median** endpoint score never left 1.0 — same sticky
basin as Run A.

### Root cause (mechanism)

Training term (as shipped in item 32):

```python
ep = mean_batch( |pred - truth| / max(|truth|, 1e-30) )   # Ac-225, last step
loss += 25 * ep
```

That formula matches the **checkpoint definition** (good intention: train
what you score). As an **unbounded mean**, it is a terrible early training
signal:

1. **Init / early preds** can be huge vs tiny normalized Ac-225 → single-batch
   mean relative error `1e16`–`1e27`.
2. Weighted by 25, total loss is dominated by `endpoint` (`loss ~ 1e28`).
3. Gradients shove Ac-225 predictions **down**. Driving preds toward **zero**
   collapses relative error from astronomical values to **~1.0**.
4. Relative error 1.0 is a wide, flat attractor for “near-zero product”:
   many scenarios look the same to the median score.
5. Once there, Smooth-L1 on the rest of the trajectory does not strongly
   punish zero Ac-225 (trace scale), and the median `best` stops moving.

So: **matching the metric algebraically ≠ a stable surrogate loss.**
The score we want (median relative error) is not what SGD sees (mean of
unclipped ratios). Mean ≠ median; unclipped ratios ≠ robust regression.

Late-time data upweight and Ac-225 log boost help the trajectory term but
cannot compete with a `1e16` endpoint mean early on.

### Why “wait longer” was unlikely after epoch 225
Rule of thumb used in-session: if `best` is still ≈1.0 by epoch 200–500,
treat as stuck. A2b hit that bar with **zero movement** of `best` from
epoch 25→225 while Run A historically stayed glued near 1 for thousands of
epochs after the same fingerprint. Not a mathematical proof — a repeated
empirical pattern plus a clear loss-landscape story.

---

## What we are trying to fix **now** (A2c intent)

**Problem statement (one sentence):**  
Make the training objective pull Ac-225 **endpoint accuracy** without an
unbounded relative-error mean that forces zero-collapse.

**Concrete fix direction (next code change):**
1. Keep scoring `best` as median endpoint relative error (unchanged poster
   definition).
2. Replace the **train** endpoint term with a **robust** surrogate of that
   residual, e.g.:
   - Smooth-L1 / Huber on `(pred - truth) / clamp(|truth|, floor)`, and/or
   - log-ratio `|log(pred+ε) - log(truth+ε)|` on the endpoint only,
   so a wild outlier contributes O(1)–O(10), not `1e27`.
3. Keep A2b’s other knobs: curriculum **off**, late data power, Ac-225 late
   log boost, QSSA, FP64, expmix, v2.
4. New Colab run folder (**A2c**), fresh start (no resume from A2/A2b).
5. Still run **B** as the Sprint-6 control before claiming anything.

**Out of scope until A2c finishes with a measured `train_summary.json`:**
- Revising Results-6 / poster % claims
- Declaring Sprint 6 “fixed accuracy”
- Quoting A / A2 / A2b as success

**Artifacts to keep (history, not headlines):**
- `IsotopePINN_runs/A/` — objective mismatch after physics bugfix
- `IsotopePINN_runs/A2/` — curriculum × endpoint score mismatch
- `IsotopePINN_runs/A2b/` — bare relative endpoint loss → collapse

---

## Judge / mentor one-liner

> We fixed two real bugs in the physics residual, then discovered the
> optimizer was not training the quantity on the poster. Aligning that
> objective with a naive mean relative loss created a zero-collapse
> attractor; the next iteration uses a robust endpoint surrogate while
> keeping the evaluation metric unchanged.

---

## Related code / notebooks

- Loss: `v3_pilstm/physics/integrated_loss.py` (`ac225_endpoint_rel_loss`,
  `data_trajectory_loss`)
- Trainer: `v3_pilstm/train_pi_lstm.py` (`PILSTM_ENDPOINT_WEIGHT`, …)
- Collapse diagnostic: `v3_pilstm/scripts/diagnose_ac225_collapse.py`
- Notebooks: `colab_runs/RUN_A2_endpoint_ISEF.ipynb`,
  `colab_runs/RUN_A2b_endpoint_no_curriculum_ISEF.ipynb`
- Chronology: `docs/UPGRADE_LOG.md` items 32–33
