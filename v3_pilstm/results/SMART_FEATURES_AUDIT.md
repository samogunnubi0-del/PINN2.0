# Smart features audit (Results-6 teacher)

**No new training.** Teacher = Results-6 `pi_lstm_best.pth`  
SHA256: `22B052AAC23BE92B0E0D6CE63CA067766D7F836AD75FD2B61834BC9BF5BC2481`  
Held-out Ac-225 endpoint median: **5.12%** (v2: **8.18%** — under this comparison's full-trajectory endpoint protocol; the canonical v63-pipeline v2 headline is **4.51%**, see `results/v63_validation_20260530.json` — protocols differ, not a 2× discrepancy)

Machine-readable: [`smart_features_audit.json`](smart_features_audit.json)

## What made the model smart (all PASS)

| Feature | R6 setting | Status |
|---------|------------|--------|
| Hard IC | `true` | PASS — Lagaris blend in `models/pi_lstm.py` |
| Energy Fourier | `8` | PASS |
| Time Fourier | `16` | PASS in ckpt; train default now **16** (was 0) |
| Integrated physics | `phys_w=20` | PASS — trapezoidal Reduced-PINN loss |
| Distill + ramp | `distill=true`, `w=5` | PASS — full early, ramps to 0 by end |
| Causal time weights | wired | PASS — `causal_time_weights` in train step |
| Overshoot penalty | `overshoot_w=20` | PASS — high-flux Ra-227 overshoot = 0 |
| Endpoint Ac-225 ckpt | `endpoint_ac225` | PASS — best score 0.0348 @ epoch 3425 |
| Pretrain frac | `0.2` | PASS — physics-first early |
| Grad balance | `false` | PASS — R6 winner; `true` quality path lost |
| Eval every | `25` | PASS — not `1` |
| L-BFGS | `iter=60` | PASS — kept; `data_w` applied; reject-if-worse + smoke |

## Dead weight (do not use as “best” / required)

- Results-7 QUALITY weights as `pi_lstm_best` (archived as `pi_lstm_best_ARCHIVE_R7_QUALITY.pth` / `pi_lstm_best_QUALITY_E3096.pth`)
- `PILSTM_FLOAT64=1` as required for quality (R6 won float32)
- `PILSTM_EVAL_EVERY=1` as quality gospel (R6 used 25)
- `PILSTM_GRAD_BALANCE=1` as required (R6 used false)
- Raw Joyo / Matyskin / cross-route as hard training fitness (OOS or wrong σ)

## L-BFGS note

Data-only polish can worsen `endpoint_ac225`. Code now:

1. Scales data loss by `data_w`
2. Saves only if `post < best_med` **and** `post < pre`
3. Logs `accepted_polish` vs `kept_best`
4. Smoke: `python v3_pilstm/scripts/smoke_lbfgs_reject.py`
