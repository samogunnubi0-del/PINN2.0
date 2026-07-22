# ISEF board pack — PI-LSTM Results-6 (no new training)

## One-sentence claim

**PI-LSTM (Results-6) is a more accurate Ac-225 endpoint surrogate of our Bateman ODE than frozen v2; Joyo literature match requires spectrum-calibrated σ(n,2n); batched inference cuts wall time without changing teacher weights.**

## Research problem / criteria / constraints

| Item | Content |
|------|---------|
| **Need** | Fast Ac-225 design scans under stiff five-species Bateman physics |
| **Success criteria** | Beat v2 on held-out Ac-225 median relative error; conformal ~90% relative coverage; inference ≪ stiff ODE for large scenario grids |
| **Constraints** | 0D neutron channels only `(n,2n)` / `(n,γ)`; no cyclotron/linac/spallation as in-scope fitness |

## Alternatives tested

1. Stiff ODE (truth teacher)  
2. v2 MLP-PINN (frozen)  
3. PI-LSTM Results-6 (science teacher — **this board**)  
4. Results-7 quality resume (archived — worse held-out)  
5. Batched / compile inference paths (additive speed)  

## Primary results (22 held-out scenarios, seed 2024)

| Species | v2 median rel | PI-LSTM R6 |
|---------|---------------|------------|
| **Ac-225** | 8.18%¹ | **5.12%** |
| Ra-226 | 16.3% | **15.6%** |
| Ra-225 | **7.43%** | 7.86% |
| Ra-227 | 14.5% | **6.62%** |
| Ac-227 | **6.74%** | 10.8% |
| High-flux Ra-227 overshoot | 0 | 0 |

¹ **Protocol reconciliation:** the v2 value 8.18% comes from `compare_v2_pilstm.json`, which scores v2 under this comparison's full-trajectory endpoint protocol on the 22 v3 held-out scenarios (seed 2024). The canonical v2/v63 headline **4.51%** (`results/v63_validation_20260530.json`) comes from the separate v63 validation pipeline (different scenario draw and scoring). The two numbers are **not directly comparable** — the 8.18% vs 4.51% gap reflects the protocol difference, not a 2× model discrepancy.

Sources: `v3_pilstm/results/compare_v2_pilstm.json`, `train_summary.json`

## Uncertainty (split conformal, PI-LSTM)

- Nominal coverage 90% (`α=0.1`)  
- Ac-225 **relative** test coverage ≈ **90.9%**  
- Source: `v3_pilstm/results/conformal_validation.json` (`model=pilstm`)

## Speed (frozen R6, CPU float32)

| Path | ms / 22 scenarios | Ac-225 gate |
|------|-------------------|-------------|
| Eager loop | ~3004 ms (~137 ms/sc) | baseline 5.12% |
| **Batched** | **~36 ms (~1.65 ms/sc)** | **PASS** (~83×) |
| `torch.compile` | failed here (no MSVC `cl`) | N/A |

Source: `v3_pilstm/results/speed_harness.json`  
KD student: not trained (teacher frozen per plan).

## Secondary: Joyo σ calibration (not primary MAPE)

Default spectrum-average σ ≈ 27 mb overpredicts Joyo sims.  
Recommended scale ≈ **0.053** → effective ≈ **1.44 mb**.  
MAPE default ≈ **1,429%** → calibrated ≈ **19.5%** (source JSON stores fractions: 14.29 and 0.195).

| Anchor | Default rel err | Calibrated rel err |
|--------|-----------------|--------------------|
| Sasaki 2023 | 1,794.5% (prediction 18.9× reference) | **1.0%** |
| Iwahashi 2022 | 1,062.7% (prediction 11.6× reference) | **38.0%** |

> **Honesty label — illustration, not validation.** This calibration fits **one free parameter** (σ scale) to **two literature anchors**, so agreement at those anchors is circular. It shows how sensitive the endpoint is to the effective σ(n,2n); it is **not** independent validation of the ODE or the surrogate.

Figure: `graphs/v3_joyo_sigma_calibration.png`  
JSON: `v3_pilstm/results/joyo_sigma_calibration.json`

## Out of scope / limits (interview)

- **Matyskin / cyclotron / linac / spallation** — wrong or missing channels (OOS)  
- **Raw Joyo with default σ** — physics mismatch, not a PI-LSTM training failure  
- **Ac-227 impurity** — PI-LSTM still weak vs v2 on thermal (n,γ) lit anchors  
- **Results-7** — longer train did not beat R6 held-out  

## Graphs (regen from R6)

- `graphs/v3_v2_vs_pilstm_ac225.png`  
- `graphs/v3_species_median_errors.png`  
- `graphs/v3_trajectory_example.png`  
- `graphs/v3_literature_anchors.png`  
- `graphs/v3_joyo_sigma_calibration.png`  

## Smart features kept (audit)

See `v3_pilstm/results/SMART_FEATURES_AUDIT.md` — hard-IC, physics, distill, Fourier, causal, overshoot, endpoint ckpt, L-BFGS (reject-if-worse).

## Weights

- **Best:** `v3_pilstm/weights/pi_lstm_best.pth` (Results-6)  
- **Archive:** `pi_lstm_best_ARCHIVE_R7_QUALITY.pth`, `pi_lstm_best_QUALITY_E3096.pth`
