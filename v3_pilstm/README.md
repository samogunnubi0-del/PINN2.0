# IsotopePINN v3 — PI-LSTM (Integrated Loss)

Parallel track to frozen v2 MLP-PINN. Implements Jaden Palmer's integrated-loss recommendation using **Reduced-PINN** (Nasiri & Dargazany 2022, [arXiv:2208.12045](https://arxiv.org/abs/2208.12045)).

## Quick start

```powershell
cd "New folder"
.\.venv\Scripts\Activate.ps1

# Smoke train (CPU, ~80 epochs)
python v3_pilstm/train_pi_lstm.py

# Full train (set epochs)
$env:PILSTM_EPOCHS="4000"
python v3_pilstm/train_pi_lstm.py

# Compare v2 vs PI-LSTM
python v3_pilstm/analysis/compare_models.py

# Split conformal UQ on v2 (90% nominal coverage)
python v3_pilstm/analysis/run_conformal_validation.py

# Preview app
streamlit run v3_pilstm/app_v3.py
```

## Checkpoint selection

By default training saves the best weights by **endpoint Ac-225 median rel error** on the
canonical 22 held-out scenarios (same metric as `compare_models.py`). Override with:

```powershell
$env:PILSTM_CKPT_METRIC="traj_median"   # legacy: full-trajectory val median
```

Checkpoints are saved via `model.save()` as `{config, state_dict}`. Legacy bare
`state_dict` files (128/4, no hard-IC) are auto-inferred but may not match Colab config.

## Uncertainty quantification (conformal)

Split conformal prediction replaces MC Dropout (per Jaden Palmer). Calibrate on the
first 11 canonical held-out scenarios; evaluate coverage on the remaining 11:

```powershell
python v3_pilstm/analysis/run_conformal_validation.py
CONFORMAL_MODEL=pilstm python v3_pilstm/analysis/run_conformal_validation.py
```

Output: `v3_pilstm/results/conformal_validation.json`

## Training recommendations (literature)

- **LSTM over MLP-PINN** for stiff ODE surrogates: recurrent memory reduces error
  accumulation on long horizons vs pointwise MLP mappings (PINN-LSTM malware dynamics,
  seismic LSTM-PINN hybrids).
- **Hard IC ansatz** (Lagaris trial solution): exact `N(t=0)=IC` stabilizes early-time
  stiff ingrowth (Ra-227 T½ ≈ 42 min).
- **Log-spaced time grid + causal physics weighting**: denser early samples + ramped
  physics loss help stiff transients without over-penalizing late-time product yield.
- **v2 distillation** (weight default 10): soft teacher targets accelerate convergence;
  keep distill moderate so LSTM still learns trajectory dynamics, not only pointwise mimicry.
- **Stiff regimes**: consider STL-PINN-style curriculum (low-stiff pretrain → joint) or
  longer schedules (5000–6000 epochs) with L-BFGS polish on the data term.

## Colab Run C (recommended GPU recipe)

Reconciled v3.2 settings for endpoint Ac-225 checkpoint selection on canonical
held-out scenarios (val seed **2025**, test/compare seed **2024**):

- **Full quality:** `PILSTM_EPOCHS=6000` (~1–3 hr on Colab A100/T4 with Run C)
- **Quick / Kaggle fast:** `MODE=fast` in notebooks (float32 + `PILSTM_EVAL_EVERY=25`)

```python
os.environ['PILSTM_CKPT_METRIC'] = 'endpoint_ac225'
os.environ['PILSTM_EPOCHS'] = '6000'
os.environ['PILSTM_PRETRAIN_FRAC'] = '0.20'
os.environ['PILSTM_GRAD_BALANCE'] = '1'
os.environ['PILSTM_DISTILL_WEIGHT'] = '5'   # ramps to 0 over epochs 60%-100%
os.environ['PILSTM_DATA_WEIGHT'] = '35'
os.environ['PILSTM_PHYSICS_WEIGHT'] = '20'
os.environ['PILSTM_CAUSAL_EPS'] = '2.5'
os.environ['PILSTM_N_TRAIN'] = '1400'
os.environ['PILSTM_N_STEPS'] = '64'
os.environ['PILSTM_TIME_FOURIER'] = '16'    # 0 to disable
os.environ['PILSTM_FOURIER'] = '8'
os.environ['PILSTM_LOG_EVERY'] = '10'
# Optional speed (Kaggle/Vast): PILSTM_EVAL_EVERY=25, PILSTM_FLOAT64=0, PILSTM_EARLY_STOP=0.01
```

Distillation weight is scheduled in `train_pi_lstm.py` (full during pretrain,
linear ramp to zero from 60%–100% of training). Optional inference polish:
`PILSTM_ENDPOINT_PROJECT=1` applies one-step trapezoidal Newton projection.

## Cloud training notebooks

| Platform | Notebook | Guide |
|----------|----------|-------|
| **Kaggle (free GPU)** | `PI_LSTM_RunC_KAGGLE.ipynb` | [KAGGLE_SETUP.md](KAGGLE_SETUP.md) |
| **Colab** | `PI_LSTM_RunC_COLAB.ipynb` | [COLAB_SETUP.md](COLAB_SETUP.md) |
| Vast.ai | `scripts/vast_one_shot.py` | [VAST_SETUP.md](VAST_SETUP.md) |

1. On PC: `python build_colab_zip.py` → upload fresh `IsotopePINN_Project.zip`
2. Open the platform notebook → GPU → Run all
3. Download `PI_LSTM_Results.zip` → copy weights/JSON/graphs locally
4. Verify: `compare_models.py`, `scripts/plot_v2_vs_pilstm.py`

## Layout

```
v3_pilstm/
  models/pi_lstm.py
  physics/integrated_loss.py
  physics/conformal.py
  physics/bateman_rhs.py
  data/trajectory_dataset.py
  train_pi_lstm.py
  analysis/compare_models.py
  analysis/run_conformal_validation.py
  analysis/endpoint_eval.py
  analysis/validate_empirical.py
  scripts/plot_v2_vs_pilstm.py
  PI_LSTM_Colab_Run.ipynb
  PI_LSTM_RunC_COLAB.ipynb
  PI_LSTM_RunC_KAGGLE.ipynb
  COLAB_SETUP.md
  KAGGLE_SETUP.md
  VAST_SETUP.md
  weights/pi_lstm_best.pth
  results/
```

v2 files (`pinn_model.py`, `train.py`, `weights/pinn_best_weights.pth`) are **not modified**.
