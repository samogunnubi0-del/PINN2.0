# PI-LSTM v3 — Google Colab Setup (Samuel)

Step-by-step guide to train PI-LSTM on a Colab GPU and bring weights back to your PC.

**Preferred notebook:** `v3_pilstm/PI_LSTM_RunC_COLAB.ipynb` (self-contained unzip + Run C).  
**No Colab GPU?** Use **[KAGGLE_SETUP.md](KAGGLE_SETUP.md)** + `PI_LSTM_RunC_KAGGLE.ipynb` (free).

## What you need locally

- Project root: the inner `New folder/` directory (contains `pinn_model.py`, `v3_pilstm/`, `weights/`, `data/`, etc.)
- Fresh zip: `python build_colab_zip.py` → `IsotopePINN_Project.zip`
- Optional: local venv for smoke tests before/after Colab

## 1. Upload project to Google Drive

**Easiest (Run C notebook):** upload only these to **My Drive root**:

1. `IsotopePINN_Project.zip`
2. `PI_LSTM_RunC_COLAB.ipynb`

Cell 1 unzips into `MyDrive/IsotopePINN/`.

**Or** sync the full folder:

```
MyDrive/
  IsotopePINN/                    ← upload whole project here
    pinn_model.py
    ra226_ac225_transmutation.py
    requirements.txt
    weights/
      pinn_best_weights.pth       ← v2 baseline (needed for compare_models)
    data/
      literature_benchmarks.csv
    v3_pilstm/
      PI_LSTM_RunC_COLAB.ipynb
      train_pi_lstm.py
      models/
      physics/
      analysis/
      weights/                    ← pi_lstm_best.pth written here after training
      results/
```

**Options:**

- **Google Drive for Desktop** — sync or copy the folder to `MyDrive/IsotopePINN/`
- **Zip fallback** — run `python build_colab_zip.py` from the parent repo, upload `IsotopePINN_Project.zip` to Drive root; the notebook will unzip into `IsotopePINN/`

## 2. Open the Colab notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Open **`PI_LSTM_RunC_COLAB.ipynb`** (a `.ipynb` file — **not** `IsotopePINN_Project.zip`):
   - **Easiest:** **File → Upload notebook** from your PC (`New folder/v3_pilstm/PI_LSTM_RunC_COLAB.ipynb`)
3. **Runtime → Change runtime type → T4 GPU** (or better)

> If Colab shows **`Unexpected token 'P', "PK"`**, you opened a **ZIP** as if it were a notebook. ZIP files start with the bytes `PK`. Upload/open only `.ipynb`; run Cell 1 to extract the project zip.

## 3. Run cells in order (`PI_LSTM_RunC_COLAB.ipynb`)

| Cell | Purpose |
|------|---------|
| 1 | GPU + mount Drive + unzip + verify `val Ac225 med` / `EVAL_EVERY` |
| 2 | Deps check + backup any existing weights |
| 3 | **Full training (Run C)** — `MODE='quality'` (float64) or `'fast'` |
| 4 | Verify Run C config + protected Drive copies |
| 5 | `compare_models.py`, empirical, Joyo, plots |
| 6 | Zip + browser download |

### Training environment variables (Cell 3, Run C)

| Variable | Value | Meaning |
|----------|-------|---------|
| `PILSTM_FLOAT64` | `1` (quality) / `0` (fast) | Double precision vs speed |
| `PILSTM_EVAL_EVERY` | `1` / `25` | Checkpoint eval frequency |
| `PILSTM_EPOCHS` | `6000` | Full Run C |
| `PILSTM_N_TRAIN` | `1400` | Structured ODE scenarios |
| `PILSTM_N_VAL` / `PILSTM_N_TEST` | `22` | Canonical held-out sets |
| `PILSTM_N_STEPS` | `64` | Log-spaced time grid |
| `PILSTM_BATCH` | `16` | Batch size |
| `PILSTM_HIDDEN` | `256` | LSTM hidden size |
| `PILSTM_FOURIER` | `8` | Fourier energy bands |
| `PILSTM_TIME_FOURIER` | `16` | Time Fourier features |
| `PILSTM_HARD_IC` | `1` | Hard initial-condition ansatz |
| `PILSTM_DATA_WEIGHT` | `35` | Supervised trajectory loss |
| `PILSTM_PHYSICS_WEIGHT` | `20` | Integrated Bateman physics loss |
| `PILSTM_MASS_WEIGHT` | `10` | Mass conservation penalty |
| `PILSTM_DISTILL` / `PILSTM_DISTILL_WEIGHT` | `1` / `5` | Distill from frozen v2 teacher |
| `PILSTM_OVERSHOOT_WEIGHT` | `20` | Ra-227 overshoot penalty |
| `PILSTM_CAUSAL_EPS` | `2.5` | Causal early-time weighting |
| `PILSTM_CKPT_METRIC` | `endpoint_ac225` | Best-checkpoint metric |
| `PILSTM_LOG_EVERY` | `10` | Print frequency |
| `PILSTM_EARLY_STOP` | `0` / `0.01` | Optional stop when best < threshold |

| `PILSTM_PRETRAIN_FRAC` | `0.15` | Fraction of epochs for physics-first pretrain |
| `PILSTM_LBFGS_ITER` | `60` | Final L-BFGS polish iterations (0 = off) |
| `PILSTM_LOG_WEIGHT` | `2.0` | Log-space loss weight for trace species |
| `PILSTM_GRAD_BALANCE` | `0` | Optional data/physics grad-norm balancing (slower) |

Distillation needs `weights/pinn_best_weights.pth` (the frozen v2 model) in the upload.
Eval is now **full-trajectory** Ac-225 error (all timesteps), directly comparable to v2.
Expect **~2–4 hours** on a T4 for 6000 epochs (varies with GPU quota).

## 4. Download results to your PC

After Cell 8 downloads `PI_LSTM_Results.zip`, unzip and copy into your local repo:

```
PI_LSTM_Results/v3_pilstm/weights/pi_lstm_best.pth
  → New folder/v3_pilstm/weights/pi_lstm_best.pth

PI_LSTM_Results/v3_pilstm/results/*.json
  → New folder/v3_pilstm/results/

PI_LSTM_Results/graphs/v3_*.png
  → New folder/graphs/
```

Poster graphs generated automatically in Cell 7:
- `v3_v2_vs_pilstm_ac225.png` — Ac-225 accuracy focus
- `v3_species_median_errors.png` — all five species
- `v3_trajectory_example.png` — example Ac-225 ingrowth curve
- `v3_literature_anchors.png` — literature vs ODE/v2/PI-LSTM
- `v3_joyo_sigma_calibration.png` — Joyo σ calibration (if run)

Weights also persist on Drive at `MyDrive/IsotopePINN/v3_pilstm/weights/pi_lstm_best.pth` if the session disconnects.

## 5. Verify locally

```powershell
cd "New folder"
.\.venv\Scripts\Activate.ps1

python v3_pilstm/analysis/compare_models.py
python v3_pilstm/scripts/plot_v2_vs_pilstm.py
python v3_pilstm/analysis/validate_empirical.py

streamlit run v3_pilstm/app_v3.py
```

## 6. Literature benchmarks (optional)

See `data/literature_benchmarks_README.md` for Tier A/B/C sources and how to fill `data/literature_benchmarks.csv`, then re-run Cell 7 or `validate_empirical.py` locally.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Unexpected token 'P', "PK"` | You opened a **`.zip` as a notebook** (or tried to JSON-parse a zip). Use **File → Upload notebook** for `PI_LSTM_Colab_Run.ipynb`; run **Cell 2** to unzip `IsotopePINN_Project.zip` into `MyDrive/IsotopePINN/` |
| `Missing after setup` | Ensure full project is at `MyDrive/IsotopePINN/`, or upload zip to Drive root |
| No GPU | Runtime → Change runtime type → GPU |
| Session disconnect | Re-mount Drive; weights saved on each val improvement — skip to compare/download cells |
| `v2 weights missing` | Include `weights/pinn_best_weights.pth` in the uploaded folder |
| Slow on CPU | Do not run 4000-epoch full train without GPU |
