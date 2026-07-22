# PI-LSTM Run C — Kaggle GPU Setup (free)

## 🍼 Easiest path (use this!)

New: **`PI_LSTM_ISEF_EASY.ipynb`** — one editable letter (`RUN = "A"`), automatic
babysitting, and plain-English instructions. Three steps:

1. On your PC: `python build_colab_zip.py` → upload `C:\Users\ogunn\Downloads\IsotopePINN_Project.zip` as a Kaggle **Dataset**.
2. Upload `v3_pilstm/PI_LSTM_ISEF_EASY.ipynb` as a new notebook, attach your dataset (**Add Data**), set **Settings → Accelerator: GPU** and **Internet: On**.
3. Press **Run All**. Pick `SMOKE` first (~15 min rehearsal), then `A`, `B`, `C`, `EVIDENCE` in that order.

The notebook prints exactly which files to download from the **Output** panel and
where they go in your local repo. Everything below is the original manual recipe —
you only need it if you want full control.

---

Train the same Run C recipe that hit **~0.51%** Ac-225 median rel error on Colab, using Kaggle’s free GPU.

## Why Kaggle

- Free GPU (T4 / P100), no Colab quota burn
- Session ~9–12 hours — enough for **fast** mode; **quality** mode may need a long session or early-stop
- Download results from the notebook **Output** tab

## 1. Build a fresh zip on your PC

From the parent repo (folder that contains `build_colab_zip.py`):

```powershell
cd "c:\Users\ogunn\Downloads\New folder"
python build_colab_zip.py
```

Upload **`C:\Users\ogunn\Downloads\IsotopePINN_Project.zip`** (not an old copy).

The zip must include a `train_pi_lstm.py` that prints `val Ac225 med` and supports `PILSTM_EVAL_EVERY`.

## 2. Create a Kaggle Dataset

1. [kaggle.com](https://www.kaggle.com) → **Datasets** → **New Dataset**
2. Upload `IsotopePINN_Project.zip`
3. Title e.g. `isotope-pinn-project` → Create

## 3. Open the notebook

1. Upload `v3_pilstm/PI_LSTM_RunC_KAGGLE.ipynb` (Code → New Notebook → File → Upload, or copy-paste)
2. **Settings** (right panel):
   - **Accelerator → GPU**
   - **Internet → On**
3. **Add Data** → attach your dataset
4. **Run All**

## 4. Modes (Cell 3)

| Mode | Env knobs | Expect |
|------|-----------|--------|
| **`fast`** (default) | float32, `EVAL_EVERY=25`, batch 32, early-stop at 1% | ~2–4 hr on T4 |
| **`quality`** | float64, eval every epoch (Colab A100 match) | ~5–8+ hr on T4 |

Change in Cell 3:

```python
MODE = 'fast'      # or 'quality'
```

## 5. After training

1. Cell 4 verifies Run C config (`hidden=256`, `time_fourier=16`, `hard_ic=True`)
2. Cell 5 runs compare / validate / plots
3. Cell 6 writes `/kaggle/working/PI_LSTM_Results.zip`
4. Download from **Output** → copy into your local `New folder/New folder/`

Protected copy: `/kaggle/working/pi_lstm_best_RUN_C_FINAL.pth`

## If the session dies mid-train

Best weights are already at:

`/kaggle/working/IsotopePINN/v3_pilstm/weights/pi_lstm_best.pth`

Re-run Cells 1–2, **skip Cell 3**, run Cells 4–6.

## Sanity checks

Logs must look like:

```text
epoch    10/6000 | ... | val Ac225 med=0.90.. best=...@...
```

If you see `traj Ac225` instead of `val Ac225 med`, the zip is stale — rebuild and re-upload the dataset.

## Colab fallback

If you get Colab GPU again: use `PI_LSTM_RunC_COLAB.ipynb` with the same zip on Drive (`MODE = 'quality'` by default).
