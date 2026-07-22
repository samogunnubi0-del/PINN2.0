# Deploy IsotopePINN to Streamlit Community Cloud

Public demo: **https://lhyjrhmwzxqfpuuwsux7zh.streamlit.app**  
Source repo: **https://github.com/samogunnubi0-del/PINN2.0**

## What the live app shows (current)

| Surface | Content |
|---------|---------|
| **Main file `app.py`** | Interactive **v2 MLP-PINN** screening + new **PI-LSTM** tab with Results-6 metrics/figures |
| **Optional** `v3_pilstm/app_v3.py` | Live PI-LSTM vs ODE trajectory (run locally or as a second Cloud app) |

Headline Results-6 numbers come from committed JSON under `v3_pilstm/results/`.

## Required repo files

- `app.py`, `pinn_model.py`, `ra226_ac225_transmutation.py`, `train.py` helpers as imported
- `weights/pinn_best_weights.pth` (SHA-256 prefix `4f461387`)
- `v3_pilstm/weights/pi_lstm_best.pth` (SHA-256 prefix `22b052aa`) — for local `app_v3.py`
- `results/v63_validation_20260530.json`
- `v3_pilstm/results/compare_v2_pilstm.json`, `speed_harness.json`, `conformal_validation.json`
- `graphs/v3_*.png`
- `requirements-streamlit-cloud.txt` (not full `requirements.txt` on Cloud)

## Deploy / update steps

1. Push `main` on **PINN2.0** (this repo).
2. In [share.streamlit.io](https://share.streamlit.io/) → your app → **Settings**:
   - Repository: `samogunnubi0-del/PINN2.0`
   - Branch: `main`
   - Main file: `app.py`
   - Requirements: `requirements-streamlit-cloud.txt`
   - Python: 3.11
3. **Reboot app** (or wait for auto-redeploy on push).
4. Optional second app: main file `v3_pilstm/app_v3.py` for live PI-LSTM trajectories.

## Cold start

Free tier apps sleep after inactivity. First open after sleep may take ~45–90 seconds (PyTorch + weight load). Model load is cached via `@st.cache_resource`.

## Local test

```powershell
cd "C:\Users\ogunn\Downloads\New folder"
.\.venv\Scripts\Activate.ps1   # if present
pip install -r requirements.txt
streamlit run app.py
# flagship live trajectory:
streamlit run v3_pilstm/app_v3.py
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Old ~4.5%-only hero | Confirm Cloud points at **PINN2.0** `main`, then reboot |
| No PI-LSTM figures | Ensure `graphs/v3_*.png` and compare JSON are on `main` |
| "No PINN weights" | Commit `weights/pinn_best_weights.pth` |
| Build OOM / slow torch | Use `requirements-streamlit-cloud.txt` (CPU wheels) |
