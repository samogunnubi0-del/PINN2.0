# IsotopePINN

Physics-informed neural network surrogate for **Actinium-225** production planning in the Ra-226 transmutation chain — relevant to **targeted alpha therapy (TAT)** radiopharmaceutical supply.

**Investigator:** Sam Ogunnubi  
**Technical mentor:** Jaden Palmer (PhD researcher, ARTISANS Lab, NC State Nuclear Engineering)

**Live demo:** [https://lhyjrhmwzxqfpuuwsux7zh.streamlit.app](https://lhyjrhmwzxqfpuuwsux7zh.streamlit.app)  
(Deploy details: [Streamlit Cloud](docs/STREAMLIT_CLOUD_DEPLOY.md) · `requirements-streamlit-cloud.txt`)

**Collaboration one-pager:** [Ac225_PINN_Executive_OnePager.html](Ac225_PINN_Executive_OnePager.html) (open in a browser)  
**ISEF board pack (PI-LSTM):** [v3_pilstm/results/ISEF_BOARD_PACK.md](v3_pilstm/results/ISEF_BOARD_PACK.md)

> Seeking collaborators and reviewers (nuclear data, impurity modeling, experimental anchors, ISEF packaging). Open a GitHub issue or email **sam.ogunnubi0@gmail.com**.

---

## Summary

| Item | Detail |
|------|--------|
| **Problem** | Ac-225 is scarce; planning irradiation (flux, energy, time) requires many stiff ODE solves |
| **Approach** | 0D five-species Bateman ODE reference (NNDC/JENDL) + physics-informed surrogates (v2 MLP-PINN and v3 PI-LSTM) |
| **Validation** | Six independent checks vs ODE for v2; held-out Ac-225 median **~4.5%** (v2 solo protocol). PI-LSTM Results-6 beats frozen v2 on the paired endpoint protocol (**5.12%** vs **8.18%**) |
| **Not validated** | Laboratory reactor data, patient pharmacokinetics, or 3D transport (MCNP/OpenMC) |

This repo is **Python / PyTorch / Streamlit**. The single `.html` file is a printable outreach brief, not the model code.

---

## Two model tracks

| Track | What it is | Canonical artifact |
|-------|------------|-------------------|
| **v2 MLP-PINN (frozen demo)** | Original Streamlit app + differential physics loss | `weights/pinn_best_weights.pth`, `app.py` |
| **v3 PI-LSTM (Results-6)** | LSTM + exact/`expmix` physics loss, conformal UQ, speed harness | `v3_pilstm/weights/pi_lstm_best.pth` |

### v2 results (ODE reference validation)

| Check | Result |
|-------|--------|
| Empty-target safety (no production from zero inventory) | PASS |
| Production scenario (14 MeV, full Ra-226 feed) | PASS (&lt;10% Ac-225 vs ODE) |
| Decay-chain ingrowth (Ra-225 → Ac-225, no flux) | PASS |
| Species quality gate | PASS |
| PINN vs ODE correlation | PASS |
| Held-out scenarios (22 cases) | **4.51%** median Ac-225 error |

Full report: [`results/v63_validation_20260530.json`](results/v63_validation_20260530.json)

Weights checksum (SHA-256 prefix): `4f461387` — file [`weights/pinn_best_weights.pth`](weights/pinn_best_weights.pth)  
PI-LSTM Results-6 checksum (SHA-256 prefix): `22b052aa` — file [`v3_pilstm/weights/pi_lstm_best.pth`](v3_pilstm/weights/pi_lstm_best.pth)

### v3 PI-LSTM Results-6 (paired comparison protocol)

| Metric | Value |
|--------|-------|
| Ac-225 endpoint median relative error (22 held-out) | **5.12%** |
| Frozen v2 on the **same** comparison protocol | 8.18% |
| Batched CPU inference | ~**1.65 ms**/scenario (~83× vs eager loop) |
| Split conformal Ac-225 relative coverage (α=0.1) | ~**90.9%** |

Sources: `v3_pilstm/results/compare_v2_pilstm.json`, `speed_harness.json`, `conformal_validation.json`, [`ISEF_BOARD_PACK.md`](v3_pilstm/results/ISEF_BOARD_PACK.md).

> The v2 solo headline **4.51%** and the paired-protocol v2 value **8.18%** are **not directly comparable** — different scoring pipelines. Use 4.51% for the frozen demo story; use 5.12% vs 8.18% for PI-LSTM progress vs v2.

---

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **Overview → Validation → About** in the app.

### PI-LSTM (v3)

```bash
streamlit run v3_pilstm/app_v3.py
# optional retrain (GPU recommended):
python v3_pilstm/train_pi_lstm.py
```

---

## Reproduce validation (v2)

```bash
pip install -r requirements.txt
python test_single.py
python analysis/validate_predictor.py
python analysis/evaluate_quality_gate.py
python analysis/correlation_check.py
```

Expected: all gates PASS; held-out Ac-225 median ~0.045 vs ODE.

### PI-LSTM compare / conformal / speed (after weights present)

```bash
python v3_pilstm/analysis/compare_models.py
python v3_pilstm/analysis/run_conformal_validation.py
python v3_pilstm/scripts/speed_harness.py
```

---

## Training (optional)

Full v2 retrain is GPU-heavy (~4k epochs with physics pretrain):

```bash
python train.py
```

Colab-friendly entry: [`IsotopePINN_Colab_Run.ipynb`](IsotopePINN_Colab_Run.ipynb). Env vars are documented in `train.py` / `v3_pilstm/COLAB_SETUP.md`.

---

## Deploy (Streamlit Cloud)

1. This repo is already on GitHub: [samogunnubi0-del/PINN2.0](https://github.com/samogunnubi0-del/PINN2.0)
2. Connect at [share.streamlit.io](https://share.streamlit.io/) → main file `app.py`
3. Requirements file: **`requirements-streamlit-cloud.txt`** (CPU PyTorch)
4. Ensure `weights/pinn_best_weights.pth` and `results/v63_validation_20260530.json` are present (they are in this repo)

Details: [`docs/STREAMLIT_CLOUD_DEPLOY.md`](docs/STREAMLIT_CLOUD_DEPLOY.md)

First load after idle sleep may take ~45–90 seconds (PyTorch + model load).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `app.py` | Interactive demo (Overview, Screening, Validation, Methods, About) |
| `pinn_model.py`, `train.py` | v2 PINN architecture and training |
| `ra226_ac225_transmutation.py` | Stiff ODE reference (Radau) |
| `test_single.py` | Scenario integrity tests (Trio A/B/C) |
| `baseline_lstm.py` | Ablation baseline (not the flagship) |
| `analysis/` | Held-out validation, quality gate, correlation |
| `results/` | Validation JSON (including v63 report) |
| `weights/` | Canonical v2 checkpoint |
| `v3_pilstm/` | PI-LSTM models, physics losses, Results-6 weights, board pack |
| `graphs/` | Key comparison / calibration figures |
| `docs/DATA_ASSUMPTIONS.md` | Nuclear data sources and modeling scope |
| `Ac225_PINN_Executive_OnePager.html` | Outreach / collaborator one-pager (browser) |
| `ISEF_Planning/project_history.md` | Research and mentorship history |

---

## Modeling scope

This is a **0D lumped** well-mixed target model (scalar flux and energy), not a full reactor or patient dose model. Chemistry, recovery yield, and shipping delays are post-processed in the app, not inside the PINN loss. Cyclotron / linac / spallation routes are out of scope for the fitness function used here.

## Limitations

- All reported errors are **model vs ODE**, not vs experiment.
- Largest v2 errors near **epithermal (~9.5%)** and **threshold (~8.5%)** neutron energies.
- PI-LSTM is still weaker than v2 on some **Ac-227 / thermal (n,γ)** anchors.
- Do **not** use for regulatory release or clinical dosing without separate assay and qualified review.

## References

Methods and citations are listed in the app **About** tab (Raissi et al. PINN framework; NNDC/JENDL nuclear data). Mentorship narrative: [`ISEF_Planning/project_history.md`](ISEF_Planning/project_history.md).

## License

MIT — see [LICENSE](LICENSE).
