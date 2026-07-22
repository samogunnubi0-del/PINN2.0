# Vast.ai — PI-LSTM Run C (copy-paste playbook)

## A. Rent the machine (~3 min)

1. [cloud.vast.ai](https://cloud.vast.ai) → **Search**
2. Filters: GPU RAM ≥ 16 GB, Disk ≥ 30 GB, CUDA ≥ 12
3. Pick **A100 40GB** (~$0.50–1.50/hr) — best value for this job
4. **Rent** → Image: `pytorch/pytorch` or Jupyter + CUDA 12
5. Disk: **40 GB**, type: **on-demand** (not interruptible)
6. Wait for **Running** → click **Open Jupyter**

---

## B. Upload zip (~1 min)

1. Jupyter file browser → go to **`/workspace`**
2. **Upload** `IsotopePINN_Project.zip` from your PC  
   (`C:\Users\ogunn\Downloads\IsotopePINN_Project.zip`)
3. **Do not click the zip** in Jupyter (UTF-8 error is harmless — ignore it)

If upload lands at `/IsotopePINN_Project.zip` instead, the script copies it automatically.

---

## C. One command to start everything

**Jupyter → New → Terminal**, paste:

```bash
# If zip is only at root:
cp -n /IsotopePINN_Project.zip /workspace/ 2>/dev/null || true

# Download one-shot script (or skip if already in zip after unzip):
cd /workspace
unzip -q -o IsotopePINN_Project.zip "v3_pilstm/scripts/vast_one_shot.sh" -d IsotopePINN 2>/dev/null || true
sed -i 's/\r$//' IsotopePINN/v3_pilstm/scripts/vast_one_shot.sh 2>/dev/null || true
bash IsotopePINN/v3_pilstm/scripts/vast_one_shot.sh
```

**Or manual full block** (if script missing from zip):

```bash
cp -n /IsotopePINN_Project.zip /workspace/ 2>/dev/null || true
cd /workspace && unzip -q -o IsotopePINN_Project.zip -d IsotopePINN
cd /workspace/IsotopePINN
find v3_pilstm -type f \( -name '*.py' -o -name '*.sh' \) -exec sed -i 's/\r$//' {} +
pip install -q numpy scipy matplotlib pandas
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

export PYTHONPATH="/workspace/IsotopePINN:/workspace/IsotopePINN/v3_pilstm"
export PILSTM_FLOAT64=1 PILSTM_CKPT_METRIC=endpoint_ac225 PILSTM_EPOCHS=6000
export PILSTM_PRETRAIN_FRAC=0.20 PILSTM_GRAD_BALANCE=1 PILSTM_DISTILL=1
export PILSTM_DISTILL_WEIGHT=5 PILSTM_DATA_WEIGHT=35 PILSTM_PHYSICS_WEIGHT=20
export PILSTM_MASS_WEIGHT=10 PILSTM_CAUSAL_EPS=2.5 PILSTM_N_TRAIN=1400
export PILSTM_N_STEPS=64 PILSTM_N_VAL=22 PILSTM_N_TEST=22 PILSTM_BATCH=16
export PILSTM_HIDDEN=256 PILSTM_FOURIER=8 PILSTM_TIME_FOURIER=16 PILSTM_HARD_IC=1
export PILSTM_OVERSHOOT_WEIGHT=20 PILSTM_LBFGS_ITER=60 PILSTM_LOG_WEIGHT=2.0
export PYTHONUNBUFFERED=1

nohup python3 -u v3_pilstm/train_pi_lstm.py > /workspace/pi_lstm_train_log.txt 2>&1 &
echo "PID=$!  — safe to close browser tab"
tail -f /workspace/pi_lstm_train_log.txt
```

Press **Ctrl+C** to stop watching (training keeps running).

---

## D. What to expect (timeline)

| Time | Log shows |
|------|-----------|
| 0 min | `PI-LSTM training \| device=cuda ...` |
| 20–45 min | *(silent — building 1444 ODE scenarios on CPU)* |
| ~30–45 min | `Distillation teacher loaded from ...` |
| shortly after | `epoch     1/6000 \| ...` |
| every ~300 epochs | `epoch   300/6000 \| ...` |
| ~3–6 hr total | `Done. Test traj Ac-225 ...` |

**Check anytime** (new terminal tab):

```bash
tail -20 /workspace/pi_lstm_train_log.txt
ps aux | grep train_pi_lstm
```

---

## E. Package + download (when Done appears)

```bash
cd /workspace/IsotopePINN
export PYTHONPATH="/workspace/IsotopePINN:/workspace/IsotopePINN/v3_pilstm"
export PILSTM_N_STEPS=64
python3 v3_pilstm/analysis/compare_models.py

python3 - <<'PY'
import glob, zipfile
out = "/workspace/PI_LSTM_Results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("v3_pilstm/weights/pi_lstm_best.pth")
    for p in glob.glob("v3_pilstm/results/*.json"):
        zf.write(p, p)
    for p in glob.glob("graphs/v3_*.png"):
        zf.write(p, p)
    zf.write("/workspace/pi_lstm_train_log.txt", "pi_lstm_train_log.txt")
print("Wrote", out)
PY

ls -lh /workspace/PI_LSTM_Results.zip
```

Jupyter file browser → `/workspace/PI_LSTM_Results.zip` → **right-click → Download**

---

## F. Destroy instance

Vast console → **Destroy** (stops billing immediately)

---

## G. Copy to your PC

Unzip into your project:

- `v3_pilstm/weights/pi_lstm_best.pth` → rename backup to `pi_lstm_best_RUN_C_FINAL.pth`
- `v3_pilstm/results/*.json`
- `graphs/v3_*.png`

---

## Good result

`compare_v2_pilstm.json` → PI-LSTM Ac-225 **< 4.5%** (beats v2). Target from Colab: **~0.5%**.

---

## Cost estimate

A100 @ $1/hr × ~4 hr ≈ **$4** (well under your $5 deposit)
