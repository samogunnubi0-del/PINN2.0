#!/usr/bin/env bash
# PI-LSTM Run C — full Vast.ai setup (deps + train + package)
# Paste into Jupyter terminal after uploading IsotopePINN_Project.zip to /workspace
set -eu

WORK=/workspace
ZIP="$WORK/IsotopePINN_Project.zip"
ALT_ZIP=/IsotopePINN_Project.zip
PROJECT="$WORK/IsotopePINN"
LOG="$WORK/pi_lstm_train_log.txt"
EPOCHS="${PILSTM_EPOCHS:-6000}"

echo "=== 1. Locate zip ==="
if [[ ! -f "$ZIP" && -f "$ALT_ZIP" ]]; then
  cp "$ALT_ZIP" "$ZIP"
fi
if [[ ! -f "$ZIP" ]]; then
  echo "ERROR: Upload IsotopePINN_Project.zip to /workspace first (Jupyter upload)."
  exit 1
fi
ls -lh "$ZIP"

echo "=== 2. Unzip (skip stale pi_lstm weights) ==="
mkdir -p "$PROJECT"
if [[ ! -f "$PROJECT/v3_pilstm/train_pi_lstm.py" ]]; then
  python3 - <<'PY'
import os, zipfile
work = "/workspace"
project = os.path.join(work, "IsotopePINN")
zip_path = os.path.join(work, "IsotopePINN_Project.zip")
skip = "v3_pilstm/weights/pi_lstm_best.pth"
with zipfile.ZipFile(zip_path) as zf:
    for m in zf.infolist():
        if m.is_dir():
            continue
        rel = m.filename.replace("\\", "/").lstrip("/")
        if ".." in rel.split("/") or rel.endswith(skip):
            continue
        dest = os.path.join(project, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(m) as src, open(dest, "wb") as dst:
            dst.write(src.read())
print("Unzipped OK")
PY
else
  echo "Already extracted — skip unzip"
fi

cd "$PROJECT"
find v3_pilstm -type f \( -name '*.py' -o -name '*.sh' \) -exec sed -i 's/\r$//' {} + 2>/dev/null || true

echo "=== 3. GPU check ==="
nvidia-smi -L

echo "=== 4. Python deps ==="
python3 -c "import numpy, scipy, torch" 2>/dev/null || {
  pip install -q numpy scipy matplotlib pandas
  pip install -q torch --index-url https://download.pytorch.org/whl/cu124
}
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== 5. GPU code patches ==="
python3 - <<'PY'
from pathlib import Path
p = Path("v3_pilstm/physics/distill.py")
s = p.read_text(encoding="utf-8")
old1 = "flat = features.reshape(b * s, f).to(self.model_dtype)\n        out = self.model(flat)\n        return out.reshape(b, s, 5)"
new1 = "flat = features.reshape(b * s, f).to(device=self.device, dtype=self.model_dtype)\n        out = self.model(flat)\n        return out.reshape(b, s, 5).to(features.device)"
old2 = "map_location=device)\n        self.model.eval()\n        for p in self.model.parameters():\n            p.requires_grad_(False)\n        self.device = device\n        # v2 was trained/exported in float32; keep teacher in float32 for stability.\n        self.model_dtype = torch.float32"
new2 = "map_location=device)\n        self.device = device\n        self.model_dtype = torch.float32\n        self.model.to(device=device, dtype=self.model_dtype)\n        self.model.eval()\n        for p in self.model.parameters():\n            p.requires_grad_(False)"
for o, n in ((old1, new1), (old2, new2)):
    if n not in s and o in s:
        s = s.replace(o, n)
p.write_text(s, encoding="utf-8")
p = Path("v3_pilstm/analysis/endpoint_eval.py")
s = p.read_text(encoding="utf-8")
o = "out = endpoint.numpy()[0]\n        else:\n            out = traj.numpy()[0, -1]"
n = "out = endpoint.detach().cpu().numpy()[0]\n        else:\n            out = traj.detach().cpu().numpy()[0, -1]"
if n not in s and o in s:
    s = s.replace(o, n)
    p.write_text(s, encoding="utf-8")
print("Patches OK")
PY

echo "=== 6. Run C env ==="
export PYTHONPATH="$PROJECT:$PROJECT/v3_pilstm"
export PILSTM_FLOAT64=1
export PILSTM_CKPT_METRIC=endpoint_ac225
export PILSTM_EPOCHS="$EPOCHS"
export PILSTM_PRETRAIN_FRAC=0.20
export PILSTM_GRAD_BALANCE=1
export PILSTM_DISTILL=1
export PILSTM_DISTILL_WEIGHT=5
export PILSTM_DATA_WEIGHT=35
export PILSTM_PHYSICS_WEIGHT=20
export PILSTM_MASS_WEIGHT=10
export PILSTM_CAUSAL_EPS=2.5
export PILSTM_N_TRAIN=1400
export PILSTM_N_STEPS=64
export PILSTM_N_VAL=22
export PILSTM_N_TEST=22
export PILSTM_BATCH=16
export PILSTM_HIDDEN=256
export PILSTM_FOURIER=8
export PILSTM_TIME_FOURIER=16
export PILSTM_HARD_IC=1
export PILSTM_OVERSHOOT_WEIGHT=20
export PILSTM_LBFGS_ITER=60
export PILSTM_LOG_WEIGHT=2.0
export PYTHONUNBUFFERED=1

echo "=== 7. Train (background — safe to close browser tab) ==="
echo "Log: $LOG"
nohup python3 -u v3_pilstm/train_pi_lstm.py > "$LOG" 2>&1 &
echo "PID=$!"
echo "Watch: tail -f $LOG"
sleep 3
tail -5 "$LOG" || true
