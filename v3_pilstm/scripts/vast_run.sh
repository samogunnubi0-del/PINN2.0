#!/usr/bin/env bash
# PI-LSTM Run C — one-shot training on Vast.ai (or any Linux GPU box)
# Usage (after IsotopePINN_Project.zip is in /workspace):
#   cd /workspace && bash vast_run.sh
set -euo pipefail

WORK="${WORK:-/workspace}"
ZIP="${ZIP:-$WORK/IsotopePINN_Project.zip}"
PROJECT="$WORK/IsotopePINN"
EPOCHS="${PILSTM_EPOCHS:-6000}"

echo "=== GPU ==="
nvidia-smi -L || { echo "No GPU"; exit 1; }

if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP"
  echo "Upload IsotopePINN_Project.zip to /workspace first (Jupyter upload or scp)."
  exit 1
fi

if [[ -f "$PROJECT/v3_pilstm/train_pi_lstm.py" ]]; then
  echo "=== Project already extracted — skip unzip ==="
else
echo "=== Unzip (skip stale pi_lstm weights) ==="
mkdir -p "$PROJECT"
python3 - <<'PY'
import os, zipfile, sys
work = os.environ.get("WORK", "/workspace")
project = os.path.join(work, "IsotopePINN")
zip_path = os.environ.get("ZIP", os.path.join(work, "IsotopePINN_Project.zip"))
skip_suffix = "v3_pilstm/weights/pi_lstm_best.pth"
with zipfile.ZipFile(zip_path) as zf:
    for m in zf.infolist():
        if m.is_dir():
            continue
        rel = m.filename.replace("\\", "/").lstrip("/")
        if ".." in rel.split("/"):
            continue
        if rel.endswith(skip_suffix):
            print("Skip stale:", rel)
            continue
        dest = os.path.join(project, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(m) as src, open(dest, "wb") as dst:
            dst.write(src.read())
print("Unzipped to", project)
PY
fi

cd "$PROJECT"
# Fix CRLF if zip was built on Windows
find v3_pilstm -name '*.py' -o -name '*.sh' | while read -r f; do sed -i 's/\r$//' "$f" 2>/dev/null || true; done
export PYTHONPATH="$PROJECT:$PROJECT/v3_pilstm:${PYTHONPATH:-}"

echo "=== GPU fixes (idempotent) ==="
python3 - <<'PY'
from pathlib import Path
p = Path("v3_pilstm/physics/distill.py")
s = p.read_text(encoding="utf-8")
old1 = """flat = features.reshape(b * s, f).to(self.model_dtype)
        out = self.model(flat)
        return out.reshape(b, s, 5)"""
new1 = """flat = features.reshape(b * s, f).to(device=self.device, dtype=self.model_dtype)
        out = self.model(flat)
        return out.reshape(b, s, 5).to(features.device)"""
old2 = """map_location=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        # v2 was trained/exported in float32; keep teacher in float32 for stability.
        self.model_dtype = torch.float32"""
new2 = """map_location=device)
        self.device = device
        self.model_dtype = torch.float32
        self.model.to(device=device, dtype=self.model_dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)"""
for old, new in ((old1, new1), (old2, new2)):
    if new not in s and old in s:
        s = s.replace(old, new)
p.write_text(s, encoding="utf-8")
p = Path("v3_pilstm/analysis/endpoint_eval.py")
s = p.read_text(encoding="utf-8")
old = "out = endpoint.numpy()[0]\n        else:\n            out = traj.numpy()[0, -1]"
new = "out = endpoint.detach().cpu().numpy()[0]\n        else:\n            out = traj.detach().cpu().numpy()[0, -1]"
if new not in s and old in s:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
print("Patches OK")
PY

echo "=== Run C training (epochs=$EPOCHS) ==="
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

python3 -u v3_pilstm/train_pi_lstm.py 2>&1 | tee "$WORK/pi_lstm_train_log.txt"

echo "=== Verify Run C ==="
python3 - <<'PY'
import json, sys
sys.path.insert(0, "v3_pilstm")
from models.pi_lstm import PhysicsInformedLSTM
m = PhysicsInformedLSTM.load("v3_pilstm/weights/pi_lstm_best.pth")
cfg = m.config
print(json.dumps(cfg, indent=2))
assert cfg.get("hidden_dim") == 256
assert cfg.get("n_time_fourier") == 16
assert cfg.get("hard_ic") is True
print("Run C OK")
PY

echo "=== Results scripts ==="
export PILSTM_N_STEPS=64
python3 v3_pilstm/analysis/compare_models.py || true
python3 v3_pilstm/analysis/validate_empirical.py || true
python3 v3_pilstm/scripts/plot_v2_vs_pilstm.py || true

echo "=== Package zip ==="
python3 - <<'PY'
import glob, os, zipfile
work = os.environ.get("WORK", "/workspace")
project = os.path.join(work, "IsotopePINN")
out = os.path.join(work, "PI_LSTM_Results.zip")
rels = ["v3_pilstm/weights/pi_lstm_best.pth"]
rels += glob.glob(os.path.join(project, "v3_pilstm/results/*.json"))
rels += glob.glob(os.path.join(project, "graphs/v3_*.png"))
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in rels:
        if os.path.isfile(path):
            zf.write(path, os.path.relpath(path, project))
    log = os.path.join(work, "pi_lstm_train_log.txt")
    if os.path.isfile(log):
        zf.write(log, "pi_lstm_train_log.txt")
print("Wrote", out)
PY

echo ""
echo "DONE. Download $WORK/PI_LSTM_Results.zip then DESTROY the Vast instance."
