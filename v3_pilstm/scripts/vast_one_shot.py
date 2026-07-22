#!/usr/bin/env python3
"""PI-LSTM Run C — full Vast.ai setup (no bash / CRLF issues)."""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/workspace"))
ZIP = WORK / "IsotopePINN_Project.zip"
ALT_ZIP = Path("/IsotopePINN_Project.zip")
PROJECT = WORK / "IsotopePINN"
LOG = WORK / "pi_lstm_train_log.txt"
SKIP_WEIGHTS = "v3_pilstm/weights/pi_lstm_best.pth"


def fix_crlf(root: Path) -> None:
    for pattern in ("**/*.py", "**/*.sh"):
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            data = p.read_bytes()
            if b"\r" in data:
                p.write_bytes(data.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))


def unzip_project() -> None:
    train_py = PROJECT / "v3_pilstm" / "train_pi_lstm.py"
    if train_py.is_file():
        print("Already extracted — skip unzip")
        return
    print("Unzipping project...")
    with zipfile.ZipFile(ZIP) as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            rel = m.filename.replace("\\", "/").lstrip("/")
            if ".." in rel.split("/") or rel.endswith(SKIP_WEIGHTS):
                continue
            dest = PROJECT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(m))
    print("Unzipped OK")


def patch_gpu_fixes() -> None:
    distill = PROJECT / "v3_pilstm" / "physics" / "distill.py"
    s = distill.read_text(encoding="utf-8")
    reps = [
        (
            "flat = features.reshape(b * s, f).to(self.model_dtype)\n        out = self.model(flat)\n        return out.reshape(b, s, 5)",
            "flat = features.reshape(b * s, f).to(device=self.device, dtype=self.model_dtype)\n        out = self.model(flat)\n        return out.reshape(b, s, 5).to(features.device)",
        ),
        (
            "map_location=device)\n        self.model.eval()\n        for p in self.model.parameters():\n            p.requires_grad_(False)\n        self.device = device\n        # v2 was trained/exported in float32; keep teacher in float32 for stability.\n        self.model_dtype = torch.float32",
            "map_location=device)\n        self.device = device\n        self.model_dtype = torch.float32\n        self.model.to(device=device, dtype=self.model_dtype)\n        self.model.eval()\n        for p in self.model.parameters():\n            p.requires_grad_(False)",
        ),
    ]
    for old, new in reps:
        if new not in s and old in s:
            s = s.replace(old, new)
    distill.write_text(s, encoding="utf-8")

    ep = PROJECT / "v3_pilstm" / "analysis" / "endpoint_eval.py"
    s = ep.read_text(encoding="utf-8")
    old = "out = endpoint.numpy()[0]\n        else:\n            out = traj.numpy()[0, -1]"
    new = "out = endpoint.detach().cpu().numpy()[0]\n        else:\n            out = traj.detach().cpu().numpy()[0, -1]"
    if new not in s and old in s:
        s = s.replace(old, new)
        ep.write_text(s, encoding="utf-8")
    print("Patches OK")


def ensure_deps() -> None:
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import torch  # noqa: F401
        print("Deps OK")
        return
    except ImportError:
        pass
    print("Installing numpy scipy matplotlib pandas torch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy", "matplotlib", "pandas"])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "torch",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])


def main() -> None:
    print("=== 1. Locate zip ===")
    if not ZIP.is_file() and ALT_ZIP.is_file():
        import shutil
        shutil.copy2(ALT_ZIP, ZIP)
    if not ZIP.is_file():
        sys.exit("ERROR: Upload IsotopePINN_Project.zip to /workspace first.")
    print(ZIP, f"({ZIP.stat().st_size / 1e6:.1f} MB)")

    print("=== 2. Unzip ===")
    PROJECT.mkdir(parents=True, exist_ok=True)
    unzip_project()
    fix_crlf(PROJECT)

    print("=== 3. GPU ===")
    subprocess.run(["nvidia-smi", "-L"], check=False)

    print("=== 4. Deps ===")
    ensure_deps()
    import torch
    print(f"torch {torch.__version__} cuda {torch.cuda.is_available()}")

    os.chdir(PROJECT)
    print("=== 5. Patches ===")
    patch_gpu_fixes()

    print("=== 6. Env + train ===")
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{PROJECT}:{PROJECT / 'v3_pilstm'}",
        "PILSTM_FLOAT64": "1",
        "PILSTM_CKPT_METRIC": "endpoint_ac225",
        "PILSTM_EPOCHS": os.environ.get("PILSTM_EPOCHS", "6000"),
        "PILSTM_PRETRAIN_FRAC": "0.20",
        "PILSTM_GRAD_BALANCE": "1",
        "PILSTM_DISTILL": "1",
        "PILSTM_DISTILL_WEIGHT": "5",
        "PILSTM_DATA_WEIGHT": "35",
        "PILSTM_PHYSICS_WEIGHT": "20",
        "PILSTM_MASS_WEIGHT": "10",
        "PILSTM_CAUSAL_EPS": "2.5",
        "PILSTM_N_TRAIN": "1400",
        "PILSTM_N_STEPS": "64",
        "PILSTM_N_VAL": "22",
        "PILSTM_N_TEST": "22",
        "PILSTM_BATCH": "16",
        "PILSTM_HIDDEN": "256",
        "PILSTM_FOURIER": "8",
        "PILSTM_TIME_FOURIER": "16",
        "PILSTM_HARD_IC": "1",
        "PILSTM_OVERSHOOT_WEIGHT": "20",
        "PILSTM_LBFGS_ITER": "60",
        "PILSTM_LOG_WEIGHT": "2.0",
        "PYTHONUNBUFFERED": "1",
    })
    train = PROJECT / "v3_pilstm" / "train_pi_lstm.py"
    log_f = open(LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(train)],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=PROJECT,
    )
    print(f"Training PID={proc.pid}")
    print(f"Log: {LOG}")
    print("Safe to close browser. Watch: tail -f", LOG)
    time.sleep(3)
    if LOG.is_file():
        lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-5:]:
            print(line)


if __name__ == "__main__":
    main()
