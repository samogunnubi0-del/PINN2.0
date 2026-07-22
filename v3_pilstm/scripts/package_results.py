"""
Build a complete PI_LSTM_Results.zip for download / local backup.

Usage (from project root):
    python v3_pilstm/scripts/package_results.py
    python v3_pilstm/scripts/package_results.py --output ../PI_LSTM_Results_FIXED.zip
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import zipfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = PROJECT_ROOT / "v3_pilstm"
WEIGHTS = V3_ROOT / "weights" / "pi_lstm_best.pth"
DEFAULT_OUT = PROJECT_ROOT.parent / "PI_LSTM_Results_FIXED.zip"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V3_ROOT))

from models.pi_lstm import PhysicsInformedLSTM  # noqa: E402


def _checkpoint_report() -> dict:
    if not WEIGHTS.exists():
        return {"ok": False, "error": f"missing weights: {WEIGHTS}"}
    blob = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "config" in blob:
        cfg = blob["config"]
    else:
        sd = blob["state_dict"] if isinstance(blob, dict) else blob
        cfg = PhysicsInformedLSTM._config_from_state_dict(sd)
    ok = bool(cfg.get("hidden_dim", 0) >= 128)
    return {"ok": ok, "config": cfg, "weights_bytes": WEIGHTS.stat().st_size}


def _collect_files() -> list[Path]:
    patterns = [
        V3_ROOT / "weights" / "pi_lstm_best.pth",
        PROJECT_ROOT / "analysis" / "benchmark_progress.json",
        PROJECT_ROOT / "analysis" / "benchmark_progress.md",
        V3_ROOT / "results" / "graph_manifest.json",
    ]
    patterns.extend(Path(p) for p in glob.glob(str(V3_ROOT / "results" / "*.json")))
    patterns.extend(Path(p) for p in glob.glob(str(PROJECT_ROOT / "graphs" / "v3_*.png")))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in patterns:
        p = p.resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        out.append(p)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package PI-LSTM v3 results zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = _checkpoint_report()
    print(json.dumps(report, indent=2))
    if not report.get("ok"):
        raise SystemExit(f"Checkpoint check failed: {report.get('error', report)}")

    files = _collect_files()
    if len(files) < 10:
        raise SystemExit(f"Expected >=10 artifacts, found {len(files)}. Run compare/validate/plot first.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            zf.write(path, arcname=rel)
            print("Added:", rel)

    print(f"\nWrote {args.output} ({len(files)} files)")


if __name__ == "__main__":
    main()
