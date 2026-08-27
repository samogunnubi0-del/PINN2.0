import os
import zipfile
import shutil
from pathlib import Path

ROOT = Path(r"c:\Users\ogunn\Downloads\New folder\New folder").resolve()
OUT_ZIP = Path(r"c:\Users\ogunn\Downloads\New folder\IsotopePINN_repo.zip")
ALT_ZIP = ROOT / "IsotopePINN_repo.zip"

EXCLUDE_DIRS = {
    ".venv", "venv", "__pycache__", ".git", "archive", "kaggle_results",
    "kaggle_kernel", "kaggle_dataset_staging", ".streamlit", ".cursor", "scratch",
    "tmp", "smoke_tmp", "colab_runs"
}

SKIP_FILES = {
    ".env", "nuclear.code-workspace", "_gen_colab_nbs.py", "_gen_colab_runs.py",
    "build_ncsu_pitch_deck_v2.py", "rebuild_isotopepinn_repo_zip.py"
}

SKIP_SUFFIXES = (".pyc", ".pyo", ".tmp", ".zip", ".log")

def should_include(full: Path, rel: Path) -> bool:
    if rel.name in SKIP_FILES:
        return False
    if rel.name.endswith(SKIP_SUFFIXES):
        return False
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    # Exclude large CSV/log files over 1 MB in results/
    if "results" in rel.parts:
        try:
            if full.stat().st_size > 1_000_000:
                return False
        except OSError:
            return False
    return True

def create_zip():
    print(f"Building complete {OUT_ZIP} from {ROOT}...")
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
        
    count = 0
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
            for fn in filenames:
                full = Path(dirpath) / fn
                rel = full.relative_to(ROOT)
                if not should_include(full, rel):
                    continue
                # Archive name inside zip starts with "IsotopePINN/"
                arc = "IsotopePINN/" + "/".join(rel.parts)
                zf.write(full, arcname=arc)
                count += 1
                
    print(f"Zip created! Size: {OUT_ZIP.stat().st_size / 1024 / 1024:.2f} MB ({count} files)")
    
    # Verify key required files are present
    with zipfile.ZipFile(OUT_ZIP) as zf:
        namelist = set(zf.namelist())
        required = [
            "IsotopePINN/pinn_model.py",
            "IsotopePINN/train.py",
            "IsotopePINN/app.py",
            "IsotopePINN/v3_pilstm/train_pi_lstm.py",
            "IsotopePINN/v3_pilstm/models/pi_lstm.py",
            "IsotopePINN/v3_pilstm/data/trajectory_dataset.py",
            "IsotopePINN/v3_pilstm/weights/pi_lstm_best.pth",
        ]
        for r in required:
            status = "OK" if r in namelist else "MISSING"
            print(f" [{status}] {r}")

    shutil.copy2(OUT_ZIP, ALT_ZIP)
    print(f"Copied to nested location: {ALT_ZIP}")

if __name__ == "__main__":
    create_zip()
