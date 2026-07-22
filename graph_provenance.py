"""
Audit trail for PNGs under graphs/: hashes, run identity, data paths.
Used by train.py (Kaggle/local), sync script, and auxiliary plotters.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Any

_MANIFEST_NAME = "graph_manifest.json"
_LAST_TRAINING_RUN = "last_training_run.json"
_LAST_GRAPH_TXT = "LAST_GRAPH_WRITE.txt"
_SYNC_LOG = "sync_log.txt"

_active_project_root: pathlib.Path | None = None
_agent_logged_kaggle_redirect: bool = False


def _kaggle_input_anchor() -> pathlib.Path:
    """Resolved /kaggle/input so ``relative_to`` matches code paths under that tree (Windows + Linux)."""
    return pathlib.Path("/kaggle/input").resolve()


def under_kaggle_input_tree(path: pathlib.Path) -> bool:
    """True if ``path`` resolves under the read-only Kaggle input mount."""
    try:
        path.resolve().relative_to(_kaggle_input_anchor())
        return True
    except ValueError:
        return False


def _kaggle_writable_project_root(project_root: pathlib.Path) -> pathlib.Path:
    """Kaggle mounts /kaggle/input read-only; never write manifests under a code path there."""
    root = project_root.resolve()
    if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
        return root
    if not under_kaggle_input_tree(root):
        return root
    out = pathlib.Path(os.environ.get("PINN_OUTPUT_ROOT", "/kaggle/working")).resolve()
    # #region agent log
    global _agent_logged_kaggle_redirect
    if not _agent_logged_kaggle_redirect:
        _agent_logged_kaggle_redirect = True
        try:
            import json
            import time

            _dbg = pathlib.Path(__file__).resolve().parent / "debug-7b01da.log"
            with _dbg.open("a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "7b01da",
                            "timestamp": int(time.time() * 1000),
                            "location": "graph_provenance.py:_kaggle_writable_project_root",
                            "message": "redirect_results_to_writable_root",
                            "hypothesisId": "H1-kaggle-input-anchor",
                            "data": {
                                "project_root": str(root),
                                "anchor": str(_kaggle_input_anchor()),
                                "writable_root": str(out),
                            },
                        },
                        default=str,
                    )
                    + "\n"
                )
        except OSError:
            pass
    # #endregion
    return out


def training_host() -> str:
    return "kaggle" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "local"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _results_dir(project_root: pathlib.Path) -> pathlib.Path:
    root = _kaggle_writable_project_root(project_root)
    d = root / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_manifest(project_root: pathlib.Path) -> dict[str, Any]:
    path = _results_dir(project_root) / _MANIFEST_NAME
    if not path.is_file():
        return {"version": 1, "artifacts": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "artifacts": {}}


def record_graph_write(
    project_root: pathlib.Path,
    graph_abs_path: pathlib.Path,
    *,
    producer: str,
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hash PNG after save; update graph_manifest.json; refresh LAST_GRAPH_WRITE.txt for train.py."""
    graph_abs_path = graph_abs_path.resolve()
    write_root = _kaggle_writable_project_root(project_root)
    write_root = write_root.resolve()
    try:
        rel = graph_abs_path.relative_to(write_root).as_posix()
    except ValueError:
        try:
            rel = graph_abs_path.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            rel = str(graph_abs_path)

    if not graph_abs_path.is_file():
        raise FileNotFoundError(f"Graph not found after save: {graph_abs_path}")

    stat = graph_abs_path.stat()
    entry: dict[str, Any] = {
        "relative_path": rel,
        "absolute_path": str(graph_abs_path),
        "sha256": sha256_file(graph_abs_path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "written_at_utc": utc_now_iso(),
        "producer": producer,
        "run_id": run_id,
        "host": training_host(),
    }
    if extra:
        entry.update(extra)

    manifest = load_manifest(write_root)
    manifest.setdefault("version", 1)
    manifest.setdefault("artifacts", {})
    manifest["artifacts"][rel] = entry
    manifest["updated_at_utc"] = entry["written_at_utc"]
    _atomic_write_json(_results_dir(write_root) / _MANIFEST_NAME, manifest)

    short = entry["sha256"][:12]
    print(
        f"PROVENANCE WROTE {rel} | bytes={entry['size_bytes']} sha256={short}… | "
        f"run_id={run_id} producer={producer} host={entry['host']}"
    )

    if producer == "train.py":
        txt = _results_dir(write_root) / _LAST_GRAPH_TXT
        lines = [
            f"written_at_utc={entry['written_at_utc']}",
            f"run_id={run_id}",
            f"relative_path={rel}",
            f"sha256={entry['sha256']}",
            f"producer={producer}",
            f"host={entry['host']}",
        ]
        if extra:
            for k in ("data_path", "train_script_rev"):
                if k in extra:
                    lines.append(f"{k}={extra[k]}")
        txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return entry


def training_run_start(
    project_root: pathlib.Path,
    *,
    run_id: str,
    train_script_rev: str,
    data_path: pathlib.Path,
    loss_plot_path: pathlib.Path,
    parity_plot_path: pathlib.Path,
) -> None:
    global _active_project_root
    _active_project_root = _kaggle_writable_project_root(project_root).resolve()
    project_root = _active_project_root
    data_path = data_path.resolve()
    try:
        dst = data_path.stat()
        data_meta = {"data_path": str(data_path), "data_mtime_ns": dst.st_mtime_ns, "data_size": dst.st_size}
    except OSError:
        data_meta = {"data_path": str(data_path), "data_mtime_ns": None, "data_size": None}

    payload = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": utc_now_iso(),
        "train_script_rev": train_script_rev,
        "host": training_host(),
        "loss_plot_path": str(loss_plot_path.resolve()),
        "parity_plot_path": str(parity_plot_path.resolve()),
        **data_meta,
    }
    _atomic_write_json(_results_dir(project_root) / _LAST_TRAINING_RUN, payload)


def training_run_finalize(
    project_root: pathlib.Path,
    *,
    run_id: str,
    status: str,
    loss_plot_path: pathlib.Path,
    parity_plot_path: pathlib.Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """status examples: 'graphs_saved', 'complete', 'incomplete', 'failed'."""
    project_root = _kaggle_writable_project_root(project_root).resolve()
    path = _results_dir(project_root) / _LAST_TRAINING_RUN
    base: dict[str, Any] = {}
    if path.is_file():
        try:
            base = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    base["run_id"] = run_id
    base["status"] = status
    base["ended_at_utc"] = utc_now_iso()
    base["loss_plot_path"] = str(loss_plot_path.resolve())
    base["parity_plot_path"] = str(parity_plot_path.resolve())
    if extra:
        base.update(extra)
    _atomic_write_json(path, base)
    global _active_project_root
    _active_project_root = None


def _atexit_mark_incomplete() -> None:
    root = _active_project_root
    if root is None:
        return
    path = _results_dir(root) / _LAST_TRAINING_RUN
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if data.get("status") != "running":
        return
    data["status"] = "incomplete"
    data["ended_at_utc"] = utc_now_iso()
    data["note"] = "process exited before training_run_finalize (killed, exception, or early sys.exit)"
    _atomic_write_json(path, data)


atexit.register(_atexit_mark_incomplete)


def new_run_id() -> str:
    return str(uuid.uuid4())


def append_sync_log(project_root: pathlib.Path, message: str) -> None:
    line = f"{utc_now_iso()} {message}\n"
    p = _results_dir(project_root) / _SYNC_LOG
    with p.open("a", encoding="utf-8") as f:
        f.write(line)
