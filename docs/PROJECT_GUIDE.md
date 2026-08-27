# Project Guide

This repository contains two related model tracks plus the material used to
validate and present them. Use this guide as the stable map of the workspace.

## Start Here

| Need | Go to |
|---|---|
| Understand the project and run the demo | `README.md` |
| Check which claims are currently supported | `docs/CURRENT_STATUS.md` |
| Run the v2 Streamlit demo | `app.py` |
| Run the v3 research app | `v3_pilstm/app_v3.py` |
| Reproduce v2 checks | `test_single.py` and `analysis/` |

## Workspace Map

| Location | Contents | Working rule |
|---|---|---|
| Root Python files | v2 demo, ODE reference, training, and test entry points | Keep only launchable entry points and core modules here |
| `weights/` | v2 checkpoints | `pinn_best_weights.pth` is the app target; pair every claimed result with a checksum |
| `results/` | Machine-readable v2 run and validation records | Do not overwrite a record without updating its artifact identity |
| `analysis/` | Reproducible v2 evaluation scripts and outputs | Use for held-out metrics and quality gates |
| `graphs/` | Figures generated from recorded results | Regenerate rather than hand-edit figures |
| `v3_pilstm/` | PI-LSTM source, weights, analysis, and research results | Treat as a separate model track with its own evidence |
| `docs/` | Scope, provenance, deployment, claim, and status documentation | Put decisions and human-facing explanations here |
| `colab_runs/`, `kaggle_kernel/`, `kaggle_results/` | Remote-training notebooks and imported outputs | Preserve as experiment provenance, not runtime dependencies |
| `poster/`, `ISEF_Planning/`, `*.html` | Presentation and outreach materials | Keep claims synchronized with validated records |
| `archive/`, `backups/` | Historical snapshots | Read-only unless an explicit restoration is needed |

## File Placement Rules

- New v2 evaluation output belongs in `results/` or `analysis/validation/`.
- New v3 evaluation output belongs in `v3_pilstm/results/`.
- New figures belong in `graphs/` or `v3_pilstm/graphs/`, alongside their source
  script or a reference to it.
- Temporary exports, notebook fragments, logs, and build scraps should be ignored
  or placed under `archive/`; they should not live in the root.
- Do not replace a canonical checkpoint in place without recording its SHA-256,
  creation date, training source, and validation output.
- Update `results/artifact_inventory.json` whenever a canonical checkpoint or
  validation record changes.

## Before a Demo, Submission, or Claim

1. Read `docs/CURRENT_STATUS.md`.
2. Confirm the checkpoint checksum matches the validation record.
3. Run the relevant validation script for the model track being presented.
4. Use the matching protocol and metric in posters, decks, and the README.
