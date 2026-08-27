# Current Project Status

Last checked: 2026-08-04

Machine-readable inventory: `results/artifact_inventory.json`.

## What to use now

| Purpose | Entry point / artifact | Status |
|---|---|---|
| v2 Streamlit demo | `app.py` | Primary demo entry point |
| v2 checkpoint loaded by the app | `weights/pinn_best_weights.pth` | Present; needs evidence revalidation |
| v2 ODE reference | `ra226_ac225_transmutation.py` | Reference implementation |
| v2 validation commands | `test_single.py`, `analysis/validate_predictor.py` | Run before making performance claims |
| v3 PI-LSTM research track | `v3_pilstm/app_v3.py` | Active research track; separate from v2 demo |

## Artifact-integrity warning

The saved validation record `results/v63_validation_20260530.json` identifies its
validated v2 checkpoint with SHA-256 prefix `7c21debe`. The checkpoint currently
at `weights/pinn_best_weights.pth` has SHA-256:

```
77bede87ea783bade5a82700bac3ab3af1fb20d05b03d67ea9ba06532cc2b857
```

That means the record cannot currently prove the reported v2 validation results
for the file the app loads. Treat the historical `6/6 PASS` and `4.51%` held-out
Ac-225 error as results for the recorded artifact, not verified claims about the
current checkpoint, until one of these actions is completed:

1. Restore the checkpoint whose SHA-256 begins `7c21debe`, or
2. Re-run the v2 validation suite and update the validation record with the
   current checkpoint checksum.

The README also contains older v3 checksum text that does not match the current
`v3_pilstm/weights/pi_lstm_best.pth` file. Its current SHA-256 begins `c94d6234`.

## Reporting rule

Only report a metric when the model file, its checksum, and its validation record
agree. Keep comparisons between v2 and v3 tied to the exact protocol named in the
source result file.
