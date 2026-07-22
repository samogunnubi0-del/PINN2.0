# Predictor Quality Workflow

Use this sequence after code changes. The agent should not run the full training
job unless you explicitly approve it.

## 1. Full retraining

```powershell
Remove-Item Env:PINN_RESUME -ErrorAction SilentlyContinue
$env:PINN_DATA_CACHE="1"
$env:PINN_LBFGS_MAX_ITER="100"
.\venv\Scripts\python.exe train.py
```

This writes:

- `pinn_trained_weights.pth`
- `pinn_loss_history.png`
- `pinn_ac225_pred_vs_true.png`
- `pinn_validation_summary.csv`

## 2. Held-out ODE validation

```powershell
.\venv\Scripts\python.exe analysis\validate_predictor.py
```

This writes:

- `analysis\validation\heldout_validation_details.csv`
- `analysis\validation\heldout_validation_summary.csv`
- `analysis\validation\heldout_parity_all_species.png`

## 3. Predictor quality gate

```powershell
.\venv\Scripts\python.exe analysis\evaluate_quality_gate.py
```

This writes:

- `analysis\validation\predictor_quality_gate.json`

The model should not be called a strong predictor until the quality gate passes
or the remaining failures are clearly explained as limitations.
