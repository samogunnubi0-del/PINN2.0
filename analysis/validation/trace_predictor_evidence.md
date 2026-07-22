# Trace Daughter Predictor Evidence

This report focuses on the project claim: predicting trace daughter products, especially Ac-225 and Ac-227, in stiff radioactive decay chains.

## Aggregate Held-Out Trace Errors

- Ra-225: median=4.091e-02, p95=1.253e-01, n=22
- Ac-225: median=4.512e-02, p95=9.924e-02, n=22
- Ra-227: median=1.337e-02, p95=3.471e-02, n=22
- Ac-227: median=1.921e-02, p95=4.680e-02, n=22

## Impurity Decision Evidence

Decision confusion table saved to `trace_impurity_decision_confusion.csv`.
Mean decision accuracy: 1.000
Worst false-usable rate: 0.000

## Before/After Comparison

Before/after median relative error:
- Ac-225: 3.956e-01 -> 4.512e-02
- Ac-227: 9.963e-01 -> 1.921e-02
- Ra-225: 9.789e-01 -> 4.091e-02
- Ra-227: 1.000e+00 -> 1.337e-02

## ISEF-Critical Canaries

- fast14_virgin_ac225_zero_collapse: pass, n=1, median_rel=3.872e-02, zero_pred_rate=0.000
- threshold_virgin_ac225_zero_collapse: pass, n=5, median_rel=8.516e-02, zero_pred_rate=0.000
- empty_phantom_atoms: pass, n=12, median_rel=0.000e+00, zero_pred_rate=1.000
- recycled_impurity_decision: pass, n=9, decision_acc=1.000

Trace error plot saved to `trace_error_by_regime_case.png`.
