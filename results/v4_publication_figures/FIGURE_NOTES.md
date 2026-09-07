# V4 Publication Figure Notes

These figures were regenerated from the frozen V4 ZIP without training or tuning.

## Figure 1: Locked model comparison

- 48 locked modeled-proxy families.
- The set was locked for the V4 comparison protocol but had previously been evaluated in the V2 geometry diagnostic; it is not a fully untouched external test.
- LSTM median error: 1.83%; p95: 3.93%.
- GP median error: 1.97%; p95: 3.79%.
- The paired-bootstrap 95% interval for LSTM p95 minus GP p95 is [-0.43, 0.37] percentage points, so the ordering is not resolved.

## Figure 2: Reliability and domain behavior

- Calibrated joint coverage is 95.8% for both models.
- Raw GP +/-2 SD joint coverage is only 2.1%, so GP uncertainty must be calibrated before use.
- The shared guard rejects 100% of explicit outside-range cases and falsely rejects 0.0% of locked cases.
- Accuracy is not reported for rejected outside-range predictions.

## Figure 3: Ac-225 schedule propagation

- GP Ac-225 inventory p95 error: 2.04%.
- LSTM Ac-225 inventory p95 error: 3.03%.
- These are modeled normalized inventories propagated with exact five-nuclide equations, not measured activities.

## Claim boundary

The figures support a frozen architecture comparison on modeled finite-geometry/TENDL proxy data. They do not establish experimental, reactor, clinical, production-cost, or universal under-3% performance.
