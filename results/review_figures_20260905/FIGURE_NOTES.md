# Figure explanation notes

AI-assisted study material. NOT FOR SUBMISSION.

## 1. V3 runtime

The same 10,000 modeled schedules took 672.68 seconds with Radau and 7.93 seconds with my LSTM. Exact Bateman took 0.14 seconds.

These are local CPU measurements of simplified equations. They do not compare supercomputers or full neutron transport. The recorded training-only break-even estimate is about 62,633 predictions versus Radau; it excludes other development costs.

## 2. V3 matched accuracy

On the specific matched test, my LSTM's reported median was lower than the MLP's. I evaluated four time grids and report the largest grid median.

The result uses a numerical denominator floor and excludes small/zero reference cases. One training seed does not establish a general architectural advantage.

## 3. V3 screening errors

The median was 3.94%, but the 95th percentile was 183.84%. That means a low typical error did not prevent some very large mistakes.

This is a different task and metric from the matched endpoint test. The productive threshold is 1.8852 Bq. Excluded near-zero cases have separate diagnostics in the frozen JSON.

## 4. V4 rate errors

LSTM has a slightly lower median; GP has a slightly lower 95th percentile. The paired uncertainty analysis cannot reliably separate their p95 performance.

These are two reaction-rate errors per family, pooled across 48 families. They are not 96 independent experiments or a measurement of real Ac-225 production accuracy.

## 5. V4 uncertainty coverage

The GP's original bands covered both reference rates in just one of 48 cases. After calibration, both methods covered both rates in 46 of 48 cases.

Raw LSTM uncertainty combines ensemble variation and an error floor. Neither the raw bands nor calibrated intervals include every experimental or nuclear-data uncertainty.

## 6. V4 domain checks

A separate input checker stopped all 184 deliberately outside-range cases. That is why neither model should be trusted just because it returns a number.

Bounds and training-distance checks cannot detect every kind of distribution change. These results do not establish accurate extrapolation or a universal safety guarantee.

## 7. V4 Ac-225 propagation

I passed each model's reaction rates into the exact decay equations. This shows how errors in those rates affect the resulting amount of Ac-225.

Three schedules of the same proxy families do not establish real production accuracy. This test is separate from the V3 trajectory and runtime experiments.
