# V4 Current Project Status

**Updated:** August 28, 2026  
**Current phase:** Frozen computational study; external Berkeley validation pending

## Final research goal

Test whether a constrained surrogate can quickly screen modeled Ra-226 to Ac-225 production conditions, report calibrated uncertainty, reject unfamiliar inputs, and pass every accepted finalist back to trusted physics for verification.

The goal is not to replace Radau, Bateman, transport, reactor, or experimental calculations. The intended value is rapid screening across many candidate conditions.

## What is complete

- Reduced five-nuclide reaction and decay system, including the Ac-227 impurity branch.
- 1,400 Radau-generated V3 trajectories and a frozen 6,000-epoch physics-informed LSTM.
- Parameter-matched MLP comparison, shifted testing, and a 10,000-schedule decision audit.
- V4 frozen comparison between a Gaussian process and a five-seed LSTM ensemble.
- Shared data split: 160 train, 24 selection, 24 calibration, and 48 locked test families.
- Calibrated uncertainty, sparse-corner testing, explicit outside-range rejection, and paired-bootstrap comparison.
- Exact five-nuclide propagation of frozen V4 rate predictions to three Ac-225 schedules.
- Reproducible publication figures in PNG and vector PDF formats, with source and output hashes.

## Frozen V4 result

| Metric | GP | LSTM ensemble |
|---|---:|---:|
| Locked median rate error | 1.97% | 1.83% |
| Locked p95 rate error | 3.79% | 3.93% |
| Calibrated 95% joint coverage | 95.8% | 95.8% |
| Explicit outside-range rejection | 100% | 100% |
| Ac-225 inventory p95 error | 2.04% | 3.03% |

Both models passed every predeclared gate. The paired bootstrap did not resolve their ordering, so there is no scientific winner. GP is only the provisional leader because its observed p95 is slightly lower.

## Important findings

- The 48-case V4 set was locked for this comparison protocol but had previously been evaluated in the V2 geometry diagnostic. It must not be described as a completely untouched external test.
- GP raw +/-2 SD intervals had only 2.1% joint coverage. GP uncertainty is usable only after calibration.
- All 184 explicit outside-range cases were rejected. Accuracy is not claimed for rejected cases.
- V4 uses static one-step geometry inputs, so this experiment cannot prove that LSTM memory helps.
- The first Berkeley reconstruction was outside V3 support and overpredicted activity by about 600 times. This failure was preserved rather than tuned away.

## What is still required

1. Obtain or document the unavailable Berkeley inputs: target-integrated spectrum or fluence, geometry/beam corrections, beam-current history, spectrum uncertainty, and an Ac-227 detection limit if available.
2. Reconstruct the 33 and 40 MeV Berkeley cases with units and reference times checked.
3. Run the frozen external comparison once without selecting or tuning on its answer.
4. Obtain one expert review of the reaction chain, experiment reconstruction, and claim wording.
5. Freeze the final tables, write the student-authored paper/abstract, and assemble the poster.

If unavailable Berkeley information cannot be obtained, record it as a limitation and finish the project using the preserved outside-domain failure. Do not invent missing inputs or retrain until the two points agree.

## Readiness estimate

- **Computational model development and internal testing:** approximately 90% complete.
- **Complete science-fair evidence package:** approximately 70-75% complete.

The remaining gap is external evidence and reporting, not another architecture search. Stop changing GP/LSTM designs unless the Berkeley reconstruction reveals a documented implementation error.

## Safe current claim

> On a frozen modeled finite-geometry/TENDL proxy task, both GP and five-seed LSTM surrogates achieved below-4% p95 rate error, 95.8% calibrated joint coverage, and 100% rejection of explicit outside-range cases. Their ordering was not statistically resolved. Experimental accuracy remains unestablished.

## Figure use

- **Figure 1:** primary accuracy and GP/LSTM comparison figure.
- **Figure 2:** primary uncertainty and domain-safety figure.
- **Figure 3:** supporting Ac-225 schedule-propagation figure.

All three must be labeled as modeled-proxy evidence on the poster.
