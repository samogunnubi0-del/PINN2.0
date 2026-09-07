# IsotopePINN: Read-Only Code Audit

**Prepared for Samuel Ogunnubi | September 7, 2026**

Internal AI-assisted engineering review. Not a student-authored submission, experimental validation, or certification of scientific correctness.

## Bottom Line

**The completed V4 results still replay correctly, but the project has defects worth addressing before extending it.** This review identified seven code/reliability findings and one reporting finding. This is a count of substantiated findings in the reviewed paths, not an estimate of every bug in the repository.

The most important numerical defect is that a long single interval can make the shared Bateman propagator produce `inf`/`nan` for Ra-227 and Ac-227. Other problems concern restarting runs, reading scientific data, uncertainty calculations, and distinguishing time savings from money savings. An older poster also contains claims that the authoritative project records explicitly do not support.

**No existing code, notebook, checkpoint, data, graph, ledger, or presentation was edited. No model was retrained. This document is the only project artifact created by this review.** Tests used temporary operating-system locations where necessary, and Python bytecode writing was disabled.

## Findings At A Glance

P1 means address before relying on the affected workflow. P2 means a conditional defect or reliability gap that should be addressed before broader use. Priority does not imply that the frozen run was corrupted.

| ID | Priority | Finding | Completed V4 result affected? |
|---|---|---|---|
| A1 | P1 | Exact propagator overflows for long intervals | Not reproduced on its short comparison schedules |
| A2 | P1 | Completed runs/checkpoints can be reused under an incompatible request | No evidence the archived run was mixed; future restart risk |
| A3 | P1 reporting | Unmarked older poster contains unsupported validation and speed claims | Presentation problem, not a change to stored metrics |
| A4 | P2 | GP observation-noise variance is not rescaled with normalized targets | Frozen metrics remain observed results; important for future noisy teacher data |
| A5 | P2 | General calibration helper silently caps an impossible finite-sample quantile | Frozen comparison uses a separate, correctly guarded helper |
| A6 | P2 | Malformed cross-section rows are silently discarded | No demonstrated corruption in the supplied TENDL tables |
| A7 | P2 | Cost-claim eligibility can pass when the workflow costs more | No real cost-superiority claim has been established |
| A8 | P2 | Training checkpoints are written directly, despite an atomic-write preflight | Interruption risk; no corruption found in the returned ZIP |

## A1. Long Intervals Break The Exact Propagator

**Location:** [integrated_loss.py:98](<C:/Users/ogunn/Downloads/New folder/New folder/v3_pilstm/physics/integrated_loss.py:98>), used by [hybrid_model.py:86](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/hybrid_model.py:86>).

`_phi1_diff` computes a difference of decaying exponentials using `exp(-m*dt) * sinh(d*dt)`. This avoids cancellation for similar decay constants, but `sinh` overflows at large arguments even when the final physical answer should be finite. Multiplying an overflowing term by a tiny exponential does not recover that answer. The later nonnegative clamp does not remove NaNs.

**Reproduced:** Initial normalized inventory `[1, 0, 0, 0, 0]`, `(n,2n)` rate `1e-5 /h`, capture rate `1e-6 /h`, and one 1,500-hour interval produced finite Ra-226, Ra-225, and Ac-225, but non-finite Ra-227 and Ac-227. This occurred under both `ODE_DATA_VERSION=v1` and `v2`. A SciPy matrix exponential of the same normalized five-species system remained finite. At 2,000 and 4,000 hours the two impurity-branch outputs were NaN under the default version.

**Why it matters:** A longer irradiation or cooling calculation can lose the impurity result even though its input is accepted as valid. The function's statement that it is exact at any interval is numerically too strong.

**Suggested fix, not applied:** Use a stable decaying-exponential/`expm1` formulation with a properly handled equal-rate limit, or a stable matrix-exponential fallback. Guard inactive branches as well as selected branches so gradients remain finite. Assert finite outputs. Test 0, 24, 240, 720, 1,500, and 4,000 hours, equal/near-equal rates, gradients, and agreement between one long interval and several shorter intervals.

**Past-result boundary:** The V4 comparison uses irradiation intervals of 24, 120, and 240 hours and cooling intervals of 0, 24, and 72 hours. The reproduced overflow does not establish an error in those saved results. Existing V3 tests also passed. Repairing this helper does not require discarding trained weights; any affected downstream calculation should receive a new, explicitly versioned evaluation rather than overwriting old evidence.

## A2. Restart Can Return A Different Experiment's Result

**Locations:** [model_comparison.py:1313](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/model_comparison.py:1313>), [model_comparison.py:593](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/model_comparison.py:593>), and [model_comparison.py:794](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/model_comparison.py:794>).

The completed-result shortcut checks only protocol, model kind, and `run_complete`. It returns before validating the requested dataset, seed list, hidden width, training settings, or artifact manifest. Per-seed checkpoint reuse checks some architecture fields but not the full data/training request.

**Reproduced without writing files:** A mocked completed LSTM summary with seed `[42]` and hidden width `8` was accepted when requesting seeds `[314, 1618]`, hidden width `64`, 9,999 epochs, and a nonexistent dataset. It printed `Already complete` and returned the old summary.

**Why it matters:** Reusing a Drive directory can make the notebook appear to have completed a new request when it has actually loaded the old experiment. A partially completed run can also reuse old members with changed batch size, patience, or other unbound settings. `--overwrite` is not a complete solution because checkpoint resume remains independently enabled unless disabled.

**Suggested fix, not applied:** Create one immutable request fingerprint including data and split hashes, code identity, seeds, architecture, optimizer/scheduler settings, batch size, epoch budget, and patience. Store it in every progress checkpoint, completed member, and result summary. Require agreement before reuse and verify required artifact hashes. Changed requests should use a new run identity. Add tests for every mismatch and for missing/corrupt result artifacts.

**Past-result boundary:** The returned official ZIP passed hash and metric replay. This finding does not demonstrate that its members were mixed. It is a restart-contract defect, not a reason to retrain the completed comparison.

## A3. The Older Poster Still Makes Withdrawn Claims

**Locations:** [ISEF_Official_Poster_36x48.html:317](<C:/Users/ogunn/Downloads/New folder/ISEF_Official_Poster_36x48.html:317>), [line 356](<C:/Users/ogunn/Downloads/New folder/ISEF_Official_Poster_36x48.html:356>), and [line 420](<C:/Users/ogunn/Downloads/New folder/ISEF_Official_Poster_36x48.html:420>).

The poster describes a 40-hour solver run, a 1.00x Joyo national-laboratory match, a 1,000x speedup, and FDA compliance. It lacks the withdrawal banner present on the old academic-paper HTML. These statements conflict with the current project evidence and mix older methods with final-result language.

The authoritative [project record](<C:/Users/ogunn/Downloads/New folder/New folder/docs/PROJECT_RECORD_INDEX.md>) instead records an 84.9x V3 speedup against Radau for the specific 10,000-schedule reduced problem, a 56.7x slowdown against the exact analytic baseline, and a failed out-of-domain Berkeley transfer. A fitted spectrum fraction is calibration, not independent proof of a model's experimental accuracy. No regulatory compliance is established by the saved model-comparison results.

**Suggested fix, not applied:** Do not send or submit this poster in its present form. Mark it as withdrawn/historical and create a separate current poster whose numbers and captions each point to a saved artifact. Preserve the old file for history. Treat the similarly overclaiming academic draft as historical too; its existing banner already acknowledges the issue.

**Compute required:** None. This is a reporting correction, not a request for another training run. This review did not independently certify or interpret an FDA impurity standard; the issue is that the poster claims compliance without supporting project evidence.

## A4. GP Noise Variance Uses The Wrong Scale

**Location:** [multifidelity_models.py:239](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/multifidelity_models.py:239>).

The code converts supplied relative measurement uncertainty into log-rate variance, then passes that variance as `alpha` while using `normalize_y=True`. Scikit-learn standardizes the target but adds the supplied `alpha` to the covariance diagonal without automatically dividing it by the target variance. Consequently, the supplied noise and fitted targets are in different units. This was checked against the installed implementation and the [official GaussianProcessRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).

**Reproduced:** With 10% supplied relative uncertainty, the code passed `alpha = 0.00995033` for each output. For test log-target variances `0.00428571` and `0.01714286`, the corresponding normalized variances should have been approximately `2.32174` and `0.580436` under the code's noise transformation. This was an eight-row local CPU diagnostic with kernel optimization disabled, not retraining a project model.

**Why it matters:** Future Monte Carlo or measured teacher uncertainties could be substantially underweighted or overweighted. Conformal calibration is a separate interval adjustment; it does not correct how training observations were weighted.

**Suggested fix, not applied:** Explicitly standardize each output and its noise variance together, then invert the transform at prediction, or scale `alpha` consistently with scikit-learn's target normalization, including constant-target handling. Test invariance to rescaling log targets and check a deliberately noisy observation's influence.

**Past-result boundary:** Frozen GP fitting did not supply measured per-case uncertainties; it used a tiny floor and a fitted WhiteKernel. Its reported errors and coverage remain replayable observations. This audit does not establish that this defect caused the poor raw GP joint coverage. Any corrected GP experiment must be versioned separately; it need not trigger an LSTM/A100 rerun.

## A5. A Second Calibration Helper Overstates Small-Sample Support

**Location:** [multifidelity_models.py:67](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/multifidelity_models.py:67>), called by the general GP and ridge calibration methods.

The helper calculates the corrected conformal rank and silently caps it at the number of available cases. When the requested rank is larger than the dataset, this turns an unavailable finite bound into a finite interval. The separate frozen-comparison helper correctly raises an error in this situation.

**Reproduced:** Three calibration scores `[1, 2, 3]` with requested coverage `0.95` returned the finite value `3`. The required rank is `ceil((3+1)*0.95) = 4`, which is unavailable. Under continuous exchangeable scores, using the maximum of three supports 3/4 coverage, not a distribution-free 95% guarantee. See the finite-sample construction in [Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511).

**Suggested fix, not applied:** Reject insufficient calibration size or explicitly return an unbounded interval. Reuse one audited quantile implementation while keeping marginal and joint calibration definitions distinct. Validate finite scores and add small-sample tests to both GP and ridge paths.

**Past-result boundary:** The frozen GP/LSTM run uses 24 calibration families and `model_comparison.finite_sample_quantile`, which already has the guard. This finding does not invalidate its observed 95.83% joint coverage. It also means two Berkeley experiments must not be represented as enough to calibrate a finite distribution-free 95% interval using this procedure.

## A6. Cross-Section Import Hides Bad Rows

**Location:** [spectral_exact_baseline.py:39](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/spectral_exact_baseline.py:39>).

The loader converts both columns with `errors="coerce"` and drops every row that fails conversion. Although intended to tolerate a header, it also silently removes malformed scientific measurements in the middle of a table. Interpolation then fills across the missing region.

**Reproduced with an in-memory table:** Rows `(0,0)`, `(10,"CORRUPTED")`, and `(20,4)` loaded as only `(0,0)` and `(20,4)`. Querying 10 eV returned an interpolated cross section of 2 rather than reporting the invalid input.

**Suggested fix, not applied:** Permit only a recognized optional header. Reject any malformed data row with its line number. Check finite, nonnegative energies and cross sections and a strictly increasing grid. Preserve the source-file hash. Add tests for interior missing/non-numeric values and infinite energies.

**Past-result boundary:** Supplied TENDL tables passed existing spectrum tests. This is a silent-input-corruption vulnerability, not evidence that the current tables are corrupted. Tightening import validation requires no model training.

## A7. A Cost Claim Can Pass While Costs Increase

**Location:** [cost_evidence.py:160](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/cost_evidence.py:160>).

`measured_cost_claim_eligible` is determined by blockers based on elapsed time and other supplied gates. Dollar costs are computed separately and never used to block a money-saving interpretation. The flag can also be true without any supplied prices.

**Reproduced with illustrative prices, not real market estimates:** A 1,000-candidate reference workflow took 1,000 seconds and cost 1 currency unit. The complete surrogate workflow took 13 seconds but cost 1,100.002 currency units. The function nevertheless returned `measured_cost_claim_eligible=True` and no blockers.

**Why it matters:** Faster hardware or expensive inference can save seconds while losing money. Your stated research goal includes both, so those conclusions need separate gates.

**Suggested fix, not applied:** Separate time-saving eligibility from monetary-cost-saving eligibility. The money-saving flag should require complete, documented prices, comparable accounting boundaries, and lower total cost. Compute a separate monetary break-even using priced fixed and per-candidate costs. If this flag is intended to authorize only a time-cost claim, rename it and explicitly prevent interpreting it as financial savings.

**Past-result boundary:** No measured real-world cost advantage is established in the authoritative results, so no such claim is being withdrawn here. This is a future reporting-gate correction and needs only small arithmetic tests.

## A8. Atomic Preflight Does Not Make Training Saves Atomic

**Locations:** [model_comparison.py:651](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/model_comparison.py:651>) and [model_comparison.py:688](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/model_comparison.py:688>), compared with [build_model_comparison_colabs.py:478](<C:/Users/ogunn/Downloads/New folder/New folder/v4_transport_surrogate/build_model_comparison_colabs.py:478>).

The A100 notebook preflight saves to a temporary file and replaces the destination. Actual progress and best-member saves call `torch.save` directly on their final paths. An interruption can therefore leave a partial file. Subsequent resume trusts the path's existence and attempts to load it, with no alternate generation to recover from.

**Evidence status:** Confirmed by source inspection. An actual Colab disconnection was not induced, and no corruption was found in the returned result archive. This is an implementation-level reliability gap, not an observed failed run.

**Suggested fix, not applied:** Use one checkpoint-writing helper in both preflight and training. Write a temporary file, close it, validate it, and replace the destination; retain a previous good generation and bind it to the request fingerprint. Add an interrupted-save test verifying that the prior good checkpoint remains usable. A mounted cloud filesystem's behavior still needs practical validation; a local rename alone is not a universal cloud-durability guarantee.

**Compute required:** No retraining of completed models. Test with a tiny temporary checkpoint before the next genuine training run.

## What Passed

- **92 V4 unit tests passed**, covering frozen splits, hashes, uncertainty inputs, explicit out-of-range rejection, source sampling, tally normalization, teacher-record validation, exact short-interval propagation, and notebook-release contracts.
- **10 V3 decision-support/reporting tests passed**, including exact-versus-Radau checks and separation of the two mesh-error estimands.
- **115 Python files parsed successfully** across V3, V4, and the two shared root physics/model modules. Syntax success is not execution or scientific validation.
- **All 30 code cells across five V4 notebooks compiled.** No notebook was launched in Colab during this review.
- **Official V4 result ZIP replay passed:** 32 members, 30 collector-recorded hashes, 27 nested artifact hashes, metric/coverage checks, and paired-bootstrap ordering replay.
- The replay retained `claim_eligible=false`, a provisional GP lead, and no statistically supported winner.

| Replayed metric | GP | Five-member LSTM |
|---|---:|---:|
| Median rate error | 1.9708% | 1.8323% |
| p95 rate error | 3.7906% | 3.9309% |
| Worst rate error | 5.5870% | 5.6145% |
| Calibrated joint coverage | 95.8333% | 95.8333% |

The replay checks the returned evidence package. It does not rerun all predictions from training source, independently validate the nuclear model, or turn modeled reference values into measurements.

## Limitations That Are Not Software Bugs

- The present V4 reference remains a finite-geometry/TENDL modeled proxy. Its agreement is not reactor, experimental, regulatory, or clinical validation.
- V4 uses sequence length one. Comparing these architectures does not establish a benefit from long-term LSTM memory. Five ensemble members do not constitute five independent repeats of the entire comparison protocol.
- The 48 comparison families had been examined in an earlier geometry diagnostic. They were locked for this comparison, not never previously seen by the project.
- Rejecting all sampled explicit out-of-range inputs demonstrates guard behavior, not accurate extrapolation. Neither GP nor LSTM becomes trustworthy outside its domain simply by being the preferred model.
- Native GP joint coverage was poor. Calibrated coverage and interval width must both be reported; uncertainty is not the same as point accuracy.
- The five-species chain omits physical processes. The unrun transport pilot has separate, disclosed geometry/source assumptions. Passing its schema does not certify those assumptions.
- Berkeley target-integrated spectra, uncertainty/normalization information, measurement timing/recovery clarification, and an Ac-227 numerical limit remain external evidence needs. This audit did not search for newly released Berkeley files or check email replies.
- The exact analytic baseline already outperforms the neural surrogate on the reduced constant-rate problem. Fixing the above bugs alone does not prove production-time or cost savings.

## Recommended Order When You Authorize Changes

1. Stop using the unsupported poster; create a current evidence-linked version. No compute or professor response needed.
2. Add regression tests and repair long-interval propagation. Preserve frozen source/weights and version downstream reruns.
3. Strengthen result identity and checkpoint saves before another Colab training session. Reuse completed evidence, not an unverified directory.
4. Fix table validation and separate money-saving from time-saving gates. These are small local tasks.
5. Fix the general GP noise/calibration paths before fitting new uncertain transport data. Any new fit is a separately labeled experiment, not a replacement for the frozen comparison.
6. Pursue the missing experimental inputs and expert review in parallel. Their arrival helps assess physical validity; it will not automatically lower either model's error.

**Do not buy another A100 run just to address this report.** First repair and test the affected helpers, preserve existing evidence, and decide whether any specific evaluation actually needs repeating. No current finding demonstrates that the completed five-seed LSTM needs retraining.

## Review Scope And Reproducibility

Live source root: `C:/Users/ogunn/Downloads/New folder/New folder`.

The outer folder was reviewed only for current/historical reporting and publication-figure context, not treated as the authoritative training source. Existing user modifications were left intact. Reviewed areas included V4 comparison/training/calibration, baseline folding, hybrid propagation, transport normalization and evidence gates, notebook generation/release tests, selected V3 training/provenance/model/data/decision paths, and selected outreach/poster files. Not every historical script, UI route, or notebook was executed or inspected line by line.

Local test runtime: Python 3.12.10, PyTorch 2.11.0+cpu, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0. Tests used the live project's existing `.venv`; nothing was installed. No GPU, OpenMC transport, external irradiation, full Colab install, or full training run was performed. Mathematical and malformed-input probes ran locally; mocked input/checkpoint reads did not create fake project results.

Commands used from the live root, with `PYTHONDONTWRITEBYTECODE=1` and a temporary Matplotlib cache:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s v4_transport_surrogate -p 'test_*.py' -v
.\.venv\Scripts\python.exe -B -m unittest v3_pilstm.analysis.test_decision_support v3_pilstm.analysis.test_postlock_reporting -v
.\.venv\Scripts\python.exe -B v4_transport_surrogate/audit_result_package.py 'C:\Users\ogunn\Downloads\V4_GP_LSTM_FINAL_c3a56472508c.zip'
```

Reviewed artifact identities (SHA-256):

```text
Official result ZIP
86fd62a67ab932c26eef755146c81310396e7924418b6b98daa2df734365ca24

v3_pilstm/physics/integrated_loss.py
c2c1d06b3c7ec6959ecf972a46a7a64b7a3cd812d79b34c69cd4ac3fe32e6e91

v4_transport_surrogate/model_comparison.py
b4bdab797709e124fe1e849aa55864757f0ac415260fa036f664b761cd32215f

v4_transport_surrogate/multifidelity_models.py
f3fe319899bfe0c6f34f6aa3d3eec789530a2db53a53c990df6e09aea323720c

v4_transport_surrogate/spectral_exact_baseline.py
ea8ef1265edee487a0409331584d39642f298003ef6ad7fe227e39900bd3a1d2

v4_transport_surrogate/cost_evidence.py
528ad75d8a70c10060cdecb405f289a2cd29395a168b9b31880ea4ac78d074b6
```

The exact original c3 code/data bundle is still needed for a full source-to-result reproduction of the frozen Colab experiment. Today's local source review and result-table replay are not substitutes for preserving that bundle.
