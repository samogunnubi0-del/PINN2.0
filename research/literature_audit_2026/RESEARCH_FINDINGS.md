# Ac-225 V3 Literature Audit Findings

Date: 2026-08-12  
Final comparison: 137 sources  
Broad discovery pool: 2281 records

**Bottom line:** The strongest defensible contribution is not the use of an LSTM or a 1-3% synthetic median by itself. It is a frozen, failure-aware surrogate that identifies useful irradiation/cooling/harvest schedules for the Ra-226 fast-neutron pathway, verifies finalists with trusted physics, and is tested against the Berkeley experiments.

## Q1. How are Ra-226 irradiation and harvest schedules selected now?

Published studies normally choose a facility, target, beam or neutron field, irradiation duration, and a fixed cooling/separation plan, then calculate or measure the resulting inventory. The Berkeley experiments used fixed 7.51-day and 3.69-day irradiations followed by 18.9-day and 20.8-day decay intervals. Joyo studies likewise evaluate specified reactor positions and timing assumptions. Established ORIGEN and OpenMC workflows already accept multistep time, flux, power, or source-rate histories.

**Project action:** Do not claim that changing histories are new. Define a constrained schedule space and prove that V3 can screen it usefully.

**Confidence:** High

## Q2. Has irradiation plus cooling been jointly optimized for this exact pathway?

Optimization is common in nuclear engineering, accelerator tuning, reactor control, fuel management, and isotope-route design. Fixed timing scans and separation-schedule reasoning also exist for Ac-225. However, this audit did not identify a paper that combines a frozen physics-informed recurrent surrogate with joint irradiation/cooling optimization for the specific Ra-226(n,2n)Ra-225 to Ac-225 pathway and then verifies the selected schedules against a trusted solver and Berkeley data. That is a defensible research gap, but only as 'not identified in this search,' not as a universal first-ever claim.

**Project action:** Make the new action schedule discovery, not merely another endpoint predictor. Verify all Pareto finalists with the exact/trusted solver.

**Confidence:** Moderate

## Q3. What Ac-227 limit is scientifically meaningful?

There is no single context-free percentage that your model should call 'safe.' Published thorium-spallation material is often discussed near 0.1-0.3% Ac-227 activity at end of bombardment, and one decay-energy spectroscopy measurement found 0.142 +/- 0.005% for one sample. A pharmacokinetic study found a very small dose contribution for one antibody treatment scenario, while FDA guidance says impurity limits must be justified through product expiry using retention, stability, dosimetry, and measurement evidence. The ratio changes with time because Ac-225 and Ac-227 have very different half-lives.

**Project action:** Report Ac-227/Ac-225 at a named reference time and run a threshold sensitivity analysis. Ask an expert to choose the operational constraint.

**Confidence:** High for the conclusion; application-specific for any threshold

## Q4. Which variables dominate Ac-225 yield uncertainty?

For this pathway, the most important upstream uncertainties are the energy-dependent Ra-226(n,2n) cross section, neutron spectral shape above the 6.425 MeV threshold, absolute fluence, target amount and geometry, and any moderation that opens competing capture pathways. Timing and decay constants control the Ra-225 to Ac-225 conversion. Measured product recovery adds a separate chemistry uncertainty: Berkeley recovered only about 37-60% of the produced Ac-225 in the tested separations, so produced activity and recovered activity cannot be treated as the same target.

**Project action:** Run a no-retraining Sobol or Monte Carlo sensitivity study with the trusted solver; keep nuclear yield, chemical recovery, and measurement uncertainty as separate layers.

**Confidence:** High for variable identification; ranking requires sensitivity calculations

## Q5. Do established solvers already handle changing neutron histories?

Yes. ORIGEN accepts arrays of time and flux or power, and OpenMC accepts different source rates for each depletion timestep, including zero-rate decay intervals. Both can represent piecewise-changing histories. Your recurrent architecture may still be useful for extremely repeated screening or richer control sequences, but handling time dependence is not itself novel.

**Project action:** Benchmark V3 against piecewise histories and measure end-to-end optimization cost. Do not compare only with the already-cheap five-nuclide Bateman solve.

**Confidence:** High

## Q6. Are recurrent and physics-constrained nuclear surrogates new?

No. Dense neural depletion surrogates, Gaussian-process nuclide surrogates, multi-task irradiation modules, inverse-depletion networks, recurrent reactor models, and physics-constrained CNN-LSTM radionuclide models all exist. A 2026 preprint even uses a graph neural network for a 690-isotope stiff nuclear reaction network. Your exact combination and pathway may still be uncommon, but LSTM plus physics plus nuclear dynamics is not enough by itself for a strong novelty claim.

**Project action:** Center novelty on what the model enables and proves: reliable decision screening, experimental transfer, and verified schedule tradeoffs.

**Confidence:** High

## Q7. Have neural surrogates been placed inside nuclear optimization loops?

Yes. Published examples include accelerator tuning, microreactor control, PWR fuel management, coupled nuclear experiments, neutron-component design, and wider nuclear multi-objective studies. Good practice is to use the surrogate to find promising candidates and then evaluate finalists with the original high-fidelity model.

**Project action:** Your optimizer must report candidate recall, Pareto-front quality, and exact-solver verification rather than treating surrogate predictions as final truth.

**Confidence:** High

## Q8. How should the model detect unfamiliar or unsafe inputs?

The nuclear and scientific-ML literature emphasizes uncertainty, validation envelopes, and model discrepancy, but no single OOD score is universally reliable. For this project, the most defensible near-term detector is transparent: normalize each input, measure distance from the training envelope, flag out-of-range values and unusual combinations, and send flagged cases to the trusted solver. An ensemble or conformal interval could improve this later, but would require additional training or calibration data.

**Project action:** Add a no-retrain applicability-domain gate now. Never allow an optimizer to select an unverified OOD schedule.

**Confidence:** Moderate

## Q9. Which metrics matter more than median prediction error?

For a decision tool, ranking and optimization metrics are essential: Spearman or Kendall rank correlation, top-k recall, feasibility classification, Pareto hypervolume, coverage of the trusted Pareto front, and optimization regret after exact verification. Pointwise median error can look excellent while rare false positives or tail errors send the optimizer toward bad schedules. Your current p95 and zero-case failures make this especially important.

**Project action:** Use the frozen V3 to rank a held-out candidate set and compare the selected top schedules with exact-solver rankings before claiming practical utility.

**Confidence:** High

## Q10. What is needed to reproduce the Berkeley experiments?

For each 33 and 40 MeV experiment you need the machine-readable neutron spectrum at the Ra target, absolute fluence and uncertainty, target activity or mass and chemical form, position and solid angle, irradiation duration/current history, end-of-bombardment and separation timestamps, produced and recovered activities, recovery factors, and impurity detection information. The thesis already provides most scalar values; Professor Bernstein's spectrum-reproduction code is the key machine-readable bridge. A two-group collapse rule must be frozen before looking at model error.

**Project action:** Write a dated external-validation protocol first. Score produced Ac-225 separately from recovered Ac-225 and treat non-observation of Ac-227 as a detection-limit statement, not exact zero.

**Confidence:** High

## Q11. What would make this comparable to a top ISEF research story?

The transferable lesson from top research projects is not to invent every ingredient from scratch. It is to combine established ideas into a new, testable action, compare against strong alternatives, and validate that action on evidence the method did not train on. For this project, that means using recurrence, nuclear constraints, and surrogate optimization to discover schedules, then proving whether those schedules remain good under trusted physics and real Berkeley measurements. A polished model with only synthetic median accuracy is weaker than a smaller model that answers a real decision question and exposes where it fails.

**Project action:** Build the project around one clear new action, one independent benchmark, one rigorous control, and one visible failure map. That is the scientific structure to imitate, not another winner's exact topic.

**Confidence:** High as research-design guidance; award outcomes are never predictable

## Recommended Research Question

Can a frozen physics-informed recurrent surrogate reliably identify and rank irradiation, cooling, and harvest schedules for the Ra-226(n,2n)Ra-225 to Ac-225 pathway that improve Ac-225 yield while controlling Ac-227 risk, with finalists verified by a trusted solver and evaluated against published Berkeley irradiation data?

## Review Boundary

This is a scoping review. The source matrix explicitly labels title/abstract screening, official-source review, and full-text close reading. It must not be described as 137 complete full-text peer reviews.
