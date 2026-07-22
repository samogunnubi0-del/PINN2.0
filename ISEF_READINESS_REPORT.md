# IsotopePINN — ISEF Readiness Report & Improvement Roadmap

**Prepared:** 2026-07-18 · **Target:** Regional/state affiliated fair (Jan–Mar 2027) → Regeneron ISEF 2027 (May 8–14, Los Angeles)
**Method:** Full code audit + artifact audit + ISEF rules research + competitive-landscape research + nuclear-science positioning research (5 parallel reviewers; sources cited inline).

---

## 1. Executive summary

You have a genuinely strong foundation: a physics-informed surrogate for the Ra-226 → Ac-225 medical-isotope chain, a real validation harness (Trio A/B/C safety tests, quality gates, conformal UQ), an honest iteration log, and a deployed demo. PINN projects can win PHYS at ISEF — a PINN project took **First Award in Physics & Astronomy at ISEF 2025** (Aiden Kwon, PHYS001) — and a simulation+AI nuclear-adjacent project won at ISEF 2026 (Zack O'Leary, "M.A.N.T.I.S.").

**But the project is not yet ISEF-ready, and the gap is not polish — it is scientific validity.** The audit found that the headline number (4.5% held-out error) measures agreement with your own ODE, and **your own ODE currently disagrees with the only in-domain published experimental anchors by 10–18×**, driven by a cross-section model that is **30–60× off measured/evaluated values at 14 MeV**. A good judge who pulls one reference can dismantle the board. The good news: every critical issue is fixable with public data (EXFOR, JENDL-5/ENDF/B-VIII) inside the ~7-month runway to regional fairs, and fixing them converts your biggest weakness into your strongest judging story — *"I found my own model was wrong, fixed it with evaluated nuclear data, and now my surrogate reproduces an independent national-lab result within published uncertainty."*

**Readiness scorecard (as of today):**

| Dimension | Score | One-line verdict |
|---|---|---|
| Engineering / code quality | 🟡 Good | Real constraints, real UQ machinery; no seeds, weak reproducibility |
| Validation integrity | 🔴 At risk | "Held-out" was trained toward; baselines untrained; no fresh locked test |
| Physics data fidelity | 🔴 Critical flaw | σ(n,2n) model 30–60× off at 14 MeV; ODE 10–18× off Joyo anchors |
| Documentation consistency | 🔴 At risk | 3.7% vs 4.51% vs 8.18% across docs; 100× unit errors; stale failure report |
| ISEF rules compliance | 🟡 Unstarted | Forms, abstract, research plan, AI-disclosure strategy not yet done |
| Competitive positioning | 🟡 Partial | Right category exists (PHYS), right story exists — not yet written as science |
| Presentation assets | 🟡 Partial | Poster is v2-only; several banned/broken/missing graphs |

---

## 2. Critical findings (the stuff that loses medals)

### 2.1 🔴 P0 — The physics data problem (judge kill-shot #1)

- Your ODE's sigmoid σ(n,2n) saturates at **27 mb at 14 MeV** (`ra226_ac225_transmutation.py:56-61, 79-89`). Your own collected literature says **755.7 mb (JENDL)** and **1.60 ± 0.20 b** (O'Connor 1960, the only direct measurement, in EXFOR) at 14–14.5 MeV (`data/literature_benchmarks.csv` row 1). A spectrum-averaged one-group value is being used as a pointwise σ(E).
- Consequently `v3_pilstm/results/empirical_validation.json` shows the reference ODE **overpredicts the JAEA Joyo feasibility results (Sasaki 2023) by 17.9× and Iwahashi 2022 by 10.6×**.
- Your surrogate at 4.5% of a solver that is ~10–18× off reality is, physically, ~10–18× off reality. This is the single most important thing to fix.
- **Fix (P0):** download evaluated σ(E) from JENDL-5 / ENDF/B-VIII (free, browser-level), fold over a representative fast spectrum (Joyo MK-III spectrum is published), re-derive one-group σ values, re-run the ODE, and show it lands on the Joyo ORIGEN-class endpoint **15.4 ± 6.2 GBq from 1 g Ra-226 per 45-day cycle** *within the published nuclear-data uncertainty*. That uncertainty band (±40–55%) is literally the open problem JAEA itself flags — matching it *with your own uncertainty bands* is a research-grade result, not a weakness.

### 2.2 🔴 P0 — The 100× unit errors in your board materials (judge kill-shot #2)

- `v3_pilstm/results/ISEF_BOARD_PACK.md:55-57` prints "MAPE default ≈ **14.3%** → calibrated ≈ **0.20%**". The JSON values are **fractions**: 14.29 = **1,429%** and 0.195 = **19.5%**. Same bug in `analysis/benchmark_progress.md` ("0.0547%" for what is 5.5%).
- On a poster this reads as either incompetence or concealment of the literature mismatch. Fix every fraction-vs-percent rendering before anything is printed.
- Also re-label the Joyo σ-calibration honestly: fitting **one free parameter to two anchor points** and reporting 0.01/0.38 errors is circular. It is an illustration, not validation — say so.

### 2.3 🔴 P0 — The "held-out" set is really a dev set

- Training augmentation deliberately "stresses the exact trace-daughter cases that held-out validation probes" with "the same named regimes" (`train.py:555, 1162`); 12 dev iterations were gated on these canaries; v3 checkpoints were selected every 25 epochs on the same 22 scenarios, and the shipped recipe was chosen by test-set score ("Results-6 winning held-out recipe", `train_pi_lstm.py:244`).
- **Fix (P0):** freeze all design decisions, generate a **fresh test set from new seeds/scenario definitions you have never inspected**, evaluate **once**, and report that number. Report both numbers on the poster: "development-held-out: X%, untouched locked test: Y%." Judges reward this honesty enormously.

### 2.4 🔴 P0 — Missing baselines ("why a PINN at all?")

- `baseline_lstm.py` is a 30-line stub that was never trained. The `VanillaIsotopeNN` ablation exists in code but has no committed result. And for a *linear* ODE the obvious exact baselines — the **analytic Bateman solution / matrix exponential** — are never mentioned.
- **Fix (P0):** (a) train the vanilla LSTM with matched budget and report; (b) run the no-physics-loss ablation (show mass-conservation violations / negative activities); (c) add one slide: "the exact solution exists for 5 species — the point is a *parametric surrogate over (flux, energy, time, inventories)* that answers planning sweeps in milliseconds, amortized; the exact solver cannot do that without re-solving per scenario."

### 2.5 🔴 P0 — Documentation contradictions (a judge diffing your repo)

| Contradiction | Where | Truth |
|---|---|---|
| "3.7%" vs "4.51%" | `docs/ISEF_CLAIMS.md:26,82` (incl. the elevator pitch!), `docs/PINN_ARCHITECTURE_FIX.md:164,180` vs README/poster | 3.7% was v62+guards; **4.51% is canonical v63** — update everywhere |
| v2 "8.18%" vs "4.51%" | `compare_v2_pilstm.json` vs `v63_validation_20260530.json` | Different protocols (endpoint vs pipeline) — add an explicit reconciliation footnote in the board pack |
| `analysis/FAILURE_CASE_ANALYSIS.md` | Pre-fix model, Ac-225 MAPE 13,102% | Stamp "historical pre-v5" header or move to `archive/` — it currently contradicts every claim |
| Harvest demo broken | `analysis/demo_outputs/harvest_summary.md`: "optimal harvest ~0 h, peak 0.00 atoms" | Fix or remove; the 342 h claim is already flagged unsupported |
| "twelve thousand epochs" | `docs/presentation_script_for_professor.txt` | Shipped v63 is 4k; the 12k run was rejected (7.27%) |
| 500× speedup | README, claims, QA, poster outline | **No committed evidence file**; existing harness shows 142× and it's a batching effect; PI-LSTM eager (113.6 ms) is ~10× *slower* than v2 (12 ms) — re-run an honest benchmark, commit it, and claim "throughput under batching" |
| `benchmark_progress.md` | says "30-epoch smoke test… not production-ready" | `train_summary.json` shows a 6000-epoch run — regenerate |

### 2.6 🟠 P1 — Reproducibility & stats

- **No `torch.manual_seed` anywhere in the project**; `requirements.txt` uses floors only; `weights/pinn_best_weights.pth` is modified in git despite `docs/V2_FROZEN.md` declaring it frozen; five PI-LSTM weight variants with unclear lineage.
- Conformal UQ is currently vacuous where it matters: Ac-225 90% interval ≈ **±295%**, Ra-226 absolute coverage 72.7% (below 90% nominal), n_cal = n_test = 11.
- Mass units inconsistent: "virgin_1g" scenarios use 6.022e23 atoms = **226 g** (`trajectory_dataset.py:67`) vs correct 2.664e21 for 1 g in `validate_empirical.py:29`.
- **Fix:** seed everything, pin versions, re-freeze weights with checksums in a manifest, enlarge calibration/test n (≥100 scenarios), report p50 **and p95/worst-case** errors, fix the 1 g normalization, and either improve or honestly down-scope the conformal claims.

---

## 3. The science upgrade that wins (highest ROI)

Your validation ladder today: PINN ≈ ODE (self-consistency). Winners in your space anchor simulation to **something real** (Kwon built a turbulence chamber; Nath used telescope data; O'Leary used cosmic-ray muons). Your cheapest equivalent anchors, all public:

1. **EXFOR** (IAEA/NNDC, free web retrieval): the only direct σ(n,2n) measurement on Ra-226 (O'Connor & Perkin 1960, 1.60 ± 0.20 b @ 14.5 MeV). Parameter-level reality check.
2. **JENDL-5 / ENDF/B-VIII evaluated libraries** (free): replace constant-σ with σ(E) folded over the Joyo spectrum → code-vs-code comparison against the published Joyo endpoints already in your `literature_benchmarks.csv`.
3. **Decay-leg empirical validation** — the only leg with true experimental time series: McDevitt 2017 (ITU Th-229 generator milking, 39 mCi/cycle) and Melville 2007 Ra-225/Ac-225 secular equilibrium. Validates Bateman ingrowth at φ=0 against *measurements*, already rows 14–16 of your CSV.
4. **Cross-route anchors** (Matyskin 2024, Kuznetsov 2014, Snow 2025, Hogle 2016) as honest route-labeled sanity checks.

**Reframe the headline claim** (this is the sentence your poster should be built around):

> "My surrogate reproduces an independent national-lab (JAEA Joyo) production estimate **within published nuclear-data uncertainty**, quantifies that same uncertainty, and answers planning sweeps ~10²–10³× faster than stiff ODE solves — making it decision-grade for a production route whose dominant open problem *is* the ±55% cross-section uncertainty."

**Research questions to add (pick 1–2 as your stated hypotheses):**
- *H1 (physics-constraint value):* at fixed training budget, do physics-constrained models beat a plain LSTM out-of-distribution, and does the gap grow with extrapolation distance? (You have all machinery for this.)
- *H2 (UQ for planning):* is surrogate error small relative to nuclear-data uncertainty — i.e., is the model decision-grade?
- *H3 (conservation):* does the semi-analytic backbone guarantee conservation where vanilla PINNs violate it?
- *H4 (emergent optimization):* can the surrogate recover the published 17.5-day milking optimum (Iwahashi) as an emergent result?

**Novelty framing (honest version):** no published PINN/PI-LSTM surrogate exists for a Bateman *transmutation* chain with parametric (flux, energy, time, inventory) inputs aimed at isotope-production planning. A 5-species 0D chain is below SOTA numerical difficulty, so do **not** claim a numerical breakthrough; claim (1) the application niche, (2) the hybrid semi-analytic-backbone + hard-IC + physics-loss design, (3) stiffness curriculum. Cite Nasiri & Dargazany 2022 as stiffness-mitigation kin (weak-form loss; it solves one scenario at a time — you amortize over scenarios), not as prior transmutation work. Cite the real SOTA (CRAM/TTA depletion, DeepONet surrogates, point-kinetics PINNs) so judges see you know the field.

---

## 4. ISEF rules & logistics checklist (2026–27 cycle)

**Timeline:** research window ≤12 months, none before Jan 2026 → regional/state fair Jan–Mar 2027 (last affiliated fair date **April 12, 2027**) → **ISEF May 8–14, 2027, Los Angeles**. Check findafair.societyforscience.org for your fair's exact deadline — this is your real deadline.

**Forms (computational project, no human subjects/animals/hazards):**

| Item | When | Notes |
|---|---|---|
| Form 1 (Adult Sponsor checklist) | Before work | Signed by adult sponsor |
| Form 1A + Research Plan | Before experimentation | Rationale, question/hypothesis, procedures, risk & safety, data analysis, bibliography |
| Form 1B | After work, before competition | Local SRC signs |
| **Form 2A – Student Support Disclosure (NEW, all projects)** | With paperwork | **This is where you disclose AI coding assistance** |
| Abstract | After work | **250 words max**, purpose/procedure/data/conclusions; no acknowledgments, no mentor names, no logos |
| Form 7 (Continuation) | Only if you competed with an earlier version | Display at booth; judges score current-year work only |
| SRC pre-approval | **Not needed** | Simulation of radioisotope production triggers no radiation rules (no sources handled); run the ISEF Rules Wizard to document this |

**⚠️ AI-policy flag (important for you specifically):** the 2027 rules allow AI as a resource *with citation/acknowledgment*, but "**a student may not use generative AI to write the research plan, abstract, poster or to create citations.**" Your repo shows heavy AI-tool usage traces (`docs/cursor_handoff.md`, Copilot workflows). That is fine for *code* if disclosed on Form 2A — but you must be able to defend every line of the science and every written word as your own in interviews, and the written materials must be your own words. Plan for this now, not in April.

**Category:** enter **PHYS — Physics & Astronomy**, subcategory **Nuclear & Particle Physics (NUC)** if you emphasize the transmutation physics, or **Theoretical/Computational (THE)** if you emphasize the surrogate methodology. PHYS is proven hospitable to PINNs (2025 First Award). Judges are assigned by subcategory expertise — pick whose questions you want.

**Judging rubric (100 pts):** Research Question 10 · Design & Methodology 15 · Execution 20 · Creativity & Potential Impact 20 · **Presentation 35** (poster 10 + **interview 25**). Expect ~7 fifteen-minute technical interviews. The interview is where projects this technical actually win or lose.

**Display rules that bite:** booth ≤ 76 cm × 122 cm × 240 cm; no QR codes/URLs/emails on the board; no handouts; **every graphic individually credited** (incl. "Graph created by Finalist using Python/matplotlib"); visible without electricity/internet (so the Streamlit demo is a laptop backup, not the board). Research paper + data book: not required but **"strongly recommended for judging"** — you currently have neither a paper, an abstract, a formatted bibliography, nor a dated lab notebook.

---

## 5. Prioritized roadmap

### P0 — Must fix before your regional fair (now → ~Oct 2026)

| # | Task | Effort | Why |
|---|---|---|---|
| 1 | Replace sigmoid σ with JENDL-5/ENDF σ(E) folded over fast spectrum; re-run ODE; compare to Joyo 15.4 ± 6.2 GBq anchor | 1–2 weeks | Fixes kill-shot #1 |
| 2 | Fix every fraction-vs-percent bug (board pack, benchmark_progress); re-label Joyo calibration as illustration | 1 day | Fixes kill-shot #2 |
| 3 | Freeze design → generate fresh locked test set (new seeds/regimes) → evaluate once → report both dev and locked numbers | 2–3 days | Validation integrity |
| 4 | Train `baseline_lstm.py` with matched budget; run no-physics-loss ablation; add Bateman-exact baseline slide | 3–5 days | "Why a PINN?" |
| 5 | Reconcile all numbers across README/claims/QA/poster (4.51% canonical; footnote the 8.18% protocol difference); quarantine stale `FAILURE_CASE_ANALYSIS.md`; fix or remove broken harvest demo; honest speed benchmark committed to `results/` | 2–3 days | Consistency |
| 6 | Fix 1 g = 2.664e21 atoms normalization in `trajectory_dataset.py` | 1 hour | Unit correctness |
| 7 | Seeds + pinned requirements + weights manifest with checksums | 1 day | Reproducibility |

### P1 — High-impact upgrades (Oct–Dec 2026)

| # | Task | Effort |
|---|---|---|
| 8 | Decay-leg empirical validation vs McDevitt 2017 / Melville 2007 (real measurements, φ=0 leg) | 1 week |
| 9 | UQ story: propagate σ/flux uncertainty through surrogate; enlarge conformal n; report bands like JAEA's ±55% | 1–2 weeks |
| 10 | Recover published milking optimum (H4) as an "emergent result" demo | 3–5 days |
| 11 | Write the **research paper** (intro/methods/results/limitations) + 250-word abstract + formatted bibliography (in your own words — see AI rule) | 2–3 weeks, ongoing |
| 12 | Dated lab notebook export (your iteration log is 80% there — make it dated and continuous) | ongoing |
| 13 | New poster incorporating v3 + the fixed validation story; credit every graphic; remove banned graphs (`error_histogram.png`, `flux_sensitivity.png`, `ac225_growth.png`, `ac225_yield_heatmap.png`); fix `isef_sensitivity.png` filename reference | 1 week |
| 14 | Forms 1/1A/1B/2A + research plan + Rules Wizard documentation | 2 weekends |

### P2 — Stretch differentiators (Dec 2026 – fair)

| # | Task |
|---|---|
| 15 | Email one isotope-production group (university cyclotron / national lab user office) for feedback on the surrogate's planning utility — a single quoted email transforms the "who needs 500×?" answer |
| 16 | Energy-dependent σ(E) as a model *input* (spectrum-aware surrogate), not just folded constant |
| 17 | Second-solver cross-check of the ODE (Bateman closed form on short chains) |

### Judge attack-questions to rehearse (top 10)
1. "Your ground truth is a model — why trust 4.5%?" → validation-ladder slide: ODE → evaluated nuclear data → Joyo national-lab endpoint within uncertainty → decay-leg experiments.
2. "Baseline besides the ODE?" → trained LSTM + ablation + Bateman-exact framing.
3. "Where does it fail?" → volunteer the failure-mode figure first (winners do this).
4. "Why physics-informed?" → ablation: mass-conservation violations without it.
5. "Who wrote the code?" → know every hyperparameter cold; disclose AI/mentor help per Form 2A.
6. "Who needs 500×?" → scenario sweeps for irradiation/milking scheduling; cite the field's ±55% uncertainty problem (and any P2-#15 contact).
7. "How is this not a lookup table?" → held-out *regimes*, not just samples.
8. "Uncertainties?" → p50 + p95 + worst-case + UQ bands relative to nuclear-data uncertainty.
9. "Why is your ODE correct?" → JENDL/ENDF σ(E), Joyo endpoint agreement, second-solver check.
10. "What changes for a real facility?" → one concrete planning scenario with numbers.

---

## 6. What NOT to do

- Don't print anything with the Joyo "0.20% MAPE" claim as currently worded.
- Don't claim "99.8% accuracy," the 342 h harvest optimum, or 500× speedup until each is re-backed.
- Don't claim a numerical-methods breakthrough — claim the application niche + hybrid design + UQ.
- Don't let AI tools write your abstract/poster/research plan (explicitly prohibited; disclosure required for coding help).
- Don't delete the messy history — date-stamp it. Judges *love* "here's where I was wrong and how I found out."

## 7. Sources (key)

- ISEF 2027 Rules Book; Grand Award criteria; Categories; Display & Safety rules; AI Use Table — societyforscience.org (retrieved 2026-07-18)
- ISEF 2025 Grand Awards (Kwon, PHYS 1st); ISEF 2026 special awards (O'Leary); finalist directory — societyforscience.org
- Sasaki et al. 2023, *J. Nucl. Sci. Technol.* (Joyo Ra-226 feasibility, 15.4 ± 6.2 GBq) — doi:10.1080/00223131.2023.2243941
- Reviews: Springer s41181-024-00239-1 (Ac-225 production routes); MDPI Processes 10(7):1239
- Local evidence: `v3_pilstm/results/empirical_validation.json`, `joyo_sigma_calibration.json`, `data/literature_benchmarks.csv`, `docs/papers/Nasiri_Dargazany_2022_Reduced_PINN.pdf` (arXiv:2208.12045)

*Full evidence trails live in the five reviewer briefs; every claim above is traceable to a file path or URL.*

---

## 8. Amendment — 2026-07-18 improvement sprints (executed)

Since this report was written, three sprints executed against it. **P0 items 2, 5, 6, 7 are DONE** (percent bugs, doc reconciliation, 1 g normalization, seeds); items 1, 3, 4 are code-complete pending full-budget retrains. New capability: exact exponential-integrator physics loss, trainable matched-budget baseline, large-n + jackknife+/CV+ conformal, adaptive per-species weights, stiffness curriculum scaffold, deep-ensemble runner, honest speed benchmark. See `docs/UPGRADE_LOG.md` (why each change), `v3_pilstm/results/smoke_20260718.json` (10/10 checks).

**Real-data integration + a major discovery (supersedes §2.1's framing):** evaluated JENDL-5/ENDF-B-VIII σ(E), EXFOR 21405, NuDat half-lives, and 10 new literature anchors were fetched and integrated behind `ODE_DATA_VERSION=v2` (`data/evaluated/`, `docs/DATA_PROVENANCE.md`). Two discoveries changed the physics story:

1. **The cross section was never the dominant error.** Installing the correct ~28× larger evaluated σ(n,2n) made the Joyo overprediction *worse* (369–2152×) — exposing the **monoenergetic-flux approximation** (all flux treated as >6.42 MeV threshold neutrons) as the real gap. Bonus: a bare fission-spectrum fold of the evaluated table gives 26.7 mb ≈ the old synthetic 27 mb — the legacy constant was a fission-spectrum average all along.
2. **Spectrum folding closes the gap.** With `SPECTRUM_MODE=twogroup` and an inferred above-threshold fraction **f\* = 1.24×10⁻³ [7.4×10⁻⁴, 1.7×10⁻³]** (exact inversion on Sano's band, order-of-magnitude plausible for a sodium-cooled fast breeder), the ODE now reproduces the JAEA Joyo anchor at **1.00×** of 15.4 ± 6.2 GBq — *within published national-lab uncertainty* (`results/ode_data_v2_spectrum_20260718.json`, `graphs/ode_v2_spectrum_anchors.png`). The remaining dominant uncertainty is the spectrum shape — which is exactly the field's open problem.

**Updated headline framing for the poster:** "I progressively replaced every synthetic assumption in my reference model with evaluated nuclear data; each fix revealed the next-deeper assumption (cross section → flux spectrum); the final model reproduces an independent national-lab production estimate within its published uncertainty — and my surrogate learns both flux regimes." Next: Kaggle retrains against v2 physics with spectrum as a scenario-level input (design in `docs/DATA_PROVENANCE.md` §5).
