# Data Provenance — why the ODE got a versioned data layer (2026-07-18)

This document records **where every number in the physics model comes from**,
why each source was chosen, what was wrong before, what changed under
`ODE_DATA_VERSION=v2`, and what is still uncertain. It exists so the ISEF
story is honest: *"we found our synthetic cross section was ~28× too small,
fetched the real evaluated data, put it behind a flag, and measured what it
actually changes."*

Student-facing retrieval instructions (how to reproduce every download with
free, no-login sources): **`data/evaluated/README_retrieval.md`**. Raw
downloads (ENDF/JENDL `.dat` files, EXFOR JSON, NuDat HTML, OSTI PDFs) are
kept in `data/evaluated/_raw/` for audit.

---

## 1. The version flag

| | `ODE_DATA_VERSION=v1` (default) | `ODE_DATA_VERSION=v2` |
|---|---|---|
| σ(n,2n)(E) | smooth sigmoid, saturates at 27 mb | JENDL-5 evaluated σ(E) table, pointwise interpolation (ENDF/B-VIII.0 adopted the **same** evaluation — verified identical, max deviation 0.0 b) |
| σ(n,γ)(E) | 12.8 b × 1/v | JENDL-5 evaluated table ≥ 1 keV; 1/v tail below 1 keV anchored at the **experimental** 13.8 b thermal point (Bagheri 2015, EXFOR 31760) |
| Half-lives | hard-coded (1600 y / 14.8 d / 9.92 d / 42.2 min / 21.772 y) | NuDat 3 live retrieval (1603±8 y / 14.9±2 d / 10.0±1 d / 42.2±5 min / 21.772±3 y) |
| Why keep it? | All existing v2-PINN / v3-PI-LSTM checkpoints and smoke tests were produced against v1 — **bit-preserved**, still the default | The real physics reference for the next training round |

v1 behavior is bit-identical to the pre-2026-07-18 code (verified by direct
comparison against hand-computed legacy rates). Nothing changes unless you
explicitly set `ODE_DATA_VERSION=v2`.

---

## 2. Dataset-by-dataset provenance

### New (real) data — retrieved 2026-07-18 by Data_Hunter

| File | Contents | Source & retrieval route | Why this source |
|---|---|---|---|
| `data/evaluated/jendl5_ra226_n2n_sigmaE.csv` | JENDL-5 σ(n,2n)(E), 13 pts, threshold 6.4218 MeV, peak 2.53 b @10 MeV, 755.7 mb @14 MeV | IAEA-NDS mirror zip `https://www-nds.iaea.org/public/download-endf/JENDL-5/n/`, machine-parsed (`parse_endf.py`), **not transcribed** | JENDL-5 is the current major-library evaluation for Ra-226; free, citable, machine-readable |
| `data/evaluated/endfb8_ra226_n2n_sigmaE.csv` | ENDF/B-VIII.0 σ(n,2n)(E) | same mirror, ENDF-B-VIII.0 tree | Cross-check: ENDF/B-VIII.0 adopted the JENDL evaluation for Ra-226 (EVAL-AUG88 N.Takagi TIT) — we verified the tables are **identical** (max dev 0.0 b) rather than assuming it |
| `data/evaluated/jendl5_ra226_ngamma_sigmaE.csv`, `endfb8_ra226_ngamma_sigmaE.csv` | Evaluated σ(n,γ)(E) — **zero below 1 keV in both libraries** | same | The zero tabulation is itself a finding: the libraries do not evaluate thermal capture for Ra-226, so the thermal leg must be anchored experimentally |
| `data/evaluated/exfor_ra226_n2n.csv` | σ(n,2n)=1.60±0.20 b, σ(n,3n)=0.63±0.07 b @14.5 MeV (O'Connor & Perkin 1960) | EXFOR entry 21405 via IAEA-NDS GitHub mirror (`exfor_json`), identical to EXFOR master | The **only** experimental (n,2n) measurement that exists for Ra-226; full EXFOR error budget preserved in the file header |
| `data/evaluated/exfor_ra226_ngamma_thermal.csv` | Five real thermal σ(n,γ) measurements (15 b 1950; 23±1 b 1953; 19 b 1949; 14.0±4.0 b Kukleva 2015; **13.8±0.3 b Bagheri 2015**) | EXFOR entries 11727/12262/12282/31745/31760 | The two 2015 reactor-activation values agree; Bagheri 13.8±0.3 b has the smallest uncertainty → recommended anchor. Historical values diverge high (documented, not hidden) |
| `data/evaluated/halflives_nndc.csv` | NuDat 3 half-lives with uncertainties | `https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc=<NUC>&unc=nds`, live fetch 2026-07-18 | NuDat/ENSDF is the authoritative decay-data source; fetched live with the uncertainty notation preserved |
| `data/evaluated/literature_benchmarks_additions.csv` | 14 rows: new anchors + citation corrections | EXFOR full entries, OSTI full-text PDFs (Hogle 2016 = OSTI 1253240, Snow 2025 = OSTI 3028837), MDPI open access, abstract-book PDF | Full-text verification where freely available (marked VERIFIED-FULLTEXT vs VERIFIED-ABSTRACT per row) |

### Old (synthetic) data — what v1 used and why

| Item | v1 value | Original rationale | Why it was wrong |
|---|---|---|---|
| σ(n,2n) | sigmoid → 27 mb above threshold | "JENDL-5 spectrum-averaged fast-reactor value" | A **spectrum-averaged** number used as a **pointwise** σ(E). The evaluated pointwise value at 14 MeV is 755.7 mb — the sigmoid is **~28× too small** there (verified: 27.99× in code). Peak evaluated is 2.53 b @10 MeV (~94× the sigmoid) |
| σ(n,γ) thermal | 12.8 b, attributed to "ENDF/B-VIII thermal" | textbook-looking thermal capture value | Mislabeled: 12.8 b traces to **Mughabghab's Atlas of Neutron Resonances**, not ENDF. ENDF/B-VIII.0 and JENDL-5 have **no evaluated thermal capture at all** (zero below 1 keV). The best experimental value is 13.8±0.3 b (Bagheri 2015). Fixed in `data/empirical/cross_section_bands.csv` |
| Half-lives | 1600 y, 14.8 d, 9.92 d | older NuDat / common literature values | NuDat 3 currently gives 1603±8 y, 14.9±2 d, 10.0±1 d. Small shifts — but a judge can check, so v2 uses the fetched values |
| Threshold | 6.42 MeV | ENDF | Confirmed: evaluated table starts at 6.4218 MeV |

---

## 3. What actually changed when we ran the anchors (the key result)

`analysis/validate_ode_v2.py` → `results/ode_data_v2_validation_20260718.json`.
Ratios are ODE prediction ÷ measurement (no tuning applied):

| Anchor | Measurement | v1 ratio | v2 ratio |
|---|---|---|---|
| Sano 2024 Joyo 45 d | 15.4±6.2 GBq | **18.9×** | **369.6×** |
| Iwahashi 2022 Joyo 60 d + 8 d cool | ~30 GBq | 11.6× | 1042.6× |
| Iwahashi 2022 milking (3× @17.5 d) | 15.7 GBq/cycle | 24.0× | 2152.6× |
| Hogle 2016 Ac-227, 3.01 d | 22.9±1.1 kBq/µg | 0.76× | 0.82× |
| Hogle 2016 Ac-227, 7 d | 45.1±2.2 kBq/µg | 0.90× | **0.98×** |
| Hogle 2016 Ac-227, 26.09 d | 51.8±7.4 kBq/µg | 2.87× | 3.11× |
| Snow 2025 φ=0 ingrowth, 17 d | 126.8±12.6 Bq | 1.00× | 1.00× |

**This is the opposite of what we hoped — and it is the most useful result of
the whole exercise.** Installing the *correct, 28× larger* evaluated (n,2n)
cross section makes the Joyo overprediction ~20–90× **worse**, not better.
That proves the dominant error in the Joyo comparison was never the cross
section: it is the **monoenergetic-flux approximation** — the ODE scenario
treats the full Joyo core flux (5.7×10¹⁵ n/cm²/s) as if every neutron sat at
14.5 (or 10) MeV, above the 6.42 MeV threshold. In reality only the small
fast tail of the reactor spectrum can drive (n,2n) at all. The old 27 mb
"spectrum-averaged" constant was accidentally compensating for that spectrum
error — two wrongs making a roughly-right-looking 10–19× overprediction.
(The earlier non-destructive Joyo calibration found the same thing from the
other direction: the effective one-group σ that reproduces Joyo is ~1.44 mb.)

Meanwhile, where the physics scenario is *actually* pointwise-valid, v2 helps
or matches:
- **Hogle HFIR thermal (n,γ) leg**: v2's 13.8 b anchor moves 3.01 d / 7 d from
  0.76×/0.90× to 0.82×/0.98× of measurement. The 26.09 d near-saturation point
  is overpredicted ~3× by *both* versions — pointing to missing long-irradiation
  physics (self-shielding/burnup in the real capsule), not to cross sections.
- **Snow 2025 φ=0 ingrowth** (pure decay leg): both versions land on the
  measured 126.8 Bq (ratio 1.00) — the Bateman Ra-225→Ac-225 leg is solid
  under either half-life set.

---

## 3a. The completion (same day, second pass): spectrum-aware flux weighting

`SPECTRUM_MODE=mono|watt|twogroup` (default `mono` = pointwise v2, unchanged).
Reactor scenarios now fold the evaluated σ(E) over a **documented parametric
spectrum** — ⟨σ⟩ = ∫σ(E)φ(E)dE / ∫φ(E)dE — because the (n,2n) yield lives
entirely in the small tail above the 6.4218 MeV threshold. The measured Joyo
MK-III spectrum is paywalled, so both spectra are labelled **assumptions**
with citable forms (Watt 1952 / ENDF-standard a=0.988 MeV, b=2.249 MeV⁻¹;
two-group = Watt-shaped tail above threshold + thermal slow group, fraction
`SPECTRUM_FAST_FRACTION`).

Results (`analysis/validate_ode_spectrum.py` →
`results/ode_data_v2_spectrum_20260718.json`, figure
`graphs/ode_v2_spectrum_anchors.png`):

| Joyo anchor | Measurement | v1 | v2-mono | v2-Watt | v2-two-group (f\*) |
|---|---|---|---|---|---|
| Sano 2024, 45 d | 15.4±6.2 GBq | 18.95× | 369.6× | 18.61× | **1.00×** |
| Iwahashi 2022, 60 d + 8 d | ~30 GBq | 11.63× | 1042.6× | 11.44× | **0.57×** |
| Iwahashi milking 3×@17.5 d | 15.7 GBq/cycle | 24.03× | 2152.6× | 23.78× | **1.14×** |

- **Inferred above-threshold fraction:** f\* = **1.24×10⁻³ [7.4×10⁻⁴,
  1.7×10⁻³]** (exact ODE inversion on Sano's ±6.2 GBq band) → ⟨σ⟩ = 1.66 mb.
  Reported as an *effective parameter with uncertainty*, not a tuned truth.
- **Plausibility:** f\* is ~16× smaller than a bare fission spectrum's
  >6.42 MeV tail (2.00%) — order-of-magnitude plausible for a sodium-cooled
  MOX fast breeder (tail degraded by Na/Fe/O/U-238 scattering; Joyo softness
  varies strongly by position — Aoyama 2005 J. Nucl. Radiochem. Sci. 6(3);
  Iwahashi 2022 MDPI Processes 10(7):1239 Fig. 5). Not verifiable against the
  paywalled MK-III table.
- **The 27 mb mystery solved:** folding the evaluated table over a bare Watt
  fission spectrum gives ⟨σ⟩ = **26.7 mb ≈ the legacy synthetic 27 mb**. The
  v1 constant was effectively a *fission* spectrum average; a real reactor
  spectrum at the irradiation position is ~16× softer. v1-Watt agreement
  (18.95× vs 18.61×) confirms it.
- Sano (15.4 GBq) and Iwahashi (30 GBq) mutually disagree by ~2×; no single f
  satisfies both exactly — f\* splits the difference (1.00× / 0.57× / 1.14×).

---

## 4. What remains uncertain (say this out loud at ISEF)

1. **Only one experimental (n,2n) point exists** — 1.60±0.20 b @14.5 MeV
   (EXFOR 21405, 1960). Everything else about the shape is evaluation/theory.
2. **The evaluated shape is theory-based and disagrees with that one point**:
   interpolated evaluated σ at 14.5 MeV is ~0.53 b, i.e. ~0.33× the
   measurement (the experiment is ~3× the evaluation). A real, citable
   tension — not resolved here.
3. **The spectrum shape is the largest remaining modeling uncertainty.** The
   Joyo MK-III measured spectrum is paywalled (ref [13] in Iwahashi 2022), so
   §3a's Watt / two-group spectra are parametric assumptions. f\* is an
   *inferred effective parameter* (1.24×10⁻³ [0.74–1.74×10⁻³]), sensitive to
   which anchor is inverted (Sano vs Iwahashi differ ~2×); the milking-cycle
   anchor additionally depends on our documented post-EOB harvest assumption
   (see the validation JSONs). A published MK-III spectrum table would replace
   the whole f\* exercise with a direct fold.
4. **No evaluated thermal capture exists in the libraries** — our thermal
   (n,γ) leg below 1 keV is a 1/v extrapolation anchored at a single modern
   experiment (13.8±0.3 b); the 2015 Kukleva value (14.0±4.0 b) brackets it.
5. **Long-irradiation (n,γ) physics**: the ~3× overprediction of the 26-day
   HFIR point is unexplained by either data version (likely capsule
   self-shielding / burnup, which the point-model ODE does not have).
6. **Interpolation choices**: linear-linear on the tabulated grids, σ held
   constant above 20 MeV, zero below the 6.4218 MeV threshold. These are
   documented conventions, not library-official reconstructions.
7. **The 26.09-d Hogle row's φ=2.0×10¹⁵ is the nominal HFIR position flux**;
   the real capsule saw a time-varying spectrum.

---

## 5. ISEF framing — old surrogate vs new-physics reference

- All existing **v2-PINN and v3-PI-LSTM checkpoints were trained against the
  v1 reference** (the synthetic sigmoid ODE). They are, deliberately, left
  untouched: v1 remains the default and all 10 smoke checks still pass.
- **The next step is retraining against v2 physics on Kaggle** — now with
  **spectrum-aware scenarios**: monoenergetic for D-T accelerator scenarios,
  spectrum-folded (`watt` / `twogroup`) for reactor scenarios. The mono-vs-
  spectrum distinction should become a *scenario-level input the surrogate can
  learn from* — a genuinely interesting ML angle (one model, two flux
  regimes). Design (NOT yet implemented): `IsotopeEnvironment` already accepts
  a per-scenario `spectrum=` constructor argument; the dataset hook is
  `v3_pilstm/data/trajectory_dataset.py:273` where
  `env = IsotopeEnvironment(phi=sc.phi, neutron_energy_ev=sc.energy_ev)` is
  built — `TrajectoryScenario` would gain a `spectrum` field sampled from
  {"mono", "watt", "twogroup"} (with `SPECTRUM_FAST_FRACTION` sampled log-
  uniformly for twogroup), passed through to the constructor, and exposed to
  the model as an extra input feature (e.g. one-hot mode + log10 f). The
  physics-loss path (`bateman_rhs.reaction_rates`) already respects the same
  env flags.
- That comparison — *old surrogate trained on synthetic physics vs the new
  evaluated, spectrum-aware reference* — is itself an analysis opportunity: it
  measures how much of the model's apparent skill was learning a
  wrong-but-smooth surrogate, and how the PI-LSTM's physics loss behaves when
  the physics gets 28× stiffer in the (n,2n) channel.
- What we can already claim: the data pipeline is now real end-to-end
  (EXFOR + JENDL-5 + ENDF/B-VIII.0 + NuDat, machine-parsed, audit-trailed),
  versioned, spectrum-aware, and regression-guarded.
