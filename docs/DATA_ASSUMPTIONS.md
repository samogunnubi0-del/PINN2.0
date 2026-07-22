# Data And Source Assumptions

This file tracks which scientific assumptions are fed into the ODE/PINN training
pipeline and which assumptions stay in deterministic dashboard post-processing.

## ODE Data Versions (2026-07-18): `ODE_DATA_VERSION=v1` / `v2`

The physics model now reads its nuclear data through a version flag
(`ODE_DATA_VERSION` environment variable, default `v1`). Full provenance and
the "why": **`docs/DATA_PROVENANCE.md`**.

| Item | v1 (default, legacy — bit-preserved) | v2 (evaluated, retrieved 2026-07-18) |
| --- | --- | --- |
| Ra-226(n,2n) σ(E) | sigmoid threshold, saturates at 27 mb (~28× too small at 14 MeV) | JENDL-5 evaluated table pointwise (= ENDF/B-VIII.0, verified identical): 755.7 mb @14 MeV, peak 2.53 b @10 MeV, threshold 6.4218 MeV |
| Ra-226(n,γ) σ(E) | 12.8 b × 1/v (12.8 b is Mughabghab Atlas, NOT ENDF) | JENDL-5 table ≥1 keV; 1/v tail below 1 keV anchored at experimental 13.8±0.3 b (Bagheri 2015, EXFOR 31760); libraries give ZERO thermal capture below 1 keV |
| Ra-226 / Ra-225 / Ac-225 / Ra-227 / Ac-227 half-lives | 1600 y / 14.8 d / 9.92 d / 42.2 min / 21.772 y (hard-coded) | NuDat 3 live: 1603±8 y / 14.9±2 d / 10.0±1 d / 42.2±5 min / 21.772±3 y (`data/evaluated/halflives_nndc.csv`) |

Rules:
- **v1 stays the default.** All existing v2-PINN / v3-PI-LSTM checkpoints were
  trained against v1; smoke checks run under v1 and must keep passing.
- v2 applies to `ra226_ac225_transmutation.py` (ODE) and
  `v3_pilstm/physics/bateman_rhs.py::reaction_rates` (PI-LSTM physics loss).
  Under v2 the `sigma_ra226` / `sigma_ngamma` overrides are inert (the
  evaluated tables are absolute).
- v2 data files live in `data/evaluated/` and are **required** — the loaders
  fail loudly if they are missing (no silent fallback to synthetic).
- Measured v1-vs-v2 anchor comparison:
  `results/ode_data_v2_validation_20260718.json` (script:
  `analysis/validate_ode_v2.py`; figure: `graphs/ode_v2_literature_anchors.png`).
  Headline: v2 makes the monoenergetic Joyo overprediction *worse* (369–2153×
  vs 12–24×), proving the dominant error there is the flux-spectrum
  approximation, not the cross section; v2 *improves* the thermal (n,γ) Hogle
  points (0.98× at 7 d) and matches the Snow φ=0 ingrowth exactly.

## Fed Into ODE / PINN Training

These values affect generated training rows and therefore must be source-backed
before claiming predictor-grade credibility.

| Item | Current value / model | Used in | Source status |
| --- | --- | --- | --- |
| Ra-226 half-life | 1600 y | ODE + PINN physics loss | NNDC/ENSDF/NuDat decay-data value |
| Ra-225 half-life | 14.8 d | ODE + PINN physics loss | NNDC/ENSDF/NuDat decay-data value |
| Ac-225 half-life | 9.920 d | ODE + PINN physics loss | NNDC/ENSDF/NuDat decay-data value |
| Ra-227 half-life | 42.2 min | ODE + PINN physics loss | NNDC/ENSDF/NuDat decay-data value |
| Ac-227 half-life | 21.772 y | ODE + PINN physics loss | IAEA LiveChart / DDEP value |
| Ra-226(n,2n) threshold | 6.422 MeV | ODE + PINN physics loss | JENDL-3.2 / JENDL-4.0 Ra-226 cross-section table |
| Ra-226(n,2n) effective cross section | 26.69 mb fission-spectrum average; table also lists 755.7 mb at 14 MeV | ODE + PINN physics loss | JENDL-3.2 / JENDL-4.0 Ra-226 cross-section table |
| Ra-226(n,gamma) thermal cross section | 12.78-12.79 b at 0.0253 eV; resonance integral about 282-286 b | ODE + PINN physics loss | JENDL-3.2 / JENDL-4.0 Ra-226 cross-section table |
| Neutron spectrum model | scalar energy with threshold and 1/v approximations | ODE + PINN physics loss | Must be stated as simplified model |

## Kept Out Of PINN Training

These belong in deterministic post-processing, not the neural-network loss.

| Item | Current value / model | Used in | Source status |
| --- | --- | --- | --- |
| Chemical recovery yield | UI slider, default 90% | Dashboard optimizer | Literature reports >90%, >95%, and >98% depending on resin/process |
| Cooling/transport time | UI slider, default 5 days | Dashboard optimizer | Engineering assumption |
| Ac-227 impurity limit | strict 0.15% activity impurity | Dashboard optimizer | Literature/regulatory-style strict constraint; cite directly before final claim |
| Activity conversion | A = lambda * N | Dashboard optimizer | Standard nuclear decay law |

## Training Data Coverage Targets

Before each full retrain, confirm the training log reports coverage in all regimes:

- Thermal near 0.025 eV.
- Epithermal/resonance-like stress cases.
- Near-threshold fast cases around 6.42 MeV.
- Fast production cases around 14 MeV.
- Pure Ra-226 targets.
- Recycled/interrupted targets with nonzero daughters.
- Empty or near-empty inventory edge cases.

## Source Links To Cite

- NNDC NuDat / ENSDF for isotope half-lives and decay data:
  - https://www.nndc.bnl.gov/nudat3/
  - https://www.nndc.bnl.gov/ensdf/
- IAEA LiveChart / DDEP for Ac-227 decay data:
  - https://nds.iaea.org/relnsd/ddep?NUCID=227AC
- JENDL Ra-226 evaluated neutron cross-section tables:
  - https://wwwndc.jaea.go.jp/jendl/j32/Tabsigs/Ra226.HTML
  - https://wwwndc.jaea.go.jp/cgi-bin/Tab80WWW.cgi?iso=Ra226&lib=J32
  - https://wwwndc.jaea.go.jp/cgi-bin/Tab80WWW.cgi?/data/JENDL/JENDL-4-prc/intern/Ra226.intern=
- JENDL evaluation background for minor actinide neutron data:
  - Journal of Nuclear Science and Technology, evaluation of neutron nuclear data for minor nuclides.
- Ac-225 / Ra separation and recovery:
  - Improved Ac-225/Bi-213 production using DGA/TEHDGA resin, reported overall Ac-225 yield exceeding 98%.
  - Optimization of cation exchange for Ac-225 separation from radioactive thorium/radium/other metals, reported >90% total process recovery and 95-98% step recoveries.
  - Eichrom Actinium-225 separation application notes for DGA resin behavior.

## Next Source Tasks

- Replace source notes above with full bibliography entries in the final paper/poster.
- Decide whether the ODE should use the JENDL fission-spectrum average value or the 14-MeV table value for the fast route; document the choice clearly.
- Keep the 0.15% Ac-227 activity-impurity limit labeled as a strict regulatory-style constraint unless a direct official acceptance specification is found.
