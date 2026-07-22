# Literature benchmarks — how to fill `literature_benchmarks.csv`



Published anchors for optional **external** comparison of PI-LSTM and v2 PINN against literature — not for training labels.



## Honest ISEF framing



**No public empirical reactor time-series** exists for Ra-226(n,2n)→Ac-225 with controlled `(φ, E, t) → atom inventories`. Literature reports:



- One direct **cross-section** measurement (O'Connor 1960, 14.5 MeV)

- **Simulation endpoints** from Joyo feasibility studies (GBq activities at campaign milestones)

- **Cross-route experiments** (γ,n, p,2n) that do not validate the neutron-transmutation ODE directly

- **Decay-leg** generator milking (229Th→225Ra→225Ac) that validates the Bateman chain, not neutron production



For ISEF, state clearly:



> *We validate self-consistency against Radau5 Bateman integration on held-out scenarios, then compare endpoint activities to peer-reviewed reactor **simulations**, cross-route accelerators, and decay-chain milking data — not to unpublished experimental irradiation curves.*



Comparing only model vs ODE is **internal validation**. This CSV supports **external sanity checks** where data exist.



---



## Master source table (by tier)



### Tier 1 — Real experimental measurements (reactor / target irradiation)



| Source | `source_type` | Route | Key quantity | Caveat for (n,2n) ODE |

|--------|---------------|-------|--------------|------------------------|

| **Hogle et al. 2016, ORNL HFIR** ([OSTI 1253240](https://www.osti.gov/biblio/1253240)) | `empirical` | Reactor thermal | 227Ac growth vs time (45.1 kBq/µg Ra at 7 d) | **Wrong channel** — (n,γ)→227Ac, not (n,2n)→225Ac |

| **Kuznetsov et al. 2014, SM reactor** ([PDF](http://www.ssc.smr.ru/media/journals/izvestia/2014/2014_6_129_135.pdf)) | `empirical` | Reactor thermal trap | 225Ra 6.0×10⁹ Bq @ 25 d (φ=1.5×10¹⁵); 227Ac 7.8×10¹⁰ Bq/g @ 42 d total | Thermal (n,γ) chain; 225Ra is parent, not separated Ac-225 |

| **Matyskin et al. 2024, Penn State BNR** ([DOI 10.1016/j.nucmedbio.2024.108940](https://doi.org/10.1016/j.nucmedbio.2024.108940)) | `empirical` | Reactor + Gd converter | 225Ac 800 Bq at equilibrium; 225Ra 6 kBq EOB | **Photonuclear (γ,n)** in reactor — not direct fast (n,2n) |

| **Snow et al. 2025, INL** ([OSTI 3028837](https://www.osti.gov/biblio/3028837)) | `cross_route` | Linac bremsstrahlung | 335.4 kBq/g/mA/h peak @ 17 d | (γ,n) on Ra-226; rate metric only |

| **Tadokoro & Maeda 2023, linac** | `cross_route` | Linac (γ,n) | 4.12 MBq Ac-225 (two milkings) | Cross-route |

| **Maslov 2006, MT-25 microtron** | `cross_route` | Electron bremsstrahlung | 550 Bq/(µA·h·mg) | Rate only |

| **Higashi et al. 2022** | `cross_route` | Cyclotron (p,2n) | 2.23 MBq @ 5 h EOB | Cross-route |

| **Morgenstern & Apostolidis 2005** | `cross_route` | Cyclotron (p,2n) | 484.7 MBq @ 45.3 h | Cross-route |

| **Melville et al. 2007, linac** ([Appl Radiat Isot 65(12)](https://doi.org/10.1016/j.apradiso.2007.06.012)) | `cross_route` | Linac (γ,n) | ~29 µCi Ac-225 at equilibrium after 3 h (40 mg Ra needles) | Ra-225/Ac-225 secular equilibrium stock |

| **ANL FNAL spallation** (Tri-Lab 2016) | `cross_route` | Spallation on **232Th** | 2.08 mCi Ac-225 | Different target; real measured yield |

| **US6299666B1 patent** | `cross_route` | Cyclotron 15.2 MeV | 15% Ac-225/Ra ratio | Fractional yield only |



### Tier 2 — Peer-reviewed simulations (neutronics / ORIGEN)



| Source | `source_type` | Route | Key quantity | Poster label |

|--------|---------------|-------|--------------|--------------|

| **Sasaki et al. 2023, Joyo** ([DOI 10.1080/00223131.2023.2243941](https://doi.org/10.1080/00223131.2023.2243941)) | `simulation` | Fast reactor MCNP | 15.4 GBq 225Ac @ 45 d | "Literature simulation" |

| **Iwahashi et al. 2022, Joyo ORIGEN** ([MDPI Processes 10(7)1239](https://www.mdpi.com/2227-9710/10/7/1239)) | `simulation` | Fast reactor ORIGEN | ~30 GBq @ 60 d + 8 d cool | "Literature simulation" |



### Tier 3 — Cross-section / yield anchors (no Ac-225 activity)



| Source | `source_type` | What it is | Use |

|--------|---------------|------------|-----|

| **O'Connor & Perkin 1960** ([DOI 10.1016/0022-1902(60)80227-8](https://doi.org/10.1016/0022-1902(60)80227-8)) | `cross_route` | σ(n,2n)=1.60±0.20 b @ 14.5 MeV | EXFOR anchor; no activity |



### Tier 4 — Decay-leg generator milking (229Th → 225Ra → 225Ac)



| Source | `source_type` | Route | Key quantity | Caveat |

|--------|---------------|-------|--------------|--------|

| **McDevitt et al. 2017, ITU Karlsruhe** ([PMC5565267](https://pmc.ncbi.nlm.nih.gov/articles/PMC5565267/)) | `decay_leg` | 229Th generator | 39 mCi (1.44 GBq) 225Ac per 9-week cycle | Validates Bateman ingrowth, not neutron production |

| **ORNL 229Th campaign** (Mirzadeh / McDevitt reviews) | `decay_leg` | 229Th generator | ~100 mCi (3.7 GBq) per 8-week campaign | Same; dominant clinical supply route |



**Historical note:** Clinical 229Th stocks trace to **232Th(n,γ)→233U→229Th**, not Ra-226 irradiation.



---



## `source_type` values



| `source_type` | What it is | Use in poster |

|---------------|------------|---------------|

| `empirical` | Measured activities from real irradiation or separation | Strongest external anchor (label route honestly) |

| `simulation` | Peer-reviewed neutronics/ORIGEN/MCNP endpoint yields | Order-of-magnitude check; label **"literature simulation"** |

| `cross_route` | Real data on **other** production routes (γ,n, p,2n, spallation) or σ-only rows | Supporting context — do not claim (n,2n) ODE validation |

| `decay_leg` | 229Th generator milking / 225Ra→225Ac ingrowth | Bateman-chain sanity check; not neutron transmutation |



Current CSV: **27 rows** (see `literature_benchmarks.csv`; counted 2026-07-18 after the real-data merge below; previously 17).

---

## Real-data update 2026-07-18

Merged `data/evaluated/literature_benchmarks_additions.csv` (retrieved 2026-07-18
by Data_Hunter from EXFOR / IAEA-NDS / OSTI full texts) into the CSV via
`scripts/merge_literature_additions.py` (auditable, idempotent).

**Citation corrections applied:**
- "Sasaki et al. 2023 Joyo" → **Sano et al. 2024**, JNST 61:509 (first author is A. Sano; year 2024).
- "McDevitt et al. 2017 ITU Karlsruhe, PMC5565267" → **Scheinberg & McDevitt 2011**, Curr. Radiopharm. 4(4):306-320.
- Melville 2007 → **Appl. Radiat. Isot. 65(9):1014-1022**, DOI 10.1016/j.apradiso.2007.03.018 (was 65(12)).
- O'Connor & Perkin row upgraded with the EXFOR accession (**21405.002**).
- Kuznetsov SM-reactor rows now have an accessible English citation: **Kuznetsov et al. 2012, Radiochemistry 54(4):383-387**, DOI 10.1134/S1066362212040121 (same group, same experiments as the 2014 Izvestia SSC RAS paper).

**New anchors added (10 rows):**
- EXFOR 21405.003: σ(n,3n)=0.63±0.07 b @14.5 MeV (competing loss channel).
- JENDL-5 / ENDF/B-VIII.0 MAT 8834 MF3 MT16: evaluated σ(n,2n)=755.7 mb @14 MeV (identical tables).
- EXFOR 31760 (Bagheri 2015): thermal σ(n,γ)=13.8±0.3 b — recommended modern value.
- EXFOR 31745 (Kukleva 2015): thermal σ(n,γ)=14.0±4.0 b — independent modern check.
- EXFOR 12262 (Butler & Adam 1953): thermal σ(n,γ)=23±1 b — historical, diverges high.
- Snow 2025 full text: **phi=0 ingrowth** — 285.2 Bq Ra-225 (EOB) → 126.8±12.6 Bq Ac-225 @17 d post-EOB. Real Bateman decay-leg validation point.
- Hogle 2016 full text: Ac-227 series 3.01 d (22.9±1.1 kBq/µg) and 26.09 d (51.8±7.4 kBq/µg) points.
- Takaki group 11ICI: Joyo TMC 14.5±3.1 GBq Ra-225 per g per 60-d cycle (Ra-225 production anchor).

These rows feed the v1-vs-v2 ODE comparison in `analysis/validate_ode_v2.py` →
`results/ode_data_v2_validation_20260718.json` (see `docs/DATA_PROVENANCE.md`).



---



## Column definitions



| Column | Required | Description |

|--------|----------|-------------|

| `source_citation` | yes | Author, year, DOI or journal — copy to poster footnotes |

| `source_type` | yes | `empirical` \| `simulation` \| `cross_route` \| `decay_leg` |

| `phi_n_cm2_s` | if known | Effective or total neutron flux (n/cm²/s) |

| `energy_ev` | if known | Representative neutron energy (eV); use ~1.45e7 for fast (n,2n) studies |

| `time_h` | if known | Irradiation + cooling time to measurement (hours) |

| `A_Ac225_Bq` | preferred | Reported Ac-225 activity (Bq) |

| `N_Ac225` | optional | Atom count if paper gives it; else leave blank |

| `N_Ra226_0` | recommended | Initial Ra-226 atoms (1 g ≈ 2.664e21) |

| `notes` | yes | Table/figure, uncertainty, assumptions |



Provide **either** `A_Ac225_Bq` **or** `N_Ac225` for activity comparison rows. Rows with neither (e.g. cross-section-only, rate-only, wrong-isotope) are listed but skipped in MAPE.



## Unit conversion



Ac-225 half-life: **9.920 d** (NNDC). If you only have activity:



```

N_Ac225 = A_Ac225_Bq / λ,   λ = ln(2) / (9.920 × 24 × 3600) s⁻¹

```



GBq → Bq: multiply by `1e9`.  

mCi → Bq: multiply by `3.7e7`.



## Workflow



1. Read the paper; add **one row per (source, conditions, quantity)**.

2. Set `source_type` honestly (`simulation` for Joyo/ORIGEN tables; `decay_leg` for generator milking).

3. Run validation:



   ```powershell

   cd "New folder"

   .\.venv\Scripts\Activate.ps1

   python v3_pilstm/analysis/validate_empirical.py

   ```



4. Output: `v3_pilstm/results/empirical_validation.json` and console MAPE.



## Poster tip



Plot **2–4 real literature points** (simulation and cross-route labeled) on the same axis as model curves. Always cite `source_citation` in footnotes. Never present simulation endpoints as experimental measurements.


