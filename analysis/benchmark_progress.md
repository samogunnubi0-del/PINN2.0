# Literature benchmark progress — IsotopePINN v2 & PI-LSTM v3

*Generated: 2026-07-18T21:59:17.605358+00:00*

## Model status

| Model | Status | Key metric |
|-------|--------|------------|
| **v2 MLP-PINN** (frozen) | Production | Held-out 6/6 PASS; Ac-225 median 4.51% vs ODE |
| **v3 PI-LSTM** | Trained (6000 epochs) | Ac-225 median 5.5% vs ODE on test split (full trajectory); held-out endpoint 5.1% (v2: 8.2%) |

## v2 vs v3 on held-out synthetic scenarios (22 random, vs Radau5 ODE)

| Species | v2 median rel error | v3 PI-LSTM median rel error |
|---------|--------------------:|----------------------------:|
| Ra-226 | 16.3% | 15.6% |
| Ra-225 | 7.4% | 7.9% |
| Ac-225 | 8.2% | 5.1% |
| Ra-227 | 14.5% | 6.6% |
| Ac-227 | 6.7% | 10.8% |

High-flux Ra-227 overshoot: v2=0.00×, v3=0.0×

## Literature benchmark fit

**17 rows** in CSV — 3 with computable activity endpoints (2 Joyo fast-reactor simulations + 1 photonuclear context row), 7 reference-only (wrong route/channel), 7 no activity.

| ID | Tier | Type | Quantity | Literature A(Ac-225) | v2 vs lit | v3 vs lit | Status |
|----|------|------|----------|---------------------:|-----------|-----------|--------|
| `lit_00_o_connor___perkin_1960` | T3 | cross_route | σ(n,2n) cross-section (no activity) | — | n/a | n/a | skipped |
| `lit_01_sasaki_et_al__2023_joyo_sim` | T2 | simulation | Ac-225 activity (Bq) | 1.540e+10 Bq | 967.3% (10.7× over — outside 5× band) | 96.1% (25.5× under — outside 5× band) | compared |
| `lit_02_iwahashi_et_al__2022_joyo_origen` | T2 | simulation | Ac-225 activity (Bq) | 3.000e+10 Bq | 364.9% (4.6× over — within 5× band) | 98.4% (2.0× over — within 5× band) | compared |
| `lit_03_hogle_et_al__2016_ornl_hfir_ra_226_irrad` | T1 | empirical | 227Ac activity (wrong channel for A | — | n/a | n/a | skipped |
| `lit_04_higashi_et_al__2022_cyclotron_ac_225_pro` | T3 | cross_route | Ac-225 activity (Bq) | 2.230e+06 Bq | n/a | n/a | reference_only |
| `lit_05_morgenstern___apostolidis_2005_cyclotron` | T3 | cross_route | Ac-225 activity (Bq) | 4.847e+08 Bq | n/a | n/a | reference_only |
| `lit_06_snow_et_al__2025_inl_photonuclear` | T3 | cross_route | Production rate (no absolute EOB ac | — | n/a | n/a | skipped |
| `lit_07_tadokoro___maeda_2023_linac` | T3 | cross_route | Ac-225 activity (Bq) | 4.120e+06 Bq | n/a | n/a | reference_only |
| `lit_08_maslov_et_al__2006_mt_25_microtron` | T3 | cross_route | Production rate (no absolute EOB ac | — | n/a | n/a | skipped |
| `lit_09_anl_fnal_spallation_ac_225_measurement` | T3 | cross_route | Ac-225 activity (Bq) | 7.696e+07 Bq | n/a | n/a | reference_only |
| `lit_10_us6299666b1_patent_table_1` | T3 | cross_route | Fractional yield ratio (no absolute | — | n/a | n/a | skipped |
| `lit_11_kuznetsov_et_al__2014_sm_reactor_table_2` | T1 | empirical | See notes | — | n/a | n/a | skipped |
| `lit_12_matyskin_et_al__2024_penn_state_bnr` | T1 | empirical | Ac-225 activity (Bq) | 8.000e+02 Bq | 100.0% (31261.2× under) | 90.2% (10.2× under) | compared |
| `lit_13_kuznetsov_et_al__2014_sm_reactor_table_2` | T1 | empirical | 227Ac activity (wrong channel for A | — | n/a | n/a | skipped |
| `lit_14_mcdevitt_et_al__2017_itu_karlsruhe` | T4 | decay_leg | Ac-225 activity (Bq) | 1.443e+09 Bq | n/a | n/a | reference_only |
| `lit_15_ornl_229th_generator_campaign` | T4 | decay_leg | Ac-225 activity (Bq) | 3.700e+09 Bq | n/a | n/a | reference_only |
| `lit_16_melville_et_al__2007_linac_ra_225_ac_225` | T3 | cross_route | Ac-225 activity (Bq) | 1.070e+06 Bq | n/a | n/a | reference_only |

### Apples-to-oranges notes

- **lit_00_o_connor___perkin_1960**: σ(n,2n)=1.60±0.20 barn on 226Ra at 14.5 MeV; EXFOR anchor — no Ac-225 activity reported (cross-section sanity check only)
- **lit_01_sasaki_et_al__2023_joyo_sim**: Joyo/ORIGEN simulation endpoint — order-of-magnitude check only; ODE cross-sections may differ from evaluated neutronics in paper
- **lit_02_iwahashi_et_al__2022_joyo_origen**: Joyo/ORIGEN simulation endpoint — order-of-magnitude check only; ODE cross-sections may differ from evaluated neutronics in paper
- **lit_03_hogle_et_al__2016_ornl_hfir_ra_226_irrad**: HFIR reactor Ra-226 irradiation; 45.1 kBq 227Ac per µg Ra-226 at 7 d EOB; thermal capture (n,γ) 227Ac channel — validates impurity leg, N_Ra226_0 = 1 µg basis
- **lit_04_higashi_et_al__2022_cyclotron_ac_225_pro**: Cross-route production (γ,n, p,2n, etc.) — neutron ODE not applicable
- **lit_05_morgenstern___apostolidis_2005_cyclotron**: Cross-route production (γ,n, p,2n, etc.) — neutron ODE not applicable
- **lit_06_snow_et_al__2025_inl_photonuclear**: 39 MeV bremsstrahlung on Ra-226; peak production rate 335.4 kBq Ac-225 per g Ra per mA per h at 17 d post-EOB; (γ,n) cross-route — rate only, no absolute EOB activity tabulated
- **lit_07_tadokoro___maeda_2023_linac**: Cross-route production (γ,n, p,2n, etc.) — neutron ODE not applicable
- **lit_08_maslov_et_al__2006_mt_25_microtron**: Production rate 550 Bq/(µA·h·mg Ra); ~20 h irradiation; electron-bremsstrahlung (γ,n) cross-route — rate only
- **lit_09_anl_fnal_spallation_ac_225_measurement**: Cross-route production (γ,n, p,2n, etc.) — neutron ODE not applicable
- **lit_10_us6299666b1_patent_table_1**: 15% Ac-225/Ra activity ratio at 15.2 MeV proton irradiation; cyclotron cross-route — fractional yield only
- **lit_11_kuznetsov_et_al__2014_sm_reactor_table_2**: 3.16 mg RaCO₃; 25 d irradiation (600 h); **225Ra 6.0e9 Bq** measured after 17 d cooldown — parent of Ac-225, not Ac-225 activity; thermal reactor (n,γ) chain
- **lit_12_matyskin_et_al__2024_penn_state_bnr**: Reactor photonuclear (γ,n) — compared for completeness but not (n,2n) validation
- **lit_13_kuznetsov_et_al__2014_sm_reactor_table_2**: 3.16 mg RaCO₃ (8.42e18 Ra-226 atoms); 25 effective d + 17 d cool; **227Ac 7.8e10 Bq/g → 2.465e8 Bq for this target** measured — thermal (n,γ) 227Ac impurity channel (complements Hogle HFIR)
- **lit_14_mcdevitt_et_al__2017_itu_karlsruhe**: 229Th generator decay chain — validates Bateman leg, not neutron transmutation
- **lit_15_ornl_229th_generator_campaign**: 229Th generator decay chain — validates Bateman leg, not neutron transmutation
- **lit_16_melville_et_al__2007_linac_ra_225_ac_225**: Cross-route production (γ,n, p,2n, etc.) — neutron ODE not applicable

## Summary

- **v2** passes internal held-out validation (6/6) but shares the reference ODE's cross-section gap on the Joyo simulation anchors (Sasaki et al. 2023 Joyo sim: ODE 18.9× over, v2 10.7× over, v3 25.5× under; Iwahashi et al. 2022 Joyo ORIGEN: ODE 11.6× over, v2 4.6× over, v3 2.0× over) — an ODE physics gap (σ(n,2n) model), not a PINN regression.
- **v3 PI-LSTM** (Results-6, 6000 epochs) tracks the ODE teacher on held-out scenarios (Ac-225 endpoint median 5.1%) and beats v2 under the compare protocol (5.1% vs 8.2%), but inherits the same ODE literature gap.
- **14** rows are structurally non-comparable to the (n,2n) neutron ODE; photonuclear rows are context only (no pass/fail verdict).

## Files

- Machine-readable: `analysis/benchmark_progress.json`
- Empirical validation: `v3_pilstm/results/empirical_validation.json`
- v2 vs v3 compare: `v3_pilstm/results/compare_v2_pilstm.json`
