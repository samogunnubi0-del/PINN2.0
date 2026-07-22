# data/evaluated — Real Nuclear Data Retrieved 2026-07-18 (Data_Hunter)

Everything in this folder was **actually downloaded** from primary sources (no
hand transcription unless explicitly marked). Raw downloads are kept in `_raw/`
for audit. Retrieval date for all items: **2026-07-18**.

## Files

| File | What | Source (how a student can reproduce) |
|---|---|---|
| `exfor_ra226_n2n.csv` | O'Connor & Perkin 1960: σ(n,2n)=1.60±0.20 b, σ(n,3n)=0.63±0.07 b @ 14.5 MeV (EXFOR **21405**) | EXFOR web https://www-nds.iaea.org/exfor/ → search "entry 21405"; or GitHub mirror https://github.com/IAEA-NDS/exfor_json (`json/214/21405.json`) |
| `exfor_ra226_ngamma_thermal.csv` | Five REAL thermal σ(n,γ) measurements @0.0253 eV: 15 b (1950), 23±1 b (1953), 19 b (1949), **14.0±4.0 b** (Kukleva 2015), **13.8±0.3 b** (Bagheri 2015) | same EXFOR routes, entries 11727/12262/12282/31745/31760 |
| `jendl5_ra226_n2n_sigmaE.csv` | JENDL-5 evaluated σ(n,2n)(E), 13 pts, threshold 6.4218 MeV, **0.7557 b @ 14 MeV**, peak 2.53 b @ 10 MeV | IAEA mirror https://www-nds.iaea.org/public/download-endf/JENDL-5/n/n_088-Ra-226_8834.zip |
| `endfb8_ra226_n2n_sigmaE.csv` | ENDF/B-VIII.0 evaluated σ(n,2n)(E) — *identical table* (ENDF adopted the JENDL eval for Ra-226) | https://www-nds.iaea.org/public/download-endf/ENDF-B-VIII.0/n/n_8834_88-Ra-226.zip |
| `jendl5_ra226_ngamma_sigmaE.csv`, `endfb8_ra226_ngamma_sigmaE.csv` | Evaluated σ(n,γ)(E) — **zero below 1 keV in both libraries** (no evaluated thermal capture!) | same zips; parsed with `parse_endf.py` |
| `halflives_nndc.csv` | NuDat 3 half-lives fetched live: Ra-226 **1603±8 y**, Ra-225 **14.9±2 d**, Ac-225 **10.0±1 d**, Ra-227 **42.2±5 min**, Ac-227 **21.772±3 y** | https://www.nndc.bnl.gov/nudat3/ → type nuclide → "Decay Radiation" |
| `literature_benchmarks_additions.csv` | new rows for `data/literature_benchmarks.csv` (same format) + citation corrections | see per-row notes |
| `_raw/` | audit trail: ENDF/JENDL .dat files, EXFOR JSON + raw entries, NuDat HTML, Hogle 2016 & Snow 2025 full-text PDFs (OSTI) | — |

## Key facts for the ODE

1. **σ(n,2n) at 14 MeV = 755.7 mb (evaluated) vs 27 mb in the synthetic sigmoid — ~28× too small.** At 14.5 MeV interpolation of the evaluated curve gives ~0.51 b, vs the *measured* 1.60±0.20 b (O'Connor). The measurement is ~2–3× the evaluation at 14.5 MeV — a real, citable tension for the PINN to respect.
2. **σ(n,γ) thermal: use 13.8±0.3 b (Bagheri 2015, EXFOR 31760)**. Note ENDF/B-VIII.0 & JENDL-5 both lack thermal capture entirely (0 below 1 keV); the old `cross_section_bands.csv` "12.8 b ENDF/B-VIII thermal" label is therefore wrong (12.8 b traces to Mughabghab's Atlas, not ENDF).
3. **Half-life updates vs typical code values**: Ra-226 1603 y (not 1600), Ac-225 10.0 d per current NuDat (literature mixes 9.92 d and 10.0 d — Snow 2025 uses 9.92 d, Matyskin uses Ra-225 14.8 d; NuDat says 14.9 d).
4. **Joyo (Sano et al. 2024)**: 15.4±6.2 GBq Ac-225 per g Ra-226 per 45-d core-center cycle; 55.3–129.1 GBq/yr at 6 cycles. Iwahashi 2022 (open access MDPI): ~30 GBq after 60 d + 8 d cool, 15.7 GBq/cycle with 3× milking at **17.5 d** optimum, ~47 GBq/yr; ~10 GBq Ac-227 co-produced; threshold 6.4 MeV confirmed.

## Access notes (for ISEF "how I obtained real data")
- **EXFOR** (IAEA): free, no login. Web form: Target 88-RA-226, Reaction (N,2N), Quantity SIG. GitHub mirrors (IAEA-NDS/exfor_json, IAEA-NDS/EXFOR-Entry-File) give raw files without the Cloudflare-protected web form.
- **ENDF/B-VIII.0 & JENDL-5**: free zip-per-isotope downloads from the IAEA mirror `https://www-nds.iaea.org/public/download-endf/` (directory listing enabled). NNDC (https://www.nndc.bnl.gov/) offers the same via "Sigma"/ENDF retrieval.
- **NuDat 3**: free, no login.
- **OSTI.gov**: Hogle 2016 (OSTI 1253240) and Snow 2025 (OSTI 3028837) have free full-text PDFs (`https://www.osti.gov/servlets/purl/<id>`).
- **MDPI Processes 10(7):1239** (Iwahashi 2022): free open access.
- **Paywalled/blocked**: Sano et al. JNST 2024 full text (T&F; abstract via ADS/Waseda Pure), Melville 2007 (Elsevier), Matyskin 2024 full text (Elsevier; abstract via PubMed PMID 39002498), Sasaki-era Joyo one-group cross sections (figures only).
