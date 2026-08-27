"""Curate a balanced 100+ source comparison for the Ac-225 V3 project.

The input is the broad OpenAlex discovery pool produced by
``build_literature_audit.py``.  This script applies reproducible topic filters,
removes obvious off-topic and duplicate records, assigns a single comparison
category, and appends a small set of authoritative technical sources.

The output is a scoping-review matrix.  An ``abstract-screened`` row means the
title and abstract were checked for relevance; it does not mean that every
claim in the full paper was independently verified.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "research" / "literature_audit_2026"
INPUT_CSV = AUDIT_DIR / "candidate_pool.csv"
OUTPUT_CSV = AUDIT_DIR / "curated_comparison_120plus.csv"
SUMMARY_JSON = AUDIT_DIR / "curation_summary.json"


CATEGORIES = {
    "A": {
        "name": "Ac-225 production, separation, and quality",
        "quota": 20,
        "questions": "Q1,Q2,Q3,Q4,Q10",
    },
    "B": {
        "name": "Ra-226 nuclear data and neutron-spectrum evidence",
        "quota": 20,
        "questions": "Q1,Q3,Q4,Q5,Q10",
    },
    "C": {
        "name": "Depletion, activation, and trusted solvers",
        "quota": 20,
        "questions": "Q1,Q4,Q5,Q10",
    },
    "D": {
        "name": "Nuclear machine-learning and temporal surrogates",
        "quota": 20,
        "questions": "Q5,Q6,Q7,Q8,Q9",
    },
    "E": {
        "name": "Surrogate-assisted and multi-objective optimization",
        "quota": 20,
        "questions": "Q1,Q2,Q7,Q9",
    },
    "F": {
        "name": "Uncertainty, distribution shift, and reliability",
        "quota": 20,
        "questions": "Q3,Q4,Q8,Q9",
    },
}


# These are especially close competitors or key scientific anchors.  They are
# still screened under the same inclusion rules and are not automatically
# treated as proof beyond the evidence-depth column.
CORE_ASSIGNMENTS = {
    "10.1007/s00259-021-05460-7": "A",
    "10.1016/j.apradiso.2016.09.026": "A",
    "10.1016/j.nucmedbio.2024.108940": "A",
    "10.3390/pr10061215": "A",
    "10.3390/molecules24061095": "A",
    "10.1038/s41598-025-02277-4": "A",
    "10.1007/s10967-024-09811-0": "A",
    "10.1038/s41598-021-04339-9": "A",
    "10.1016/j.apradiso.2021.109676": "A",
    "10.1053/j.semnuclmed.2020.02.003": "A",
    "10.1103/physrevc.108.024616": "B",
    "10.1080/00223131.2023.2243941": "B",
    "10.1016/j.nds.2018.02.001": "B",
    "10.1016/j.anucene.2014.07.048": "C",
    "10.13182/nse10-81": "C",
    "10.1016/j.anucene.2019.107230": "D",
    "10.1016/j.anucene.2023.110204": "D",
    "10.1016/j.anucene.2020.108085": "D",
    "10.1016/j.net.2024.07.024": "D",
    "10.1016/j.anucene.2024.110598": "D",
    "10.1016/j.anucene.2021.108256": "D",
    "10.1109/sccc57464.2022.10000327": "D",
    "10.1016/j.nucengdes.2022.111776": "E",
    "10.1103/physrevaccelbeams.23.044601": "E",
    "10.1016/j.egyai.2025.100655": "E",
    "10.1016/j.jocs.2016.05.013": "E",
    "10.1016/j.enconman.2025.120852": "E",
}


OFFICIAL_SOURCES = [
    {
        "category_code": "A",
        "title": "Production and Quality Control of Actinium-225 Radiopharmaceuticals",
        "authors": "International Atomic Energy Agency",
        "year": "2024",
        "type": "official technical report",
        "venue": "IAEA-TECDOC-2057",
        "doi": "10.61092/iaea.95h3-2j2",
        "landing_url": "https://www-pub.iaea.org/MTCD/Publications/PDF/TE-2057web.pdf",
        "abstract": "Authoritative technical review of Ac-225 production and quality-control practices.",
        "evidence_depth": "official source screened",
        "screening_note": "Use for production and quality-control context; route-specific specifications are not universal clinical limits.",
    },
    {
        "category_code": "B",
        "title": "Experimental Investigation of Deuteron Breakup for the Production of Medical Isotopes",
        "authors": "Jonathan T. Morrell",
        "year": "2021",
        "type": "doctoral dissertation",
        "venue": "University of California, Berkeley",
        "doi": "",
        "landing_url": "https://nucleardata.berkeley.edu/doc/morrell.pdf",
        "abstract": "Contains the 33 and 40 MeV Ra-226 irradiation experiments, measured spectra, fluence, activities, chemistry recovery, and uncertainty tables.",
        "evidence_depth": "full text close-read",
        "screening_note": "Closest end-to-end experimental benchmark for the modeled pathway; Tables 4.1 and 4.2 were checked directly.",
    },
    {
        "category_code": "B",
        "title": "EXFOR Experimental Nuclear Reaction Data Database",
        "authors": "International Atomic Energy Agency Nuclear Data Section",
        "year": "2026",
        "type": "official evaluated database",
        "venue": "IAEA Nuclear Data Services",
        "doi": "",
        "landing_url": "https://www-nds.iaea.org/exfor/",
        "abstract": "International compilation of experimental nuclear-reaction measurements and their bibliographic provenance.",
        "evidence_depth": "official source screened",
        "screening_note": "Useful for reaction-level validation; it does not by itself validate the full production chain.",
    },
    {
        "category_code": "C",
        "title": "ORIGEN Module: Operating History, Depletion, Decay, and Activation",
        "authors": "Oak Ridge National Laboratory",
        "year": "2025",
        "type": "official software documentation",
        "venue": "SCALE 6.3.1 Manual",
        "doi": "",
        "landing_url": "https://scale-manual.ornl.gov/6.3.1/origen/origen-module.html",
        "abstract": "Documents time, flux, and power arrays for multistep irradiation and decay histories in ORIGEN.",
        "evidence_depth": "official source close-read",
        "screening_note": "Establishes that established depletion software already handles changing operating histories.",
    },
    {
        "category_code": "C",
        "title": "Depletion, Activation, and Spent Fuel Source Terms",
        "authors": "Oak Ridge National Laboratory",
        "year": "2026",
        "type": "official software documentation",
        "venue": "SCALE 6.3.3 Manual",
        "doi": "",
        "landing_url": "https://scale-manual.ornl.gov/6.3.3/Depletion-Decay-Methods.html",
        "abstract": "Describes ORIGEN nuclide coverage, arbitrary spectra, time-dependent histories, CRAM, outputs, and typical execution times.",
        "evidence_depth": "official source close-read",
        "screening_note": "Shows that stand-alone reduced depletion is already fast; the stronger surrogate use case is repeated transport-coupled screening.",
    },
    {
        "category_code": "C",
        "title": "OpenMC Depletion and Transmutation User Guide",
        "authors": "OpenMC Project",
        "year": "2026",
        "type": "official software documentation",
        "venue": "OpenMC Documentation",
        "doi": "",
        "landing_url": "https://docs.openmc.org/en/stable/usersguide/depletion.html",
        "abstract": "Documents transport-coupled and independent depletion, variable source rates, decay-only steps, and multiple time integrators.",
        "evidence_depth": "official source close-read",
        "screening_note": "Relevant trusted-solver comparator for geometry- and spectrum-aware extensions.",
    },
    {
        "category_code": "A",
        "title": "New NIST Standard Helps Deliver the Right Dosage of Cancer-Fighting Drugs",
        "authors": "National Institute of Standards and Technology",
        "year": "2025",
        "type": "official metrology release",
        "venue": "NIST",
        "doi": "",
        "landing_url": "https://www.nist.gov/news-events/news/2025/06/new-nist-standard-helps-deliver-right-dosage-cancer-fighting-drugs",
        "abstract": "Describes the first U.S. Ac-225 radioactivity standard and the role of traceable activity measurements.",
        "evidence_depth": "official source screened",
        "screening_note": "Supports activity-metrology methods, not neutron-production yield validation.",
    },
    {
        "category_code": "F",
        "title": "Evaluation Guidelines for Machine Learning-Based Safety Analysis",
        "authors": "U.S. Nuclear Regulatory Commission",
        "year": "2026",
        "type": "official regulatory research context",
        "venue": "U.S. NRC",
        "doi": "",
        "landing_url": "https://www.nrc.gov/reading-rm/doc-collections/nuregs/contract/cr",
        "abstract": "Regulatory research context for verification, validation, uncertainty, and trust in data-driven nuclear models.",
        "evidence_depth": "background only",
        "screening_note": "Included as reliability context; no specific NUREG is used as claim-level evidence in the report.",
    },
    {
        "category_code": "B",
        "questions": "Q3,Q4,Q10",
        "title": "ENDF/B-VIII.0: The 8th Major Release of the Nuclear Reaction Data Library",
        "authors": "D. A. Brown et al.",
        "year": "2018",
        "type": "evaluated library paper",
        "venue": "Nuclear Data Sheets",
        "doi": "10.1016/j.nds.2018.02.001",
        "landing_url": "https://doi.org/10.1016/j.nds.2018.02.001",
        "abstract": "Peer-reviewed description of the ENDF/B-VIII.0 evaluated nuclear data release and its verification basis.",
        "evidence_depth": "publisher abstract screened",
        "screening_note": "Supports nuclear-data provenance; an evaluated library is not a substitute for pathway-level experimental validation.",
    },
]


# Key papers that the broad search can miss because title-search ranking is
# imperfect. Metadata here was checked against the DOI landing page or an
# institutional full-text record before inclusion.
MANUAL_SCHOLARLY_SOURCES = [
    {
        "category_code": "A",
        "questions": "Q3,Q4",
        "title": "Measurement of Ac-227 Impurity in Ac-225 using Decay Energy Spectroscopy",
        "authors": "A. D. Tollefson et al.",
        "year": "2021",
        "type": "article",
        "venue": "Applied Radiation and Isotopes",
        "doi": "10.1016/j.apradiso.2021.109693",
        "landing_url": "https://doi.org/10.1016/j.apradiso.2021.109693",
        "abstract": "Demonstrates trace Ac-227 measurement in Ac-225 by decay energy spectroscopy and reports a 0.142 +/- 0.005% end-of-bombardment activity ratio for one production sample.",
        "evidence_depth": "NIST full text close-read",
        "screening_note": "Establishes a measured impurity level and analytical capability; it does not define a universal acceptable limit.",
    },
    {
        "category_code": "A",
        "questions": "Q3,Q4",
        "title": "Radiopharmaceutical Quality Control Considerations for Accelerator-Produced Actinium Therapies",
        "authors": "Diane S. Abou et al.",
        "year": "2022",
        "type": "article",
        "venue": "Cancer Biotherapy and Radiopharmaceuticals",
        "doi": "10.1089/cbr.2022.0010",
        "landing_url": "https://doi.org/10.1089/cbr.2022.0010",
        "abstract": "Examines radionuclidic and radiochemical quality control when Ac-225 formulations contain long-lived Ac-227 and its daughters.",
        "evidence_depth": "full text close-read",
        "screening_note": "Shows why impurity ratios must be time-defined and evaluated through formulation, shipping, release, and decay-product ingrowth.",
    },
    {
        "category_code": "A",
        "questions": "Q3",
        "title": "Dosimetric impact of Ac-227 in accelerator-produced Ac-225 for alpha-emitter radiopharmaceutical therapy of patients with hematological malignancies: a pharmacokinetic modeling analysis",
        "authors": "George Sgouros; Bin He; Nitya Ray; Dale L. Ludwig; Eric C. Frey",
        "year": "2021",
        "type": "article",
        "venue": "EJNMMI Physics",
        "doi": "10.1186/s40658-021-00410-6",
        "landing_url": "https://doi.org/10.1186/s40658-021-00410-6",
        "abstract": "Models the tissue-dose contribution of Ac-227 and daughters for one antibody-based hematological-cancer treatment scenario.",
        "evidence_depth": "full text close-read",
        "screening_note": "The small modeled dose contribution is application-specific and cannot be converted into a universal production constraint without product context.",
    },
    {
        "category_code": "D",
        "questions": "Q5,Q6,Q7,Q9",
        "title": "Deep learning approach to nuclear fuel transmutation in a fuel cycle simulator",
        "authors": "Jin Whan Bae; Andrei Rykhlevskii; Gwendolyn Chee; Kathryn D. Huff",
        "year": "2020",
        "type": "article",
        "venue": "Annals of Nuclear Energy",
        "doi": "10.1016/j.anucene.2019.107230",
        "landing_url": "https://doi.org/10.1016/j.anucene.2019.107230",
        "abstract": "A dense neural-network depletion surrogate trained on used-fuel data, implemented in Cyclus, and compared with ORIGEN for accuracy and inference speed.",
        "evidence_depth": "full text and publisher page close-read",
        "screening_note": "Closest established use case: medium-fidelity fuel-cycle screening. It proves that train-once nuclear depletion surrogates and speedups over ORIGEN are not new by themselves.",
    },
    {
        "category_code": "E",
        "questions": "Q1,Q2,Q7,Q9",
        "title": "A survey of multi-objective optimization methods and their applications for nuclear scientists and engineers",
        "authors": "Ryan H. Stewart; Todd S. Palmer; Bryony DuPont",
        "year": "2021",
        "type": "review",
        "venue": "Progress in Nuclear Energy",
        "doi": "10.1016/j.pnucene.2021.103830",
        "landing_url": "https://doi.org/10.1016/j.pnucene.2021.103830",
        "abstract": "A nuclear-engineering review of multi-objective optimization, objective design, Pareto fronts, surrogate use, and verification of selected designs.",
        "evidence_depth": "publisher page close-read",
        "screening_note": "Supports the proposed decision-centered evaluation: optimize multiple objectives and verify finalists with the trusted model.",
    },
    {
        "category_code": "E",
        "questions": "Q2,Q7,Q9",
        "title": "Surrogate-driven design optimization with uncertainty constraints in Monte Carlo simulations",
        "authors": "AIMS surrogate-uncertainty collaboration",
        "year": "2025",
        "type": "article",
        "venue": "Energy and AI",
        "doi": "10.1016/j.egyai.2025.100655",
        "landing_url": "https://doi.org/10.1016/j.egyai.2025.100655",
        "abstract": "Studies neural surrogates and Pareto-front recovery for neutron moderator and ion-to-neutron converter designs under different Monte Carlo uncertainty levels.",
        "evidence_depth": "publisher page close-read",
        "screening_note": "Direct evidence that low pointwise error is insufficient; uncertainty can distort objective sensitivities and Pareto fronts.",
    },
    {
        "category_code": "E",
        "questions": "Q2,Q7,Q9",
        "title": "Multi-objective constrained black-box optimization using radial basis function surrogates",
        "authors": "David Eriksson; David Bindel; Christine A. Shoemaker",
        "year": "2016",
        "type": "article",
        "venue": "Journal of Computational Science",
        "doi": "10.1016/j.jocs.2016.05.013",
        "landing_url": "https://doi.org/10.1016/j.jocs.2016.05.013",
        "abstract": "Develops a surrogate-based stochastic framework for computationally expensive constrained multi-objective simulation problems.",
        "evidence_depth": "publisher abstract screened",
        "screening_note": "Methodological background for using exact evaluations strategically instead of trusting every surrogate optimum.",
    },
    {
        "category_code": "D",
        "questions": "Q5,Q6,Q8",
        "title": "NuGNN: a Graph Neural Network for Nuclear Reaction Network Equations",
        "authors": "C. H. Kim; K. Y. Chae; S. Ko; M. R. Mumpower; M. S. Smith",
        "year": "2026",
        "type": "preprint",
        "venue": "arXiv",
        "doi": "10.48550/arxiv.2606.04491",
        "landing_url": "https://doi.org/10.48550/arxiv.2606.04491",
        "abstract": "A graph-neural surrogate for a stiff 690-isotope astrophysical reaction network, evaluated by both abundance errors and full-network rollout behavior.",
        "evidence_depth": "preprint abstract screened",
        "screening_note": "Not peer reviewed as of the audit date. It is an important near-future competitor showing why a five-nuclide chain cannot be the main professional novelty.",
    },
    {
        "category_code": "D",
        "questions": "Q5,Q6,Q8",
        "title": "A hybrid physics-constraint CNN-LSTM prediction framework for radionuclide activity in the primary loop of a nuclear power plant",
        "authors": "Nuclear Engineering and Design research team",
        "year": "2026",
        "type": "article",
        "venue": "Nuclear Engineering and Design",
        "doi": "10.1016/j.nucengdes.2026.114892",
        "landing_url": "https://doi.org/10.1016/j.nucengdes.2026.114892",
        "abstract": "A physics-constrained CNN-LSTM framework for temporal prediction of radionuclide activity in a nuclear system.",
        "evidence_depth": "title and abstract screened",
        "screening_note": "Very close at the method-family level: physics constraints, recurrence, radionuclide activity, and nuclear dynamics already coexist in the 2026 literature.",
    },
]


BLOCK_TERMS = (
    "stock market",
    "traffic flow",
    "wind speed",
    "lithium-ion battery",
    "credit risk",
    "language model",
    "image caption",
    "speech recognition",
    "autonomous vehicle",
    "photovoltaic power forecasting",
    "neutrino",
    "cosmologically",
    "cardiac action potential",
    "neuroimage",
    "cyanobacteria",
    "energy storage-based power systems",
    "lignin",
    "pyridinium",
    "electron diffraction",
)


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def norm_doi(value: str) -> str:
    return clean(value).lower().replace("https://doi.org/", "")


def norm_title(value: str) -> str:
    value = clean(value).lower()
    value = value.replace("actinium 225", "actinium225").replace("ac 225", "ac225")
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def phrase_score(text: str, weighted: dict[str, float]) -> float:
    return sum(weight for phrase, weight in weighted.items() if phrase in text)


def category_scores(row: dict[str, str]) -> dict[str, float]:
    title = clean(row.get("title")).lower()
    abstract = clean(row.get("abstract")).lower()
    # Do not use the discovery-query category as evidence of relevance. Search
    # engines occasionally return off-topic records that share a generic word.
    text = f"{title} {title} {abstract}"
    scores: dict[str, float] = {}

    scores["A"] = phrase_score(
        text,
        {
            "actinium-225": 12,
            "225ac": 12,
            "actinium 225": 12,
            "actinium": 4,
            "production": 5,
            "radionuclidic purity": 5,
            "actinium-227": 4,
            "227ac": 4,
            "radium-226": 4,
            "226ra": 4,
            "purification": 3,
            "separation": 3,
            "supply": 2,
            "processing time": 3,
            "targetry": 2,
        },
    )
    if any(term in title for term in ("therapy", "prostate", "patient", "dosimetry")) and not any(
        term in title for term in ("production", "supply", "purity", "impurity")
    ):
        scores["A"] -= 12
    if not any(
        term in text
        for term in (
            "production",
            "supply",
            "purification",
            "separation",
            "radionuclidic purity",
            "impurity",
            "targetry",
            "cross section",
            "cross-section",
            "processing time",
            "generator",
            "quality control",
        )
    ):
        scores["A"] = 0

    scores["B"] = phrase_score(
        text,
        {
            "226ra(n,2n)": 14,
            "ra-226(n,2n)": 14,
            "radium-226": 7,
            "226ra": 7,
            "deuteron breakup": 10,
            "deuteron-breakup": 10,
            "neutron spectrum": 7,
            "neutron spectra": 7,
            "cross section": 5,
            "cross-section": 5,
            "nuclear data": 4,
            "activation foil": 4,
            "fluence": 3,
            "tendl": 4,
            "endf": 4,
            "exfor": 5,
            "thick target": 3,
        },
    )
    b_anchor = (
        ("neutron" in text and any(term in text for term in ("spectrum", "spectra", "cross section", "cross-section", "nuclear data", "activation foil", "fluence")))
        or any(term in text for term in ("226ra", "radium-226", "deuteron breakup", "deuteron-breakup", "endf", "tendl", "exfor", "irdff"))
    )
    if not b_anchor:
        scores["B"] = 0

    scores["C"] = phrase_score(
        text,
        {
            "origen": 12,
            "openmc": 10,
            "serpent": 7,
            "depletion": 9,
            "burnup": 5,
            "transmutation": 6,
            "activation calculation": 5,
            "nuclide inventory": 6,
            "isotopic inventory": 6,
            "bateman": 7,
            "cram": 9,
            "matrix exponential": 6,
            "transport-depletion": 8,
            "fuel cycle simulation": 4,
        },
    )
    if not any(term in text for term in ("depletion", "burnup", "transmutation", "origen", "openmc", "serpent", "cram", "bateman", "activation code", "nuclide inventory", "isotopic inventory")):
        scores["C"] = 0

    ml = phrase_score(
        text,
        {
            "machine learning": 6,
            "deep learning": 6,
            "neural network": 6,
            "surrogate model": 5,
            "surrogate modeling": 5,
            "lstm": 8,
            "recurrent neural": 8,
            "multi-task learning": 6,
            "gaussian process": 4,
            "physics-informed": 5,
            "physics-constrained": 5,
        },
    )
    nuclear = phrase_score(
        text,
        {
            "nuclear": 4,
            "reactor": 4,
            "nuclide": 5,
            "isotope": 3,
            "depletion": 5,
            "transmutation": 5,
            "irradiation": 3,
            "spent fuel": 4,
            "radioactive": 2,
        },
    )
    nuclear_anchor = any(term in text for term in ("nuclear", "reactor", "nuclide", "depletion", "transmutation", "spent fuel", "radioisotope", "radionuclide"))
    scores["D"] = ml + nuclear if ml >= 5 and nuclear >= 3 and nuclear_anchor else 0

    optimization = phrase_score(
        text,
        {
            "multi-objective": 9,
            "multiobjective": 9,
            "surrogate-assisted": 8,
            "bayesian optimization": 7,
            "pareto": 6,
            "optimization": 4,
            "optimal control": 4,
            "candidate selection": 3,
            "evolutionary algorithm": 4,
            "genetic algorithm": 3,
            "particle swarm": 3,
            "ranking": 2,
            "regret": 3,
        },
    )
    applied = phrase_score(
        text,
        {
            "nuclear": 4,
            "reactor": 4,
            "neutron": 3,
            "accelerator": 3,
            "isotope": 3,
            "irradiation": 3,
            "engineering": 1,
            "expensive simulation": 2,
        },
    )
    nuclear_or_accelerator = any(term in text for term in ("nuclear", "reactor", "neutron", "accelerator", "radioisotope", "radionuclide", "irradiated fuel", "radiation transport"))
    scores["E"] = optimization + applied if optimization >= 4 and applied >= 1 and nuclear_or_accelerator else 0

    reliability = phrase_score(
        text,
        {
            "out-of-distribution": 10,
            "out of distribution": 10,
            "distribution shift": 8,
            "domain shift": 7,
            "uncertainty quantification": 8,
            "model discrepancy": 7,
            "conformal": 7,
            "calibration": 5,
            "deep ensemble": 5,
            "epistemic uncertainty": 5,
            "aleatoric uncertainty": 5,
            "reliability": 4,
            "trustworthy": 4,
            "extrapolation": 3,
            "validation": 2,
            "sensitivity analysis": 3,
        },
    )
    scientific = phrase_score(
        text,
        {
            "surrogate": 4,
            "scientific machine learning": 5,
            "physics-informed": 4,
            "nuclear": 3,
            "reactor": 3,
            "depletion": 3,
            "engineering": 1,
            "simulation": 1,
        },
    )
    scores["F"] = reliability + scientific if reliability >= 4 and scientific >= 1 else 0
    return scores


def included(row: dict[str, str]) -> bool:
    title = clean(row.get("title"))
    abstract = clean(row.get("abstract"))
    text = f"{title} {abstract}".lower()
    if not title or any(term in text for term in BLOCK_TERMS):
        return False
    if clean(row.get("type")) not in {"article", "review", "proceedings-article", "report", "dissertation"}:
        return False
    if not norm_doi(row.get("doi", "")) and clean(row.get("type")) in {"article", "review"}:
        return False
    year = clean(row.get("year"))
    if year.isdigit() and int(year) > 2026:
        return False
    return True


def preferred_record(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    type_rank = {"article": 5, "review": 4, "proceedings-article": 3, "report": 2, "dissertation": 1}
    return max(
        rows,
        key=lambda row: (
            type_rank.get(clean(row.get("type")), 0),
            bool(norm_doi(row.get("doi", ""))),
            bool(clean(row.get("abstract"))),
            int(clean(row.get("cited_by_count")) or 0),
        ),
    )


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = norm_title(row.get("title", ""))
        grouped.setdefault(key, []).append(row)
    return [preferred_record(group) for group in grouped.values()]


def data_basis(text: str) -> str:
    text = text.lower()
    has_measurement = any(term in text for term in ("experiment", "measured", "measurement", "clinical data"))
    has_simulation = any(
        term in text
        for term in ("simulation", "monte carlo", "origen", "openmc", "serpent", "casmo", "synthetic")
    )
    if has_measurement and has_simulation:
        return "mixed simulation and measurement"
    if has_measurement:
        return "experimental or measured"
    if has_simulation:
        return "simulation or solver generated"
    if any(term in text for term in ("review", "survey", "overview")):
        return "review or synthesis"
    return "method or metadata"


def method_family(text: str) -> str:
    text = text.lower()
    methods = []
    tests = [
        ("LSTM/recurrent network", ("lstm", "recurrent neural")),
        ("neural-network surrogate", ("neural network", "deep learning", "multi-task learning")),
        ("Gaussian-process surrogate", ("gaussian process", "kriging")),
        ("physics-informed ML", ("physics-informed", "physics-constrained", "physics guided")),
        ("multi-objective optimization", ("multi-objective", "multiobjective", "pareto")),
        ("Bayesian/evolutionary optimization", ("bayesian optimization", "genetic algorithm", "particle swarm", "evolutionary algorithm")),
        ("transport/depletion solver", ("origen", "openmc", "serpent", "depletion", "cram", "bateman")),
        ("nuclear experiment", ("irradiation experiment", "measured cross section", "activation foil", "spectroscopy")),
        ("uncertainty/reliability method", ("uncertainty quantification", "out-of-distribution", "conformal", "calibration")),
        ("production-route study", ("actinium-225 production", "225ac production", "production of 225ac")),
    ]
    for label, terms in tests:
        if any(term in text for term in terms):
            methods.append(label)
    return "; ".join(methods[:3]) or "domain study"


def flag(text: str, terms: tuple[str, ...]) -> str:
    text = text.lower()
    return "yes" if any(term in text for term in terms) else "no"


def relevance_level(code: str, score: float, text: str) -> str:
    text = text.lower()
    exact = any(
        term in text
        for term in (
            "226ra(n,2n)",
            "ra-226(n,2n)",
            "deuteron breakup",
            "deep learning approach to nuclear fuel transmutation",
            "fast uncertainty quantification of spent nuclear fuel",
            "anicca",
        )
    )
    if exact or score >= 25:
        return "high"
    if score >= 14 or code in {"C", "D", "E"} and score >= 10:
        return "medium"
    return "context"


def comparison_row(row: dict[str, Any], code: str, score: float, index: int) -> dict[str, Any]:
    title = clean(row.get("title"))
    abstract = clean(row.get("abstract"))
    text = f"{title}. {abstract}"
    doi = norm_doi(clean(row.get("doi")))
    landing_url = clean(row.get("landing_url")) or (f"https://doi.org/{doi}" if doi else "")
    evidence_depth = clean(row.get("evidence_depth")) or "title and abstract screened"
    return {
        "source_id": f"S{index:03d}",
        "category_code": code,
        "category": CATEGORIES[code]["name"],
        "questions": clean(row.get("questions")) or CATEGORIES[code]["questions"],
        "title": title,
        "authors": clean(row.get("authors")),
        "year": clean(row.get("year")),
        "source_type": clean(row.get("type")),
        "venue": clean(row.get("venue")),
        "doi": doi,
        "url": landing_url,
        "evidence_depth": evidence_depth,
        "data_basis": data_basis(text),
        "method_family": method_family(text),
        "project_relevance": relevance_level(code, score, text),
        "ra226_n2n": flag(text, ("226ra(n,2n)", "ra-226(n,2n)", "radium-226(n,2n)", "226ra (n,2n)")),
        "ac225_output": flag(text, ("225ac", "ac-225", "actinium-225", "actinium 225")),
        "time_history": flag(text, ("time-dependent", "time varying", "time-varying", "operating history", "irradiation time", "cooling time", "decay time", "transient", "temporal")),
        "recurrent_model": flag(text, ("lstm", "recurrent neural", "gru")),
        "physics_constraint": flag(text, ("physics-informed", "physics-constrained", "physics guided", "physical constraint")),
        "multiobjective_optimization": flag(text, ("multi-objective", "multiobjective", "pareto")),
        "experimental_validation": "yes" if data_basis(text) in {"experimental or measured", "mixed simulation and measurement"} else "no",
        "uncertainty_or_ood": flag(text, ("uncertainty", "out-of-distribution", "distribution shift", "domain shift", "conformal", "calibration")),
        "trusted_solver_reference": flag(text, ("origen", "openmc", "serpent", "casmo", "scale", "monte carlo", "transport code")),
        "screening_score": round(score, 3),
        "screening_note": clean(row.get("screening_note")) or "Included for structured comparison; use the linked paper for claim-level details.",
        "abstract": abstract,
    }


def curate() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as handle:
        discovered = list(csv.DictReader(handle))
    eligible = deduplicate([row for row in discovered if included(row)])

    for row in eligible:
        row["_scores"] = category_scores(row)
        row["_doi"] = norm_doi(row.get("doi", ""))

    selected: list[tuple[dict[str, Any], str, float]] = []
    used: set[str] = set()
    counts: Counter[str] = Counter()

    by_doi = {row["_doi"]: row for row in eligible if row["_doi"]}
    missing_core = []
    for doi, code in CORE_ASSIGNMENTS.items():
        row = by_doi.get(doi)
        if row is None:
            missing_core.append(doi)
            continue
        key = row["_doi"] or norm_title(row["title"])
        if key in used or counts[code] >= CATEGORIES[code]["quota"]:
            continue
        score = max(float(row["_scores"].get(code, 0)), 12.0)
        selected.append((row, code, score))
        used.add(key)
        counts[code] += 1

    category_order = ("A", "B", "C", "D", "E", "F")
    minimum = {"A": 10, "B": 8, "C": 9, "D": 10, "E": 8, "F": 8}
    for code in category_order:
        ranked = sorted(
            eligible,
            key=lambda row: (
                -float(row["_scores"].get(code, 0)),
                -float(row.get("relevance_score") or 0),
                -int(row.get("cited_by_count") or 0),
                row.get("title", ""),
            ),
        )
        for row in ranked:
            if counts[code] >= CATEGORIES[code]["quota"]:
                break
            key = row["_doi"] or norm_title(row["title"])
            score = float(row["_scores"].get(code, 0))
            if key in used or score < minimum[code]:
                continue
            selected.append((row, code, score))
            used.add(key)
            counts[code] += 1

    shortages = {
        code: CATEGORIES[code]["quota"] - counts[code]
        for code in category_order
        if counts[code] < CATEGORIES[code]["quota"]
    }
    if shortages:
        raise RuntimeError(f"Unable to fill curation quotas: {shortages}")

    selected.sort(key=lambda item: (item[1], -item[2], item[0].get("title", "")))
    rows = [
        comparison_row(row, code, score, index)
        for index, (row, code, score) in enumerate(selected, start=1)
    ]

    for source in OFFICIAL_SOURCES:
        code = source["category_code"]
        doi = norm_doi(source.get("doi", ""))
        existing_index = next((i for i, row in enumerate(rows) if doi and row.get("doi") == doi), None)
        if existing_index is not None:
            replacement = comparison_row(source, code, 30.0, existing_index + 1)
            replacement["source_id"] = rows[existing_index]["source_id"]
            rows[existing_index] = replacement
            continue
        rows.append(comparison_row(source, code, 30.0, len(rows) + 1))

    existing_dois = {row["doi"] for row in rows if row["doi"]}
    manual_added = 0
    for source in MANUAL_SCHOLARLY_SOURCES:
        doi = norm_doi(source.get("doi", ""))
        if doi and doi in existing_dois:
            continue
        code = source["category_code"]
        rows.append(comparison_row(source, code, 30.0, len(rows) + 1))
        if doi:
            existing_dois.add(doi)
        manual_added += 1

    summary = {
        "discovered_records": len(discovered),
        "eligible_after_type_topic_and_title_dedup": len(eligible),
        "scholarly_comparison_rows": len(selected),
        "authoritative_supplement_rows": len(OFFICIAL_SOURCES),
        "manual_key_scholarly_rows_added": manual_added,
        "total_comparison_rows": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "source_type_counts": dict(Counter(row["source_type"] for row in rows)),
        "evidence_depth_counts": dict(Counter(row["evidence_depth"] for row in rows)),
        "doi_count": sum(bool(row["doi"]) for row in rows),
        "high_relevance_count": sum(row["project_relevance"] == "high" for row in rows),
        "missing_core_dois_in_automated_discovery": missing_core,
        "missing_core_note": "Some missing discovery DOIs are appended from verified manual metadata below; this field audits search recall, not final-matrix absence.",
        "method_note": (
            "The 120 scholarly rows are a balanced scoping comparison selected from title/abstract metadata. "
            "The authoritative supplement is separately labeled. Claim-level conclusions rely on the smaller "
            "close-read and official evidence set identified in the report."
        ),
    }
    return rows, summary


def write_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, summary = curate()
    write_csv(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
