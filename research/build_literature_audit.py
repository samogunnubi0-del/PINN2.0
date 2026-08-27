"""Build a reproducible scholarly-source candidate pool for the Ac-225 audit.

OpenAlex is used for discovery and bibliographic metadata. The resulting pool
is screened and curated separately; discovery metadata is not treated as proof
that a paper supports a scientific claim.
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "research" / "literature_audit_2026"
RAW_DIR = OUT_DIR / "openalex_raw"

SEARCHES = [
    {
        "category": "Ac-225 production and supply",
        "questions": "Q1,Q3,Q4",
        "query": "actinium-225 production accelerator reactor radium thorium",
        "limit": 80,
    },
    {
        "category": "Ra-226 fast-neutron pathway",
        "questions": "Q1,Q2,Q3,Q4,Q5,Q10",
        "query": "radium-226 neutron actinium-225 radium-225 production",
        "limit": 80,
    },
    {
        "category": "Berkeley experiment and neutron spectrum",
        "questions": "Q4,Q10",
        "query": "deuteron breakup beryllium neutron spectrum radium-226 actinium-225 Morrell",
        "limit": 50,
    },
    {
        "category": "Ac-227 impurity and radionuclidic purity",
        "questions": "Q2,Q3,Q4",
        "query": "actinium-227 impurity actinium-225 production radionuclidic purity",
        "limit": 60,
    },
    {
        "category": "Depletion and transport solvers",
        "questions": "Q1,Q4,Q5,Q10",
        "query": "ORIGEN OpenMC SERPENT depletion isotope inventory validation",
        "limit": 90,
    },
    {
        "category": "Nuclear depletion ML surrogates",
        "questions": "Q5,Q6,Q7",
        "query": "machine learning surrogate nuclear depletion nuclide inventory",
        "limit": 100,
    },
    {
        "category": "Nuclear dynamics and recurrent models",
        "questions": "Q5,Q6",
        "query": "LSTM recurrent neural network nuclear reactor dynamics surrogate",
        "limit": 90,
    },
    {
        "category": "Physics-informed stiff dynamics",
        "questions": "Q4,Q5,Q6",
        "query": "physics-informed neural network stiff ODE chemical kinetics surrogate",
        "limit": 90,
    },
    {
        "category": "Nuclear surrogate optimization",
        "questions": "Q1,Q2,Q7,Q9",
        "query": "machine learning surrogate multi-objective optimization nuclear reactor",
        "limit": 100,
    },
    {
        "category": "Surrogate-assisted optimization metrics",
        "questions": "Q7,Q9",
        "query": "surrogate-assisted optimization ranking regret top-k candidate selection",
        "limit": 80,
    },
    {
        "category": "Uncertainty and nuclear data",
        "questions": "Q3,Q4,Q8,Q10",
        "query": "uncertainty quantification nuclear depletion cross section isotope inventory",
        "limit": 90,
    },
    {
        "category": "OOD and scientific surrogate reliability",
        "questions": "Q8,Q9",
        "query": "out-of-distribution detection scientific machine learning surrogate reliability uncertainty",
        "limit": 90,
    },
    {
        "category": "Closest depletion surrogate competitors",
        "questions": "Q5,Q6,Q7,Q9",
        "query": "deep learning nuclear fuel transmutation ORIGEN Cyclus surrogate",
        "limit": 60,
    },
    {
        "category": "Closest depletion surrogate competitors",
        "questions": "Q5,Q6,Q7,Q9",
        "query": "Gaussian process discharged fuel nuclide composition surrogate SERPENT",
        "limit": 50,
    },
    {
        "category": "Closest depletion surrogate competitors",
        "questions": "Q5,Q6,Q7,Q9",
        "query": "inverse depletion artificial neural network used nuclear fuel",
        "limit": 60,
    },
    {
        "category": "Closest depletion surrogate competitors",
        "questions": "Q5,Q6,Q7,Q9",
        "query": "ANICCA irradiation module multi-task learning CRAM SERPENT2",
        "limit": 40,
    },
    {
        "category": "Closest recurrent nuclear competitors",
        "questions": "Q5,Q6,Q8",
        "query": "physics constrained CNN LSTM radionuclide activity nuclear",
        "limit": 60,
    },
    {
        "category": "Medical isotope optimization",
        "questions": "Q1,Q2,Q3,Q7,Q9",
        "query": "medical isotope production irradiation schedule optimization surrogate machine learning",
        "limit": 80,
    },
    {
        "category": "Ra-226 nuclear data",
        "questions": "Q3,Q4,Q10",
        "query": "radium-226 n 2n cross section neutron measurement",
        "limit": 60,
    },
    {
        "category": "Deuteron breakup neutron source",
        "questions": "Q3,Q4,Q10",
        "query": "thick target deuteron breakup beryllium neutron spectrum measurement",
        "limit": 70,
    },
    {
        "category": "Stiff reaction-network surrogates",
        "questions": "Q4,Q5,Q6",
        "query": "stiff nuclear reaction network neural surrogate solver",
        "limit": 60,
    },
    {
        "category": "Surrogate validation and OOD",
        "questions": "Q8,Q9",
        "query": "surrogate model validation out of distribution uncertainty engineering simulation",
        "limit": 70,
    },
    {
        "category": "Ac-225 production and supply",
        "questions": "Q1,Q2,Q3,Q4",
        "query": "actinium-225 production review radium-226 thorium-232 quality control",
        "limit": 80,
    },
    {
        "category": "Ac-227 impurity and radionuclidic purity",
        "questions": "Q2,Q3,Q4",
        "query": "actinium-227 impurity in actinium-225 measurement dosimetry radionuclidic purity",
        "limit": 70,
    },
    {
        "category": "Ac-225 separation and generators",
        "questions": "Q1,Q2,Q3,Q10",
        "query": "radium-225 actinium-225 separation generator production",
        "limit": 70,
    },
    {
        "category": "Medical-isotope scheduling",
        "questions": "Q1,Q2,Q3,Q7,Q9",
        "query": "medical radionuclide production irradiation cooling time optimization",
        "limit": 70,
    },
    {
        "category": "Ra-226 nuclear data",
        "questions": "Q3,Q4,Q10",
        "query": "226Ra n 2n 225Ra cross section",
        "limit": 60,
    },
    {
        "category": "Deuteron breakup neutron source",
        "questions": "Q3,Q4,Q10",
        "query": "deuteron breakup neutron spectra beryllium thick target activation foil time of flight",
        "limit": 90,
    },
    {
        "category": "Nuclear data for isotope production",
        "questions": "Q3,Q4,Q10",
        "query": "nuclear data medical radioisotope production cross sections uncertainty",
        "limit": 90,
    },
    {
        "category": "Evaluated neutron data libraries",
        "questions": "Q3,Q4,Q10",
        "query": "TENDL ENDF EXFOR IRDFF nuclear reaction data evaluation neutron",
        "limit": 80,
    },
    {
        "category": "Neutron-spectrum measurement",
        "questions": "Q3,Q4,Q10",
        "query": "activation foil neutron spectrum unfolding measurement",
        "limit": 80,
    },
    {
        "category": "Depletion solver validation",
        "questions": "Q1,Q4,Q5,Q10",
        "query": "ORIGEN depletion validation isotopic assay spent fuel",
        "limit": 90,
    },
    {
        "category": "Depletion solver validation",
        "questions": "Q1,Q4,Q5,Q10",
        "query": "OpenMC depletion validation nuclide inventory",
        "limit": 70,
    },
    {
        "category": "Depletion solver validation",
        "questions": "Q1,Q4,Q5,Q10",
        "query": "Serpent depletion validation nuclide inventory burnup",
        "limit": 80,
    },
    {
        "category": "Depletion numerical methods",
        "questions": "Q1,Q4,Q5",
        "query": "CRAM burnup depletion matrix exponential solver",
        "limit": 70,
    },
    {
        "category": "Depletion numerical methods",
        "questions": "Q1,Q4,Q5",
        "query": "Bateman equations isotope depletion activation solver",
        "limit": 70,
    },
    {
        "category": "Activation solver validation",
        "questions": "Q1,Q4,Q5,Q10",
        "query": "FISPACT activation code validation nuclear inventory",
        "limit": 70,
    },
    {
        "category": "Nuclear depletion ML surrogates",
        "questions": "Q5,Q6,Q7,Q8,Q9",
        "query": "neural network surrogate nuclide inventory depletion transmutation",
        "limit": 100,
    },
    {
        "category": "Nuclear depletion ML surrogates",
        "questions": "Q5,Q6,Q7,Q8,Q9",
        "query": "machine learning spent nuclear fuel composition surrogate",
        "limit": 100,
    },
    {
        "category": "Nuclear dynamics and recurrent models",
        "questions": "Q5,Q6,Q8",
        "query": "recurrent neural network reactor transient surrogate nuclear",
        "limit": 90,
    },
    {
        "category": "Physics-informed nuclear dynamics",
        "questions": "Q5,Q6,Q8",
        "query": "physics informed neural network nuclear reactor kinetics surrogate",
        "limit": 90,
    },
    {
        "category": "Stiff reaction-network surrogates",
        "questions": "Q4,Q5,Q6,Q8",
        "query": "nuclear reaction network neural surrogate stiff equations",
        "limit": 80,
    },
    {
        "category": "Nuclear surrogate optimization",
        "questions": "Q1,Q2,Q7,Q9",
        "query": "surrogate assisted optimization nuclear reactor design",
        "limit": 100,
    },
    {
        "category": "Nuclear surrogate optimization",
        "questions": "Q1,Q2,Q7,Q9",
        "query": "multiobjective optimization nuclear engineering surrogate model",
        "limit": 100,
    },
    {
        "category": "Accelerator surrogate optimization",
        "questions": "Q1,Q2,Q7,Q9",
        "query": "machine learning surrogate optimization particle accelerator",
        "limit": 90,
    },
    {
        "category": "Neutron-source optimization",
        "questions": "Q1,Q2,Q7,Q9",
        "query": "Bayesian optimization neutron source design Monte Carlo surrogate",
        "limit": 80,
    },
    {
        "category": "Isotope-production optimization",
        "questions": "Q1,Q2,Q3,Q7,Q9",
        "query": "machine learning optimization isotope production irradiation schedule",
        "limit": 80,
    },
    {
        "category": "Scientific ML reliability",
        "questions": "Q8,Q9",
        "query": "out-of-distribution detection scientific machine learning physics surrogate",
        "limit": 90,
    },
    {
        "category": "Scientific ML reliability",
        "questions": "Q8,Q9",
        "query": "conformal prediction scientific machine learning engineering surrogate",
        "limit": 80,
    },
    {
        "category": "Scientific ML reliability",
        "questions": "Q8,Q9",
        "query": "model validation uncertainty nuclear machine learning surrogate",
        "limit": 90,
    },
    {
        "category": "Surrogate optimization reliability",
        "questions": "Q7,Q8,Q9",
        "query": "surrogate error optimization regret ranking candidate selection",
        "limit": 80,
    },
]

INCLUDE_TERMS = {
    "actinium": 6,
    "ac-225": 7,
    "225ac": 7,
    "radium": 5,
    "ra-226": 6,
    "226ra": 6,
    "ra-225": 6,
    "225ra": 6,
    "ac-227": 5,
    "227ac": 5,
    "depletion": 4,
    "transmutation": 4,
    "nuclide": 4,
    "isotope inventory": 5,
    "origen": 6,
    "openmc": 5,
    "serpent": 4,
    "neutron": 2,
    "surrogate": 4,
    "physics-informed": 4,
    "lstm": 5,
    "recurrent": 3,
    "multi-objective": 3,
    "optimization": 2,
    "uncertainty": 2,
    "out-of-distribution": 4,
    "model discrepancy": 3,
    "validation": 2,
}

EXCLUDE_TERMS = {
    "stock market": -8,
    "wind speed": -6,
    "traffic flow": -6,
    "battery state": -5,
    "lithium-ion": -4,
    "financial": -6,
    "language model": -5,
    "image classification": -5,
}

ALLOWED_TYPES = {
    "article",
    "review",
    "book-chapter",
    "dissertation",
    "report",
    "proceedings-article",
    "preprint",
}


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, slots in index.items():
        positions.extend((slot, word) for slot in slots)
    positions.sort()
    return " ".join(word for _, word in positions)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    return value.lower().replace("https://doi.org/", "").replace("http://doi.org/", "")


def author_string(authorships: list[dict[str, Any]]) -> str:
    names = [
        a.get("author", {}).get("display_name", "")
        for a in authorships
        if a.get("author", {}).get("display_name")
    ]
    if len(names) <= 6:
        return "; ".join(names)
    return "; ".join(names[:6]) + "; et al."


def location_fields(location: dict[str, Any] | None) -> tuple[str, str, str]:
    if not location:
        return "", "", ""
    source = location.get("source") or {}
    venue = source.get("display_name") or location.get("raw_source_name") or ""
    publisher = source.get("host_organization_name") or ""
    url = location.get("landing_page_url") or ""
    return normalize_text(venue), normalize_text(publisher), url


def relevance_score(record: dict[str, Any]) -> float:
    text = " ".join(
        [record.get("title", ""), record.get("abstract", ""), record.get("category", "")]
    ).lower()
    score = 0.0
    for term, weight in INCLUDE_TERMS.items():
        if term in text:
            score += weight
    for term, weight in EXCLUDE_TERMS.items():
        if term in text:
            score += weight
    score += min(record.get("cited_by_count", 0), 100) / 25.0
    if record.get("doi"):
        score += 1.5
    if record.get("type") in {"article", "review"}:
        score += 1.0
    if record.get("is_oa"):
        score += 0.25
    return round(score, 3)


def fetch(search: dict[str, Any]) -> dict[str, Any]:
    params = {
        "search": search["query"],
        "per-page": str(search["limit"]),
        "select": ",".join(
            [
                "id",
                "doi",
                "display_name",
                "publication_year",
                "publication_date",
                "type",
                "primary_location",
                "authorships",
                "cited_by_count",
                "open_access",
                "abstract_inverted_index",
                "is_retracted",
                "is_paratext",
            ]
        ),
    }
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "IsotopePINN-student-literature-audit/2026"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def collect() -> list[dict[str, Any]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records_by_key: dict[str, dict[str, Any]] = {}
    categories: dict[str, set[str]] = defaultdict(set)
    questions: dict[str, set[str]] = defaultdict(set)

    for index, search in enumerate(SEARCHES, start=1):
        payload = fetch(search)
        raw_path = RAW_DIR / f"{index:02d}_{re.sub(r'[^a-z0-9]+', '_', search['category'].lower()).strip('_')}.json"
        raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for work in payload.get("results", []):
            if work.get("is_retracted") or work.get("is_paratext"):
                continue
            if work.get("type") not in ALLOWED_TYPES:
                continue
            title = normalize_text(work.get("display_name") or "")
            if not title:
                continue
            doi = normalize_doi(work.get("doi"))
            key = doi or work.get("id") or title.lower()
            venue, publisher, landing_url = location_fields(work.get("primary_location"))
            open_access = work.get("open_access") or {}
            record = records_by_key.get(key)
            if record is None:
                record = {
                    "key": key,
                    "openalex_id": work.get("id") or "",
                    "doi": doi,
                    "title": title,
                    "authors": author_string(work.get("authorships") or []),
                    "year": work.get("publication_year") or "",
                    "publication_date": work.get("publication_date") or "",
                    "type": work.get("type") or "",
                    "venue": venue,
                    "publisher": publisher,
                    "landing_url": landing_url,
                    "oa_url": open_access.get("oa_url") or "",
                    "is_oa": bool(open_access.get("is_oa")),
                    "cited_by_count": int(work.get("cited_by_count") or 0),
                    "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
                }
                records_by_key[key] = record
            categories[key].add(search["category"])
            questions[key].update(search["questions"].split(","))
        time.sleep(0.15)

    records = []
    for key, record in records_by_key.items():
        record["categories"] = "; ".join(sorted(categories[key]))
        record["questions"] = ",".join(sorted(questions[key]))
        record["category"] = sorted(categories[key])[0]
        record["relevance_score"] = relevance_score(record)
        records.append(record)
    records.sort(
        key=lambda item: (
            -float(item["relevance_score"]),
            -int(item["cited_by_count"]),
            str(item["title"]),
        )
    )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "relevance_score",
        "questions",
        "categories",
        "title",
        "authors",
        "year",
        "type",
        "venue",
        "publisher",
        "doi",
        "cited_by_count",
        "is_oa",
        "landing_url",
        "oa_url",
        "openalex_id",
        "abstract",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = collect()
    write_csv(OUT_DIR / "candidate_pool.csv", rows)
    summary = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "discovery_index": "OpenAlex",
        "searches": SEARCHES,
        "unique_candidates": len(rows),
        "with_doi": sum(bool(row["doi"]) for row in rows),
        "peer_review_like": sum(row["type"] in {"article", "review"} for row in rows),
        "note": "Candidate discovery only. Claim-level screening and close reading are separate.",
    }
    (OUT_DIR / "discovery_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
