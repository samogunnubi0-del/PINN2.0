"""Build the readable HTML synthesis for the 2026 Ac-225 literature audit."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "research" / "literature_audit_2026"
MATRIX = AUDIT / "curated_comparison_120plus.csv"
SUMMARY = AUDIT / "curation_summary.json"
OUTPUT = ROOT / "Ac225_Research_Audit_137_Sources_2026.html"
MD_OUTPUT = AUDIT / "RESEARCH_FINDINGS.md"


def esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def link(label: str, url: str) -> str:
    return f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(label)}</a>'


KEY_SOURCES = {
    "morrell": ("Morrell dissertation, Chapter 4", "https://nucleardata.berkeley.edu/doc/morrell.pdf"),
    "breakup": ("Morrell et al., Physical Review C (2023)", "https://doi.org/10.1103/PhysRevC.108.024616"),
    "joyo": ("Iwahashi et al., Joyo uncertainty study", "https://doi.org/10.1080/00223131.2023.2243941"),
    "joyo_design": ("Iwahashi et al., Joyo production study", "https://doi.org/10.3390/pr10071239"),
    "iaea": ("IAEA-TECDOC-2057", "https://doi.org/10.61092/iaea.95h3-2j2"),
    "origen": ("SCALE ORIGEN operating-history documentation", "https://scale-manual.ornl.gov/6.3.1/origen/origen-module.html"),
    "origen_methods": ("SCALE depletion and activation methods", "https://scale-manual.ornl.gov/6.3.3/Depletion-Decay-Methods.html"),
    "openmc": ("OpenMC depletion guide", "https://docs.openmc.org/en/stable/usersguide/depletion.html"),
    "bae": ("Bae et al., neural depletion in Cyclus", "https://doi.org/10.1016/j.anucene.2019.107230"),
    "alba": ("Albà et al., fast spent-fuel UQ", "https://doi.org/10.1016/j.anucene.2023.110204"),
    "gp": ("Gaussian-process fuel-composition surrogate", "https://doi.org/10.1016/j.anucene.2020.108085"),
    "anicca": ("Casas-Molina et al., ANICCA irradiation module", "https://doi.org/10.1016/j.net.2024.07.024"),
    "inverse": ("Khuwaileh and Almomani, inverse depletion", "https://doi.org/10.1016/j.anucene.2024.110598"),
    "cnn_lstm": ("Physics-constraint CNN-LSTM radionuclide model", "https://doi.org/10.1016/j.nucengdes.2026.114892"),
    "nugnn": ("NuGNN preprint", "https://doi.org/10.48550/arXiv.2606.04491"),
    "moo": ("Stewart et al., nuclear multi-objective optimization survey", "https://doi.org/10.1016/j.pnucene.2021.103830"),
    "accelerator": ("Edelen et al., accelerator optimization speedup", "https://doi.org/10.1103/PhysRevAccelBeams.23.044601"),
    "microreactor": ("Microreactor multi-objective control optimization", "https://doi.org/10.1016/j.nucengdes.2022.111776"),
    "uncertainty_opt": ("Surrogate optimization under Monte Carlo uncertainty", "https://doi.org/10.1016/j.egyai.2025.100655"),
    "qc": ("Abou et al., actinium quality control", "https://doi.org/10.1089/cbr.2022.0010"),
    "des": ("Tollefson et al., Ac-227 decay-energy spectroscopy", "https://doi.org/10.1016/j.apradiso.2021.109693"),
    "dosimetry": ("Sgouros et al., Ac-227 dosimetric impact", "https://doi.org/10.1186/s40658-021-00410-6"),
    "fda": ("FDA product-quality considerations", "https://www.fda.gov/media/152472/download"),
    "nist": ("NIST Ac-225 radioactivity standard", "https://www.nist.gov/news-events/news/2025/06/new-nist-standard-helps-deliver-right-dosage-cancer-fighting-drugs"),
}


def cite(*keys: str) -> str:
    return ", ".join(link(*KEY_SOURCES[key]) for key in keys)


QUESTIONS = [
    {
        "id": "Q1",
        "question": "How are Ra-226 irradiation and harvest schedules selected now?",
        "answer": (
            "Published studies normally choose a facility, target, beam or neutron field, irradiation duration, and a fixed cooling/separation plan, then calculate or measure the resulting inventory. "
            "The Berkeley experiments used fixed 7.51-day and 3.69-day irradiations followed by 18.9-day and 20.8-day decay intervals. Joyo studies likewise evaluate specified reactor positions and timing assumptions. "
            "Established ORIGEN and OpenMC workflows already accept multistep time, flux, power, or source-rate histories."
        ),
        "confidence": "High",
        "sources": cite("morrell", "joyo", "joyo_design", "origen", "openmc"),
        "project_action": "Do not claim that changing histories are new. Define a constrained schedule space and prove that V3 can screen it usefully.",
    },
    {
        "id": "Q2",
        "question": "Has irradiation plus cooling been jointly optimized for this exact pathway?",
        "answer": (
            "Optimization is common in nuclear engineering, accelerator tuning, reactor control, fuel management, and isotope-route design. Fixed timing scans and separation-schedule reasoning also exist for Ac-225. "
            "However, this audit did not identify a paper that combines a frozen physics-informed recurrent surrogate with joint irradiation/cooling optimization for the specific Ra-226(n,2n)Ra-225 to Ac-225 pathway and then verifies the selected schedules against a trusted solver and Berkeley data. "
            "That is a defensible research gap, but only as 'not identified in this search,' not as a universal first-ever claim."
        ),
        "confidence": "Moderate",
        "sources": cite("morrell", "moo", "accelerator", "microreactor", "uncertainty_opt"),
        "project_action": "Make the new action schedule discovery, not merely another endpoint predictor. Verify all Pareto finalists with the exact/trusted solver.",
    },
    {
        "id": "Q3",
        "question": "What Ac-227 limit is scientifically meaningful?",
        "answer": (
            "There is no single context-free percentage that your model should call 'safe.' Published thorium-spallation material is often discussed near 0.1-0.3% Ac-227 activity at end of bombardment, and one decay-energy spectroscopy measurement found 0.142 +/- 0.005% for one sample. "
            "A pharmacokinetic study found a very small dose contribution for one antibody treatment scenario, while FDA guidance says impurity limits must be justified through product expiry using retention, stability, dosimetry, and measurement evidence. "
            "The ratio changes with time because Ac-225 and Ac-227 have very different half-lives."
        ),
        "confidence": "High for the conclusion; application-specific for any threshold",
        "sources": cite("des", "qc", "dosimetry", "fda", "iaea"),
        "project_action": "Report Ac-227/Ac-225 at a named reference time and run a threshold sensitivity analysis. Ask an expert to choose the operational constraint.",
    },
    {
        "id": "Q4",
        "question": "Which variables dominate Ac-225 yield uncertainty?",
        "answer": (
            "For this pathway, the most important upstream uncertainties are the energy-dependent Ra-226(n,2n) cross section, neutron spectral shape above the 6.425 MeV threshold, absolute fluence, target amount and geometry, and any moderation that opens competing capture pathways. "
            "Timing and decay constants control the Ra-225 to Ac-225 conversion. Measured product recovery adds a separate chemistry uncertainty: Berkeley recovered only about 37-60% of the produced Ac-225 in the tested separations, so produced activity and recovered activity cannot be treated as the same target."
        ),
        "confidence": "High for variable identification; ranking requires sensitivity calculations",
        "sources": cite("morrell", "breakup", "joyo", "iaea"),
        "project_action": "Run a no-retraining Sobol or Monte Carlo sensitivity study with the trusted solver; keep nuclear yield, chemical recovery, and measurement uncertainty as separate layers.",
    },
    {
        "id": "Q5",
        "question": "Do established solvers already handle changing neutron histories?",
        "answer": (
            "Yes. ORIGEN accepts arrays of time and flux or power, and OpenMC accepts different source rates for each depletion timestep, including zero-rate decay intervals. Both can represent piecewise-changing histories. "
            "Your recurrent architecture may still be useful for extremely repeated screening or richer control sequences, but handling time dependence is not itself novel."
        ),
        "confidence": "High",
        "sources": cite("origen", "origen_methods", "openmc"),
        "project_action": "Benchmark V3 against piecewise histories and measure end-to-end optimization cost. Do not compare only with the already-cheap five-nuclide Bateman solve.",
    },
    {
        "id": "Q6",
        "question": "Are recurrent and physics-constrained nuclear surrogates new?",
        "answer": (
            "No. Dense neural depletion surrogates, Gaussian-process nuclide surrogates, multi-task irradiation modules, inverse-depletion networks, recurrent reactor models, and physics-constrained CNN-LSTM radionuclide models all exist. "
            "A 2026 preprint even uses a graph neural network for a 690-isotope stiff nuclear reaction network. Your exact combination and pathway may still be uncommon, but LSTM plus physics plus nuclear dynamics is not enough by itself for a strong novelty claim."
        ),
        "confidence": "High",
        "sources": cite("bae", "alba", "gp", "anicca", "inverse", "cnn_lstm", "nugnn"),
        "project_action": "Center novelty on what the model enables and proves: reliable decision screening, experimental transfer, and verified schedule tradeoffs.",
    },
    {
        "id": "Q7",
        "question": "Have neural surrogates been placed inside nuclear optimization loops?",
        "answer": (
            "Yes. Published examples include accelerator tuning, microreactor control, PWR fuel management, coupled nuclear experiments, neutron-component design, and wider nuclear multi-objective studies. "
            "Good practice is to use the surrogate to find promising candidates and then evaluate finalists with the original high-fidelity model."
        ),
        "confidence": "High",
        "sources": cite("moo", "accelerator", "microreactor", "uncertainty_opt"),
        "project_action": "Your optimizer must report candidate recall, Pareto-front quality, and exact-solver verification rather than treating surrogate predictions as final truth.",
    },
    {
        "id": "Q8",
        "question": "How should the model detect unfamiliar or unsafe inputs?",
        "answer": (
            "The nuclear and scientific-ML literature emphasizes uncertainty, validation envelopes, and model discrepancy, but no single OOD score is universally reliable. For this project, the most defensible near-term detector is transparent: normalize each input, measure distance from the training envelope, flag out-of-range values and unusual combinations, and send flagged cases to the trusted solver. "
            "An ensemble or conformal interval could improve this later, but would require additional training or calibration data."
        ),
        "confidence": "Moderate",
        "sources": cite("alba", "uncertainty_opt", "origen_methods"),
        "project_action": "Add a no-retrain applicability-domain gate now. Never allow an optimizer to select an unverified OOD schedule.",
    },
    {
        "id": "Q9",
        "question": "Which metrics matter more than median prediction error?",
        "answer": (
            "For a decision tool, ranking and optimization metrics are essential: Spearman or Kendall rank correlation, top-k recall, feasibility classification, Pareto hypervolume, coverage of the trusted Pareto front, and optimization regret after exact verification. "
            "Pointwise median error can look excellent while rare false positives or tail errors send the optimizer toward bad schedules. Your current p95 and zero-case failures make this especially important."
        ),
        "confidence": "High",
        "sources": cite("moo", "uncertainty_opt", "accelerator"),
        "project_action": "Use the frozen V3 to rank a held-out candidate set and compare the selected top schedules with exact-solver rankings before claiming practical utility.",
    },
    {
        "id": "Q10",
        "question": "What is needed to reproduce the Berkeley experiments?",
        "answer": (
            "For each 33 and 40 MeV experiment you need the machine-readable neutron spectrum at the Ra target, absolute fluence and uncertainty, target activity or mass and chemical form, position and solid angle, irradiation duration/current history, end-of-bombardment and separation timestamps, produced and recovered activities, recovery factors, and impurity detection information. "
            "The thesis already provides most scalar values; Professor Bernstein's spectrum-reproduction code is the key machine-readable bridge. A two-group collapse rule must be frozen before looking at model error."
        ),
        "confidence": "High",
        "sources": cite("morrell", "breakup"),
        "project_action": "Write a dated external-validation protocol first. Score produced Ac-225 separately from recovered Ac-225 and treat non-observation of Ac-227 as a detection-limit statement, not exact zero.",
    },
    {
        "id": "Q11",
        "question": "What would make this comparable to a top ISEF research story?",
        "answer": (
            "The transferable lesson from top research projects is not to invent every ingredient from scratch. It is to combine established ideas into a new, testable action, compare against strong alternatives, and validate that action on evidence the method did not train on. "
            "For this project, that means using recurrence, nuclear constraints, and surrogate optimization to discover schedules, then proving whether those schedules remain good under trusted physics and real Berkeley measurements. A polished model with only synthetic median accuracy is weaker than a smaller model that answers a real decision question and exposes where it fails."
        ),
        "confidence": "High as research-design guidance; award outcomes are never predictable",
        "sources": cite("morrell", "bae", "alba", "moo", "uncertainty_opt"),
        "project_action": "Build the project around one clear new action, one independent benchmark, one rigorous control, and one visible failure map. That is the scientific structure to imitate, not another winner's exact topic.",
    },
]


COMPETITORS = [
    ("Bae et al. (2020)", "Dense NN depletion surrogate inside Cyclus", "UDB/ORIGEN-derived used-fuel data", "Measured inventory-level comparison and speed", "Train-once nuclear depletion acceleration is established", "https://doi.org/10.1016/j.anucene.2019.107230"),
    ("Albà et al. (2024)", "NN for spent-fuel composition, decay heat, SA/UQ", "CASMO5 plus measured PWR assemblies", "Measurement comparison and 10x+ cost reduction", "External measurements and uncertainty are stronger than V3 today", "https://doi.org/10.1016/j.anucene.2023.110204"),
    ("Løvbak et al. (2021)", "Gaussian-process nuclide-composition surrogate", "High-fidelity fuel calculations", "Probabilistic prediction", "Surrogate uncertainty is not unique to this project", "https://doi.org/10.1016/j.anucene.2020.108085"),
    ("Casas-Molina et al. (2024)", "Multi-task irradiation module replacing CRAM in ANICCA", "Serpent2 depletion data", "Fuel-cycle integration", "A learned irradiation module in a real code already exists", "https://doi.org/10.1016/j.net.2024.07.024"),
    ("CNN-LSTM radionuclide model (2026)", "Physics-constrained recurrent radionuclide prediction", "Nuclear-plant temporal data", "Temporal activity prediction", "Very close method family; pathway and decision task must carry novelty", "https://doi.org/10.1016/j.nucengdes.2026.114892"),
    ("NuGNN (2026 preprint)", "Graph surrogate for 690-isotope stiff network", "Large simulated astrophysical network", "Full rollout and final abundance behavior", "Network scale dwarfs the current five-nuclide chain", "https://doi.org/10.48550/arXiv.2606.04491"),
    ("Morrell/Berkeley", "Measured Ra-226(n,2n) production with deuteron-breakup neutrons", "Two real irradiations", "Spectrum, fluence, cross section, produced/recovered activity", "Closest experimental truth; the best next validation", "https://nucleardata.berkeley.edu/doc/morrell.pdf"),
    ("Joyo studies", "Fast-reactor Ra-226(n,2n) production and UQ", "Detailed reactor calculations and nuclear data", "Yield and uncertainty under reactor conditions", "Reactor relevance requires transport and geometry beyond two groups", "https://doi.org/10.1080/00223131.2023.2243941"),
    ("Nuclear MOO literature", "Surrogate-assisted multi-objective decision search", "Many nuclear systems", "Pareto fronts plus high-fidelity verification", "Optimization alone is not new; pathway-specific verified discovery may be", "https://doi.org/10.1016/j.pnucene.2021.103830"),
]


def load_rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_rows_html(rows: list[dict[str, str]]) -> str:
    rendered = []
    for row in rows:
        href = row.get("url", "")
        title_html = link(row["title"], href) if href else esc(row["title"])
        rendered.append(
            "<tr "
            f'data-category="{esc(row["category_code"])}" '
            f'data-depth="{esc(row["evidence_depth"])}" '
            f'data-text="{esc(" ".join([row["title"], row["authors"], row["venue"], row["doi"], row["category"], row["questions"]]).lower())}">'
            f'<td class="mono">{esc(row["source_id"])}</td>'
            f'<td><span class="category-dot cat-{esc(row["category_code"])}"></span>{esc(row["category_code"])}</td>'
            f'<td>{title_html}<small>{esc(row["authors"])}</small></td>'
            f'<td>{esc(row["year"])}</td>'
            f'<td>{esc(row["method_family"])}</td>'
            f'<td>{esc(row["data_basis"])}</td>'
            f'<td><span class="depth">{esc(row["evidence_depth"])}</span></td>'
            f'<td>{esc(row["project_relevance"])}</td>'
            "</tr>"
        )
    return "".join(rendered)


def question_html() -> str:
    chunks = []
    for item in QUESTIONS:
        chunks.append(
            f'<article class="answer" id="{item["id"].lower()}">'
            f'<div class="q-id">{item["id"]}</div><div>'
            f'<h3>{esc(item["question"])}</h3>'
            f'<p>{esc(item["answer"])}</p>'
            f'<p class="source-line"><strong>Evidence:</strong> {item["sources"]}</p>'
            f'<p class="action"><strong>What this means for your project:</strong> {esc(item["project_action"])}</p>'
            f'<span class="confidence">Confidence: {esc(item["confidence"])}</span>'
            '</div></article>'
        )
    return "".join(chunks)


def competitor_html() -> str:
    return "".join(
        "<tr>"
        f"<td>{link(name, url)}</td><td>{esc(method)}</td><td>{esc(data)}</td>"
        f"<td>{esc(validation)}</td><td>{esc(lesson)}</td></tr>"
        for name, method, data, validation, lesson, url in COMPETITORS
    )


def category_legend(summary: dict) -> str:
    names = {
        "A": "Ac-225 production and quality",
        "B": "Ra-226 data and neutron spectra",
        "C": "Trusted depletion and activation",
        "D": "Nuclear ML and temporal surrogates",
        "E": "Optimization and decision search",
        "F": "Uncertainty and reliability",
    }
    counts = Counter()
    for category, count in summary["category_counts"].items():
        code = next(key for key, name in names.items() if category.startswith(name.split()[0]) or (key == "B" and category.startswith("Ra-226")) or (key == "C" and category.startswith("Depletion")) or (key == "D" and category.startswith("Nuclear")) or (key == "E" and category.startswith("Surrogate")) or (key == "F" and category.startswith("Uncertainty")))
        counts[code] += count
    return "".join(
        f'<div><span class="category-dot cat-{code}"></span><strong>{code}</strong><span>{esc(name)}</span><b>{counts[code]}</b></div>'
        for code, name in names.items()
    )


def build_html(rows: list[dict[str, str]], summary: dict) -> str:
    close_read = sum("close-read" in row["evidence_depth"] for row in rows)
    official = sum(row["source_type"].startswith("official") or row["source_type"] == "evaluated library paper" for row in rows)
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A transparent 137-source research audit and novelty test for Sam Ogunnubi's Ac-225 V3 surrogate project.">
  <title>Ac-225 V3 Research Audit | 137 Sources</title>
  <style>
    :root {{
      --ink:#13242b; --text:#354850; --muted:#687980; --paper:#fff; --wash:#f3f6f4;
      --line:#d6dfdc; --teal:#087f73; --amber:#b56a00; --red:#a94234; --blue:#2f67a4;
      --violet:#76519a; --green:#4d7b3d; --shadow:0 12px 32px rgba(19,36,43,.09);
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--text); background:var(--wash); font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.55; letter-spacing:0; }}
    a {{ color:#086b76; text-decoration-thickness:1px; text-underline-offset:3px; }}
    a:hover {{ color:#0a4d57; }}
    .topbar {{ position:sticky; top:0; z-index:10; display:flex; align-items:center; justify-content:space-between; gap:20px; padding:10px max(20px,calc((100vw - 1180px)/2)); background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }}
    .topbar strong {{ color:var(--ink); }}
    .topbar nav {{ display:flex; flex-wrap:wrap; gap:14px; font-size:.84rem; }}
    .topbar nav a {{ color:var(--text); text-decoration:none; }}
    header {{ background:#13242b; color:white; }}
    .hero {{ max-width:1180px; margin:auto; min-height:520px; padding:72px 24px 54px; display:grid; grid-template-columns:minmax(0,1.35fr) minmax(300px,.65fr); align-items:end; gap:48px; }}
    .eyebrow {{ margin:0 0 12px; color:#77d4c8; font-weight:750; text-transform:uppercase; font-size:.78rem; }}
    h1 {{ margin:0; max-width:880px; font-family:Georgia,Times New Roman,serif; font-size:clamp(2.6rem,6vw,5.5rem); line-height:.98; font-weight:650; letter-spacing:0; }}
    .hero .lede {{ max-width:780px; margin:24px 0 0; color:#d9e5e2; font-size:1.12rem; }}
    .verdict {{ border-left:4px solid #e0a645; padding:2px 0 2px 20px; }}
    .verdict strong {{ display:block; margin-bottom:8px; color:#ffd38a; font-size:.82rem; text-transform:uppercase; }}
    .verdict p {{ margin:0; color:white; font-size:1.04rem; }}
    main {{ max-width:1180px; margin:auto; padding:42px 24px 80px; }}
    section {{ padding:44px 0; border-bottom:1px solid var(--line); }}
    section:last-child {{ border-bottom:0; }}
    .section-head {{ display:grid; grid-template-columns:220px 1fr; gap:32px; margin-bottom:28px; }}
    .section-head .number {{ color:var(--teal); font-weight:800; font-size:.82rem; text-transform:uppercase; }}
    h2 {{ margin:0; color:var(--ink); font-family:Georgia,Times New Roman,serif; font-size:clamp(1.8rem,3vw,3rem); line-height:1.08; letter-spacing:0; }}
    h3 {{ color:var(--ink); letter-spacing:0; }}
    .section-head p {{ max-width:760px; margin:12px 0 0; color:var(--muted); }}
    .stats {{ display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--line); background:white; box-shadow:var(--shadow); }}
    .stat {{ padding:24px; border-right:1px solid var(--line); }}
    .stat:last-child {{ border-right:0; }}
    .stat b {{ display:block; color:var(--ink); font-family:Georgia,serif; font-size:2.2rem; }}
    .stat span {{ color:var(--muted); font-size:.86rem; }}
    .callout {{ margin-top:24px; padding:22px 24px; border-left:4px solid var(--teal); background:#eaf5f2; }}
    .callout.warning {{ border-left-color:var(--amber); background:#fff6e7; }}
    .callout strong {{ color:var(--ink); }}
    .legend {{ display:grid; grid-template-columns:repeat(3,1fr); border:1px solid var(--line); background:white; }}
    .legend > div {{ display:grid; grid-template-columns:12px 22px 1fr auto; align-items:center; gap:8px; padding:15px 16px; border-right:1px solid var(--line); border-bottom:1px solid var(--line); font-size:.86rem; }}
    .category-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; }}
    .cat-A {{ background:#087f73; }} .cat-B {{ background:#2f67a4; }} .cat-C {{ background:#76519a; }}
    .cat-D {{ background:#b56a00; }} .cat-E {{ background:#a94234; }} .cat-F {{ background:#4d7b3d; }}
    .answers {{ display:grid; gap:12px; }}
    .answer {{ display:grid; grid-template-columns:64px 1fr; gap:18px; padding:24px; background:white; border:1px solid var(--line); }}
    .q-id {{ color:white; background:var(--ink); width:48px; height:48px; display:grid; place-items:center; font-weight:800; }}
    .answer h3 {{ margin:0 0 10px; font-size:1.2rem; }}
    .answer p {{ margin:8px 0; }}
    .source-line {{ color:var(--muted); font-size:.9rem; }}
    .action {{ padding:13px 15px; background:#f0f5f4; border-left:3px solid var(--teal); }}
    .confidence {{ display:inline-block; margin-top:5px; color:#54666d; font-size:.78rem; font-weight:700; text-transform:uppercase; }}
    .goal {{ padding:28px; color:white; background:#087f73; box-shadow:var(--shadow); }}
    .goal h3 {{ margin:0 0 10px; color:white; font-family:Georgia,serif; font-size:1.55rem; }}
    .goal p {{ margin:0; font-size:1.07rem; }}
    .split {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px; }}
    .panel {{ padding:24px; background:white; border:1px solid var(--line); }}
    .panel h3 {{ margin-top:0; }}
    .panel ul {{ margin:0; padding-left:20px; }}
    .panel li {{ margin:8px 0; }}
    .claim-ladder {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--line); border:1px solid var(--line); }}
    .claim-ladder > div {{ background:white; padding:22px; }}
    .claim-ladder strong {{ display:block; color:var(--ink); margin-bottom:8px; }}
    table {{ border-collapse:collapse; width:100%; background:white; font-size:.86rem; }}
    th {{ position:sticky; top:45px; z-index:2; padding:12px; text-align:left; color:white; background:#23383f; }}
    td {{ padding:12px; vertical-align:top; border-bottom:1px solid var(--line); }}
    tr:hover td {{ background:#f6f9f8; }}
    td small {{ display:block; margin-top:4px; color:var(--muted); }}
    .table-wrap {{ overflow:auto; max-height:720px; border:1px solid var(--line); }}
    .competitor-wrap {{ overflow:auto; border:1px solid var(--line); }}
    .controls {{ display:grid; grid-template-columns:1fr auto auto; gap:10px; margin-bottom:12px; }}
    input, select, button {{ min-height:42px; border:1px solid #aebdb9; background:white; color:var(--ink); font:inherit; }}
    input, select {{ padding:8px 11px; }}
    button {{ padding:8px 14px; cursor:pointer; font-weight:700; }}
    button:hover {{ background:#edf4f2; }}
    .mono {{ font-family:Consolas,monospace; }}
    .depth {{ color:#4e646b; font-size:.78rem; }}
    .table-count {{ margin:10px 0 0; color:var(--muted); font-size:.84rem; }}
    .roadmap {{ display:grid; gap:1px; background:var(--line); border:1px solid var(--line); }}
    .roadmap article {{ display:grid; grid-template-columns:120px 1fr 220px; gap:24px; padding:20px; background:white; }}
    .roadmap time {{ color:var(--teal); font-weight:800; }}
    .roadmap h3 {{ margin:0 0 6px; font-size:1.02rem; }}
    .roadmap p {{ margin:0; }}
    .roadmap .done {{ color:var(--muted); font-size:.86rem; }}
    footer {{ padding:28px 24px; color:#d5e0dd; background:#13242b; text-align:center; font-size:.82rem; }}
    @media (max-width:850px) {{
      .hero {{ grid-template-columns:1fr; min-height:auto; padding-top:54px; }}
      .section-head {{ grid-template-columns:1fr; gap:8px; }}
      .stats {{ grid-template-columns:1fr 1fr; }} .stat:nth-child(2) {{ border-right:0; }}
      .legend {{ grid-template-columns:1fr; }} .split,.claim-ladder {{ grid-template-columns:1fr; }}
      .controls {{ grid-template-columns:1fr; }} .roadmap article {{ grid-template-columns:1fr; gap:8px; }}
      .topbar nav {{ display:none; }}
    }}
    @media (max-width:520px) {{ .stats {{ grid-template-columns:1fr; }} .stat {{ border-right:0; border-bottom:1px solid var(--line); }} .answer {{ grid-template-columns:1fr; }} }}
    @media print {{ .topbar,.controls,button {{ display:none !important; }} body {{ background:white; }} main {{ max-width:none; }} section {{ break-inside:avoid; }} .table-wrap {{ max-height:none; overflow:visible; }} th {{ position:static; }} }}
  </style>
</head>
<body>
  <div class="topbar"><strong>Ac-225 V3 Research Audit</strong><nav><a href="#verdict">Verdict</a><a href="#questions">Core questions</a><a href="#novelty">Novelty</a><a href="#roadmap">Plan</a><a href="#sources">{len(rows)} sources</a></nav></div>
  <header>
    <div class="hero">
      <div><p class="eyebrow">Scoping review and novelty test | August 12, 2026</p><h1>What the literature says your project should become.</h1><p class="lede">A transparent comparison of {len(rows)} scholarly and authoritative sources against the final frozen V3 evidence, the Berkeley experiments, established depletion tools, nuclear surrogates, and optimization research.</p></div>
      <div class="verdict"><strong>Bottom line</strong><p>The project is scientifically promising, but the 0.97-1.57% synthetic median is not the main novelty. The strongest path is a verified decision tool that discovers useful irradiation, cooling, and harvest schedules for the Ra-226 fast-neutron route.</p></div>
    </div>
  </header>
  <main>
    <section id="verdict">
      <div class="section-head"><div class="number">01 / Audit Scope</div><div><h2>Large enough to find the field, honest enough to show the limits.</h2><p>The broad discovery pass found {summary['discovered_records']:,} candidate records. A reproducible screen selected 120 balanced scholarly comparisons and added {len(rows) - 120} key official or manually verified sources.</p></div></div>
      <div class="stats"><div class="stat"><b>{len(rows)}</b><span>sources in final comparison</span></div><div class="stat"><b>{summary['doi_count']}</b><span>sources with DOI records</span></div><div class="stat"><b>{official}</b><span>official/evaluated sources</span></div><div class="stat"><b>{close_read}</b><span>full-text or official close reads</span></div></div>
      <div class="callout warning"><strong>Important method boundary:</strong> this is a scoping review, not {len(rows)} complete full-text peer reviews. The searchable matrix labels evidence depth. The core answers below rely mainly on direct thesis reading, official documentation, and publisher or institutional records for the closest papers.</div>
      <div class="legend">{category_legend(summary)}</div>
    </section>

    <section id="questions">
      <div class="section-head"><div class="number">02 / Core Questions</div><div><h2>Direct answers to the questions that decide the project.</h2><p>Each answer distinguishes what is already established, what this audit did not find, and what evidence would turn your project into a stronger scientific contribution.</p></div></div>
      <div class="answers">{question_html()}</div>
    </section>

    <section id="novelty">
      <div class="section-head"><div class="number">03 / Novelty</div><div><h2>Your novelty cannot be “I used an LSTM.”</h2><p>Every major ingredient already exists somewhere. The combination becomes meaningful only if it enables and validates a new scientific action.</p></div></div>
      <div class="goal"><h3>Recommended research question</h3><p>Can a frozen physics-informed recurrent surrogate reliably identify and rank irradiation, cooling, and harvest schedules for the Ra-226(n,2n)Ra-225 to Ac-225 pathway that improve Ac-225 yield while controlling Ac-227 risk, with finalists verified by a trusted solver and evaluated against published Berkeley irradiation data?</p></div>
      <div class="split">
        <div class="panel"><h3>Already established</h3><ul><li>ORIGEN and OpenMC model changing histories.</li><li>Neural nuclear-depletion surrogates train once and infer quickly.</li><li>Physics-informed and recurrent nuclear models exist.</li><li>Surrogate-assisted nuclear optimization exists.</li><li>The Ra-226 fast-neutron route and Berkeley irradiations already exist.</li></ul></div>
        <div class="panel"><h3>Potential contribution</h3><ul><li>A frozen, leakage-resistant Ac-225 pathway surrogate with matched architecture evidence.</li><li>External transfer to the two Berkeley experiments without retraining.</li><li>A pathway-specific schedule Pareto front for yield, impurity, and time.</li><li>Decision reliability measured by ranking, regret, and verified finalists.</li><li>A failure-aware applicability gate that sends risky schedules back to physics.</li></ul></div>
      </div>
      <div class="callout"><strong>Novelty verdict:</strong> no exact published match was identified in this search for the full combination above. That is not proof of world-first status. Before a poster says “novel,” a nuclear mentor should review the search terms, adjacent papers, and exact wording.</div>
    </section>

    <section id="competitors">
      <div class="section-head"><div class="number">04 / Closest Work</div><div><h2>The papers your judges are most likely to compare you with.</h2><p>These competitors show where V3 is strong, where it is behind, and why external validation plus decision quality matter more than another decimal place of median error.</p></div></div>
      <div class="competitor-wrap"><table><thead><tr><th>Work</th><th>What it does</th><th>Data</th><th>Validation</th><th>Lesson for V3</th></tr></thead><tbody>{competitor_html()}</tbody></table></div>
    </section>

    <section id="claims">
      <div class="section-head"><div class="number">05 / Claim Ladder</div><div><h2>What you can say now, and what has to be earned.</h2></div></div>
      <div class="claim-ladder">
        <div><strong>Supported now</strong><p>On a frozen reduced five-nuclide synthetic benchmark, seed-42 V3 reached 0.97% matched and 1.57% shifted worst-mesh median Ac-225 endpoint error and beat its parameter-matched MLP on the declared paired median criterion.</p></div>
        <div><strong>Supported after Berkeley</strong><p>The unchanged V3 was externally evaluated against two published Ra-226 secondary-neutron irradiation experiments. Report both errors and their uncertainty, even if performance is poor.</p></div>
        <div><strong>Supported after decision test</strong><p>The surrogate recovered useful schedules or a Pareto front with measured top-k recall and low exact-solver regret, while an applicability gate prevented unsafe extrapolation.</p></div>
      </div>
      <div class="callout warning"><strong>Still not supported:</strong> “1-3% reactor accuracy,” “clinical accuracy,” “all predictions below 3%,” “general LSTM superiority,” “cheaper than ORIGEN,” or “the first nuclear physics-informed recurrent surrogate.”</div>
    </section>

    <section id="roadmap">
      <div class="section-head"><div class="number">06 / Finish Plan</div><div><h2>A December plan that does not require another A100 run.</h2><p>The highest-value tasks use the frozen checkpoint and trusted calculations. Additional seeds or retraining remain optional and should follow scientific need, not panic.</p></div></div>
      <div class="roadmap">
        <article><time>Aug 12-21</time><div><h3>Freeze the external protocol</h3><p>Record spectrum collapse, units, target activity, fluence, timing, produced-versus-recovered activity, uncertainty scoring, and Ac-227 non-detection handling before seeing V3 error.</p></div><div class="done">Done when: a dated protocol and Berkeley input table exist.</div></article>
        <article><time>Aug 22-Sep 8</time><div><h3>Reproduce Berkeley inputs and run the unchanged model</h3><p>Use Professor Bernstein's spectrum code. Predict both experiments. Compare with 1.40 +/- 0.15 and 0.77 +/- 0.10 microcuries of produced Ac-225, plus the measured Ra-225 production/cross-section evidence.</p></div><div class="done">Done when: both predictions, uncertainty comparisons, and failures are frozen.</div></article>
        <article><time>Sep 9-23</time><div><h3>Define a real decision problem with a mentor</h3><p>Choose physically allowed irradiation blocks, cooling intervals, harvest times, facility constraints, and the time-defined Ac-227 objective. Do not let the optimizer invent impossible operations.</p></div><div class="done">Done when: variables, constraints, objectives, and trusted solver are written down.</div></article>
        <article><time>Sep 24-Oct 12</time><div><h3>Run a no-retrain ranking and Pareto test</h3><p>Generate a held-out schedule pool with the trusted reduced solver. Let V3 rank it. Measure Spearman/Kendall correlation, top-k recall, feasibility errors, Pareto hypervolume, and exact-solver regret.</p></div><div class="done">Done when: every V3 finalist has a trusted-solver answer.</div></article>
        <article><time>Oct 13-27</time><div><h3>Add reliability and sensitivity</h3><p>Build an input-envelope distance gate. Map V3 errors versus flux regime, yield, spectrum, and schedule. Run sensitivity to nuclear data, fluence, timing, and recovery without retraining.</p></div><div class="done">Done when: failure map and applicability domain are visible.</div></article>
        <article><time>Oct 28-Nov 15</time><div><h3>Independent review and claim freeze</h3><p>Ask an isotope-production or SCALE/ORIGEN expert to review the chain, two-group collapse, Ac-227 objective, constraints, and claim wording. Record exactly what changed.</p></div><div class="done">Done when: signed/dated notes and final claim boundary exist.</div></article>
        <article><time>Nov 16-Dec 7</time><div><h3>Write the student paper, poster, and interview defense</h3><p>Lead with the question, experimental benchmark, decision test, limitations, and exact-solver verification. Put the 1-3% synthetic median in context, not in place of the science.</p></div><div class="done">Done when: another person can reproduce the figures and you can explain every term.</div></article>
      </div>
    </section>

    <section id="sources">
      <div class="section-head"><div class="number">07 / Source Matrix</div><div><h2>Search and inspect all {len(rows)} compared sources.</h2><p>Category, method, data basis, evidence depth, and direct links are exposed so the comparison can be audited instead of trusted blindly.</p></div></div>
      <div class="controls"><input id="search" type="search" placeholder="Search title, author, DOI, method, or question" aria-label="Search sources"><select id="category" aria-label="Filter category"><option value="all">All categories</option><option value="A">A - Ac-225 production</option><option value="B">B - Ra-226 and neutron data</option><option value="C">C - Trusted solvers</option><option value="D">D - Nuclear ML</option><option value="E">E - Optimization</option><option value="F">F - Reliability</option></select><button type="button" onclick="window.print()">Print report</button></div>
      <div class="table-wrap"><table id="source-table"><thead><tr><th>ID</th><th>Area</th><th>Source</th><th>Year</th><th>Method</th><th>Data basis</th><th>Evidence depth</th><th>Relevance</th></tr></thead><tbody>{source_rows_html(rows)}</tbody></table></div>
      <p class="table-count" id="table-count">Showing {len(rows)} of {len(rows)} sources.</p>
      <div class="callout"><strong>Reproducibility:</strong> the full CSV matrix, broad candidate pool, search queries, and curation scripts are stored in <span class="mono">research/literature_audit_2026</span> and <span class="mono">research</span>. Search date: August 12, 2026.</div>
    </section>
  </main>
  <footer>Prepared as an internal research-planning audit for Sam Ogunnubi. It does not replace mentor review, radiation-safety expertise, or formal systematic-review methods.</footer>
  <script>
    const search = document.getElementById('search');
    const category = document.getElementById('category');
    const rows = [...document.querySelectorAll('#source-table tbody tr')];
    const count = document.getElementById('table-count');
    function filterRows() {{
      const q = search.value.trim().toLowerCase();
      const cat = category.value;
      let shown = 0;
      rows.forEach(row => {{
        const visible = (!q || row.dataset.text.includes(q)) && (cat === 'all' || row.dataset.category === cat);
        row.hidden = !visible;
        if (visible) shown += 1;
      }});
      count.textContent = `Showing ${{shown}} of ${{rows.length}} sources.`;
    }}
    search.addEventListener('input', filterRows);
    category.addEventListener('change', filterRows);
  </script>
</body>
</html>'''


def build_markdown(rows: list[dict[str, str]], summary: dict) -> str:
    q_sections = []
    for item in QUESTIONS:
        q_sections.append(
            f"## {item['id']}. {item['question']}\n\n{item['answer']}\n\n"
            f"**Project action:** {item['project_action']}\n\n**Confidence:** {item['confidence']}"
        )
    return "\n\n".join(
        [
            "# Ac-225 V3 Literature Audit Findings",
            f"Date: 2026-08-12  \nFinal comparison: {len(rows)} sources  \nBroad discovery pool: {summary['discovered_records']} records",
            "**Bottom line:** The strongest defensible contribution is not the use of an LSTM or a 1-3% synthetic median by itself. It is a frozen, failure-aware surrogate that identifies useful irradiation/cooling/harvest schedules for the Ra-226 fast-neutron pathway, verifies finalists with trusted physics, and is tested against the Berkeley experiments.",
            "\n\n".join(q_sections),
            "## Recommended Research Question\n\nCan a frozen physics-informed recurrent surrogate reliably identify and rank irradiation, cooling, and harvest schedules for the Ra-226(n,2n)Ra-225 to Ac-225 pathway that improve Ac-225 yield while controlling Ac-227 risk, with finalists verified by a trusted solver and evaluated against published Berkeley irradiation data?",
            f"## Review Boundary\n\nThis is a scoping review. The source matrix explicitly labels title/abstract screening, official-source review, and full-text close reading. It must not be described as {len(rows)} complete full-text peer reviews.",
        ]
    ) + "\n"


def main() -> None:
    rows = load_rows()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    OUTPUT.write_text(build_html(rows, summary), encoding="utf-8")
    MD_OUTPUT.write_text(build_markdown(rows, summary), encoding="utf-8")
    print(json.dumps({"html": str(OUTPUT), "markdown": str(MD_OUTPUT), "sources": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
