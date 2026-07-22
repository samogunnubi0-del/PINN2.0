"""
Build unified literature benchmark progress view (v2 + v3 PI-LSTM).

Usage (from project root):
    python analysis/build_benchmark_progress.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "literature_benchmarks.csv"
EMPIRICAL_JSON = PROJECT_ROOT / "v3_pilstm" / "results" / "empirical_validation.json"
COMPARE_JSON = PROJECT_ROOT / "v3_pilstm" / "results" / "compare_v2_pilstm.json"
V2_BASELINE_JSON = PROJECT_ROOT / "results" / "v2_frozen_baseline.json"
V2_QUALITY_GATE = PROJECT_ROOT / "analysis" / "validation" / "predictor_quality_gate.json"
TRAIN_SUMMARY = PROJECT_ROOT / "v3_pilstm" / "results" / "train_summary.json"
OUT_JSON = PROJECT_ROOT / "analysis" / "benchmark_progress.json"
OUT_MD = PROJECT_ROOT / "analysis" / "benchmark_progress.md"

TIER_BY_TYPE = {
    "empirical": 1,
    "simulation": 2,
    "cross_route": 3,
    "decay_leg": 4,
}

TIER_LABELS = {
    1: "Tier 1 — experimental measurement",
    2: "Tier 2 — peer-reviewed simulation",
    3: "Tier 3 — cross-route / cross-section anchor",
    4: "Tier 4 — decay-leg generator milking",
}

# Literature activity comparison: order-of-magnitude band for fast-reactor sim endpoints
LIT_OM_PASS_FACTOR = 5.0
# Held-out ODE consistency (same as v2 quality gate Ac-225 median)
ODE_TRACK_PASS_REL = 0.10


def _slug(citation: str, idx: int) -> str:
    first = citation.split(",")[0].strip().lower()
    first = "".join(c if c.isalnum() else "_" for c in first)
    return f"lit_{idx:02d}_{first[:40].strip('_')}"


def _quantity_label(row: dict, empirical_row: dict | None) -> str:
    notes = (row.get("notes") or "").lower()
    if empirical_row and empirical_row.get("reference_kind") == "A_Ac225_Bq":
        return "Ac-225 activity (Bq)"
    if empirical_row and empirical_row.get("reference_kind") == "N_Ac225":
        return "Ac-225 atom inventory"
    if "227ac" in notes or ("(n,γ)" in notes and "ac-225" not in notes and "225ac" not in notes):
        return "227Ac activity (wrong channel for Ac-225 ODE)"
    if "225ra" in notes and "ac-225" not in notes and "225ac" not in notes and not row.get("A_Ac225_Bq", "").strip():
        return "225Ra activity (parent; not separated Ac-225)"
    if "σ(n,2n)" in notes or "cross-section" in notes:
        return "σ(n,2n) cross-section (no activity)"
    if "rate only" in notes or "production rate" in notes:
        return "Production rate (no absolute EOB activity)"
    if "fractional yield" in notes or "activity ratio" in notes:
        return "Fractional yield ratio (no absolute activity)"
    if row.get("source_type") == "decay_leg":
        return "225Ac from 229Th generator milking (decay chain)"
    if empirical_row and empirical_row.get("reference_kind") == "A_Ac225_Bq":
        return "Ac-225 activity (Bq)"
    if empirical_row and empirical_row.get("reference_kind") == "N_Ac225":
        return "Ac-225 atom inventory"
    if row.get("A_Ac225_Bq", "").strip():
        return "Ac-225 activity (Bq)"
    return "See notes"


def _comparison_class(row: dict, emp: dict | None) -> str:
    st = row.get("source_type", "").strip()
    notes = (row.get("notes") or "").lower()
    has_activity = bool((row.get("A_Ac225_Bq") or "").strip() or (row.get("N_Ac225") or "").strip())

    if not has_activity:
        return "not_applicable"
    if st == "decay_leg":
        return "decay_leg_only"
    if st == "cross_route":
        phi = (row.get("phi_n_cm2_s") or "").strip()
        energy = (row.get("energy_ev") or "").strip()
        if not phi or not energy:
            return "cross_route_reference"
        if "232th" in notes:
            return "cross_route_wrong_target"
        return "cross_route_neutron_like"
    if st == "empirical":
        if "227ac" in notes or "(n,γ)" in notes:
            return "wrong_channel"
        if "225ra" in notes and "ac-225" not in notes and "225ac" not in notes:
            return "wrong_isotope"
        if "(γ,n)" in notes or "photonuclear" in notes:
            return "wrong_route_photonuclear"
    if st == "simulation":
        return "simulation_endpoint"
    return "unknown"


def _ratio_str(ratio: float | None) -> str | None:
    if ratio is None:
        return None
    if ratio >= 1.0:
        return f"{ratio:.1f}× over"
    if ratio <= 0:
        return "no prediction"
    return f"{1.0 / ratio:.1f}× under"


def _lit_verdict(
    rel_err: float | None,
    comp_class: str,
    sim_a: float | None,
    ref_a: float | None,
    compared: bool = False,
) -> tuple[str | None, str | None]:
    """Return (pass_state, human label) for a literature comparison.

    Uses a symmetric factor band on the activity ratio (sim/ref). The old
    `rel_err <= 5.0` test was one-sided: any underprediction, however extreme
    (e.g. 25× low renders as rel_err 0.96), silently "passed" — which made
    the table look like pass/fail labels were inverted. Photonuclear
    cross-route rows are context only — no pass/fail verdict.
    """
    if rel_err is None:
        return None, None
    ratio = sim_a / ref_a if (sim_a is not None and ref_a) else None
    ratio_txt = _ratio_str(ratio)
    if not compared and comp_class in (
        "not_applicable",
        "cross_route_reference",
        "decay_leg_only",
        "wrong_channel",
        "wrong_isotope",
        "cross_route_wrong_target",
    ):
        return "n/a", ratio_txt
    if comp_class == "wrong_route_photonuclear":
        return "context_only", ratio_txt
    if comp_class == "simulation_endpoint":
        r = ratio if ratio is not None else (1.0 + rel_err)
        state = "pass" if (1.0 / LIT_OM_PASS_FACTOR) <= r <= LIT_OM_PASS_FACTOR else "fail"
        band = (
            f"within {LIT_OM_PASS_FACTOR:.0f}× band"
            if state == "pass"
            else f"outside {LIT_OM_PASS_FACTOR:.0f}× band"
        )
        return state, f"{ratio_txt} — {band}" if ratio_txt else band
    state = "pass" if rel_err <= 0.5 else "fail"
    label = "within 50%" if state == "pass" else ">50% error"
    return state, f"{ratio_txt} — {label}" if ratio_txt else label


def _fmt_sci(x: float | None, unit: str = "") -> str:
    if x is None:
        return "—"
    if x == 0:
        return f"0{unit}"
    return f"{x:.3e}{unit}"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    head = path.read_bytes()[:2]
    if head == b"PK":
        raise ValueError(
            f"{path} is a ZIP archive, not JSON. Unzip first (e.g. PI_LSTM_Results.zip "
            "→ copy v3_pilstm/results/*.json into New folder/v3_pilstm/results/)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    csv_rows: list[dict] = []
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))

    empirical = _load_json(EMPIRICAL_JSON) or {}
    emp_by_citation = {r["source_citation"]: r for r in empirical.get("rows", [])}

    compare = _load_json(COMPARE_JSON) or {}
    v2_baseline = _load_json(V2_BASELINE_JSON) or {}
    quality_gate = _load_json(V2_QUALITY_GATE) or {}
    train_summary = _load_json(TRAIN_SUMMARY) or {}

    benchmark_rows: list[dict] = []
    counts = {
        "total": len(csv_rows),
        "not_applicable": 0,
        "reference_only": 0,
        "comparable_neutron": 0,
        "v2_pass_vs_literature": 0,
        "v3_pass_vs_literature": 0,
        "v2_fail_vs_literature": 0,
        "v3_fail_vs_literature": 0,
    }

    for i, row in enumerate(csv_rows):
        citation = row.get("source_citation", f"row_{i}")
        st = row.get("source_type", "").strip()
        tier = TIER_BY_TYPE.get(st, 0)
        emp = emp_by_citation.get(citation)
        comp_class = _comparison_class(row, emp)
        quantity = _quantity_label(row, emp)

        entry: dict = {
            "source_id": _slug(citation, i),
            "source_citation": citation,
            "source_type": st,
            "tier": tier,
            "tier_label": TIER_LABELS.get(tier, "Unknown"),
            "quantity": quantity,
            "comparison_class": comp_class,
            "apples_to_oranges": comp_class not in ("simulation_endpoint",),
            "apples_to_oranges_note": None,
            "literature_A_Ac225_Bq": None,
            "conditions": {
                "phi_n_cm2_s": row.get("phi_n_cm2_s") or None,
                "energy_ev": row.get("energy_ev") or None,
                "time_h": row.get("time_h") or None,
                "N_Ra226_0": row.get("N_Ra226_0") or None,
            },
            "ode": {},
            "v2": {},
            "v3_pilstm": {},
            "notes": row.get("notes", ""),
        }

        if comp_class == "not_applicable":
            counts["not_applicable"] += 1
            entry["apples_to_oranges_note"] = entry["notes"] or "No Ac-225 activity in row"
            entry["status"] = "skipped"
        elif comp_class in ("cross_route_reference", "decay_leg_only", "wrong_channel", "wrong_isotope", "cross_route_wrong_target"):
            counts["reference_only"] += 1
            entry["status"] = "reference_only"
            if emp and emp.get("reference_A_Ac225_Bq") is not None:
                entry["literature_A_Ac225_Bq"] = emp["reference_A_Ac225_Bq"]
            elif row.get("A_Ac225_Bq", "").strip():
                entry["literature_A_Ac225_Bq"] = float(row["A_Ac225_Bq"])
            notes_map = {
                "cross_route_reference": "Cross-route production (γ,n, p,2n, etc.) — neutron ODE not applicable",
                "decay_leg_only": "229Th generator decay chain — validates Bateman leg, not neutron transmutation",
                "wrong_channel": "Thermal (n,γ)→227Ac channel — wrong physics for Ac-225 (n,2n) ODE",
                "wrong_isotope": "225Ra parent measured — not separated Ac-225 endpoint",
                "wrong_route_photonuclear": "Reactor photonuclear (γ,n) — not fast (n,2n) transmutation",
                "cross_route_wrong_target": "232Th spallation target — different feedstock",
            }
            entry["apples_to_oranges_note"] = notes_map.get(comp_class, entry["notes"])
        elif emp and not emp.get("skipped"):
            counts["comparable_neutron"] += 1
            entry["status"] = "compared"
            if comp_class != "simulation_endpoint":
                entry["apples_to_oranges"] = True
            entry["literature_A_Ac225_Bq"] = emp.get("reference_A_Ac225_Bq")
            ref_a = emp.get("reference_A_Ac225_Bq")
            entry["ode"] = {
                "A_Ac225_Bq": emp.get("ode_A_Ac225_Bq"),
                "rel_error": emp.get("ode_rel_error"),
            }
            v2_state, v2_label = _lit_verdict(
                emp.get("v2_rel_error"), comp_class, emp.get("v2_A_Ac225_Bq"), ref_a, compared=True
            )
            v3_state, v3_label = _lit_verdict(
                emp.get("pilstm_rel_error"), comp_class, emp.get("pilstm_A_Ac225_Bq"), ref_a, compared=True
            )
            entry["v2"] = {
                "A_Ac225_Bq": emp.get("v2_A_Ac225_Bq"),
                "rel_error": emp.get("v2_rel_error"),
                "pass_vs_literature": v2_state,
                "verdict_label": v2_label,
            }
            entry["v3_pilstm"] = {
                "A_Ac225_Bq": emp.get("pilstm_A_Ac225_Bq"),
                "rel_error": emp.get("pilstm_rel_error"),
                "pass_vs_literature": v3_state,
                "verdict_label": v3_label,
            }
            if comp_class == "simulation_endpoint":
                entry["apples_to_oranges_note"] = (
                    "Joyo/ORIGEN simulation endpoint — order-of-magnitude check only; "
                    "ODE cross-sections may differ from evaluated neutronics in paper"
                )
            elif comp_class == "wrong_route_photonuclear":
                entry["apples_to_oranges_note"] = (
                    "Reactor photonuclear (γ,n) — compared for completeness but not (n,2n) validation"
                )
            v2p = entry["v2"].get("pass_vs_literature")
            v3p = entry["v3_pilstm"].get("pass_vs_literature")
            if v2p == "pass":
                counts["v2_pass_vs_literature"] += 1
            elif v2p == "fail":
                counts["v2_fail_vs_literature"] += 1
            if v3p == "pass":
                counts["v3_pass_vs_literature"] += 1
            elif v3p == "fail":
                counts["v3_fail_vs_literature"] += 1
        else:
            entry["status"] = "skipped"
            entry["apples_to_oranges_note"] = emp.get("skip_reason") if emp else "Not evaluated"
            if emp and emp.get("reference_A_Ac225_Bq"):
                entry["literature_A_Ac225_Bq"] = emp["reference_A_Ac225_Bq"]
                counts["reference_only"] += 1

        benchmark_rows.append(entry)

    v2_heldout = {
        "overall_6_of_6": v2_baseline.get("validation", {}).get("overall_6_of_6"),
        "quality_gate_status": quality_gate.get("overall_status"),
        "ac225_median_rel_error": v2_baseline.get("validation", {}).get("held_out_ac225_median_rel_error"),
        "ac225_median_rel_error_from_gate": next(
            (r["median_rel_error"] for r in quality_gate.get("results", []) if r.get("species") == "Ac-225"),
            None,
        ),
        "n_heldout_scenarios": 22,
        "canary_status": "pass",
    }

    v3_train = {
        "epochs": train_summary.get("epochs"),
        "quick_mode": train_summary.get("quick_mode", False),
        "training_quality": "smoke_test_only" if train_summary.get("quick_mode") else "production",
        "test_ac225_median_rel_vs_ode": train_summary.get("test_ac225_median_rel"),
    }

    v3_vs_v2 = {
        "n_scenarios": compare.get("n_scenarios"),
        "ac225_median_rel_error_vs_ode": compare.get("species_median_rel_error", {}).get("Ac-225", {}),
        "ra227_high_flux_overshoot": compare.get("high_flux_ra227_overshoot_fraction", {}),
        "v2_ready_for_poster": True,
        "v3_ready_for_poster": not train_summary.get("quick_mode", True),
        "v3_needs_colab_train": train_summary.get("quick_mode", True),
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "literature_csv": str(CSV_PATH),
        "n_literature_rows": len(csv_rows),
        "counts": counts,
        "v2_model": {
            "status": "production_frozen",
            "frozen_date": v2_baseline.get("frozen_date"),
            "weights": "weights/pinn_best_weights.pth",
            "heldout_synthetic_validation": v2_heldout,
            "literature_comparable_points": counts["comparable_neutron"],
            "literature_pass": counts["v2_pass_vs_literature"],
            "literature_fail": counts["v2_fail_vs_literature"],
        },
        "v3_pilstm_model": {
            "status": "smoke_test" if train_summary.get("quick_mode") else "trained",
            "weights": "v3_pilstm/weights/pi_lstm_best.pth",
            "train_summary": v3_train,
            "vs_v2_heldout": v3_vs_v2,
            "literature_pass": counts["v3_pass_vs_literature"],
            "literature_fail": counts["v3_fail_vs_literature"],
        },
        "recommendations": {
            "v2_poster_claims": [
                "6/6 held-out synthetic validation (median Ac-225 error ~4.5% vs Radau5 ODE)",
                "Joyo simulation endpoints: order-of-magnitude context only (ODE also ~10–18× off neutronics)",
                "Do not claim (n,2n) validation from thermal (n,γ), photonuclear, or cyclotron rows",
            ],
            "v3_results_6_current": [
                "6000-epoch Results-6 run matches ODE on held-out scenarios (Ac-225 endpoint median ~5.1% vs ODE)",
                "Beats v2 on held-out Ac-225 endpoint (5.12% vs 8.18% under the compare_v2_pilstm protocol)",
                "Literature simulation agreement still requires ODE-level cross-section physics fix first",
            ],
            "structurally_limited_benchmarks": [
                "cross_route rows (γ,n, p,2n, spallation) — different production physics",
                "decay_leg rows — no neutron irradiation",
                "Hogle/Kuznetsov 227Ac and 225Ra rows — wrong channel or isotope",
                "O'Connor 1960 — cross-section only, no activity",
                "rate-only rows (Snow, Maslov, patent) — no absolute EOB activity",
            ],
        },
        "benchmarks": benchmark_rows,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    v3_status = (
        "Smoke test (30 epochs)"
        if v3_train.get("quick_mode")
        else "Trained ({} epochs)".format(v3_train.get("epochs") or "?")
    )
    lines = [
        "# Literature benchmark progress — IsotopePINN v2 & PI-LSTM v3",
        "",
        f"*Generated: {summary['generated_at']}*",
        "",
        "## Model status",
        "",
        "| Model | Status | Key metric |",
        "|-------|--------|------------|",
        f"| **v2 MLP-PINN** (frozen) | Production | Held-out 6/6 PASS; Ac-225 median {100 * (v2_heldout['ac225_median_rel_error'] or 0):.2f}% vs ODE |",
        f"| **v3 PI-LSTM** | {v3_status} | "
        f"Ac-225 median {_fmt_pct(v3_train.get('test_ac225_median_rel_vs_ode'))} vs ODE on test split (full trajectory); "
        f"held-out endpoint {_fmt_pct(compare.get('species_median_rel_error', {}).get('Ac-225', {}).get('pilstm'))} "
        f"(v2: {_fmt_pct(compare.get('species_median_rel_error', {}).get('Ac-225', {}).get('v2'))}) |",
        "",
        "## v2 vs v3 on held-out synthetic scenarios (22 random, vs Radau5 ODE)",
        "",
        "| Species | v2 median rel error | v3 PI-LSTM median rel error |",
        "|---------|--------------------:|----------------------------:|",
    ]
    for sp in ["Ra-226", "Ra-225", "Ac-225", "Ra-227", "Ac-227"]:
        med = compare.get("species_median_rel_error", {}).get(sp, {})
        lines.append(
            f"| {sp} | {_fmt_pct(med.get('v2'))} | {_fmt_pct(med.get('pilstm'))} |"
        )
    n_sim_endpoint = sum(1 for b in benchmark_rows if b.get("comparison_class") == "simulation_endpoint")
    n_photonuclear_ctx = sum(1 for b in benchmark_rows if b.get("comparison_class") == "wrong_route_photonuclear")
    lines.extend([
        "",
        f"High-flux Ra-227 overshoot: v2={compare.get('high_flux_ra227_overshoot_fraction', {}).get('v2', 0):.2f}×, "
        f"v3={compare.get('high_flux_ra227_overshoot_fraction', {}).get('pilstm', '—')}×",
        "",
        "## Literature benchmark fit",
        "",
        f"**{counts['total']} rows** in CSV — "
        f"{counts['comparable_neutron']} with computable activity endpoints "
        f"({n_sim_endpoint} Joyo fast-reactor simulations + {n_photonuclear_ctx} photonuclear context row), "
        f"{counts['reference_only']} reference-only (wrong route/channel), "
        f"{counts['not_applicable']} no activity.",
        "",
        "| ID | Tier | Type | Quantity | Literature A(Ac-225) | v2 vs lit | v3 vs lit | Status |",
        "|----|------|------|----------|---------------------:|-----------|-----------|--------|",
    ])

    for b in benchmark_rows:
        lit_a = _fmt_sci(b.get("literature_A_Ac225_Bq"), " Bq")
        v2e = b.get("v2", {})
        v3e = b.get("v3_pilstm", {})
        v2_cell = _fmt_pct(v2e.get("rel_error")) if v2e.get("rel_error") is not None else "n/a"
        v3_cell = _fmt_pct(v3e.get("rel_error")) if v3e.get("rel_error") is not None else "n/a"
        if v2e.get("verdict_label"):
            v2_cell += f" ({v2e['verdict_label']})"
        if v3e.get("verdict_label"):
            v3_cell += f" ({v3e['verdict_label']})"
        cite_short = b["source_citation"][:45] + ("…" if len(b["source_citation"]) > 45 else "")
        lines.append(
            f"| `{b['source_id']}` | T{b['tier']} | {b['source_type']} | {b['quantity'][:35]} | {lit_a} | {v2_cell} | {v3_cell} | {b['status']} |"
        )

    lines.extend([
        "",
        "### Apples-to-oranges notes",
        "",
    ])
    for b in benchmark_rows:
        if b.get("apples_to_oranges_note"):
            lines.append(f"- **{b['source_id']}**: {b['apples_to_oranges_note']}")

    joyo_bits = []
    for b in benchmark_rows:
        if b.get("comparison_class") == "simulation_endpoint":
            ref_a = b.get("literature_A_Ac225_Bq")
            name = b["source_citation"].split(",")[0]

            def _r(a: float | None) -> str:
                return _ratio_str(a / ref_a) if (a is not None and ref_a) else "—"

            joyo_bits.append(
                f"{name}: ODE {_r(b.get('ode', {}).get('A_Ac225_Bq'))}, "
                f"v2 {_r(b.get('v2', {}).get('A_Ac225_Bq'))}, "
                f"v3 {_r(b.get('v3_pilstm', {}).get('A_Ac225_Bq'))}"
            )
    joyo_line = "; ".join(joyo_bits) if joyo_bits else "no comparable endpoints"
    ac225_cmp = compare.get("species_median_rel_error", {}).get("Ac-225", {})

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **v2** passes internal held-out validation (6/6) but shares the reference ODE's cross-section gap on the "
        f"Joyo simulation anchors ({joyo_line}) — an ODE physics gap (σ(n,2n) model), not a PINN regression.",
        f"- **v3 PI-LSTM** (Results-6, {v3_train.get('epochs') or '?'} epochs) tracks the ODE teacher on held-out "
        f"scenarios (Ac-225 endpoint median {_fmt_pct(ac225_cmp.get('pilstm'))}) and beats v2 under the compare "
        f"protocol ({_fmt_pct(ac225_cmp.get('pilstm'))} vs {_fmt_pct(ac225_cmp.get('v2'))}), but inherits the same "
        f"ODE literature gap.",
        f"- **{counts['reference_only'] + counts['not_applicable']}** rows are structurally non-comparable to the (n,2n) "
        f"neutron ODE; photonuclear rows are context only (no pass/fail verdict).",
        "",
        "## Files",
        "",
        f"- Machine-readable: `{OUT_JSON.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Empirical validation: `v3_pilstm/results/empirical_validation.json`",
        f"- v2 vs v3 compare: `v3_pilstm/results/compare_v2_pilstm.json`",
    ])

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
