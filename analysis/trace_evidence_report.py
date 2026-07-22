"""
Build judge-facing evidence artifacts for trace daughter prediction quality.

Run after:
    python analysis/validate_predictor.py
    python analysis/evaluate_quality_gate.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "analysis" / "validation"
SUMMARY_PATH = VALIDATION_DIR / "heldout_validation_summary.csv"
DETAIL_PATH = VALIDATION_DIR / "heldout_validation_details.csv"
CANARY_PATH = VALIDATION_DIR / "heldout_canary_report.csv"
BASELINE_SUMMARY_PATH = VALIDATION_DIR / "baseline_heldout_validation_summary.csv"
OUT_MD = VALIDATION_DIR / "trace_predictor_evidence.md"
OUT_ERROR_PLOT = VALIDATION_DIR / "trace_error_by_regime_case.png"
OUT_IMPURITY_CSV = VALIDATION_DIR / "trace_impurity_decision_confusion.csv"

TRACE_SPECIES = ["Ac-225", "Ac-227", "Ra-225", "Ra-227"]


def _require_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SUMMARY_PATH.exists() or not DETAIL_PATH.exists():
        raise FileNotFoundError(
            "Missing held-out validation outputs. Run analysis/validate_predictor.py first."
        )
    return pd.read_csv(SUMMARY_PATH), pd.read_csv(DETAIL_PATH)


def _plot_trace_errors(summary: pd.DataFrame) -> None:
    rows = summary[
        (summary["species"].isin(TRACE_SPECIES))
        & (summary["regime"] != "all")
        & (summary["case_type"] != "all")
    ].copy()
    if rows.empty:
        return
    rows["label"] = rows["regime"] + "\n" + rows["case_type"]
    labels = list(dict.fromkeys(rows["label"].tolist()))
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, species in enumerate(TRACE_SPECIES):
        vals = []
        for label in labels:
            g = rows[(rows["label"] == label) & (rows["species"] == species)]
            vals.append(float(g["median_rel_error"].iloc[0]) if not g.empty else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=species)

    ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.5, label="100% relative error")
    ax.set_yscale("log")
    ax.set_ylabel("Median relative error (log scale)")
    ax.set_title("Trace Daughter Prediction Error by Regime and Case Type")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_ERROR_PLOT, dpi=160)
    plt.close(fig)
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        import graph_provenance
        graph_provenance.record_graph_write(
            ROOT,
            OUT_ERROR_PLOT.resolve(),
            producer="trace_evidence_report.py",
            run_id=graph_provenance.new_run_id(),
        )
    except Exception:
        pass


def _write_impurity_confusion(details: pd.DataFrame) -> pd.DataFrame:
    imp = details[details["species"] == "Ac-227 activity impurity"].copy()
    if imp.empty:
        out = pd.DataFrame()
        out.to_csv(OUT_IMPURITY_CSV, index=False)
        return out
    rows = []
    for (regime, case_type), g in imp.groupby(["regime", "case_type"]):
        truth = g["truth_usable"].astype(bool)
        pred = g["prediction_usable"].astype(bool)
        rows.append({
            "regime": regime,
            "case_type": case_type,
            "n": int(len(g)),
            "true_usable_pred_usable": int((truth & pred).sum()),
            "true_usable_pred_unusable": int((truth & ~pred).sum()),
            "true_unusable_pred_usable": int((~truth & pred).sum()),
            "true_unusable_pred_unusable": int((~truth & ~pred).sum()),
            "decision_accuracy": float((truth == pred).mean()),
            "false_usable_rate": float((pred & ~truth).mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_IMPURITY_CSV, index=False)
    return out


def _baseline_comparison(summary: pd.DataFrame) -> list[str]:
    if not BASELINE_SUMMARY_PATH.exists():
        return [
            "No baseline comparison was generated because "
            "`baseline_heldout_validation_summary.csv` was not found.",
            "To create one, save the pre-upgrade summary under that filename, retrain, "
            "then rerun this report.",
        ]
    baseline = pd.read_csv(BASELINE_SUMMARY_PATH)
    if "case_type" not in baseline.columns:
        baseline["case_type"] = "all"
    if "regime" not in baseline.columns:
        baseline["regime"] = "all"
    current = summary[(summary["regime"] == "all") & (summary["case_type"] == "all")]
    base = baseline[(baseline["regime"] == "all") & (baseline["case_type"] == "all")]
    lines = ["Before/after median relative error:"]
    for species in TRACE_SPECIES:
        b = base[base["species"] == species]
        c = current[current["species"] == species]
        if b.empty or c.empty:
            continue
        before = float(b["median_rel_error"].iloc[0])
        after = float(c["median_rel_error"].iloc[0])
        lines.append(f"- {species}: {before:.3e} -> {after:.3e}")
    return lines


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    summary, details = _require_inputs()
    _plot_trace_errors(summary)
    confusion = _write_impurity_confusion(details)

    aggregate = summary[
        (summary["regime"] == "all")
        & (summary["case_type"] == "all")
        & (summary["species"].isin(TRACE_SPECIES))
    ]
    lines = [
        "# Trace Daughter Predictor Evidence",
        "",
        "This report focuses on the project claim: predicting trace daughter products, "
        "especially Ac-225 and Ac-227, in stiff radioactive decay chains.",
        "",
        "## Aggregate Held-Out Trace Errors",
        "",
    ]
    if aggregate.empty:
        lines.append("No aggregate trace rows found in the validation summary.")
    else:
        for row in aggregate.itertuples(index=False):
            lines.append(
                f"- {row.species}: median={row.median_rel_error:.3e}, "
                f"p95={row.p95_rel_error:.3e}, n={row.n}"
            )
    lines.extend(["", "## Impurity Decision Evidence", ""])
    if confusion.empty:
        lines.append("No impurity decision rows found.")
    else:
        lines.append(f"Decision confusion table saved to `{OUT_IMPURITY_CSV.name}`.")
        lines.append(f"Mean decision accuracy: {confusion['decision_accuracy'].mean():.3f}")
        lines.append(f"Worst false-usable rate: {confusion['false_usable_rate'].max():.3f}")
    lines.extend(["", "## Before/After Comparison", ""])
    lines.extend(_baseline_comparison(summary))
    lines.extend(["", "## ISEF-Critical Canaries", ""])
    if CANARY_PATH.exists():
        canary = pd.read_csv(CANARY_PATH)
        for row in canary.itertuples(index=False):
            status = getattr(row, "status", "unknown")
            pieces = [f"- {row.canary}: {status}, n={row.n}"]
            if hasattr(row, "median_rel_error") and pd.notna(getattr(row, "median_rel_error")):
                pieces.append(f"median_rel={getattr(row, 'median_rel_error'):.3e}")
            if hasattr(row, "zero_prediction_rate") and pd.notna(getattr(row, "zero_prediction_rate")):
                pieces.append(f"zero_pred_rate={getattr(row, 'zero_prediction_rate'):.3f}")
            if hasattr(row, "decision_accuracy") and pd.notna(getattr(row, "decision_accuracy")):
                pieces.append(f"decision_acc={getattr(row, 'decision_accuracy'):.3f}")
            lines.append(", ".join(pieces))
    else:
        lines.append("No canary report found. Run `analysis/validate_predictor.py` first.")
    lines.extend(["", f"Trace error plot saved to `{OUT_ERROR_PLOT.name}`."])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved evidence report: {OUT_MD}")
    print(f"Saved trace error plot: {OUT_ERROR_PLOT}")
    print(f"Saved impurity confusion table: {OUT_IMPURITY_CSV}")


if __name__ == "__main__":
    main()
