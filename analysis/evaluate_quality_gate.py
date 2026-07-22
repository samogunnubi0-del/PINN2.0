"""
Evaluate whether the latest trained model meets predictor-grade thresholds.

Run after:
  python analysis/validate_predictor.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
VALIDATION_SUMMARY = ROOT / "analysis" / "validation" / "heldout_validation_summary.csv"
OUT_JSON = ROOT / "analysis" / "validation" / "predictor_quality_gate.json"

THRESHOLDS = {
    "Ac-225": {"median_rel_error": 0.10, "p95_rel_error": 1.0},
    "Ac-227": {"median_rel_error": 0.20, "p95_rel_error": 1.0},
    "Ra-225": {"median_rel_error": 0.20, "p95_rel_error": 2.0},
    "Ra-227": {"median_rel_error": 0.30, "p95_rel_error": 2.0},
}
IMPURITY_THRESHOLDS = {
    "decision_accuracy": 0.90,
    "false_usable_rate": 0.05,
    "p95_abs_impurity_error": 0.0015,
}


def main() -> None:
    if not VALIDATION_SUMMARY.is_file():
        print(f"Missing {VALIDATION_SUMMARY}. Run analysis/validate_predictor.py first.")
        sys.exit(1)

    df = pd.read_csv(VALIDATION_SUMMARY)
    overall = df[df["regime"] == "all"].copy()

    results: list[dict] = []
    all_pass = True

    for species, limits in THRESHOLDS.items():
        row = overall[overall["species"] == species]
        if row.empty:
            print(f"WARN: no aggregate row for {species}")
            all_pass = False
            continue
        r = row.iloc[0]
        med = float(r["median_rel_error"])
        p95 = float(r["p95_rel_error"])
        med_pass = med <= limits["median_rel_error"]
        p95_pass = p95 <= limits["p95_rel_error"]
        status = "pass" if med_pass and p95_pass else "fail"
        if status == "fail":
            all_pass = False
        results.append(
            {
                "species": species,
                "median_rel_error": med,
                "p95_rel_error": p95,
                "median_rel_error_limit": limits["median_rel_error"],
                "median_rel_error_pass": med_pass,
                "p95_rel_error_limit": limits["p95_rel_error"],
                "p95_rel_error_pass": p95_pass,
                "status": status,
            }
        )
        print(f"{species}: median={med:.4f} (limit {limits['median_rel_error']}) "
              f"p95={p95:.4f} -> {status.upper()}")

    imp_row = overall[overall["species"] == "Ac-227 activity impurity"]
    if not imp_row.empty:
        r = imp_row.iloc[0]
        da = float(r["decision_accuracy"])
        fur = float(r["false_usable_rate"])
        p95_imp = float(r["p95_rel_error"])
        imp_pass = (
            da >= IMPURITY_THRESHOLDS["decision_accuracy"]
            and fur <= IMPURITY_THRESHOLDS["false_usable_rate"]
            and p95_imp <= IMPURITY_THRESHOLDS["p95_abs_impurity_error"]
        )
        if not imp_pass:
            all_pass = False
        results.append(
            {
                "species": "Ac-227 activity impurity",
                "decision_accuracy": da,
                "decision_accuracy_limit": IMPURITY_THRESHOLDS["decision_accuracy"],
                "false_usable_rate": fur,
                "false_usable_rate_limit": IMPURITY_THRESHOLDS["false_usable_rate"],
                "p95_abs_impurity_error": p95_imp,
                "p95_abs_impurity_error_limit": IMPURITY_THRESHOLDS["p95_abs_impurity_error"],
                "status": "pass" if imp_pass else "fail",
            }
        )
        print(f"Ac-227 impurity decision: acc={da:.3f} false_usable={fur:.3f} -> "
              f"{'PASS' if imp_pass else 'FAIL'}")

    payload = {
        "overall_status": "pass" if all_pass else "fail",
        "thresholds": THRESHOLDS,
        "impurity_decision_thresholds": IMPURITY_THRESHOLDS,
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    print(f"Wrote {OUT_JSON}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
