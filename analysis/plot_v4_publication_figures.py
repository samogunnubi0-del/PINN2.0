"""Create publication-style figures from the frozen V4 GP/LSTM result bundle.

The script reads the archived CSV and JSON artifacts directly from the ZIP. It
does not retrain, tune, or modify either model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


GP_COLOR = "#0072B2"
LSTM_COLOR = "#D55E00"
TEAL = "#009E73"
GOLD = "#E69F00"
RED = "#CC3311"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
INK = "#17242C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-zip",
        type=Path,
        default=Path(r"C:\Users\ogunn\Downloads\V4_GP_LSTM_FINAL_c3a56472508c.zip"),
        help="Frozen V4 result ZIP.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "v4_publication_figures",
        help="Directory for PNG, PDF, and provenance outputs.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ResultBundle:
    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"V4 result ZIP not found: {self.path}")
        self.archive = zipfile.ZipFile(self.path)

    def _member(self, suffix: str) -> str:
        matches = [name for name in self.archive.namelist() if name.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one ZIP member ending in {suffix!r}; found {matches}")
        return matches[0]

    def csv(self, suffix: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.archive.read(self._member(suffix))))

    def json(self, suffix: str) -> dict:
        return json.loads(self.archive.read(self._member(suffix)).decode("utf-8"))

    def close(self) -> None:
        self.archive.close()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 7.2,
            "figure.dpi": 120,
            "savefig.dpi": 360,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=LIGHT_GRAY, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=INK,
        va="top",
    )


def panel_label_inside(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.015,
        0.985,
        letter,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=INK,
        va="top",
        bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def add_parity_panel(
    ax: plt.Axes,
    gp: pd.DataFrame,
    lstm: pd.DataFrame,
    truth_col: str,
    pred_col: str,
    reaction_label: str,
    show_legend: bool,
) -> None:
    truth = gp[truth_col].to_numpy()
    all_values = np.concatenate([truth, gp[pred_col].to_numpy(), lstm[pred_col].to_numpy()])
    lo = all_values.min() * 0.94
    hi = all_values.max() * 1.06
    line = np.geomspace(lo, hi, 300)

    ax.fill_between(line, line * 0.95, line * 1.05, color="#D1D5DB", alpha=0.45, label="+/-5% band")
    ax.plot(line, line, color="#374151", linewidth=1.0, linestyle="--", label="1:1")
    ax.scatter(
        truth,
        gp[pred_col],
        s=25,
        facecolors="none",
        edgecolors=GP_COLOR,
        linewidths=0.9,
        alpha=0.9,
        label="GP",
    )
    ax.scatter(
        truth,
        lstm[pred_col],
        s=22,
        marker="x",
        color=LSTM_COLOR,
        linewidths=0.9,
        alpha=0.85,
        label="LSTM ensemble",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Reference rate (h$^{-1}$ $\mu$A$^{-1}$)")
    ax.set_ylabel(r"Predicted rate (h$^{-1}$ $\mu$A$^{-1}$)")
    ax.set_title(reaction_label)
    clean_axis(ax)
    if show_legend:
        ax.legend(frameon=False, loc="upper left", handletextpad=0.5)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    fraction = np.arange(1, ordered.size + 1) / ordered.size
    return ordered, fraction


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=360)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def figure_model_comparison(
    gp_test: pd.DataFrame,
    lstm_test: pd.DataFrame,
    paired: pd.DataFrame,
    comparison: dict,
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.2), constrained_layout=True)

    add_parity_panel(
        axes[0, 0],
        gp_test,
        lstm_test,
        "truth_n2n_per_h_per_uA",
        "prediction_n2n_per_h_per_uA",
        r"Ra-226(n,2n) rate",
        True,
    )
    panel_label(axes[0, 0], "a")

    add_parity_panel(
        axes[0, 1],
        gp_test,
        lstm_test,
        "truth_ngamma_per_h_per_uA",
        "prediction_ngamma_per_h_per_uA",
        r"Ra-226(n,$\gamma$) rate",
        False,
    )
    panel_label(axes[0, 1], "b")

    ax = axes[1, 0]
    gp_error = paired["gp_case_mean_relative_error"].to_numpy() * 100
    lstm_error = paired["lstm_case_mean_relative_error"].to_numpy() * 100
    for values, color, label in [
        (gp_error, GP_COLOR, "GP"),
        (lstm_error, LSTM_COLOR, "LSTM ensemble"),
    ]:
        x, y = ecdf(values)
        ax.step(x, y, where="post", color=color, linewidth=1.6, label=label)
    ax.set_xlabel("Case-mean relative error (%)")
    ax.set_ylabel("Fraction of locked cases")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_title("Locked-case error distribution")
    clean_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    metrics = (
        f"Median: GP {comparison['gp']['median_relative_error'] * 100:.2f}%, "
        f"LSTM {comparison['lstm']['median_relative_error'] * 100:.2f}%\n"
        f"p95: GP {comparison['gp']['p95_relative_error'] * 100:.2f}%, "
        f"LSTM {comparison['lstm']['p95_relative_error'] * 100:.2f}%"
    )
    ax.text(
        0.03,
        0.96,
        metrics,
        transform=ax.transAxes,
        va="top",
        fontsize=6.9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    panel_label(ax, "c")

    ax = axes[1, 1]
    delta = np.sort(paired["lstm_minus_gp_case_error"].to_numpy() * 100)
    x = np.arange(1, delta.size + 1)
    colors_by_sign = np.where(delta > 0, GP_COLOR, LSTM_COLOR)
    ax.axhline(0, color="#374151", linewidth=1.0)
    ax.scatter(x, delta, c=colors_by_sign, s=22, alpha=0.85, edgecolors="none")
    ax.set_xlabel("Locked families, sorted by paired difference")
    ax.set_ylabel("LSTM - GP case error (percentage points)")
    ax.set_title("Paired model difference")
    clean_axis(ax)
    bootstrap = comparison["paired_bootstrap"]
    ci_low, ci_high = [value * 100 for value in bootstrap["confidence_interval_95"]]
    observed = bootstrap["observed_difference"] * 100
    note = f"p95 difference = {observed:+.2f} pp\npaired-bootstrap 95% CI [{ci_low:+.2f}, {ci_high:+.2f}] pp"
    ax.text(
        0.03,
        0.96,
        note,
        transform=ax.transAxes,
        va="top",
        fontsize=6.9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LSTM_COLOR, label="LSTM lower error"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GP_COLOR, label="GP lower error"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    panel_label(ax, "d")

    fig.suptitle("Frozen GP/LSTM comparison on 48 modeled-proxy families", fontsize=10.5, fontweight="semibold")
    fig.text(
        0.5,
        -0.015,
        "Modeled finite-geometry/TENDL proxy evidence; this is not experimental or reactor validation.",
        ha="center",
        fontsize=6.8,
        color=GRAY,
    )
    return save_figure(fig, output_dir, "figure_1_locked_model_comparison")


def mean_log_std(frame: pd.DataFrame) -> np.ndarray:
    return frame[["log_prediction_std_n2n", "log_prediction_std_ngamma"]].mean(axis=1).to_numpy()


def figure_reliability(
    gp_test: pd.DataFrame,
    lstm_test: pd.DataFrame,
    gp_domain: pd.DataFrame,
    lstm_domain: pd.DataFrame,
    gp_result: dict,
    lstm_result: dict,
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.0), constrained_layout=True)

    ax = axes[0, 0]
    categories = [r"Raw $\pm$2 SD", "Calibrated 95%"]
    gp_values = [
        gp_result["uncertainty"]["raw_two_standard_deviations"]["joint_case_coverage"] * 100,
        gp_result["uncertainty"]["calibrated_95"]["joint_case_coverage"] * 100,
    ]
    lstm_values = [
        lstm_result["uncertainty"]["raw_two_standard_deviations"]["joint_case_coverage"] * 100,
        lstm_result["uncertainty"]["calibrated_95"]["joint_case_coverage"] * 100,
    ]
    positions = np.arange(len(categories))
    width = 0.34
    bars_gp = ax.bar(positions - width / 2, gp_values, width, color=GP_COLOR, label="GP")
    bars_lstm = ax.bar(positions + width / 2, lstm_values, width, color=LSTM_COLOR, label="LSTM ensemble")
    ax.axhline(95, color="#374151", linewidth=1.0, linestyle="--", label="95% target")
    ax.set_xticks(positions, categories)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Joint locked-case coverage (%)")
    ax.set_title("Uncertainty interval coverage")
    clean_axis(ax, "y")
    ax.legend(frameon=False, loc="upper left")
    for bars in [bars_gp, bars_lstm]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.1f}", ha="center", fontsize=6.8)
    panel_label(ax, "a")

    ax = axes[0, 1]
    group_specs = [
        ("Locked test", None),
        ("Sparse corners", "in_range_sparse_corner"),
        ("Outside range", "explicit_out_of_range"),
    ]
    gp_groups = [
        mean_log_std(gp_test),
        mean_log_std(gp_domain[gp_domain["challenge_kind"] == group_specs[1][1]]),
        mean_log_std(gp_domain[gp_domain["challenge_kind"] == group_specs[2][1]]),
    ]
    lstm_groups = [
        mean_log_std(lstm_test),
        mean_log_std(lstm_domain[lstm_domain["challenge_kind"] == group_specs[1][1]]),
        mean_log_std(lstm_domain[lstm_domain["challenge_kind"] == group_specs[2][1]]),
    ]
    centers = np.arange(1, 4)
    gp_box = ax.boxplot(
        gp_groups,
        positions=centers - 0.18,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "white", "linewidth": 1.2},
    )
    lstm_box = ax.boxplot(
        lstm_groups,
        positions=centers + 0.18,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "white", "linewidth": 1.2},
    )
    for patch in gp_box["boxes"]:
        patch.set_facecolor(GP_COLOR)
        patch.set_alpha(0.8)
    for patch in lstm_box["boxes"]:
        patch.set_facecolor(LSTM_COLOR)
        patch.set_alpha(0.8)
    ax.set_xticks(centers, [name for name, _ in group_specs])
    ax.set_ylabel("Mean predicted log-rate SD")
    ax.set_title("Does uncertainty rise away from training data?")
    clean_axis(ax, "y")
    ax.legend(
        handles=[
            Line2D([0], [0], color=GP_COLOR, linewidth=7, label="GP"),
            Line2D([0], [0], color=LSTM_COLOR, linewidth=7, label="LSTM ensemble"),
        ],
        frameon=False,
        loc="upper left",
    )
    panel_label(ax, "b")

    ax = axes[1, 0]
    warn_rates = [
        gp_result["domain_challenges"]["locked_test_false_rejection_rate"] * 100,
        gp_result["domain_challenges"]["sparse_corner_warning_rate"] * 100,
        gp_result["domain_challenges"]["explicit_ood_rejection_rate"] * 100,
    ]
    bars = ax.bar(
        ["Locked test", "Sparse corners", "Outside range"],
        warn_rates,
        color=[TEAL, GOLD, RED],
        width=0.62,
    )
    ax.set_ylim(0, 108)
    ax.set_ylabel("Rejected or warned (%)")
    ax.set_title("Shared domain-guard decisions")
    clean_axis(ax, "y")
    for bar, value in zip(bars, warn_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=7)
    panel_label(ax, "c")

    ax = axes[1, 1]
    for frame, color, label, marker in [
        (gp_test, GP_COLOR, "GP", "o"),
        (lstm_test, LSTM_COLOR, "LSTM ensemble", "x"),
    ]:
        x = frame["nearest_training_distance"].to_numpy()
        y = frame["case_mean_relative_error"].to_numpy() * 100
        ax.scatter(x, y, s=22, color=color, alpha=0.75, marker=marker, label=label)
    rho_gp = spearmanr(gp_test["nearest_training_distance"], gp_test["case_mean_relative_error"]).statistic
    rho_lstm = spearmanr(lstm_test["nearest_training_distance"], lstm_test["case_mean_relative_error"]).statistic
    ax.set_xlabel("Nearest-training distance (standardized feature space)")
    ax.set_ylabel("Case-mean relative error (%)")
    ax.set_title("Locked error versus domain distance")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.97,
        0.96,
        f"Spearman rho\nGP {rho_gp:+.2f}\nLSTM {rho_lstm:+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    panel_label(ax, "d")

    fig.suptitle("Uncertainty calibration and applicability-domain behavior", fontsize=10.5, fontweight="semibold")
    fig.text(
        0.5,
        -0.015,
        "Whiskers show the 5th-95th percentiles. Outside-range predictions are rejected, so their accuracy is not claimed.",
        ha="center",
        fontsize=6.8,
        color=GRAY,
    )
    return save_figure(fig, output_dir, "figure_2_uncertainty_and_domain")


def figure_ac225_inventory(
    gp_schedule: pd.DataFrame,
    lstm_schedule: pd.DataFrame,
    gp_result: dict,
    lstm_result: dict,
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.3), constrained_layout=True)

    ax = axes[0]
    truth = gp_schedule["truth_ac225_norm"].to_numpy()
    values = np.concatenate(
        [truth, gp_schedule["predicted_ac225_norm"].to_numpy(), lstm_schedule["predicted_ac225_norm"].to_numpy()]
    )
    lo = values.min() * 0.94
    hi = values.max() * 1.06
    line = np.geomspace(lo, hi, 300)
    ax.fill_between(line, line * 0.95, line * 1.05, color="#D1D5DB", alpha=0.45)
    ax.plot(line, line, color="#374151", linewidth=1.0, linestyle="--")
    ax.scatter(
        truth,
        gp_schedule["predicted_ac225_norm"],
        s=20,
        facecolors="none",
        edgecolors=GP_COLOR,
        linewidths=0.8,
        alpha=0.8,
        label="GP",
    )
    ax.scatter(
        truth,
        lstm_schedule["predicted_ac225_norm"],
        s=18,
        marker="x",
        color=LSTM_COLOR,
        linewidths=0.8,
        alpha=0.75,
        label="LSTM ensemble",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference normalized Ac-225 inventory")
    ax.set_ylabel("Predicted normalized Ac-225 inventory")
    ax.set_title("Inventory propagated with exact five-nuclide equations")
    clean_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    panel_label_inside(ax, "a")

    ax = axes[1]
    schedule_order = ["short", "medium", "long"]
    centers = np.arange(1, 4)
    gp_groups = [
        gp_schedule.loc[gp_schedule["schedule"] == schedule, "relative_error"].to_numpy() * 100
        for schedule in schedule_order
    ]
    lstm_groups = [
        lstm_schedule.loc[lstm_schedule["schedule"] == schedule, "relative_error"].to_numpy() * 100
        for schedule in schedule_order
    ]
    gp_box = ax.boxplot(
        gp_groups,
        positions=centers - 0.18,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "white", "linewidth": 1.2},
    )
    lstm_box = ax.boxplot(
        lstm_groups,
        positions=centers + 0.18,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        medianprops={"color": "white", "linewidth": 1.2},
    )
    for patch in gp_box["boxes"]:
        patch.set_facecolor(GP_COLOR)
        patch.set_alpha(0.8)
    for patch in lstm_box["boxes"]:
        patch.set_facecolor(LSTM_COLOR)
        patch.set_alpha(0.8)
    labels = ["24 h / 0 h", "120 h / 24 h", "240 h / 72 h"]
    ax.set_xticks(centers, labels)
    ax.set_xlabel("Irradiation / cooling schedule")
    ax.set_ylabel("Ac-225 relative error (%)")
    ax.set_title("Schedule-specific inventory error")
    clean_axis(ax, "y")
    ax.legend(
        handles=[
            Line2D([0], [0], color=GP_COLOR, linewidth=7, label="GP"),
            Line2D([0], [0], color=LSTM_COLOR, linewidth=7, label="LSTM ensemble"),
        ],
        frameon=False,
        loc="upper right",
    )
    summary = (
        f"Grid p95: GP {gp_result['ac225_inventory']['grid_p95_relative_error'] * 100:.2f}%\n"
        f"LSTM {lstm_result['ac225_inventory']['grid_p95_relative_error'] * 100:.2f}%"
    )
    ax.text(
        0.11,
        0.96,
        summary,
        transform=ax.transAxes,
        va="top",
        fontsize=6.8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    panel_label_inside(ax, "b")

    fig.suptitle("Propagation of frozen rate predictions to Ac-225 inventory", fontsize=10.5, fontweight="semibold")
    fig.text(
        0.5,
        -0.02,
        "Modeled proxy schedules only; inventories are normalized and do not represent measured production activity.",
        ha="center",
        fontsize=6.8,
        color=GRAY,
    )
    return save_figure(fig, output_dir, "figure_3_ac225_schedule_propagation")


def write_notes(output_dir: Path, comparison: dict, gp_result: dict, lstm_result: dict) -> Path:
    path = output_dir / "FIGURE_NOTES.md"
    ci = comparison["paired_bootstrap"]["confidence_interval_95"]
    notes = f"""# V4 Publication Figure Notes

These figures were regenerated from the frozen V4 ZIP without training or tuning.

## Figure 1: Locked model comparison

- 48 locked modeled-proxy families.
- The set was locked for the V4 comparison protocol but had previously been evaluated in the V2 geometry diagnostic; it is not a fully untouched external test.
- LSTM median error: {lstm_result['test_metrics']['median_relative_error'] * 100:.2f}%; p95: {lstm_result['test_metrics']['p95_relative_error'] * 100:.2f}%.
- GP median error: {gp_result['test_metrics']['median_relative_error'] * 100:.2f}%; p95: {gp_result['test_metrics']['p95_relative_error'] * 100:.2f}%.
- The paired-bootstrap 95% interval for LSTM p95 minus GP p95 is [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}] percentage points, so the ordering is not resolved.

## Figure 2: Reliability and domain behavior

- Calibrated joint coverage is {lstm_result['uncertainty']['calibrated_95']['joint_case_coverage'] * 100:.1f}% for both models.
- Raw GP +/-2 SD joint coverage is only {gp_result['uncertainty']['raw_two_standard_deviations']['joint_case_coverage'] * 100:.1f}%, so GP uncertainty must be calibrated before use.
- The shared guard rejects {gp_result['domain_challenges']['explicit_ood_rejection_rate'] * 100:.0f}% of explicit outside-range cases and falsely rejects {gp_result['domain_challenges']['locked_test_false_rejection_rate'] * 100:.1f}% of locked cases.
- Accuracy is not reported for rejected outside-range predictions.

## Figure 3: Ac-225 schedule propagation

- GP Ac-225 inventory p95 error: {gp_result['ac225_inventory']['grid_p95_relative_error'] * 100:.2f}%.
- LSTM Ac-225 inventory p95 error: {lstm_result['ac225_inventory']['grid_p95_relative_error'] * 100:.2f}%.
- These are modeled normalized inventories propagated with exact five-nuclide equations, not measured activities.

## Claim boundary

The figures support a frozen architecture comparison on modeled finite-geometry/TENDL proxy data. They do not establish experimental, reactor, clinical, production-cost, or universal under-3% performance.
"""
    path.write_text(notes, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    bundle = ResultBundle(args.input_zip)
    try:
        paired = bundle.csv("/combined/paired_case_comparison.csv")
        comparison = bundle.json("/combined/final_model_comparison.json")
        gp_test = bundle.csv("/gp/test_predictions.csv")
        lstm_test = bundle.csv("/lstm/test_predictions.csv")
        gp_domain = bundle.csv("/gp/domain_challenges.csv")
        lstm_domain = bundle.csv("/lstm/domain_challenges.csv")
        gp_schedule = bundle.csv("/gp/ac225_schedule_predictions.csv")
        lstm_schedule = bundle.csv("/lstm/ac225_schedule_predictions.csv")
        gp_result = bundle.json("/gp/model_result.json")
        lstm_result = bundle.json("/lstm/model_result.json")
    finally:
        bundle.close()

    outputs: list[Path] = []
    outputs.extend(figure_model_comparison(gp_test, lstm_test, paired, comparison, args.output_dir))
    outputs.extend(
        figure_reliability(gp_test, lstm_test, gp_domain, lstm_domain, gp_result, lstm_result, args.output_dir)
    )
    outputs.extend(figure_ac225_inventory(gp_schedule, lstm_schedule, gp_result, lstm_result, args.output_dir))
    outputs.append(write_notes(args.output_dir, comparison, gp_result, lstm_result))

    manifest_path = args.output_dir / "figure_manifest.json"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "input_zip": str(args.input_zip.resolve()),
        "input_zip_sha256": sha256_file(args.input_zip.resolve()),
        "protocol": comparison["protocol"],
        "dataset_sha256": comparison["dataset_sha256"],
        "claim_eligible": comparison["claim_eligible"],
        "outcome": comparison["outcome"],
        "outputs": {path.name: sha256_file(path) for path in outputs},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {len(outputs)} figure/note files in {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
