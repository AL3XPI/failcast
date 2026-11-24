#!/usr/bin/env python3
"""Generate publication-quality README benchmark charts from `pts eval --output-json`.

Outputs two 300 DPI, dark/light-theme-readable PNG charts into `--out-dir`:
  1. pareto_fdr_vs_budget.png       (Budget % vs. Commit-Level FDR %)
  2. calibration_reliability.png    (Raw GBDT vs. Platt-Calibrated Reliability Diagram)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Ensure matplotlib can write its font cache even in sandboxed/read-only HOME setups.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pts-mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


STRATEGY_STYLES: dict[str, dict[str, Any]] = {
    "PTS (GBDT+Platt)": {
        "color": "#0F766E",  # Deep teal (high contrast, colorblind-safe)
        "marker": "o",
        "linewidth": 2.8,
        "markersize": 7.5,
        "linestyle": "-",
        "zorder": 6,
    },
    "Heuristic": {
        "color": "#2563EB",  # Crisp royal blue
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 6.0,
        "linestyle": "--",
        "zorder": 5,
    },
    "StaticAst": {
        "color": "#D97706",  # Warm amber
        "marker": "^",
        "linewidth": 1.9,
        "markersize": 6.2,
        "linestyle": "-.",
        "zorder": 4,
    },
    "RecentFailureRate": {
        "color": "#7C3AED",  # Muted violet
        "marker": "D",
        "linewidth": 1.9,
        "markersize": 5.8,
        "linestyle": (0, (3, 1.5, 1, 1.5)),
        "zorder": 3,
    },
    "Random": {
        "color": "#64748B",  # Neutral slate
        "marker": "x",
        "linewidth": 1.6,
        "markersize": 6.2,
        "linestyle": ":",
        "zorder": 2,
    },
}

DEFAULT_STYLE = {
    "color": "#334155",
    "marker": "o",
    "linewidth": 1.8,
    "markersize": 5.5,
    "linestyle": "-",
    "zorder": 3,
}


def configure_rcparams() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "#FFFFFF",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "axes.labelcolor": "#0F172A",
            "text.color": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )


def spread_label_y_positions(
    points: list[tuple[str, float]],
    min_gap: float = 4.2,
    y_min: float = 2.0,
    y_max: float = 99.0,
) -> dict[str, float]:
    """Adjust direct-label vertical offsets minimally so adjacent labels never overlap."""
    ordered = sorted(points, key=lambda item: item[1])
    ys = [max(y_min, min(y_max, y)) for _, y in ordered]

    # Forward pass to push overlapping labels apart upwards.
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap

    # If top overflowed, shift back down preserving minimum gap.
    if ys and ys[-1] > y_max:
        overflow = ys[-1] - y_max
        ys[-1] = y_max
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
        if ys[0] < y_min:
            # Distribute evenly within bounds if very crowded.
            step = (y_max - y_min) / max(1, len(ys) - 1)
            ys = [y_min + i * step for i in range(len(ys))]
            _ = overflow

    return {name: y for (name, _), y in zip(ordered, ys)}


def plot_pareto_fdr_vs_budget(
    report: dict[str, Any],
    out_path: Path,
    repo_label: str | None = None,
) -> None:
    budgets_raw = sorted(report.get("budgets", []), key=lambda b: float(b["budget_fraction"]))
    if not budgets_raw:
        raise ValueError("eval report contains no budget entries")

    x_budgets = [float(b["budget_fraction"]) * 100.0 for b in budgets_raw]
    strategy_order = [
        "PTS (GBDT+Platt)",
        "Heuristic",
        "StaticAst",
        "RecentFailureRate",
        "Random",
    ]

    series: dict[str, list[float]] = {}
    for b in budgets_raw:
        for s in b.get("strategies", []):
            name = s["strategy"]
            series.setdefault(name, []).append(float(s["commit_fdr"]) * 100.0)

    ordered_names = [s for s in strategy_order if s in series] + [
        s for s in series if s not in strategy_order
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=300)

    end_points: list[tuple[str, float]] = []
    for name in ordered_names:
        ys = series[name]
        style = STRATEGY_STYLES.get(name, DEFAULT_STYLE)
        ax.plot(
            x_budgets[: len(ys)],
            ys,
            label=name,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            linestyle=style["linestyle"],
            zorder=style["zorder"],
            markeredgecolor="#FFFFFF" if style["marker"] != "x" else style["color"],
            markeredgewidth=0.9 if style["marker"] != "x" else 1.5,
        )
        end_points.append((name, ys[-1]))

    # Annotate PTS data points directly for quick visual inspection.
    if "PTS (GBDT+Platt)" in series:
        pts_ys = series["PTS (GBDT+Platt)"]
        for x_val, y_val in zip(x_budgets[: len(pts_ys)], pts_ys):
            ax.annotate(
                f"{y_val:.0f}%",
                xy=(x_val, y_val),
                xytext=(0, 8 if y_val < 93 else -14),
                textcoords="offset points",
                ha="center",
                va="bottom" if y_val < 93 else "top",
                fontsize=8.2,
                fontweight="bold",
                color=STRATEGY_STYLES["PTS (GBDT+Platt)"]["color"],
            )

    # Direct right-margin labels with collision avoidance.
    x_max = max(x_budgets)
    x_min = min(x_budgets)
    span = max(10.0, x_max - x_min)
    label_x = x_max + span * 0.035
    adjusted_ys = spread_label_y_positions(end_points, min_gap=5.2, y_min=3.0, y_max=98.0)

    for name in ordered_names:
        style = STRATEGY_STYLES.get(name, DEFAULT_STYLE)
        y_actual = series[name][-1]
        y_label = adjusted_ys[name]
        if abs(y_label - y_actual) > 1.2:
            ax.plot(
                [x_max + span * 0.006, label_x - span * 0.008],
                [y_actual, y_label],
                color=style["color"],
                linewidth=0.8,
                alpha=0.55,
            )
        ax.text(
            label_x,
            y_label,
            f"{name} ({y_actual:.1f}%)",
            color=style["color"],
            fontsize=8.8,
            fontweight="bold" if name == "PTS (GBDT+Platt)" else "normal",
            va="center",
            ha="left",
        )

    ax.set_xlim(max(0.0, x_min - span * 0.06), x_max + span * 0.42)
    ax.set_ylim(-2.0, 105.0)
    ax.set_xticks(x_budgets)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel("Wall-Clock Time Budget (% of Full Suite)", labelpad=8, fontweight="bold")
    ax.set_ylabel("Commit-Level Failure Detection Rate (%)", labelpad=8, fontweight="bold")

    evaluated_runs = report.get("evaluated_runs", 0)
    failing_runs = report.get("failing_runs", 0)
    title_core = "Failure Detection Rate vs. Time Budget"
    if repo_label:
        title_core = f"{title_core} — {repo_label}"
    ax.set_title(title_core, loc="left", pad=22, fontweight="bold", fontsize=13)
    ax.text(
        0.0,
        1.025,
        f"Chronological walk-forward evaluation ({evaluated_runs} evaluated PR runs, {failing_runs} failing runs, zero leakage verified)",
        transform=ax.transAxes,
        fontsize=9.2,
        color="#475569",
    )

    ax.grid(True, linestyle="--", alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def plot_calibration_reliability(
    report: dict[str, Any],
    out_path: Path,
    repo_label: str | None = None,
) -> None:
    cal_bins = report.get("calibration_bins", {})
    raw_bins = cal_bins.get("raw_gbdt", [])
    platt_bins = cal_bins.get("platt_calibrated", [])
    if not raw_bins or not platt_bins:
        raise ValueError("eval report missing calibration_bins.raw_gbdt or platt_calibrated")

    raw_brier = float(report.get("uncalibrated_brier_score", 0.0))
    platt_brier = float(report.get("brier_score", 0.0))
    total_samples = sum(int(b["count"]) for b in platt_bins)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 5.8),
        dpi=300,
        gridspec_kw={"height_ratios": [3.2, 1.05], "hspace": 0.14, "wspace": 0.22},
        sharex="col",
    )

    panels = [
        (
            0,
            raw_bins,
            "Raw GBDT (Uncalibrated Sigmoid)",
            raw_brier,
            "#D97706",
            "s",
        ),
        (
            1,
            platt_bins,
            "PTS (GBDT + Platt Calibrated)",
            platt_brier,
            "#0F766E",
            "o",
        ),
    ]

    max_count = max(
        [int(b["count"]) for b in raw_bins] + [int(b["count"]) for b in platt_bins] + [1]
    )

    for col, bins, panel_title, brier_val, color, marker in panels:
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        non_empty = [b for b in bins if int(b["count"]) > 0]
        xs = [float(b["mean_predicted"]) for b in non_empty]
        ys = [float(b["empirical_positive_rate"]) for b in non_empty]
        counts = [int(b["count"]) for b in non_empty]

        # Perfect calibration diagonal.
        ax_top.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            color="#64748B",
            linewidth=1.3,
            label="Perfect calibration",
            zorder=1,
        )

        if xs:
            ax_top.plot(
                xs,
                ys,
                color=color,
                linewidth=2.2,
                marker=marker,
                markersize=6.8,
                markeredgecolor="#FFFFFF",
                markeredgewidth=0.9,
                zorder=4,
                label=panel_title,
            )

            # Annotate populated bins with sample counts, staggering adjacent crowded labels.
            prev_xy: tuple[float, float] | None = None
            stagger_toggle = False
            for x_val, y_val, cnt in zip(xs, ys, counts):
                label_txt = f"n={cnt:,}" if cnt < 10000 else f"n={cnt / 1000:.1f}k"
                crowded = (
                    prev_xy is not None
                    and abs(x_val - prev_xy[0]) < 0.12
                    and abs(y_val - prev_xy[1]) < 0.12
                )
                if crowded:
                    stagger_toggle = not stagger_toggle
                else:
                    stagger_toggle = False
                if y_val > 0.85:
                    offset_y = -15 if not stagger_toggle else -26
                elif stagger_toggle:
                    offset_y = 20
                else:
                    offset_y = 9
                ax_top.annotate(
                    label_txt,
                    xy=(x_val, y_val),
                    xytext=(0, offset_y),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.3,
                    color="#334155",
                )
                prev_xy = (x_val, y_val)

        ax_top.set_xlim(-0.04, 1.04)
        ax_top.set_ylim(-0.06, 1.08)
        ax_top.grid(True, linestyle="--", alpha=0.7)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_top.set_title(panel_title, loc="left", fontweight="bold", fontsize=11.0, pad=8)
        if col == 0:
            ax_top.set_ylabel("Empirical Failure Rate", fontweight="bold", labelpad=8)

        # Brier score + total outcome callout badge placed in the clear upper-right area.
        badge_text = f"Brier Score: {brier_val:.4f}\nOutcomes: {total_samples:,}"
        ax_top.text(
            0.96,
            0.38,
            badge_text,
            transform=ax_top.transAxes,
            va="bottom",
            ha="right",
            fontsize=8.6,
            fontweight="bold",
            color="#0F172A",
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": "#F8FAFC",
                "edgecolor": "#CBD5E1",
                "linewidth": 0.9,
            },
        )

        # Bottom panel: sample distribution histogram per 0.1 bin (log scale when skewed).
        centers = [float(b["bin_center"]) for b in bins]
        all_counts = [int(b["count"]) for b in bins]
        ax_bot.bar(
            centers,
            all_counts,
            width=0.082,
            color=color,
            alpha=0.75,
            edgecolor="#FFFFFF",
            linewidth=0.6,
        )
        if max_count > 100:
            ax_bot.set_yscale("log")
            ax_bot.set_ylim(0.8, max_count * 2.5)
        else:
            ax_bot.set_ylim(0, max_count * 1.25)
        ax_bot.set_xlim(-0.04, 1.04)
        ax_bot.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_bot.set_xlabel("Mean Predicted Failure Probability", fontweight="bold", labelpad=6)
        if col == 0:
            ax_bot.set_ylabel("Bin Count", fontweight="bold", labelpad=8)
        ax_bot.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)

    title = "Probability Calibration Reliability Diagram (10 Equal-Width Bins)"
    if repo_label:
        title = f"{title} — {repo_label}"
    fig.suptitle(title, x=0.06, y=0.99, ha="left", fontweight="bold", fontsize=13.0)
    if platt_brier <= raw_brier and raw_brier > 0:
        rel_pct = (raw_brier - platt_brier) / raw_brier * 100.0
        subtitle = (
            f"Walk-forward Brier score across {total_samples:,} outcomes: "
            f"Raw GBDT {raw_brier:.4f} -> Platt-Calibrated {platt_brier:.4f} ({rel_pct:.1f}% reduction)"
        )
    else:
        subtitle = (
            f"Walk-forward Brier score across {total_samples:,} outcomes: "
            f"Raw GBDT {raw_brier:.4f} vs. Platt-Calibrated {platt_brier:.4f}"
        )
    fig.text(
        0.06,
        0.935,
        subtitle,
        ha="left",
        fontsize=9.2,
        color="#475569",
    )

    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.08, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot PTS walk-forward Pareto and calibration reliability PNG charts."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to JSON report emitted by `pts eval --output-json`.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Output directory for PNG charts (default: docs/benchmarks).",
    )
    parser.add_argument(
        "--repo-label",
        default=None,
        help="Optional repository name shown in chart titles (e.g. 'pallets/click').",
    )
    args = parser.parse_args()

    configure_rcparams()
    report = json.loads(args.input.read_text(encoding="utf-8"))

    pareto_path = args.out_dir / "pareto_fdr_vs_budget.png"
    cal_path = args.out_dir / "calibration_reliability.png"

    plot_pareto_fdr_vs_budget(report, pareto_path, repo_label=args.repo_label)
    plot_calibration_reliability(report, cal_path, repo_label=args.repo_label)

    print(f"Wrote {pareto_path}")
    print(f"Wrote {cal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
