#!/usr/bin/env python3
"""Generate the two compact figures used by the public project README."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finpfn-public-mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE = REPOSITORY / "reproduction/reference_results"
DEFAULT_OUTPUT = REPOSITORY / "reproduction/runs/public_figures"
COMPARISON_INPUT = REFERENCE / "summary/model_comparison.csv"
TAIL_INPUT = REFERENCE / "summary/us_tail_precision.csv"

MODELS = ["FinPFN", "Ridge", "LightGBM", "TabPFN"]
MARKETS = ["CSI500", "US"]
MODEL_COLORS = {
    "FinPFN": "#2A6FBB",
    "Ridge": "#E07A1F",
    "LightGBM": "#3A9D5D",
    "TabPFN": "#8C62AA",
}
MARKET_COLORS = {"CSI500": "#2A6FBB", "US": "#E07A1F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for regenerated figures (default: reproduction/runs/public_figures)",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def annotate_bars(axis: plt.Axes, decimals: int = 3) -> None:
    for container in axis.containers:
        axis.bar_label(
            container,
            fmt=f"%.{decimals}f",
            padding=3,
            fontsize=8.5,
        )


def save_ic_ir_overview(comparison: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(len(MODELS))
    width = 0.36

    for axis, metric, title in [
        (axes[0], "mean_ic", "Mean cross-sectional IC"),
        (axes[1], "ir", "IC information ratio (IR)"),
    ]:
        for offset_index, market in enumerate(MARKETS):
            values = (
                comparison.loc[comparison["market"] == market]
                .set_index("model")
                .reindex(MODELS)[metric]
            )
            offset = (offset_index - 0.5) * width
            axis.bar(
                x + offset,
                values,
                width,
                color=MARKET_COLORS[market],
                label="CSI 500" if market == "CSI500" else "U.S.",
            )
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, MODELS, rotation=15)
        axis.grid(axis="y", alpha=0.2)
        annotate_bars(axis)

    axes[0].set_ylim(-0.048, 0.076)
    axes[1].set_ylim(-0.60, 0.78)
    axes[0].set_ylabel("Spearman rank correlation")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "FinPFN leads statistical ranking metrics in both markets",
        y=0.99,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.81))
    figure.savefig(
        output_dir / "ic_ir_overview.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_ic_portfolio_gap(
    comparison: pd.DataFrame,
    tails: pd.DataFrame,
    output_dir: Path,
) -> None:
    us = comparison.loc[comparison["market"] == "US"].set_index("model")
    tail_40 = tails.loc[tails["k"] == 40].set_index("model").reindex(MODELS)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.1))

    for model in MODELS:
        axes[0].scatter(
            us.loc[model, "mean_ic"],
            us.loc[model, "gross_long_short_sharpe"],
            s=95,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        label_offset = {
            "FinPFN": (-5, -15),
            "Ridge": (7, -15),
            "LightGBM": (-5, 7),
            "TabPFN": (5, 7),
        }[model]
        axes[0].annotate(
            model,
            (
                us.loc[model, "mean_ic"],
                us.loc[model, "gross_long_short_sharpe"],
            ),
            xytext=label_offset,
            textcoords="offset points",
            fontsize=9,
            ha="right" if model in {"FinPFN", "LightGBM"} else "left",
        )
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    axes[0].set(
        title="U.S.: full-universe IC vs. tail Sharpe",
        xlabel="Mean cross-sectional IC",
        ylabel="Gross top-minus-bottom Sharpe",
    )
    axes[0].set_xlim(-0.002, 0.071)
    axes[0].set_ylim(-0.10, 1.73)
    axes[0].grid(alpha=0.2)

    x = np.arange(len(MODELS))
    width = 0.36
    axes[1].bar(
        x - width / 2,
        tail_40["mean_top_precision"],
        width,
        color="#2A6FBB",
        label="Top-40 precision",
    )
    axes[1].bar(
        x + width / 2,
        tail_40["mean_bottom_precision"],
        width,
        color="#E07A1F",
        label="Bottom-40 precision",
    )
    axes[1].axhline(
        float(tail_40["mean_actual_fraction"].iloc[0]),
        color="#444444",
        linestyle="--",
        linewidth=1.2,
    )
    axes[1].set(
        title="U.S.: extreme-tail selection",
        ylabel="Realized-tail precision",
    )
    axes[1].set_ylim(0, 0.25)
    axes[1].set_xticks(x, MODELS, rotation=15)
    axes[1].grid(axis="y", alpha=0.2)
    annotate_bars(axes[1])

    legend_handles = [
        Patch(facecolor="#2A6FBB", label="Top-40 precision"),
        Patch(facecolor="#E07A1F", label="Bottom-40 precision"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            label="Random selection (8%)",
        ),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    figure.suptitle(
        "Higher overall IC did not guarantee better tradable tails",
        y=0.99,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.93))
    figure.savefig(
        output_dir / "ic_portfolio_gap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_corrected_tail_precision(tails: pd.DataFrame, output_dir: Path) -> None:
    tail_sizes = sorted(tails["k"].unique())
    figure, axes = plt.subplots(
        2,
        len(tail_sizes),
        figsize=(13, 7.4),
        sharey="row",
    )
    row_limits = {
        "mean_top_precision": tails["mean_top_precision"].max() * 1.18,
        "mean_bottom_precision": tails["mean_bottom_precision"].max() * 1.18,
    }
    for column_index, k in enumerate(tail_sizes):
        subset = tails.loc[tails["k"] == k].set_index("model").reindex(MODELS)
        random_precision = float(subset["mean_actual_fraction"].iloc[0])
        for row_index, (metric, leg) in enumerate(
            [
                ("mean_top_precision", "Top"),
                ("mean_bottom_precision", "Bottom"),
            ]
        ):
            axis = axes[row_index, column_index]
            axis.bar(
                MODELS,
                subset[metric],
                color=[MODEL_COLORS[model] for model in MODELS],
            )
            axis.axhline(
                random_precision,
                color="#444444",
                linestyle="--",
                linewidth=1.2,
            )
            axis.set_title(f"{leg}-{k} precision")
            axis.set_ylim(0, row_limits[metric])
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="x", rotation=25)
            if column_index == 0:
                axis.set_ylabel("Realized-tail precision")

    legend_handles = [
        Patch(facecolor=MODEL_COLORS[model], label=model) for model in MODELS
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            label="Random-selection rate (k / 500)",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
    )
    figure.suptitle(
        "U.S. realized-tail precision by selected portfolio size",
        y=0.93,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(
        output_dir / "us_tail_precision.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    reference_figures = (REFERENCE / "figures").resolve()
    if output_dir == reference_figures:
        raise ValueError(
            "Refusing to overwrite frozen reference figures; choose a run directory"
        )

    comparison = pd.read_csv(COMPARISON_INPUT)
    tails = pd.read_csv(TAIL_INPUT)
    require_columns(
        comparison,
        {"market", "model", "mean_ic", "ir", "gross_long_short_sharpe"},
        COMPARISON_INPUT,
    )
    require_columns(
        tails,
        {
            "model",
            "k",
            "mean_actual_fraction",
            "mean_top_precision",
            "mean_bottom_precision",
        },
        TAIL_INPUT,
    )
    if set(comparison["market"]) != set(MARKETS):
        raise ValueError("Expected exactly the CSI500 and US market labels")
    if set(comparison["model"]) != set(MODELS):
        raise ValueError("Comparison input does not contain exactly four models")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_ic_ir_overview(comparison, output_dir)
    save_ic_portfolio_gap(comparison, tails, output_dir)
    save_corrected_tail_precision(tails, output_dir)
    print(f"Wrote regenerated figures to {output_dir}")
    print(f"Corrected tail-precision figure: {output_dir / 'us_tail_precision.png'}")


if __name__ == "__main__":
    main()
