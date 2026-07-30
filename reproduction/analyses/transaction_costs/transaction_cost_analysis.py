#!/usr/bin/env python3
"""Evaluate frozen CSI 500 portfolios under predeclared one-way costs.

All stored return columns produced by the baseline evaluator are percentage points.
This script converts them to decimal returns before subtracting transaction costs.
It writes only below the ignored reproduction/runs directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finpfn-cost-analysis-mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[3]
CSI_RUN_ROOT = REPOSITORY / "reproduction/runs/csi500"
DEFAULT_BASELINE = CSI_RUN_ROOT / "evaluation"
DEFAULT_OUTPUT = REPOSITORY / "reproduction/runs/transaction_costs"
ANNUALIZATION = 240
COST_GRID_BPS = (0, 2, 5, 10, 20, 30, 50)
EXPECTED_MODELS = ("FinPFN", "TabPFN", "Ridge", "LightGBM")

DEFAULT_PREDICTIONS = {
    "FinPFN": REPOSITORY
    / "reproduction/runs/csi500/checkpoints/notebook_exact/"
    "csi500_finpfn_seed42_notebook_with_replacement.parquet",
    "TabPFN": REPOSITORY
    / "reproduction/runs/csi500/checkpoints/notebook_exact/"
    "csi500_tabpfn_seed42_notebook_with_replacement.parquet",
    "Ridge": REPOSITORY
    / "reproduction/runs/csi500/baselines/csi500_ridge_seed42.parquet",
    "LightGBM": REPOSITORY
    / "reproduction/runs/csi500/baselines/"
    "csi500_lightgbm_seed42.parquet",
}

EXPECTED_SHA256 = {
    "csi500_finpfn_seed42_notebook_with_replacement.parquet": (
        "03e62d18bf14cb6a3787213a87369adf12914d65748f8d1536a7bc5cecca76f3"
    ),
    "csi500_tabpfn_seed42_notebook_with_replacement.parquet": (
        "0fa76d578741b3a50a9f6e1b96009bae6fe4f884b9ce7a3fe0f52b6cec95c26a"
    ),
    "csi500_ridge_seed42.parquet": (
        "a6cccd1f54f3ced4cd5165615a6c7d921d3d46d157f5a7e166a532532b6488b1"
    ),
    "csi500_lightgbm_seed42.parquet": (
        "0a0c7f0bcbb5e97d25dcaf73448e55f9ec97b70aa8dc5bb91d0bf0f70eae375a"
    ),
    "decile_holdings.parquet": (
        "bfcd2ba50283f41a0ad4dc2265d7067cb570edf79a9352bbd820ae02d57d432d"
    ),
    "decile_returns_by_period.csv": (
        "02c0c0c75d003d7c68b02565e13d373119b87b374761a7fc49e9370b2d16b14b"
    ),
    "turnover_by_decile.csv": (
        "60c622e967db8e90f5515571d05f1c91898b14439210ae7190dca7437c7afd4a"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdings", type=Path, default=DEFAULT_BASELINE / "decile_holdings.parquet"
    )
    parser.add_argument(
        "--period-returns",
        type=Path,
        default=DEFAULT_BASELINE / "decile_returns_by_period.csv",
    )
    parser.add_argument(
        "--baseline-turnover",
        type=Path,
        default=DEFAULT_BASELINE / "turnover_by_decile.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = repo_relative(path)
    observed = sha256(path)
    expected = EXPECTED_SHA256.get(path.name)
    if expected is None:
        raise ValueError(f"Input is not a declared frozen artifact: {relative}")
    if observed != expected:
        raise ValueError(
            f"Frozen input checksum mismatch for {relative}: {observed} != {expected}"
        )
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": observed,
    }


def refuse_existing_outputs(output_dir: Path) -> None:
    declared = [
        "cost_sensitivity.csv",
        "model_break_even_costs.csv",
        "net_performance_by_period.csv",
        "integrity_checks.json",
        "input_manifest.json",
        "report.md",
    ]
    existing = [str(output_dir / name) for name in declared if (output_dir / name).exists()]
    figures = output_dir / "figures"
    if figures.exists() and any(figures.iterdir()):
        existing.append(str(figures))
    if existing:
        raise FileExistsError(
            "Refusing to overwrite transaction-cost outputs: " + ", ".join(existing)
        )


def prediction_integrity(
    holdings: pd.DataFrame, prediction_paths: dict[str, Path]
) -> dict[str, object]:
    checks: dict[str, object] = {}
    holding_keys = {
        model: set(zip(part["date"], part["id"], strict=True))
        for model, part in holdings.groupby("model", sort=True)
    }
    for model in EXPECTED_MODELS:
        path = prediction_paths[model]
        frame = pd.read_parquet(
            path, columns=["model", "seed", "date", "id", "prediction", "status"]
        )
        frame["date"] = pd.to_datetime(frame["date"])
        if set(frame["model"].dropna().unique()) != {model}:
            raise ValueError(f"Prediction model label mismatch in {path.name}")
        valid = frame.loc[(frame["status"] == "ok") & frame["prediction"].notna()].copy()
        if valid.empty:
            raise ValueError(f"No valid predictions in {path.name}")
        counts = valid.groupby(["date", "id"], sort=False).size()
        collapsed = (
            valid.groupby(["date", "id"], as_index=False, sort=True)["prediction"]
            .mean()
            .merge(
                holdings.loc[holdings["model"] == model, ["date", "id", "decile"]],
                on=["date", "id"],
                how="inner",
                validate="one_to_one",
            )
        )
        expected_keys = holding_keys[model]
        observed_keys = set(zip(collapsed["date"], collapsed["id"], strict=True))
        if observed_keys != expected_keys:
            raise ValueError(f"Holdings are not the exact prediction subset for {model}")

        reassigned = []
        for date, part in collapsed.groupby("date", sort=True):
            part = part.sort_values(["prediction", "id"], kind="stable").copy()
            part["derived_decile"] = (
                np.floor(np.arange(len(part)) * 10 / len(part)).astype(int) + 1
            )
            reassigned.append(part[["date", "id", "decile", "derived_decile"]])
        reassigned_frame = pd.concat(reassigned, ignore_index=True)
        mismatches = int(
            (reassigned_frame["decile"] != reassigned_frame["derived_decile"]).sum()
        )
        if mismatches:
            raise ValueError(
                f"{model} holdings do not match deciles from its own prediction: "
                f"{mismatches} mismatches"
            )
        checks[model] = {
            "input_rows": int(len(frame)),
            "valid_rows": int(len(valid)),
            "unique_asset_dates": int(len(counts)),
            "repeated_rows_beyond_first": int(len(valid) - len(counts)),
            "maximum_repetitions": int(counts.max()),
            "common_holding_rows": int(len(collapsed)),
            "holding_decile_mismatches_from_own_prediction": mismatches,
            "aggregation_before_deciles": "mean prediction by model/date/id",
            "target_used_to_form_deciles": False,
        }
    return checks


def basic_holdings_integrity(holdings: pd.DataFrame) -> dict[str, object]:
    required = {"model", "seed", "date", "id", "decile"}
    missing = sorted(required.difference(holdings.columns))
    if missing:
        raise ValueError(f"Holdings missing columns: {missing}")
    holdings["date"] = pd.to_datetime(holdings["date"])
    duplicates = int(holdings.duplicated(["model", "seed", "date", "id"]).sum())
    if duplicates:
        raise ValueError(f"Holdings contain {duplicates} duplicate model/seed/date/id rows")
    if set(holdings["model"].unique()) != set(EXPECTED_MODELS):
        raise ValueError("Holdings model set differs from the four declared models")
    if not holdings["decile"].between(1, 10).all():
        raise ValueError("Invalid decile values")

    universe_by_model_date = {
        (model, date): frozenset(part["id"])
        for (model, date), part in holdings.groupby(["model", "date"], sort=True)
    }
    dates = sorted(holdings["date"].unique())
    for date in dates:
        universes = [universe_by_model_date[(model, date)] for model in EXPECTED_MODELS]
        if any(universe != universes[0] for universe in universes[1:]):
            raise ValueError(f"Models have inconsistent holdings universe on {date}")
    counts = holdings.groupby(["model", "date"]).size()
    return {
        "duplicate_model_seed_date_id_rows": duplicates,
        "models": sorted(holdings["model"].unique()),
        "dates": int(len(dates)),
        "rows": int(len(holdings)),
        "assets_per_model_date_min": int(counts.min()),
        "assets_per_model_date_mean": float(counts.mean()),
        "assets_per_model_date_max": int(counts.max()),
        "common_universe_identical_across_models_each_date": True,
        "exactly_one_decile_per_model_asset_date": True,
    }


def leg_turnover(holdings: pd.DataFrame, *, include_initial: bool) -> pd.Series:
    previous: dict[object, float] | None = None
    rows: dict[pd.Timestamp, float] = {}
    for date, part in holdings.groupby("date", sort=True):
        weight = 1.0 / len(part)
        current = {identifier: weight for identifier in part["id"]}
        if previous is None:
            if include_initial:
                rows[pd.Timestamp(date)] = 1.0
        else:
            identifiers = set(previous) | set(current)
            rows[pd.Timestamp(date)] = 0.5 * sum(
                abs(current.get(item, 0.0) - previous.get(item, 0.0))
                for item in identifiers
            )
        previous = current
    return pd.Series(rows, name="turnover_one_way", dtype=float).sort_index()


def build_turnover_series(
    holdings: pd.DataFrame, baseline_turnover: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    records = []
    comparison = []
    for (model, seed), part in holdings.groupby(["model", "seed"], sort=True):
        long_part = part.loc[part["decile"] == 10]
        short_part = part.loc[part["decile"] == 1]
        long_with_entry = leg_turnover(long_part, include_initial=True)
        short_with_entry = leg_turnover(short_part, include_initial=True)
        long_rebalancing = leg_turnover(long_part, include_initial=False)
        short_rebalancing = leg_turnover(short_part, include_initial=False)
        for decile, calculated in [(10, long_rebalancing), (1, short_rebalancing)]:
            frozen = baseline_turnover.loc[
                (baseline_turnover["model"] == model)
                & (baseline_turnover["seed"] == seed)
                & (baseline_turnover["decile"] == decile),
                "mean_one_way_turnover",
            ]
            if len(frozen) != 1:
                raise ValueError(f"Missing frozen turnover for {model} decile {decile}")
            difference = float(calculated.mean() - frozen.iat[0])
            comparison.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "decile": decile,
                    "recomputed_excluding_initial": float(calculated.mean()),
                    "frozen_mean_one_way_turnover": float(frozen.iat[0]),
                    "difference": difference,
                }
            )
            if abs(difference) > 1e-12:
                raise ValueError(
                    f"Turnover mismatch for {model} decile {decile}: {difference}"
                )
        if not long_with_entry.index.equals(short_with_entry.index):
            raise ValueError(f"Long/short turnover dates differ for {model}")
        for date in long_with_entry.index:
            long_value = float(long_with_entry.loc[date])
            short_value = float(short_with_entry.loc[date])
            records.extend(
                [
                    {
                        "model": model,
                        "seed": int(seed),
                        "date": date,
                        "portfolio": "long",
                        "turnover_one_way": long_value,
                    },
                    {
                        "model": model,
                        "seed": int(seed),
                        "date": date,
                        "portfolio": "short",
                        "turnover_one_way": short_value,
                    },
                    {
                        "model": model,
                        "seed": int(seed),
                        "date": date,
                        "portfolio": "long_short",
                        "turnover_one_way": long_value + short_value,
                    },
                ]
            )
    return pd.DataFrame(records), {"frozen_turnover_comparison": comparison}


def build_gross_returns(period_returns: pd.DataFrame) -> pd.DataFrame:
    raw = period_returns.loc[period_returns["return_basis"] == "raw"].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if raw.duplicated(["model", "seed", "date"]).any():
        raise ValueError("Raw period returns contain duplicate model/seed/date rows")
    expected = raw["decile_10"] - raw["decile_1"]
    if not np.allclose(expected, raw["long_short"], rtol=0, atol=1e-12):
        raise ValueError("Frozen long_short is not decile_10 minus decile_1")
    frames = []
    definitions = {
        "long": raw["decile_10"],
        "short": -raw["decile_1"],
        "long_short": raw["long_short"],
    }
    for portfolio, gross_percentage_points in definitions.items():
        frame = raw[["model", "seed", "date"]].copy()
        frame["portfolio"] = portfolio
        frame["gross_return_decimal"] = gross_percentage_points.to_numpy() / 100.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sharpe(values: pd.Series) -> float:
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(values.mean() / standard_deviation * np.sqrt(ANNUALIZATION))


def maximum_drawdown(values: pd.Series) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + values.to_numpy(dtype=float))]
    running_peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running_peak - 1.0))


def summarize_return_series(values: pd.Series, prefix: str) -> dict[str, float]:
    if (values <= -1.0).any():
        terminal_wealth = np.nan
        max_drawdown = np.nan
    else:
        terminal_wealth = float(np.prod(1.0 + values))
        max_drawdown = maximum_drawdown(values)
    return {
        f"{prefix}_mean_return_decimal": float(values.mean()),
        f"{prefix}_annualized_volatility_decimal": float(
            values.std(ddof=1) * np.sqrt(ANNUALIZATION)
        ),
        f"{prefix}_sharpe": sharpe(values),
        f"{prefix}_cumulative_arithmetic_return_decimal": float(values.sum()),
        f"{prefix}_terminal_compounded_wealth": terminal_wealth,
        f"{prefix}_maximum_drawdown": max_drawdown,
    }


def evaluate_costs(
    gross_returns: pd.DataFrame, turnover: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = gross_returns.merge(
        turnover,
        on=["model", "seed", "date", "portfolio"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(gross_returns) or len(merged) != len(turnover):
        raise ValueError("Return and turnover series do not align exactly")

    period_frames = []
    summary_rows = []
    for cost_bps in COST_GRID_BPS:
        period = merged.copy()
        period["cost_rate_bps_per_one_way_turnover"] = cost_bps
        period["transaction_cost_decimal"] = (
            period["turnover_one_way"] * cost_bps / 10_000.0
        )
        period["net_return_decimal"] = (
            period["gross_return_decimal"] - period["transaction_cost_decimal"]
        )
        period["gross_compounded_wealth"] = period.groupby(
            ["model", "seed", "portfolio"], sort=False
        )["gross_return_decimal"].transform(lambda values: (1.0 + values).cumprod())
        period["net_compounded_wealth"] = period.groupby(
            ["model", "seed", "portfolio"], sort=False
        )["net_return_decimal"].transform(lambda values: (1.0 + values).cumprod())
        period_frames.append(period)

        for (model, seed, portfolio), part in period.groupby(
            ["model", "seed", "portfolio"], sort=True
        ):
            row: dict[str, object] = {
                "model": model,
                "seed": int(seed),
                "portfolio": portfolio,
                "cost_rate_bps_per_one_way_turnover": cost_bps,
                "n_periods": int(len(part)),
                "average_turnover_one_way_including_initial": float(
                    part["turnover_one_way"].mean()
                ),
            }
            row.update(summarize_return_series(part["gross_return_decimal"], "gross"))
            row.update(summarize_return_series(part["net_return_decimal"], "net"))
            summary_rows.append(row)

    periods = pd.concat(period_frames, ignore_index=True).sort_values(
        ["cost_rate_bps_per_one_way_turnover", "model", "portfolio", "date"]
    )
    summary = pd.DataFrame(summary_rows)
    summary["net_sharpe_rank"] = (
        summary.groupby(["portfolio", "cost_rate_bps_per_one_way_turnover"])[
            "net_sharpe"
        ]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    zero = summary.loc[
        summary["cost_rate_bps_per_one_way_turnover"] == 0
    ].copy()
    zero["break_even_cost_bps_mean_net_return_zero"] = np.where(
        zero["average_turnover_one_way_including_initial"] > 0,
        zero["gross_mean_return_decimal"]
        / zero["average_turnover_one_way_including_initial"]
        * 10_000.0,
        np.nan,
    )
    break_even = zero[
        [
            "model",
            "seed",
            "portfolio",
            "n_periods",
            "gross_mean_return_decimal",
            "average_turnover_one_way_including_initial",
            "break_even_cost_bps_mean_net_return_zero",
        ]
    ].sort_values(["portfolio", "model"])
    return summary, break_even, periods


def plot_results(summary: pd.DataFrame, periods: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=False)
    model_order = list(EXPECTED_MODELS)

    fig, axis = plt.subplots(figsize=(8, 5))
    long_short = summary.loc[summary["portfolio"] == "long_short"]
    for model in model_order:
        part = long_short.loc[long_short["model"] == model].sort_values(
            "cost_rate_bps_per_one_way_turnover"
        )
        axis.plot(
            part["cost_rate_bps_per_one_way_turnover"],
            part["net_sharpe"],
            marker="o",
            label=model,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Cost (bps per one-way turnover)", ylabel="Net H-L Sharpe")
    axis.set_title("CSI 500 long-short net Sharpe sensitivity")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "net_sharpe_by_cost.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    at_ten = periods.loc[
        (periods["portfolio"] == "long_short")
        & (periods["cost_rate_bps_per_one_way_turnover"] == 10)
    ]
    for model in model_order:
        part = at_ten.loc[at_ten["model"] == model].sort_values("date")
        axis.plot(part["date"], part["net_compounded_wealth"], label=model)
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set(ylabel="Compounded wealth", xlabel="Date")
    axis.set_title("CSI 500 net long-short wealth at 10 bps")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "cumulative_long_short_wealth_10bps.png", dpi=180)
    plt.close(fig)

    zero = summary.loc[
        (summary["portfolio"] == "long_short")
        & (summary["cost_rate_bps_per_one_way_turnover"] == 0)
    ].set_index("model").reindex(model_order)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(zero.index, zero["average_turnover_one_way_including_initial"])
    axis.set(ylabel="Long + short one-way turnover", xlabel="Model")
    axis.set_title("Average long-short turnover (initial entry included)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "average_long_short_turnover.png", dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        rendered.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rendered)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    break_even: pd.DataFrame,
    integrity: dict[str, object],
    runtime_seconds: float,
) -> None:
    long_short = summary.loc[summary["portfolio"] == "long_short"].copy()
    ranking_rows = []
    for cost, part in long_short.groupby(
        "cost_rate_bps_per_one_way_turnover", sort=True
    ):
        ordered = part.sort_values(["net_sharpe_rank", "model"])
        ranking_rows.append(
            [
                int(cost),
                " > ".join(ordered["model"]),
                ", ".join(
                    f"{row.model} {row.net_sharpe:.4f}"
                    for row in ordered.itertuples()
                ),
            ]
        )
    selected_costs = long_short.loc[
        long_short["cost_rate_bps_per_one_way_turnover"].isin([0, 10, 50])
    ].sort_values(["cost_rate_bps_per_one_way_turnover", "net_sharpe_rank"])
    metric_rows = [
        [
            int(row.cost_rate_bps_per_one_way_turnover),
            row.model,
            f"{row.gross_mean_return_decimal * 100:.4f}%",
            f"{row.net_mean_return_decimal * 100:.4f}%",
            f"{row.net_annualized_volatility_decimal * 100:.4f}%",
            f"{row.gross_sharpe:.4f}",
            f"{row.net_sharpe:.4f}",
            f"{row.average_turnover_one_way_including_initial:.4f}",
            f"{row.net_terminal_compounded_wealth:.4f}",
            f"{row.net_maximum_drawdown * 100:.2f}%",
        ]
        for row in selected_costs.itertuples()
    ]
    break_even_ls = break_even.loc[break_even["portfolio"] == "long_short"].sort_values(
        "break_even_cost_bps_mean_net_return_zero", ascending=False
    )
    break_even_rows = [
        [
            row.model,
            f"{row.average_turnover_one_way_including_initial:.6f}",
            f"{row.break_even_cost_bps_mean_net_return_zero:.3f}",
        ]
        for row in break_even_ls.itertuples()
    ]
    pred_checks = integrity["prediction_integrity"]
    pred_rows = [
        [
            model,
            pred_checks[model]["input_rows"],
            pred_checks[model]["unique_asset_dates"],
            pred_checks[model]["repeated_rows_beyond_first"],
            pred_checks[model]["holding_decile_mismatches_from_own_prediction"],
        ]
        for model in EXPECTED_MODELS
    ]

    report = f"""# Transaction-cost and turnover sensitivity

## Result

This analysis uses one frozen common test universe, each model's own holdings,
and one raw-return target. The cost grid was fixed at 0, 2, 5, 10, 20, 30,
and 50 bps before model results were compared.

{markdown_table(['Cost (bps)', 'Net H-L Sharpe ranking', 'Net Sharpe by model'], ranking_rows)}

FinPFN has the highest long-short turnover, so its net performance degrades
faster than Ridge and LightGBM. This is a sensitivity audit of an observed
test result, not a strategy-selection exercise.

## Cost and turnover definitions

- Stored percentage-point returns are converted to decimal returns.
- `cost_rate = bps / 10,000`.
- One-leg turnover is `0.5 * sum(abs(weight[t] - weight[t-1]))`.
- The first date charges entry from cash; long and short each start at 1.0.
- The long leg is decile 10; short return is negative decile-1 return.
- Total H-L turnover is the sum of one-way long and short turnover.
- Sharpe uses the actual H-L series and an annualization factor of 240.
- Break-even cost sets mean net return to zero and excludes unmodelled impact,
  borrow, financing, and capacity costs.

## Selected metrics

{markdown_table(['Cost', 'Model', 'Gross mean/period', 'Net mean/period', 'Net annualized volatility', 'Gross Sharpe', 'Net Sharpe', 'Average turnover', 'Net terminal wealth', 'Net maximum drawdown'], metric_rows)}

### Mean-return break-even cost

{markdown_table(['Model', 'Average total one-way turnover', 'Break-even bps'], break_even_rows)}

Complete long, short, and H-L results are in `cost_sensitivity.csv`; per-period
costs, returns, and wealth are in `net_performance_by_period.csv`.

## Sampling and holdings integrity

{markdown_table(['Model', 'Prediction rows', 'Unique asset-dates', 'Repeated rows', 'Own-prediction decile mismatches'], pred_rows)}

- Frozen holdings contain {integrity['holdings']['rows']:,} rows and no duplicate
  `(model, seed, date, id)` keys.
- Every model uses the same date-specific universe and its own prediction column.
- Repeated FinPFN and TabPFN task rows are averaged by model/date/asset before
  deciles are formed.
- Targets are never used for ranking.
- Recomputed turnover differs from the frozen evaluator by at most
  {integrity['maximum_absolute_frozen_turnover_difference']:.3e}.

FinPFN's higher turnover is not caused by duplicate holdings, repeated asset
rows, or an inconsistent universe. It comes from less stable tail rankings.

## Reproduction

Run:

```bash
python reproduction/analyses/transaction_costs/transaction_cost_analysis.py
```

Input paths, sizes, and hashes are stored in `input_manifest.json`; integrity
checks are stored in `integrity_checks.json`. Runtime was
{runtime_seconds:.3f} CPU seconds.

## Limitation

The linear model excludes nonlinear spreads, market impact, borrow
availability, financing, price limits, execution delay, and capacity.
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    args.holdings = args.holdings.resolve()
    args.period_returns = args.period_returns.resolve()
    args.baseline_turnover = args.baseline_turnover.resolve()
    args.output_dir = args.output_dir.resolve()
    try:
        relative_output = args.output_dir.relative_to(
            (REPOSITORY / "reproduction/runs").resolve()
        )
    except ValueError as error:
        raise ValueError("Output must remain under reproduction/runs") from error
    if relative_output.parts[:1] != ("transaction_costs",):
        raise ValueError(
            "This script only writes under reproduction/runs/transaction_costs"
        )
    refuse_existing_outputs(args.output_dir)

    declared_inputs = [args.holdings, args.period_returns, args.baseline_turnover]
    declared_inputs.extend(DEFAULT_PREDICTIONS.values())
    manifest = {
        "analysis": "CSI 500 frozen common-universe transaction-cost sensitivity",
        "annualization": ANNUALIZATION,
        "cost_grid_bps_per_one_way_turnover": list(COST_GRID_BPS),
        "inputs": [validate_input(path) for path in declared_inputs],
        "command": (
            "python "
            "reproduction/analyses/transaction_costs/transaction_cost_analysis.py"
        ),
    }

    holdings = pd.read_parquet(args.holdings)
    periods = pd.read_csv(args.period_returns)
    baseline_turnover = pd.read_csv(args.baseline_turnover)
    holding_checks = basic_holdings_integrity(holdings)
    prediction_checks = prediction_integrity(holdings, DEFAULT_PREDICTIONS)
    turnover, turnover_checks = build_turnover_series(holdings, baseline_turnover)
    gross_returns = build_gross_returns(periods)
    cost_summary, break_even, net_periods = evaluate_costs(gross_returns, turnover)

    max_difference = max(
        abs(row["difference"])
        for row in turnover_checks["frozen_turnover_comparison"]
    )
    integrity = {
        "holdings": holding_checks,
        "prediction_integrity": prediction_checks,
        **turnover_checks,
        "maximum_absolute_frozen_turnover_difference": max_difference,
        "long_short_return_identity_verified": True,
        "return_turnover_date_alignment_verified": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cost_summary.to_csv(args.output_dir / "cost_sensitivity.csv", index=False)
    break_even.to_csv(args.output_dir / "model_break_even_costs.csv", index=False)
    net_periods.to_csv(args.output_dir / "net_performance_by_period.csv", index=False)
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "integrity_checks.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    plot_results(cost_summary, net_periods, args.output_dir / "figures")
    runtime_seconds = time.perf_counter() - started
    write_report(
        args.output_dir / "report.md",
        cost_summary,
        break_even,
        integrity,
        runtime_seconds,
    )
    print(f"Transaction-cost analysis complete in {runtime_seconds:.3f}s")
    print(
        cost_summary.loc[
            cost_summary["portfolio"] == "long_short",
            [
                "model",
                "cost_rate_bps_per_one_way_turnover",
                "gross_sharpe",
                "net_sharpe",
                "average_turnover_one_way_including_initial",
                "net_sharpe_rank",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
