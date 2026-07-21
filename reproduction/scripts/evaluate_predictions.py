#!/usr/bin/env python3
"""Evaluate every reproduced model with one corrected, raw-return pipeline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finpfn-matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from reproduction_common import MARKET_CONFIG, load_panel


CSI_OFFICIAL_REGIME_WINDOWS = [
    ("2022-03-11", "2022-03-14"),
    ("2022-03-14", "2022-03-17"),
    ("2022-04-08", "2022-04-11"),
    ("2022-04-22", "2022-04-25"),
    ("2022-08-01", "2022-08-04"),
    ("2022-10-21", "2022-10-24"),
    ("2022-10-28", "2022-10-31"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--market", choices=sorted(MARKET_CONFIG), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument(
        "--date-policy",
        choices=["intersection", "model_available"],
        default="intersection",
        help="Use common dates for headline comparability or each model's available dates",
    )
    parser.add_argument(
        "--universe-policy",
        choices=["intersection", "model_available"],
        default="intersection",
        help="Use common asset-dates for headline comparability or each model's coverage",
    )
    return parser.parse_args()


def read_prediction_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix in {".csv", ".gz"}:
        frame = pd.read_csv(path, compression="infer")
    else:
        raise ValueError(f"Unsupported prediction file: {path.name}")
    required = {"model", "seed", "date", "id", "prediction", "status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing prediction columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    if "context_date" in frame:
        frame["context_date"] = pd.to_datetime(frame["context_date"], errors="coerce")
    for column in ["prediction", "prediction_mean", "target_group_z"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["source_file"] = path.name
    if "target_group_z" not in frame:
        frame["target_group_z"] = np.nan
    return frame


def collapse_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        predictions.groupby(["model", "seed"], dropna=False)
        .agg(
            rows=("id", "size"),
            failed_rows=("status", lambda values: int((values != "ok").sum())),
        )
        .reset_index()
    )
    valid = predictions.loc[
        (predictions["status"] == "ok") & predictions["prediction"].notna()
    ].copy()
    keys = ["model", "seed", "date", "id"]
    duplicate_counts = valid.groupby(keys).size().rename("prediction_repetitions")
    valid = (
        valid.groupby(keys, as_index=False, sort=True)
        .agg(
            prediction=("prediction", "mean"),
            target_group_z=("target_group_z", "mean"),
        )
        .merge(duplicate_counts.reset_index(), on=keys, how="left", validate="one_to_one")
    )
    duplicate_summary = (
        valid.groupby(["model", "seed"])["prediction_repetitions"]
        .agg(
            unique_predictions="size",
            repeated_asset_predictions=lambda values: int((values > 1).sum()),
            maximum_repetitions="max",
        )
        .reset_index()
    )
    return valid, counts.merge(duplicate_summary, on=["model", "seed"], how="left")


def apply_date_policy(
    predictions: pd.DataFrame, policy: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    date_sets = {
        f"{model}/seed{seed}": set(group["date"])
        for (model, seed), group in predictions.groupby(["model", "seed"])
    }
    if not date_sets:
        raise ValueError("No successful predictions are available")
    common_dates = set.intersection(*date_sets.values())
    record = {
        "date_policy": policy,
        "dates_by_model_seed": {
            key: len(values) for key, values in sorted(date_sets.items())
        },
        "intersection_dates": len(common_dates),
        "intersection_date_min": min(common_dates).isoformat() if common_dates else None,
        "intersection_date_max": max(common_dates).isoformat() if common_dates else None,
    }
    if policy == "intersection":
        if not common_dates:
            raise ValueError("Prediction files have no common evaluation dates")
        predictions = predictions.loc[predictions["date"].isin(common_dates)].copy()
    return predictions, record


def apply_universe_policy(
    predictions: pd.DataFrame, policy: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    key_sets = {
        f"{model}/seed{seed}": set(zip(group["date"], group["id"], strict=True))
        for (model, seed), group in predictions.groupby(["model", "seed"])
    }
    common_keys = set.intersection(*key_sets.values())
    record = {
        "universe_policy": policy,
        "asset_dates_by_model_seed": {
            key: len(values) for key, values in sorted(key_sets.items())
        },
        "intersection_asset_dates": len(common_keys),
    }
    if policy == "intersection":
        if not common_keys:
            raise ValueError("Prediction files have no common asset-date evaluation universe")
        keys = pd.MultiIndex.from_frame(predictions[["date", "id"]])
        common_index = pd.MultiIndex.from_tuples(common_keys, names=["date", "id"])
        predictions = predictions.loc[keys.isin(common_index)].copy()
    return predictions, record


def spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 2 or x.loc[valid].nunique() < 2 or y.loc[valid].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(x.loc[valid], y.loc[valid]).statistic)


def compute_ic(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, date), group in merged.groupby(["model", "seed", "date"]):
        rows.append(
            {
                "market": group["market"].iat[0],
                "model": model,
                "seed": seed,
                "date": date,
                "n_assets": len(group),
                "ic_raw_target": spearman(group["raw_target"], group["prediction"]),
                "ic_group_standardized_target": spearman(
                    group["target_group_z"], group["prediction"]
                ),
                "raw_vs_group_target_rank_correlation": spearman(
                    group["raw_target"], group["target_group_z"]
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(["model", "seed", "date"])
    result["year"] = result["date"].dt.year
    return result


def summarize_ic(
    ic: pd.DataFrame,
    coverage: pd.DataFrame,
    market: str,
) -> pd.DataFrame:
    rows = []
    for (model, seed), group in ic.groupby(["model", "seed"]):
        values = group["ic_raw_target"]
        mean_ic = values.mean()
        std_ic = values.std(ddof=1)
        row = {
            "market": market,
            "row_type": "seed",
            "model": model,
            "seed": seed,
            "n_dates": int(values.count()),
            "mean_ic": mean_ic,
            "ic_std_ddof1": std_ic,
            "ir": mean_ic / std_ic,
            "mean_assets_per_date": group["n_assets"].mean(),
        }
        coverage_row = coverage.loc[
            (coverage["model"] == model) & (coverage["seed"] == seed)
        ]
        if len(coverage_row) == 1:
            row.update(
                {
                    "prediction_coverage": coverage_row["prediction_coverage"].iat[0],
                    "failed_rows": coverage_row["failed_rows"].iat[0],
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    aggregate_rows = []
    metric_columns = [
        "n_dates",
        "mean_ic",
        "ic_std_ddof1",
        "ir",
        "mean_assets_per_date",
        "prediction_coverage",
        "failed_rows",
    ]
    for model, group in summary.groupby("model"):
        row = {
            "market": market,
            "row_type": "seed_mean",
            "model": model,
            "seed": np.nan,
        }
        for column in metric_columns:
            row[column] = group[column].mean()
        row["ir_across_seed_std_ddof1"] = group["ir"].std(ddof=1)
        row["mean_ic_across_seed_std_ddof1"] = group["mean_ic"].std(ddof=1)
        aggregate_rows.append(row)
    return pd.concat([summary, pd.DataFrame(aggregate_rows)], ignore_index=True)


def compute_coverage(
    predictions: pd.DataFrame,
    raw: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    available_by_date = raw.groupby("date").size()
    rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"]):
        denominator = int(available_by_date.reindex(group["date"].unique()).sum())
        status_row = status.loc[(status["model"] == model) & (status["seed"] == seed)]
        rows.append(
            {
                "model": model,
                "seed": seed,
                "prediction_rows": int(len(group)),
                "available_asset_dates": denominator,
                "prediction_coverage": len(group) / denominator if denominator else np.nan,
                "failed_rows": int(status_row["failed_rows"].iat[0]),
                "repeated_asset_predictions": int(
                    status_row["repeated_asset_predictions"].fillna(0).iat[0]
                ),
                "maximum_repetitions": int(
                    status_row["maximum_repetitions"].fillna(0).iat[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def assign_deciles(group: pd.DataFrame) -> pd.DataFrame:
    group = group.dropna(subset=["prediction", "raw_return_percentage_points"]).copy()
    if len(group) < 10:
        group["decile"] = np.nan
        return group
    group = group.sort_values(["prediction", "id"], kind="stable").reset_index(drop=True)
    group["decile"] = np.floor(np.arange(len(group)) * 10 / len(group)).astype(int) + 1
    group["cross_sectional_excess_return"] = (
        group["raw_return_percentage_points"]
        - group["raw_return_percentage_points"].mean()
    )
    return group


def sharpe(values: pd.Series, annualization: int) -> float:
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(values.mean() / standard_deviation * np.sqrt(annualization))


def one_way_turnover(holdings: pd.DataFrame) -> float:
    previous: dict[object, float] | None = None
    observations = []
    for _, group in holdings.groupby("date", sort=True):
        weight = 1.0 / len(group)
        current = {identifier: weight for identifier in group["id"]}
        if previous is not None:
            identifiers = set(previous) | set(current)
            observations.append(
                0.5
                * sum(abs(current.get(item, 0.0) - previous.get(item, 0.0)) for item in identifiers)
            )
        previous = current
    return float(np.mean(observations)) if observations else np.nan


def compute_portfolios(
    merged: pd.DataFrame,
    annualization: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assigned = (
        merged.groupby(["model", "seed", "date"], group_keys=True)
        .apply(assign_deciles, include_groups=False)
        .reset_index(level=[0, 1, 2])
        .reset_index(drop=True)
        .dropna(subset=["decile"])
    )
    assigned["decile"] = assigned["decile"].astype(int)
    return_columns = {
        "raw": "raw_return_percentage_points",
        "artifact_cross_sectional_excess": "cross_sectional_excess_return",
    }
    period_frames = []
    metric_rows = []
    turnover_rows = []
    for (model, seed), model_holdings in assigned.groupby(["model", "seed"]):
        for decile, decile_holdings in model_holdings.groupby("decile"):
            turnover_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "decile": decile,
                    "mean_one_way_turnover": one_way_turnover(decile_holdings),
                }
            )
        for basis, return_column in return_columns.items():
            period = (
                model_holdings.groupby(["date", "decile"])[return_column]
                .mean()
                .unstack()
                .reindex(columns=range(1, 11))
                .dropna()
            )
            period.columns = [f"decile_{column}" for column in period.columns]
            period["long_short"] = period["decile_10"] - period["decile_1"]
            period = period.reset_index()
            period["model"] = model
            period["seed"] = seed
            period["return_basis"] = basis
            period_frames.append(period)

            bottom = period["decile_1"]
            top = period["decile_10"]
            long_short = period["long_short"]
            bottom_sharpe = sharpe(bottom, annualization)
            top_sharpe = sharpe(top, annualization)
            metric_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "return_basis": basis,
                    "annualization": annualization,
                    "n_dates": len(period),
                    "bottom_decile_sharpe": bottom_sharpe,
                    "top_decile_sharpe": top_sharpe,
                    "notebook_sharpe_spread": top_sharpe - bottom_sharpe,
                    "primary_long_short_sharpe": sharpe(long_short, annualization),
                    "bottom_cumulative_percentage_points": bottom.sum(),
                    "top_cumulative_percentage_points": top.sum(),
                    "long_short_cumulative_percentage_points": long_short.sum(),
                }
            )
    periods = pd.concat(period_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    turnover = pd.DataFrame(turnover_rows)
    holdings = assigned[["model", "seed", "date", "id", "decile"]]
    return periods, metrics, turnover, holdings


def summarize_subperiods(ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, year), group in ic.groupby(["model", "seed", "year"]):
        values = group["ic_raw_target"]
        standard_deviation = values.std(ddof=1)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "subperiod": str(year),
                "n_dates": int(values.count()),
                "mean_ic": values.mean(),
                "ic_std_ddof1": standard_deviation,
                "ir": values.mean() / standard_deviation,
            }
        )
    return pd.DataFrame(rows)


def regime_mask(dates: pd.Series) -> pd.Series:
    mask = pd.Series(False, index=dates.index)
    for start, end in CSI_OFFICIAL_REGIME_WINDOWS:
        mask |= dates.between(pd.Timestamp(start), pd.Timestamp(end))
    return mask


def compute_regimes(ic: pd.DataFrame, market: str) -> pd.DataFrame:
    if market == "us":
        return pd.DataFrame(
            [
                {
                    "market": market,
                    "model": "ALL",
                    "seed": np.nan,
                    "period": "official_regime_unavailable",
                    "definition": "VIX source and exact official episode dates are absent",
                    "n_dates": 0,
                    "mean_ic": np.nan,
                    "ic_std_ddof1": np.nan,
                    "ir": np.nan,
                }
            ]
        )
    rows = []
    mask = regime_mask(ic["date"])
    for (model, seed), group in ic.groupby(["model", "seed"]):
        group_mask = mask.loc[group.index]
        for label, selected in {
            "overall": group,
            "official_regime": group.loc[group_mask],
            "outside_official_regime": group.loc[~group_mask],
        }.items():
            values = selected["ic_raw_target"]
            standard_deviation = values.std(ddof=1)
            rows.append(
                {
                    "market": market,
                    "model": model,
                    "seed": seed,
                    "period": label,
                    "definition": "released notebook seven inclusive CSI windows",
                    "n_dates": int(values.count()),
                    "mean_ic": values.mean(),
                    "ic_std_ddof1": standard_deviation,
                    "ir": values.mean() / standard_deviation,
                }
            )
    return pd.DataFrame(rows)


def add_reported_ir(comparison: pd.DataFrame, market: str) -> pd.DataFrame:
    targets_path = Path(__file__).resolve().parents[1] / "configs/reported_targets.json"
    targets = json.loads(targets_path.read_text())["ir"][market]
    comparison["paper_reported_ir"] = comparison["model"].map(targets)
    comparison["ir_minus_paper"] = comparison["ir"] - comparison["paper_reported_ir"]
    return comparison


def save_figures(ic: pd.DataFrame, periods: pd.DataFrame, figures: Path, market: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    ic_plot = ic.groupby(["model", "date"])["ic_raw_target"].mean().unstack("model")
    axis = ic_plot.plot(figsize=(11, 5), linewidth=1.1, alpha=0.85)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(title=f"{market}: cross-sectional Spearman IC", ylabel="IC", xlabel="")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / f"ic_timeseries_{market}.png", dpi=180)
    plt.close(axis.figure)

    excess = periods.loc[periods["return_basis"] == "artifact_cross_sectional_excess"]
    for model, group in excess.groupby("model"):
        daily = group.groupby("date")[[f"decile_{i}" for i in range(1, 11)]].mean()
        axis = daily.cumsum().plot(figsize=(10, 6), linewidth=1.1, colormap="viridis")
        axis.set(
            title=f"{market} {model}: cumulative decile excess returns",
            ylabel="Percentage points (arithmetic sum)",
            xlabel="",
        )
        axis.figure.tight_layout()
        slug = model.lower().replace(" ", "_")
        axis.figure.savefig(figures / f"cumulative_deciles_{market}_{slug}.png", dpi=180)
        plt.close(axis.figure)

        long_short = group.groupby("date")["long_short"].mean().cumsum()
        axis = long_short.plot(figsize=(10, 5), linewidth=1.5)
        axis.set(
            title=f"{market} {model}: cumulative top-minus-bottom return",
            ylabel="Percentage points (arithmetic sum)",
            xlabel="",
        )
        axis.figure.tight_layout()
        axis.figure.savefig(
            figures / f"cumulative_long_short_{market}_{slug}.png", dpi=180
        )
        plt.close(axis.figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(
        [read_prediction_file(path) for path in args.predictions], ignore_index=True
    )
    valid, status = collapse_predictions(predictions)
    valid, date_scope = apply_date_policy(valid, args.date_policy)

    raw = load_panel(
        args.dataset,
        args.market,
        split="test",
        columns=["date", "id", "target"],
    ).rename(columns={"target": "raw_target"})
    if raw.duplicated(["date", "id"]).any():
        raise ValueError("Raw panel contains duplicate asset-date rows")
    source_coverage = compute_coverage(valid, raw, status)
    valid, universe_scope = apply_universe_policy(valid, args.universe_policy)
    evaluation_coverage = compute_coverage(valid, raw, status)
    coverage = source_coverage.merge(
        evaluation_coverage[
            ["model", "seed", "prediction_rows", "prediction_coverage"]
        ].rename(
            columns={
                "prediction_rows": "evaluation_prediction_rows",
                "prediction_coverage": "evaluation_prediction_coverage",
            }
        ),
        on=["model", "seed"],
        how="left",
        validate="one_to_one",
    )
    merged = valid.merge(raw, on=["date", "id"], how="left", validate="many_to_one")
    if merged["raw_target"].isna().any():
        raise ValueError("Some predictions do not match the official parquet target")
    merged["market"] = args.market
    merged["raw_return_percentage_points"] = (
        merged["raw_target"]
        * MARKET_CONFIG[args.market]["return_to_percentage_points"]
    )

    ic = compute_ic(merged)
    comparison = add_reported_ir(summarize_ic(ic, coverage, args.market), args.market)
    subperiods = summarize_subperiods(ic)
    regimes = compute_regimes(ic, args.market)
    periods, portfolio_metrics, turnover, holdings = compute_portfolios(
        merged, MARKET_CONFIG[args.market]["annualization"]
    )
    label_diagnostics = (
        ic.groupby(["model", "seed"])
        .agg(
            dates_with_group_target=("ic_group_standardized_target", "count"),
            mean_ic_group_standardized_target=(
                "ic_group_standardized_target",
                "mean",
            ),
            mean_raw_vs_group_target_rank_correlation=(
                "raw_vs_group_target_rank_correlation",
                "mean",
            ),
        )
        .reset_index()
    )

    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    ic.to_csv(args.output_dir / "ic_by_period.csv", index=False)
    subperiods.to_csv(args.output_dir / "ic_by_subperiod.csv", index=False)
    portfolio_metrics.to_csv(args.output_dir / "portfolio_metrics.csv", index=False)
    periods.to_csv(args.output_dir / "decile_returns_by_period.csv", index=False)
    regimes.to_csv(args.output_dir / "regime_metrics.csv", index=False)
    coverage.to_csv(args.output_dir / "prediction_coverage.csv", index=False)
    turnover.to_csv(args.output_dir / "turnover_by_decile.csv", index=False)
    holdings.to_parquet(args.output_dir / "decile_holdings.parquet", index=False)
    label_diagnostics.to_csv(args.output_dir / "label_ranking_diagnostics.csv", index=False)
    (args.output_dir / "evaluation_scope.json").write_text(
        json.dumps(
            {"date": date_scope, "universe": universe_scope},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    save_figures(ic, periods, args.figures_dir, args.market)

    print(comparison.to_string(index=False))
    print("\nPrimary portfolio metric: primary_long_short_sharpe")
    print(portfolio_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
