#!/usr/bin/env python3
"""Validation-only calibration audit for saved TabPFN ensemble outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finpfn-next-phase-mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY / "reproduction/next_phase/uncertainty"
VALIDATION_START = pd.Timestamp("2021-01-01")
VALIDATION_END = pd.Timestamp("2022-01-01")
ANNUALIZATION = 240
COVERAGE_GRID = (1.0, 0.8, 0.6, 0.4, 0.2)
CALIBRATION_QUANTILES = 5
FIXED_COST_BPS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, np.nan, len(pair)
    result = stats.spearmanr(pair["x"], pair["y"])
    return float(result.statistic), float(result.pvalue), len(pair)


def zscore(series: pd.Series) -> pd.Series:
    standard_deviation = series.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return series * 0.0
    return (series - series.mean()) / standard_deviation


def rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def qcut_within_date(series: pd.Series) -> pd.Series:
    ranks = series.rank(method="first", pct=True)
    return np.ceil(ranks * CALIBRATION_QUANTILES).clip(1, CALIBRATION_QUANTILES).astype(int)


def load_and_collapse(path: Path, raw_returns: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "model",
        "seed",
        "split",
        "date",
        "id",
        "prediction",
        "prediction_mean",
        "predictive_std",
        "predictive_q10",
        "predictive_q90",
        "status",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing {missing}")
    member_columns = sorted(
        [column for column in frame if column.startswith("member_") and column.endswith("_median")],
        key=lambda name: int(name.split("_")[1]),
    )
    if len(member_columns) != 8:
        raise ValueError(f"{path.name} has {len(member_columns)} member median columns")
    if set(frame["split"].dropna().unique()) != {"validation"}:
        raise ValueError("Uncertainty calibration is restricted to validation artifacts")
    frame["date"] = pd.to_datetime(frame["date"])
    if not ((frame["date"] >= VALIDATION_START) & (frame["date"] < VALIDATION_END)).all():
        raise ValueError("Prediction dates fall outside the declared validation period")
    if (frame["status"] != "ok").any():
        raise ValueError("Validation calibration requires zero failed inference rows")

    numeric = [
        "prediction",
        "prediction_mean",
        "predictive_std",
        "predictive_q10",
        "predictive_q90",
    ] + member_columns
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite uncertainty output in {path.name}")

    members = frame[member_columns].to_numpy(dtype=float)
    row_member_mean = members.mean(axis=1)
    row_member_var_population = members.var(axis=1, ddof=0)
    frame["within_group_ensemble_sd"] = members.std(axis=1, ddof=1)
    member_median = np.median(members, axis=1)
    frame["ensemble_mad"] = np.median(
        np.abs(members - member_median[:, np.newaxis]), axis=1
    )
    frame["ensemble_iqr"] = np.quantile(members, 0.75, axis=1) - np.quantile(
        members, 0.25, axis=1
    )
    frame["member_first_moment"] = row_member_mean
    frame["member_second_moment"] = row_member_var_population + row_member_mean**2
    frame["mean_median_disagreement"] = np.abs(
        frame["prediction_mean"] - frame["prediction"]
    )
    frame["predictive_interval_width"] = (
        frame["predictive_q90"] - frame["predictive_q10"]
    )

    aggregation: dict[str, tuple[str, str]] = {
        "prediction": ("prediction", "mean"),
        "prediction_mean": ("prediction_mean", "mean"),
        "within_group_ensemble_sd": ("within_group_ensemble_sd", "mean"),
        "ensemble_mad": ("ensemble_mad", "mean"),
        "ensemble_iqr": ("ensemble_iqr", "mean"),
        "mean_median_disagreement": ("mean_median_disagreement", "mean"),
        "predictive_interval_width": ("predictive_interval_width", "mean"),
        "predictive_std": ("predictive_std", "mean"),
        "member_first_moment": ("member_first_moment", "mean"),
        "member_second_moment": ("member_second_moment", "mean"),
        "group_composition_sd": ("prediction", "std"),
        "group_occurrences": ("prediction", "size"),
    }
    for column in member_columns:
        aggregation[f"collapsed_{column}"] = (column, "mean")
    collapsed = frame.groupby(
        ["model", "seed", "date", "id"], as_index=False, sort=True
    ).agg(**aggregation)
    collapsed["group_composition_sd"] = collapsed["group_composition_sd"].fillna(0.0)
    total_variance = (
        collapsed["member_second_moment"] - collapsed["member_first_moment"] ** 2
    ).clip(lower=0.0)
    collapsed["total_member_sd"] = np.sqrt(total_variance)
    collapsed = collapsed.drop(columns=["member_first_moment", "member_second_moment"])

    collapsed = collapsed.merge(
        raw_returns, on=["date", "id"], how="left", validate="many_to_one"
    )
    if collapsed["raw_return_decimal"].isna().any():
        raise ValueError("Some validation predictions do not match the raw-return panel")

    collapsed["prediction_rank"] = collapsed.groupby("date")["prediction"].transform(
        rank_pct
    )
    collapsed["return_rank"] = collapsed.groupby("date")[
        "raw_return_decimal"
    ].transform(rank_pct)
    collapsed["absolute_rank_error"] = np.abs(
        collapsed["prediction_rank"] - collapsed["return_rank"]
    )
    collapsed["prediction_z"] = collapsed.groupby("date")["prediction"].transform(zscore)
    collapsed["return_z"] = collapsed.groupby("date")["raw_return_decimal"].transform(
        zscore
    )
    collapsed["absolute_cross_sectional_z_error"] = np.abs(
        collapsed["prediction_z"] - collapsed["return_z"]
    )
    predicted_long = collapsed["prediction_rank"] > 0.9
    predicted_short = collapsed["prediction_rank"] <= 0.1
    realized_long = collapsed["return_rank"] > 0.9
    realized_short = collapsed["return_rank"] <= 0.1
    collapsed["predicted_tail"] = predicted_long | predicted_short
    collapsed["correct_tail_membership"] = np.where(
        predicted_long,
        realized_long,
        np.where(predicted_short, realized_short, np.nan),
    )

    rank_columns = []
    for member_column in member_columns:
        collapsed_name = f"collapsed_{member_column}"
        rank_name = f"rank_{member_column}"
        collapsed[rank_name] = collapsed.groupby("date")[collapsed_name].transform(rank_pct)
        rank_columns.append(rank_name)
    collapsed["member_rank_disagreement"] = collapsed[rank_columns].std(axis=1, ddof=1)
    collapsed = collapsed.drop(columns=rank_columns)

    all_dates = pd.Index(sorted(collapsed["date"].unique()))
    date_index = {date: index for index, date in enumerate(all_dates)}
    current = collapsed[["id", "date", "prediction_rank"]].copy()
    current["date_index"] = current["date"].map(date_index)
    following = current.rename(
        columns={"prediction_rank": "next_prediction_rank", "date_index": "next_index"}
    )[["id", "next_index", "next_prediction_rank"]]
    current["next_index"] = current["date_index"] + 1
    migration = current.merge(
        following, on=["id", "next_index"], how="left", validate="many_to_one"
    )
    migration["rank_instability_next"] = np.abs(
        migration["next_prediction_rank"] - migration["prediction_rank"]
    )
    collapsed = collapsed.merge(
        migration[["id", "date", "rank_instability_next"]],
        on=["id", "date"],
        how="left",
        validate="one_to_one",
    )
    return collapsed


def uncertainty_signals(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "total_member_sd",
        "within_group_ensemble_sd",
        "ensemble_mad",
        "ensemble_iqr",
        "member_rank_disagreement",
        "group_composition_sd",
        "mean_median_disagreement",
        "predictive_interval_width",
        "predictive_std",
    ]
    return [column for column in candidates if column in frame and frame[column].notna().any()]


def calibration(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_rows = []
    quantile_rows = []
    outcomes = [
        "absolute_cross_sectional_z_error",
        "absolute_rank_error",
        "rank_instability_next",
    ]
    for (model, seed), model_frame in frame.groupby(["model", "seed"], sort=True):
        date_ic = model_frame.groupby("date").apply(
            lambda part: spearman(part["prediction"], part["raw_return_decimal"])[0],
            include_groups=False,
        )
        for signal in uncertainty_signals(model_frame):
            for outcome in outcomes:
                coefficient, pvalue, count = spearman(model_frame[signal], model_frame[outcome])
                by_date = model_frame.groupby("date").apply(
                    lambda part: spearman(part[signal], part[outcome])[0],
                    include_groups=False,
                )
                calibration_rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "uncertainty_signal": signal,
                        "outcome": outcome,
                        "n_observations": count,
                        "pooled_spearman": coefficient,
                        "pooled_pvalue": pvalue,
                        "mean_datewise_spearman": float(by_date.mean()),
                        "median_datewise_spearman": float(by_date.median()),
                        "n_dates": int(by_date.notna().sum()),
                    }
                )

            date_uncertainty = model_frame.groupby("date")[signal].mean()
            coefficient, pvalue, count = spearman(date_uncertainty, -date_ic)
            calibration_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "uncertainty_signal": signal,
                    "outcome": "realized_forward_ic_deterioration",
                    "n_observations": count,
                    "pooled_spearman": coefficient,
                    "pooled_pvalue": pvalue,
                    "mean_datewise_spearman": np.nan,
                    "median_datewise_spearman": np.nan,
                    "n_dates": count,
                }
            )

            temporary = model_frame.copy()
            temporary["uncertainty_quantile"] = temporary.groupby("date")[signal].transform(
                qcut_within_date
            )
            for quantile, part in temporary.groupby("uncertainty_quantile", sort=True):
                tail = part.loc[part["predicted_tail"]]
                quantile_rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "uncertainty_signal": signal,
                        "uncertainty_quantile": int(quantile),
                        "n_asset_dates": int(len(part)),
                        "mean_absolute_cross_sectional_z_error": float(
                            part["absolute_cross_sectional_z_error"].mean()
                        ),
                        "mean_absolute_rank_error": float(part["absolute_rank_error"].mean()),
                        "mean_next_rank_instability": float(
                            part["rank_instability_next"].mean()
                        ),
                        "predicted_tail_count": int(len(tail)),
                        "tail_selection_precision": float(
                            tail["correct_tail_membership"].mean()
                        ),
                    }
                )
    return pd.DataFrame(calibration_rows), pd.DataFrame(quantile_rows)


def sharpe(values: pd.Series) -> float:
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(values.mean() / standard_deviation * np.sqrt(ANNUALIZATION))


def turnover_series(holdings: dict[pd.Timestamp, set[object]]) -> pd.Series:
    previous: dict[object, float] | None = None
    values: dict[pd.Timestamp, float] = {}
    for date in sorted(holdings):
        identifiers = holdings[date]
        current = {identifier: 1.0 / len(identifiers) for identifier in identifiers}
        if previous is None:
            values[date] = 1.0
        else:
            combined = set(previous) | set(current)
            values[date] = 0.5 * sum(
                abs(current.get(item, 0.0) - previous.get(item, 0.0))
                for item in combined
            )
        previous = current
    return pd.Series(values, dtype=float).sort_index()


def coverage_performance(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), model_frame in frame.groupby(["model", "seed"], sort=True):
        for signal in uncertainty_signals(model_frame):
            for requested_coverage in COVERAGE_GRID:
                selected_frames = []
                for date, part in model_frame.groupby("date", sort=True):
                    keep = max(10, int(np.floor(len(part) * requested_coverage)))
                    selected = part.sort_values([signal, "id"], kind="stable").head(keep).copy()
                    selected = selected.sort_values(["prediction", "id"], kind="stable")
                    selected["decile"] = (
                        np.floor(np.arange(len(selected)) * 10 / len(selected)).astype(int)
                        + 1
                    )
                    selected_frames.append(selected)
                selected = pd.concat(selected_frames, ignore_index=True)
                date_ic = selected.groupby("date").apply(
                    lambda part: spearman(part["prediction"], part["raw_return_decimal"])[0],
                    include_groups=False,
                )
                ic_std = date_ic.std(ddof=1)
                period = (
                    selected.groupby(["date", "decile"])["raw_return_decimal"]
                    .mean()
                    .unstack()
                    .dropna(subset=[1, 10])
                )
                gross_long_short = period[10] - period[1]
                long_holdings = {
                    date: set(part.loc[part["decile"] == 10, "id"])
                    for date, part in selected.groupby("date", sort=True)
                }
                short_holdings = {
                    date: set(part.loc[part["decile"] == 1, "id"])
                    for date, part in selected.groupby("date", sort=True)
                }
                total_turnover = turnover_series(long_holdings) + turnover_series(short_holdings)
                total_turnover = total_turnover.reindex(gross_long_short.index)
                net_long_short = gross_long_short - total_turnover * FIXED_COST_BPS / 10_000
                rows.append(
                    {
                        "model": model,
                        "seed": int(seed),
                        "uncertainty_signal": signal,
                        "requested_coverage": requested_coverage,
                        "realized_coverage": float(len(selected) / len(model_frame)),
                        "n_asset_dates": int(len(selected)),
                        "n_dates": int(len(date_ic)),
                        "mean_ic": float(date_ic.mean()),
                        "ic_std_ddof1": float(ic_std),
                        "ir": float(date_ic.mean() / ic_std) if ic_std else np.nan,
                        "gross_long_short_mean_return_decimal": float(
                            gross_long_short.mean()
                        ),
                        "gross_long_short_sharpe": sharpe(gross_long_short),
                        "average_total_one_way_turnover_including_initial": float(
                            total_turnover.mean()
                        ),
                        "fixed_cost_bps": FIXED_COST_BPS,
                        "net_long_short_mean_return_decimal": float(
                            net_long_short.mean()
                        ),
                        "net_long_short_sharpe": sharpe(net_long_short),
                        "mean_long_positions": float(
                            selected.loc[selected["decile"] == 10]
                            .groupby("date")
                            .size()
                            .mean()
                        ),
                        "mean_short_positions": float(
                            selected.loc[selected["decile"] == 1]
                            .groupby("date")
                            .size()
                            .mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plots(
    quantiles: pd.DataFrame, coverage: pd.DataFrame, figures_dir: Path
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=False)
    for model in sorted(quantiles["model"].unique()):
        part = quantiles.loc[quantiles["model"] == model]
        fig, axis = plt.subplots(figsize=(9, 5))
        for signal, signal_frame in part.groupby("uncertainty_signal", sort=True):
            axis.plot(
                signal_frame["uncertainty_quantile"],
                signal_frame["mean_absolute_rank_error"],
                marker="o",
                label=signal,
            )
        axis.set(
            xlabel="Within-date uncertainty quintile (1=lowest)",
            ylabel="Mean absolute percentile-rank error",
            title=f"{model}: empirical uncertainty-error curve (validation)",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{model.lower()}_error_by_uncertainty_quintile.png", dpi=180)
        plt.close(fig)

        model_coverage = coverage.loc[coverage["model"] == model]
        fig, axis = plt.subplots(figsize=(9, 5))
        for signal, signal_frame in model_coverage.groupby(
            "uncertainty_signal", sort=True
        ):
            axis.plot(
                signal_frame["realized_coverage"],
                signal_frame["gross_long_short_sharpe"],
                marker="o",
                label=signal,
            )
        axis.set(
            xlabel="Realized coverage",
            ylabel="Gross actual H-L Sharpe",
            title=f"{model}: coverage-performance curve (validation)",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{model.lower()}_coverage_performance.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    uncertainty_root = (REPOSITORY / "reproduction/next_phase/uncertainty").resolve()
    try:
        output_dir.relative_to(uncertainty_root)
    except ValueError as error:
        raise ValueError("Output must stay under reproduction/next_phase/uncertainty") from error
    outputs = [
        output_dir / "calibration_metrics.csv",
        output_dir / "coverage_performance.csv",
        output_dir / "uncertainty_quantile_metrics.csv",
        output_dir / "asset_date_uncertainty.parquet",
        output_dir / "evaluation_manifest.json",
    ]
    if any(path.exists() for path in outputs) or (
        (output_dir / "figures").exists() and any((output_dir / "figures").iterdir())
    ):
        raise FileExistsError("Refusing to overwrite uncertainty-evaluation outputs")

    dataset = args.dataset.resolve()
    raw_returns = pd.read_parquet(
        dataset,
        columns=["date", "id", "target"],
        filters=[("date", ">=", VALIDATION_START), ("date", "<", VALIDATION_END)],
    ).rename(columns={"target": "raw_return_decimal"})
    raw_returns["date"] = pd.to_datetime(raw_returns["date"])
    if raw_returns.duplicated(["date", "id"]).any():
        raise ValueError("Raw validation panel contains duplicate asset-date rows")

    frames = [load_and_collapse(path.resolve(), raw_returns) for path in args.predictions]
    asset_dates = pd.concat(frames, ignore_index=True)
    calibration_metrics, quantile_metrics = calibration(asset_dates)
    coverage_metrics = coverage_performance(asset_dates)

    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_metrics.to_csv(output_dir / "calibration_metrics.csv", index=False)
    coverage_metrics.to_csv(output_dir / "coverage_performance.csv", index=False)
    quantile_metrics.to_csv(output_dir / "uncertainty_quantile_metrics.csv", index=False)
    asset_dates.to_parquet(output_dir / "asset_date_uncertainty.parquet", index=False)
    plots(quantile_metrics, coverage_metrics, output_dir / "figures")
    manifest = {
        "analysis_split": "validation_only",
        "validation_start_inclusive": VALIDATION_START.isoformat(),
        "validation_end_exclusive": VALIDATION_END.isoformat(),
        "coverage_grid": list(COVERAGE_GRID),
        "calibration_uncertainty_quantiles": CALIBRATION_QUANTILES,
        "diagnostic_fixed_cost_bps_per_one_way_turnover": FIXED_COST_BPS,
        "common_target": "raw CSI return; cross-sectional z/rank used for errors",
        "inputs": [
            {
                "path": path.resolve().relative_to(REPOSITORY).as_posix(),
                "sha256": sha256(path.resolve()),
            }
            for path in args.predictions
        ]
        + [
            {
                "path": dataset.relative_to(REPOSITORY).as_posix(),
                "sha256": sha256(dataset),
            }
        ],
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(calibration_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
