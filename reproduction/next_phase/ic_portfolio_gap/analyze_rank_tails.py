#!/usr/bin/env python3
"""Explain the frozen CSI 500 IC-versus-portfolio gap without model tuning."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
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
OUTPUT = REPOSITORY / "reproduction/next_phase/ic_portfolio_gap"
DATASET = REPOSITORY / "30features_csi500.parquet"
COMMON_HOLDINGS = (
    REPOSITORY
    / "reproduction/results/csi500_all_models_notebook_exact/decile_holdings.parquet"
)
FROZEN_METRICS = (
    REPOSITORY
    / "reproduction/results/csi500_all_models_notebook_exact/portfolio_metrics.csv"
)
FROZEN_IC = (
    REPOSITORY
    / "reproduction/results/csi500_all_models_notebook_exact/ic_by_period.csv"
)
PREDICTIONS = {
    "FinPFN": (
        REPOSITORY
        / "reproduction/artifacts/csi500_notebook_exact/"
        "csi500_finpfn_seed42_notebook_with_replacement.parquet"
    ),
    "TabPFN": (
        REPOSITORY
        / "reproduction/artifacts/csi500_notebook_exact/"
        "csi500_tabpfn_seed42_notebook_with_replacement.parquet"
    ),
    "Ridge": (
        REPOSITORY
        / "reproduction/artifacts/predictions/csi500_baselines/"
        "csi500_ridge_seed42.parquet"
    ),
    "LightGBM": (
        REPOSITORY
        / "reproduction/artifacts/predictions/csi500_baselines/"
        "csi500_lightgbm_seed42.parquet"
    ),
}
MODELS = ["FinPFN", "Ridge", "LightGBM", "TabPFN"]
PERCENTILE_BINS = 20
TAIL_K = [10, 20, 40]
ANNUALIZATION = 240
EXPECTED_DATES = 301
EXPECTED_ASSET_DATES = 120_620
EXPECTED_MEAN_IC = {
    "FinPFN": 0.045596852913951554,
    "Ridge": 0.0374090905011318,
    "LightGBM": 0.03643387027794353,
    "TabPFN": -0.037758244755383065,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spearman(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if (
        valid.sum() < 3
        or x.loc[valid].nunique() < 2
        or y.loc[valid].nunique() < 2
    ):
        return np.nan
    return float(stats.spearmanr(x.loc[valid], y.loc[valid]).statistic)


def sharpe(values: pd.Series) -> float:
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(values.mean() / standard_deviation * np.sqrt(ANNUALIZATION))


def deterministic_ranked(part: pd.DataFrame) -> pd.DataFrame:
    part = part.sort_values(["prediction", "id"], kind="stable").copy()
    n_assets = len(part)
    positions = np.arange(n_assets)
    part["prediction_rank"] = (positions + 0.5) / n_assets
    part["prediction_percentile_bin"] = (
        np.floor(positions * PERCENTILE_BINS / n_assets).astype(int) + 1
    )
    part["prediction_decile"] = (
        np.floor(positions * 10 / n_assets).astype(int) + 1
    )
    realized = part.sort_values(["raw_return_decimal", "id"], kind="stable")
    realized_rank = pd.Series(
        (np.arange(n_assets) + 0.5) / n_assets,
        index=realized.index,
        dtype=float,
    )
    part["realized_rank"] = realized_rank.reindex(part.index)
    prediction_standard_deviation = part["prediction"].std(ddof=1)
    if prediction_standard_deviation == 0 or not np.isfinite(prediction_standard_deviation):
        part["prediction_z"] = 0.0
    else:
        part["prediction_z"] = (
            part["prediction"] - part["prediction"].mean()
        ) / prediction_standard_deviation
    part["absolute_rank_error"] = (
        part["prediction_rank"] - part["realized_rank"]
    ).abs()
    part["signed_prediction_return"] = (
        np.sign(part["prediction_z"]) * part["raw_return_decimal"]
    )
    return part


def load_panel() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    holdings = pd.read_parquet(COMMON_HOLDINGS, columns=["date", "id"])
    holdings["date"] = pd.to_datetime(holdings["date"])
    keys = (
        holdings.drop_duplicates(["date", "id"])
        .sort_values(["date", "id"])
        .reset_index(drop=True)
    )
    if len(keys) != EXPECTED_ASSET_DATES or keys["date"].nunique() != EXPECTED_DATES:
        raise ValueError("Frozen common universe does not match its recorded scope")

    raw = pd.read_parquet(DATASET, columns=["date", "id", "target"])
    raw["date"] = pd.to_datetime(raw["date"])
    keys = keys.merge(raw, on=["date", "id"], how="left", validate="one_to_one")
    if keys["target"].isna().any() or not np.isfinite(keys["target"]).all():
        raise ValueError("Raw target is missing on the common universe")
    keys = keys.rename(columns={"target": "raw_return_decimal"})

    repeated_sources: dict[str, pd.DataFrame] = {}
    model_frames = []
    for model, path in PREDICTIONS.items():
        source = pd.read_parquet(
            path,
            columns=["model", "seed", "date", "id", "group_id", "prediction", "status"],
        )
        source["date"] = pd.to_datetime(source["date"])
        source = source.loc[
            (source["model"] == model)
            & (source["seed"] == 42)
            & (source["status"] == "ok")
            & source["prediction"].notna()
        ].copy()
        repeated_sources[model] = source
        collapsed = (
            source.groupby(["date", "id"], as_index=False, sort=True)
            .agg(
                prediction=("prediction", "mean"),
                prediction_repetitions=("prediction", "size"),
                group_composition_prediction_sd=("prediction", "std"),
            )
        )
        collapsed["group_composition_prediction_sd"] = collapsed[
            "group_composition_prediction_sd"
        ].fillna(0.0)
        model_frame = keys.merge(
            collapsed, on=["date", "id"], how="left", validate="one_to_one"
        )
        if model_frame["prediction"].isna().any():
            raise ValueError(f"{model} does not cover the frozen common universe")
        model_frame["model"] = model
        model_frames.append(model_frame)
    panel = pd.concat(model_frames, ignore_index=True)
    panel = (
        panel.groupby(["model", "date"], group_keys=True, sort=False)
        .apply(deterministic_ranked, include_groups=False)
        .reset_index(level=[0, 1])
        .reset_index(drop=True)
    )
    return panel, repeated_sources


def percentile_curve(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.groupby(["model", "prediction_percentile_bin"], sort=True)
        .agg(
            n_asset_dates=("id", "size"),
            mean_prediction_rank=("prediction_rank", "mean"),
            mean_prediction=("prediction", "mean"),
            mean_raw_return_decimal=("raw_return_decimal", "mean"),
            median_raw_return_decimal=("raw_return_decimal", "median"),
            mean_realized_rank=("realized_rank", "mean"),
            mean_absolute_rank_error=("absolute_rank_error", "mean"),
            mean_signed_prediction_return_decimal=("signed_prediction_return", "mean"),
        )
        .reset_index()
    )


def rank_region_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    regions = {
        "bottom_20pct": lambda x: x["prediction_rank"] <= 0.2,
        "middle_20_to_80pct": lambda x: (x["prediction_rank"] > 0.2)
        & (x["prediction_rank"] <= 0.8),
        "top_20pct": lambda x: x["prediction_rank"] > 0.8,
        "bottom_decile": lambda x: x["prediction_decile"] == 1,
        "top_decile": lambda x: x["prediction_decile"] == 10,
        "full_universe": lambda x: pd.Series(True, index=x.index),
    }
    for (model, date), part in panel.groupby(["model", "date"], sort=False):
        for region, selector in regions.items():
            selected = part.loc[selector(part)]
            rows.append(
                {
                    "model": model,
                    "date": date,
                    "region": region,
                    "n_assets": len(selected),
                    "rank_ic": spearman(
                        selected["prediction"], selected["raw_return_decimal"]
                    ),
                    "mean_raw_return_decimal": selected["raw_return_decimal"].mean(),
                    "mean_absolute_rank_error": selected["absolute_rank_error"].mean(),
                }
            )
    per_date = pd.DataFrame(rows)
    summary = (
        per_date.groupby(["model", "region"])
        .agg(
            n_dates=("rank_ic", "count"),
            mean_assets=("n_assets", "mean"),
            mean_rank_ic=("rank_ic", "mean"),
            rank_ic_std_ddof1=("rank_ic", "std"),
            mean_raw_return_decimal=("mean_raw_return_decimal", "mean"),
            mean_absolute_rank_error=("mean_absolute_rank_error", "mean"),
        )
        .reset_index()
    )
    summary["rank_ic_ir"] = (
        summary["mean_rank_ic"] / summary["rank_ic_std_ddof1"]
    )
    return summary


def tail_precision(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (model, date), part in panel.groupby(["model", "date"], sort=False):
        predicted = part.sort_values(["prediction", "id"], kind="stable")
        realized = part.sort_values(["raw_return_decimal", "id"], kind="stable")
        for k in TAIL_K:
            predicted_bottom = predicted.head(k)
            predicted_top = predicted.tail(k)
            realized_bottom_ids = set(realized.head(k)["id"])
            realized_top_ids = set(realized.tail(k)["id"])
            rows.append(
                {
                    "model": model,
                    "date": date,
                    "k": k,
                    "actual_fraction": k / len(part),
                    "top_precision": predicted_top["id"].isin(realized_top_ids).mean(),
                    "bottom_precision": predicted_bottom["id"]
                    .isin(realized_bottom_ids)
                    .mean(),
                    "top_mean_return_decimal": predicted_top[
                        "raw_return_decimal"
                    ].mean(),
                    "bottom_mean_return_decimal": predicted_bottom[
                        "raw_return_decimal"
                    ].mean(),
                    "long_short_return_decimal": predicted_top[
                        "raw_return_decimal"
                    ].mean()
                    - predicted_bottom["raw_return_decimal"].mean(),
                    "top_mean_absolute_rank_error": predicted_top[
                        "absolute_rank_error"
                    ].mean(),
                    "bottom_mean_absolute_rank_error": predicted_bottom[
                        "absolute_rank_error"
                    ].mean(),
                }
            )
    per_date = pd.DataFrame(rows)
    summary = (
        per_date.groupby(["model", "k"])
        .agg(
            n_dates=("date", "nunique"),
            mean_actual_fraction=("actual_fraction", "mean"),
            mean_top_precision=("top_precision", "mean"),
            mean_bottom_precision=("bottom_precision", "mean"),
            mean_top_return_decimal=("top_mean_return_decimal", "mean"),
            mean_bottom_return_decimal=("bottom_mean_return_decimal", "mean"),
            mean_long_short_return_decimal=("long_short_return_decimal", "mean"),
            long_short_sharpe=("long_short_return_decimal", sharpe),
            mean_top_absolute_rank_error=("top_mean_absolute_rank_error", "mean"),
            mean_bottom_absolute_rank_error=("bottom_mean_absolute_rank_error", "mean"),
        )
        .reset_index()
    )
    return summary, per_date


def rank_stability(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, model_panel in panel.groupby("model", sort=False):
        by_date = {
            date: part.set_index("id")
            for date, part in model_panel.groupby("date", sort=True)
        }
        dates = sorted(by_date)
        for previous_date, date in zip(dates[:-1], dates[1:]):
            previous = by_date[previous_date]
            current = by_date[date]
            common = previous.index.intersection(current.index)
            rank_correlation = spearman(
                previous.loc[common, "prediction_rank"],
                current.loc[common, "prediction_rank"],
            )
            mean_rank_migration = float(
                (
                    previous.loc[common, "prediction_rank"]
                    - current.loc[common, "prediction_rank"]
                )
                .abs()
                .mean()
            )
            previous_order = (
                previous.reset_index()
                .sort_values(["prediction", "id"], kind="stable")
                .set_index("id")
            )
            current_order = (
                current.reset_index()
                .sort_values(["prediction", "id"], kind="stable")
                .set_index("id")
            )
            for k in TAIL_K:
                previous_top = set(previous_order.tail(k).index)
                current_top = set(current_order.tail(k).index)
                previous_bottom = set(previous_order.head(k).index)
                current_bottom = set(current_order.head(k).index)
                top_intersection = len(previous_top & current_top)
                bottom_intersection = len(previous_bottom & current_bottom)
                rows.append(
                    {
                        "model": model,
                        "previous_date": previous_date,
                        "date": date,
                        "k": k,
                        "common_assets": len(common),
                        "cross_date_rank_spearman": rank_correlation,
                        "mean_absolute_rank_migration": mean_rank_migration,
                        "top_overlap_fraction": top_intersection / k,
                        "bottom_overlap_fraction": bottom_intersection / k,
                        "top_jaccard": top_intersection
                        / len(previous_top | current_top),
                        "bottom_jaccard": bottom_intersection
                        / len(previous_bottom | current_bottom),
                    }
                )
    return pd.DataFrame(rows)


def date_contributions(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, date), part in panel.groupby(["model", "date"], sort=False):
        bottom = part.loc[part["prediction_decile"] == 1, "raw_return_decimal"]
        top = part.loc[part["prediction_decile"] == 10, "raw_return_decimal"]
        rows.append(
            {
                "model": model,
                "date": date,
                "n_assets": len(part),
                "ic": spearman(part["prediction"], part["raw_return_decimal"]),
                "bottom_decile_return_decimal": bottom.mean(),
                "top_decile_return_decimal": top.mean(),
                "long_short_return_decimal": top.mean() - bottom.mean(),
            }
        )
    result = pd.DataFrame(rows).sort_values(["model", "date"]).reset_index(drop=True)
    enriched = []
    for model, part in result.groupby("model", sort=False):
        part = part.copy()
        n_dates = len(part)
        total_return = part["long_short_return_decimal"].sum()
        part["mean_ic_contribution"] = part["ic"] / n_dates
        part["long_short_arithmetic_contribution_decimal"] = part[
            "long_short_return_decimal"
        ]
        part["long_short_share_of_total"] = (
            part["long_short_return_decimal"] / total_return
            if total_return != 0
            else np.nan
        )
        part["leave_one_date_out_mean_ic"] = (
            part["ic"].sum() - part["ic"]
        ) / (n_dates - 1)
        leave_one_out_sharpes = []
        for row_index in part.index:
            leave_one_out_sharpes.append(
                sharpe(part.drop(index=row_index)["long_short_return_decimal"])
            )
        part["leave_one_date_out_long_short_sharpe"] = leave_one_out_sharpes
        enriched.append(part)
    return pd.concat(enriched, ignore_index=True)


def prediction_magnitude_curve(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.copy()
    frame["absolute_prediction_z"] = frame["prediction_z"].abs()
    frame["magnitude_decile"] = (
        frame.groupby(["model", "date"])["absolute_prediction_z"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"), 10, labels=False, duplicates="drop"
            )
            + 1
        )
        .astype(int)
    )
    return (
        frame.groupby(["model", "magnitude_decile"])
        .agg(
            n_asset_dates=("id", "size"),
            mean_absolute_prediction_z=("absolute_prediction_z", "mean"),
            mean_absolute_rank_error=("absolute_rank_error", "mean"),
            mean_signed_prediction_return_decimal=("signed_prediction_return", "mean"),
            mean_raw_return_decimal=("raw_return_decimal", "mean"),
        )
        .reset_index()
    )


def asset_contributions(panel: pd.DataFrame) -> pd.DataFrame:
    tail = panel.loc[panel["prediction_decile"].isin([1, 10])].copy()
    counts = tail.groupby(["model", "date", "prediction_decile"])["id"].transform("size")
    tail["signed_weighted_return_decimal"] = np.where(
        tail["prediction_decile"] == 10,
        tail["raw_return_decimal"] / counts,
        -tail["raw_return_decimal"] / counts,
    )
    return (
        tail.groupby(["model", "id"])
        .agg(
            tail_periods=("date", "size"),
            top_periods=("prediction_decile", lambda x: int((x == 10).sum())),
            bottom_periods=("prediction_decile", lambda x: int((x == 1).sum())),
            cumulative_signed_contribution_decimal=(
                "signed_weighted_return_decimal",
                "sum",
            ),
            mean_signed_contribution_decimal=(
                "signed_weighted_return_decimal",
                "mean",
            ),
        )
        .reset_index()
    )


def group_composition_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["FinPFN", "TabPFN"]:
        part = panel.loc[panel["model"] == model]
        dispersion = part["group_composition_prediction_sd"]
        rows.append(
            {
                "model": model,
                "n_asset_dates": len(part),
                "mean_prediction_repetitions": part[
                    "prediction_repetitions"
                ].mean(),
                "mean_group_composition_prediction_sd": dispersion.mean(),
                "spearman_sd_vs_absolute_rank_error": spearman(
                    dispersion, part["absolute_rank_error"]
                ),
                "spearman_sd_vs_absolute_prediction_z": spearman(
                    dispersion, part["prediction_z"].abs()
                ),
                "mean_sd_predicted_middle_80pct": part.loc[
                    part["prediction_decile"].between(2, 9),
                    "group_composition_prediction_sd",
                ].mean(),
                "mean_sd_predicted_tail_deciles": part.loc[
                    part["prediction_decile"].isin([1, 10]),
                    "group_composition_prediction_sd",
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def validation_checks(
    panel: pd.DataFrame, dates: pd.DataFrame
) -> dict[str, object]:
    observed_ic = (
        dates.groupby("model")["ic"].mean().reindex(MODELS).to_dict()
    )
    ic_differences = {
        model: abs(observed_ic[model] - expected)
        for model, expected in EXPECTED_MEAN_IC.items()
    }
    if max(ic_differences.values()) > 1e-8:
        raise ValueError(f"IC does not reproduce frozen evaluator: {ic_differences}")

    frozen_ic = pd.read_csv(FROZEN_IC)
    frozen_ic["date"] = pd.to_datetime(frozen_ic["date"])
    frozen_ic = frozen_ic[["model", "date", "ic_raw_target"]]
    date_ic_check = dates[["model", "date", "ic"]].merge(
        frozen_ic, on=["model", "date"], how="left", validate="one_to_one"
    )
    date_ic_check["difference"] = (
        date_ic_check["ic"] - date_ic_check["ic_raw_target"]
    )
    maximum_date_ic_difference = float(date_ic_check["difference"].abs().max())
    if maximum_date_ic_difference > 1e-6:
        raise ValueError(
            "Per-date IC differs materially from frozen evaluator: "
            f"{maximum_date_ic_difference}"
        )

    frozen_holdings = pd.read_parquet(COMMON_HOLDINGS)
    frozen_holdings["date"] = pd.to_datetime(frozen_holdings["date"])
    decile_check = panel[
        ["model", "date", "id", "prediction_decile"]
    ].merge(
        frozen_holdings[["model", "date", "id", "decile"]],
        on=["model", "date", "id"],
        how="left",
        validate="one_to_one",
    )
    decile_mismatches = int(
        (decile_check["prediction_decile"] != decile_check["decile"]).sum()
    )
    if decile_mismatches:
        raise ValueError(
            f"{decile_mismatches} decile assignments differ from frozen holdings"
        )

    frozen = pd.read_csv(FROZEN_METRICS)
    frozen = frozen.loc[frozen["return_basis"] == "raw"].set_index("model")
    sharpe_checks = {}
    for model, part in dates.groupby("model"):
        observed = sharpe(part["long_short_return_decimal"])
        expected = float(frozen.loc[model, "primary_long_short_sharpe"])
        difference = abs(observed - expected)
        if difference > 1e-12:
            raise ValueError(
                f"{model} long-short Sharpe differs from frozen evaluator by {difference}"
            )
        sharpe_checks[model] = {
            "observed": observed,
            "expected": expected,
            "absolute_difference": difference,
        }
    return {
        "rows": len(panel),
        "dates": int(panel["date"].nunique()),
        "asset_dates_per_model": int(
            panel.groupby("model").size().drop_duplicates().iat[0]
        ),
        "mean_ic": observed_ic,
        "mean_ic_absolute_difference": ic_differences,
        "maximum_per_date_ic_absolute_difference": maximum_date_ic_difference,
        "per_date_ic_differences_above_1e-12": int(
            (date_ic_check["difference"].abs() > 1e-12).sum()
        ),
        "decile_assignment_mismatches": decile_mismatches,
        "long_short_sharpe": sharpe_checks,
    }


def summary_stats(
    curve: pd.DataFrame,
    regions: pd.DataFrame,
    tails: pd.DataFrame,
    stability: pd.DataFrame,
    dates: pd.DataFrame,
    assets: pd.DataFrame,
    group_diagnostics: pd.DataFrame,
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for model in MODELS:
        model_curve = curve.loc[curve["model"] == model]
        curve_monotonicity = spearman(
            model_curve["prediction_percentile_bin"],
            model_curve["mean_raw_return_decimal"],
        )
        model_regions = regions.loc[regions["model"] == model].set_index("region")
        model_stability = stability.loc[
            (stability["model"] == model) & (stability["k"] == 40)
        ]
        model_dates = dates.loc[dates["model"] == model]
        model_assets = assets.loc[assets["model"] == model].copy()
        absolute_asset_contribution = model_assets[
            "cumulative_signed_contribution_decimal"
        ].abs()
        top_five_share = (
            absolute_asset_contribution.nlargest(5).sum()
            / absolute_asset_contribution.sum()
        )
        largest_date = model_dates.loc[
            model_dates["long_short_return_decimal"].abs().idxmax()
        ]
        summaries[model] = {
            "percentile_curve_monotonicity_spearman": curve_monotonicity,
            "bin20_minus_bin1_mean_return_decimal": float(
                model_curve.loc[
                    model_curve["prediction_percentile_bin"] == 20,
                    "mean_raw_return_decimal",
                ].iat[0]
                - model_curve.loc[
                    model_curve["prediction_percentile_bin"] == 1,
                    "mean_raw_return_decimal",
                ].iat[0]
            ),
            "full_mean_ic": float(
                model_regions.loc["full_universe", "mean_rank_ic"]
            ),
            "middle_mean_ic": float(
                model_regions.loc["middle_20_to_80pct", "mean_rank_ic"]
            ),
            "bottom20_mean_ic": float(
                model_regions.loc["bottom_20pct", "mean_rank_ic"]
            ),
            "top20_mean_ic": float(
                model_regions.loc["top_20pct", "mean_rank_ic"]
            ),
            "top40_precision": float(
                tails.loc[
                    (tails["model"] == model) & (tails["k"] == 40),
                    "mean_top_precision",
                ].iat[0]
            ),
            "bottom40_precision": float(
                tails.loc[
                    (tails["model"] == model) & (tails["k"] == 40),
                    "mean_bottom_precision",
                ].iat[0]
            ),
            "top40_overlap": float(model_stability["top_overlap_fraction"].mean()),
            "bottom40_overlap": float(
                model_stability["bottom_overlap_fraction"].mean()
            ),
            "mean_rank_migration": float(
                model_stability["mean_absolute_rank_migration"].mean()
            ),
            "largest_absolute_long_short_date": largest_date["date"].isoformat(),
            "largest_absolute_long_short_return_decimal": float(
                largest_date["long_short_return_decimal"]
            ),
            "asset_absolute_contribution_top5_share": float(top_five_share),
        }
    for row in group_diagnostics.itertuples():
        summaries[row.model]["group_composition"] = {
            "mean_repetitions": row.mean_prediction_repetitions,
            "sd_vs_absolute_rank_error_spearman": (
                row.spearman_sd_vs_absolute_rank_error
            ),
            "middle_sd": row.mean_sd_predicted_middle_80pct,
            "tail_sd": row.mean_sd_predicted_tail_deciles,
        }
    return summaries


def save_figures(
    curve: pd.DataFrame,
    tails: pd.DataFrame,
    stability: pd.DataFrame,
    dates: pd.DataFrame,
) -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=False)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for model in MODELS:
        part = curve.loc[curve["model"] == model]
        axis.plot(
            part["prediction_percentile_bin"],
            part["mean_raw_return_decimal"] * 100,
            marker="o",
            markersize=3,
            label=model,
        )
    axis.set(
        title="Realized return by predicted percentile (20 bins)",
        xlabel="Predicted percentile bin (low to high)",
        ylabel="Mean raw return (%)",
        xticks=[1, 5, 10, 15, 20],
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "percentile_return_curve.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, column, title in [
        (axes[0], "mean_top_precision", "Top-k precision"),
        (axes[1], "mean_bottom_precision", "Bottom-k precision"),
    ]:
        for model in MODELS:
            part = tails.loc[tails["model"] == model]
            axis.plot(part["k"], part[column], marker="o", label=model)
        axis.set(title=title, xlabel="k", ylabel="Precision")
        axis.grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figures / "tail_precision.png", dpi=180)
    plt.close(figure)

    stability_summary = (
        stability.loc[stability["k"] == 40]
        .groupby("model")
        .agg(
            top_overlap=("top_overlap_fraction", "mean"),
            bottom_overlap=("bottom_overlap_fraction", "mean"),
        )
        .reindex(MODELS)
    )
    axis = stability_summary.plot(kind="bar", figsize=(9, 5.3))
    axis.set(
        title="Adjacent-date top/bottom 40 membership overlap",
        xlabel="",
        ylabel="Overlap fraction",
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(OUTPUT / "figures/rank_stability.png", dpi=180)
    plt.close(axis.figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for model in MODELS:
        part = dates.loc[dates["model"] == model]
        axis.scatter(
            part["ic"],
            part["long_short_return_decimal"] * 100,
            s=12,
            alpha=0.45,
            label=model,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set(
        title="Daily IC versus long-short return",
        xlabel="Cross-sectional Spearman IC",
        ylabel="Top-minus-bottom return (%)",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "figures/date_ic_vs_portfolio_return.png", dpi=180)
    plt.close(figure)


def write_report(
    summaries: dict[str, object],
    regions: pd.DataFrame,
    tails: pd.DataFrame,
    dates: pd.DataFrame,
    runtime_seconds: float,
) -> None:
    rows = []
    for model in MODELS:
        item = summaries[model]
        rows.append(
            "| "
            + " | ".join(
                [
                    model,
                    f"{item['full_mean_ic']:.4f}",
                    f"{item['middle_mean_ic']:.4f}",
                    f"{item['bottom20_mean_ic']:.4f}",
                    f"{item['top20_mean_ic']:.4f}",
                    f"{item['percentile_curve_monotonicity_spearman']:.3f}",
                    f"{item['top40_precision']:.3f}",
                    f"{item['bottom40_precision']:.3f}",
                    f"{item['top40_overlap']:.3f}",
                    f"{item['bottom40_overlap']:.3f}",
                ]
            )
            + " |"
        )
    finpfn = summaries["FinPFN"]
    ridge = summaries["Ridge"]
    lightgbm = summaries["LightGBM"]
    finpfn_dates = dates.loc[dates["model"] == "FinPFN"]
    finpfn_ic_ls_corr = spearman(
        finpfn_dates["ic"], finpfn_dates["long_short_return_decimal"]
    )
    largest_dates = (
        finpfn_dates.assign(
            absolute_long_short=finpfn_dates["long_short_return_decimal"].abs()
        )
        .nlargest(5, "absolute_long_short")[
            ["date", "ic", "long_short_return_decimal"]
        ]
    )
    largest_date_rows = "\n".join(
        f"| {row.date.date()} | {row.ic:.4f} | "
        f"{row.long_short_return_decimal * 100:.3f}% |"
        for row in largest_dates.itertuples()
    )
    report = f"""# Phase 4：FinPFN 的 IC–组合表现差距

## 执行口径

本分析只读取冻结的 301 个测试日期、120,620 个共同资产—日期和统一 raw-return target。预测先按 `(model,date,id)` 对重复 group rows 取均值，与官方 evaluator 一致。20 个 predicted-percentile bins、`k={{10,20,40}}`、20%/80% 局部区域均在分析前固定；结果是对已知测试表现的**探索性机制审计**，不用于回调 Phase 3。

| 模型 | 全体 IC | 中部 20–80% IC | bottom 20% IC | top 20% IC | 20-bin 单调性 | top-40 precision | bottom-40 precision | top-40 相邻留存 | bottom-40 相邻留存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 核心解释

1. **FinPFN 的优势并不等于更好的尾部资产识别。** 全体平均 IC 为 {finpfn['full_mean_ic']:.4f}，高于 Ridge 的 {ridge['full_mean_ic']:.4f} 和 LightGBM 的 {lightgbm['full_mean_ic']:.4f}；但 top/bottom 40 precision 分别为 {finpfn['top40_precision']:.3f}/{finpfn['bottom40_precision']:.3f}，须与两条基线在上表直接比较。局部 IC 也表明总体 rank 改善和每一侧尾部内部的精细排序不是同一个任务。
2. **FinPFN 的头尾成员更不稳定。** 相邻日期 top/bottom 40 留存为 {finpfn['top40_overlap']:.3f}/{finpfn['bottom40_overlap']:.3f}；Ridge 为 {ridge['top40_overlap']:.3f}/{ridge['bottom40_overlap']:.3f}，LightGBM 为 {lightgbm['top40_overlap']:.3f}/{lightgbm['bottom40_overlap']:.3f}。这与 Phase 1 的 FinPFN 最高换手一致。
3. **IC 与尾部组合收益只有不完全对应。** FinPFN 日期级 IC 与同日 long-short return 的 Spearman 为 {finpfn_ic_ls_corr:.3f}。IC 使用全部约 401 个资产的排序信息，而 decile 组合只使用约 40+40 个极端资产；许多中小排序改善可以提高 IC，却不会进入持仓。
4. **简单降换手不能自动保留 IC 的经济收益。** Phase 3 的 20% rank buffer 在 validation 改善净 Sharpe，但唯一一次 test 将 gross/net Sharpe 降至 3.760/-1.040；说明不稳定性是症状之一，却不能单靠 buffer 修复尾部收益。
5. **50-stock group 重复抽样是可见混杂，但不是充分解释。** FinPFN 同一资产—日期平均重复 {finpfn['group_composition']['mean_repetitions']:.3f} 次；group-composition prediction SD 与绝对 rank error 的 Spearman 为 {finpfn['group_composition']['sd_vs_absolute_rank_error_spearman']:.3f}，尾部 SD 为 {finpfn['group_composition']['tail_sd']:.4f}、中部为 {finpfn['group_composition']['middle_sd']:.4f}。这说明 group composition 会改变预测，但仅凭相关性不能断言它造成全部 tail gap。

## 最大 FinPFN 日期贡献

| 日期 | IC | long-short return |
|---|---:|---:|
{largest_date_rows}

逐日期的 leave-one-out IC 与 long-short Sharpe、收益占比见 `date_contributions.csv`；逐资产尾部贡献见 `asset_contributions.csv`。FinPFN 绝对资产贡献最大的 5 个 ID 占全部绝对资产贡献的 {finpfn['asset_absolute_contribution_top5_share']:.2%}，因此报告同时保留日期和资产集中度，避免把总结果误解为均匀分布。

## 对预声明假设的判断

- **“FinPFN 改善中部排序但不改善极端选择”**：由中部/尾部局部 IC、20-bin 曲线和 top/bottom precision 联合判断；总体上得到支持，但不是“优势只存在于中部”的强结论。
- **“FinPFN 极端预测较不稳定”**：得到支持；相邻成员留存更低、rank migration 更大，并与较高换手一致。
- **“FinPFN 通过许多小排序改善获得更高 IC”**：得到支持；全体 IC 优势大于尾部 precision/组合优势。
- **“高换手移除或反转统计优势”**：成本侵蚀得到支持；但 Phase 3 的 buffer 失败说明高换手并非唯一因果机制。
- **“group-wise task construction 影响 global tail ranking”**：存在可测 composition dispersion，属于合理混杂；当前 artifact 不足以给出因果证明。

数据中只有匿名技术/财务 features 和资产 ID，没有文档化 sector、size、volatility 或 liquidity 标签，因此没有从含义不明确的列名推断经济分组。完整数值见本目录 CSV，图见 `figures/`。本地 CPU runtime 为 {runtime_seconds:.3f} 秒。
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    declared = [
        OUTPUT / "percentile_return_curve.csv",
        OUTPUT / "tail_precision.csv",
        OUTPUT / "tail_precision_by_period.csv",
        OUTPUT / "rank_stability.csv",
        OUTPUT / "date_contributions.csv",
        OUTPUT / "rank_region_metrics.csv",
        OUTPUT / "prediction_magnitude_curve.csv",
        OUTPUT / "asset_contributions.csv",
        OUTPUT / "group_composition_diagnostics.csv",
        OUTPUT / "analysis_summary.json",
        OUTPUT / "input_manifest.json",
        OUTPUT / "report.md",
    ]
    if any(path.exists() for path in declared) or (
        (OUTPUT / "figures").exists() and any((OUTPUT / "figures").iterdir())
    ):
        raise FileExistsError("Refusing to overwrite Phase 4 outputs")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    panel, _ = load_panel()
    curve = percentile_curve(panel)
    regions = rank_region_metrics(panel)
    tails, tails_by_period = tail_precision(panel)
    stability = rank_stability(panel)
    dates = date_contributions(panel)
    magnitude = prediction_magnitude_curve(panel)
    assets = asset_contributions(panel)
    group_diagnostics = group_composition_diagnostics(panel)
    checks = validation_checks(panel, dates)
    summaries = summary_stats(
        curve, regions, tails, stability, dates, assets, group_diagnostics
    )

    curve.to_csv(OUTPUT / "percentile_return_curve.csv", index=False)
    tails.to_csv(OUTPUT / "tail_precision.csv", index=False)
    tails_by_period.to_csv(OUTPUT / "tail_precision_by_period.csv", index=False)
    stability.to_csv(OUTPUT / "rank_stability.csv", index=False)
    dates.to_csv(OUTPUT / "date_contributions.csv", index=False)
    regions.to_csv(OUTPUT / "rank_region_metrics.csv", index=False)
    magnitude.to_csv(OUTPUT / "prediction_magnitude_curve.csv", index=False)
    assets.to_csv(OUTPUT / "asset_contributions.csv", index=False)
    group_diagnostics.to_csv(
        OUTPUT / "group_composition_diagnostics.csv", index=False
    )
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "input_manifest.json").write_text(
        json.dumps(
            {
                "analysis_scope": "frozen_test_exploratory_mechanism_audit",
                "percentile_bins": PERCENTILE_BINS,
                "tail_k": TAIL_K,
                "annualization": ANNUALIZATION,
                "inputs": {
                    "dataset": {"path": str(DATASET.relative_to(REPOSITORY)), "sha256": sha256(DATASET)},
                    "common_holdings": {
                        "path": str(COMMON_HOLDINGS.relative_to(REPOSITORY)),
                        "sha256": sha256(COMMON_HOLDINGS),
                    },
                    **{
                        f"{model}_predictions": {
                            "path": str(path.relative_to(REPOSITORY)),
                            "sha256": sha256(path),
                        }
                        for model, path in PREDICTIONS.items()
                    },
                },
                "validation_checks": checks,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    save_figures(curve, tails, stability, dates)
    runtime_seconds = time.perf_counter() - started
    write_report(summaries, regions, tails, dates, runtime_seconds)
    print(json.dumps(summaries, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
