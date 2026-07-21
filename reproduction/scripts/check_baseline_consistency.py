#!/usr/bin/env python3
"""Check that Ridge and LightGBM portfolio results are independently formed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluate_predictions import compute_portfolios
from reproduction_common import MARKET_CONFIG, load_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--lightgbm", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ridge-periods", type=Path, required=True)
    parser.add_argument("--lightgbm-periods", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_predictions(path: Path, expected_model: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    if set(frame["model"].unique()) != {expected_model}:
        raise ValueError(f"Unexpected model label in {path.name}")
    if set(frame["status"].unique()) != {"ok"}:
        raise ValueError(f"Non-success prediction rows in {path.name}")
    if frame.duplicated(["date", "id"]).any():
        raise ValueError(f"Duplicate asset-date predictions in {path.name}")
    if frame["prediction"].isna().any():
        raise ValueError(f"Missing predictions in {path.name}")
    return frame[["date", "id", "prediction"]].copy()


def correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    if method == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    return float(stats.spearmanr(x, y).statistic)


def holdings_overlap(holdings: pd.DataFrame, decile: int) -> pd.DataFrame:
    selected = holdings.loc[holdings["decile"] == decile]
    rows = []
    ridge = selected.loc[selected["model"] == "Ridge"]
    lightgbm = selected.loc[selected["model"] == "LightGBM"]
    common_dates = sorted(set(ridge["date"]) & set(lightgbm["date"]))
    for date in common_dates:
        ridge_ids = set(ridge.loc[ridge["date"] == date, "id"])
        lightgbm_ids = set(lightgbm.loc[lightgbm["date"] == date, "id"])
        intersection = ridge_ids & lightgbm_ids
        union = ridge_ids | lightgbm_ids
        rows.append(
            {
                "date": date,
                "decile": decile,
                "ridge_count": len(ridge_ids),
                "lightgbm_count": len(lightgbm_ids),
                "intersection_count": len(intersection),
                "overlap_fraction_of_smaller_decile": (
                    len(intersection) / min(len(ridge_ids), len(lightgbm_ids))
                ),
                "jaccard": len(intersection) / len(union),
            }
        )
    return pd.DataFrame(rows)


def saved_long_short(path: Path, expected_model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[
        (frame["model"] == expected_model) & (frame["return_basis"] == "raw"),
        ["date", "long_short"],
    ]
    return frame.rename(columns={"long_short": expected_model})


def main() -> None:
    args = parse_args()
    ridge = read_predictions(args.ridge, "Ridge").rename(
        columns={"prediction": "ridge_prediction"}
    )
    lightgbm = read_predictions(args.lightgbm, "LightGBM").rename(
        columns={"prediction": "lightgbm_prediction"}
    )
    predictions = ridge.merge(
        lightgbm, on=["date", "id"], how="inner", validate="one_to_one"
    )
    if len(predictions) != len(ridge) or len(predictions) != len(lightgbm):
        raise ValueError("Ridge and LightGBM prediction universes differ")

    daily_spearman = (
        predictions.groupby("date")
        .apply(
            lambda group: stats.spearmanr(
                group["ridge_prediction"], group["lightgbm_prediction"]
            ).statistic,
            include_groups=False,
        )
        .rename("prediction_spearman")
        .reset_index()
    )

    raw = load_panel(
        args.dataset,
        "csi500",
        split="test",
        columns=["date", "id", "target"],
    )
    predictions = predictions.merge(
        raw, on=["date", "id"], how="left", validate="one_to_one"
    )
    if predictions["target"].isna().any():
        raise ValueError("Predictions do not match the official raw targets")

    long_frames = []
    for model, column in {
        "Ridge": "ridge_prediction",
        "LightGBM": "lightgbm_prediction",
    }.items():
        part = predictions[["date", "id", "target", column]].rename(
            columns={column: "prediction"}
        )
        part["model"] = model
        part["seed"] = 42
        part["raw_return_percentage_points"] = (
            part["target"]
            * MARKET_CONFIG["csi500"]["return_to_percentage_points"]
        )
        long_frames.append(part)
    combined = pd.concat(long_frames, ignore_index=True)
    periods, _, _, holdings = compute_portfolios(
        combined, MARKET_CONFIG["csi500"]["annualization"]
    )

    overlap = pd.concat(
        [holdings_overlap(holdings, 1), holdings_overlap(holdings, 10)],
        ignore_index=True,
    )
    long_short = (
        periods.loc[periods["return_basis"] == "raw", ["date", "model", "long_short"]]
        .pivot(index="date", columns="model", values="long_short")
        .reset_index()
    )
    long_short["difference_ridge_minus_lightgbm"] = (
        long_short["Ridge"] - long_short["LightGBM"]
    )

    saved = saved_long_short(args.ridge_periods, "Ridge").merge(
        saved_long_short(args.lightgbm_periods, "LightGBM"),
        on="date",
        validate="one_to_one",
    )
    saved_check = long_short.merge(
        saved,
        on="date",
        suffixes=("_recomputed", "_saved"),
        validate="one_to_one",
    )
    ridge_saved_max_difference = float(
        (saved_check["Ridge_recomputed"] - saved_check["Ridge_saved"]).abs().max()
    )
    lightgbm_saved_max_difference = float(
        (
            saved_check["LightGBM_recomputed"]
            - saved_check["LightGBM_saved"]
        ).abs().max()
    )

    top_overlap = overlap.loc[overlap["decile"] == 10]
    bottom_overlap = overlap.loc[overlap["decile"] == 1]
    summary = {
        "input_prediction_paths": {
            "Ridge": args.ridge.resolve().as_posix(),
            "LightGBM": args.lightgbm.resolve().as_posix(),
        },
        "sha256": {
            "Ridge": sha256(args.ridge),
            "LightGBM": sha256(args.lightgbm),
            "evaluator": sha256(Path(__file__).with_name("evaluate_predictions.py")),
        },
        "rows": len(predictions),
        "dates": predictions["date"].nunique(),
        "prediction_pearson": correlation(
            predictions["ridge_prediction"],
            predictions["lightgbm_prediction"],
            "pearson",
        ),
        "prediction_spearman_all_rows": correlation(
            predictions["ridge_prediction"],
            predictions["lightgbm_prediction"],
            "spearman",
        ),
        "per_date_prediction_spearman": {
            "mean": daily_spearman["prediction_spearman"].mean(),
            "median": daily_spearman["prediction_spearman"].median(),
            "std_ddof1": daily_spearman["prediction_spearman"].std(ddof=1),
            "minimum": daily_spearman["prediction_spearman"].min(),
            "maximum": daily_spearman["prediction_spearman"].max(),
        },
        "top_decile_overlap": {
            "mean_fraction_of_smaller_decile": top_overlap[
                "overlap_fraction_of_smaller_decile"
            ].mean(),
            "mean_jaccard": top_overlap["jaccard"].mean(),
            "mean_intersection_count": top_overlap["intersection_count"].mean(),
        },
        "bottom_decile_overlap": {
            "mean_fraction_of_smaller_decile": bottom_overlap[
                "overlap_fraction_of_smaller_decile"
            ].mean(),
            "mean_jaccard": bottom_overlap["jaccard"].mean(),
            "mean_intersection_count": bottom_overlap["intersection_count"].mean(),
        },
        "long_short_series": {
            "pearson": correlation(long_short["Ridge"], long_short["LightGBM"], "pearson"),
            "spearman": correlation(
                long_short["Ridge"], long_short["LightGBM"], "spearman"
            ),
            "maximum_absolute_difference_percentage_points": long_short[
                "difference_ridge_minus_lightgbm"
            ].abs().max(),
            "mean_absolute_difference_percentage_points": long_short[
                "difference_ridge_minus_lightgbm"
            ].abs().mean(),
        },
        "saved_evaluator_series_recompute_max_absolute_difference": {
            "Ridge": ridge_saved_max_difference,
            "LightGBM": lightgbm_saved_max_difference,
        },
        "decile_source_confirmation": (
            "compute_portfolios groups by model/seed/date and assign_deciles sorts "
            "each group by its own prediction column with id only as a tie-breaker"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_spearman.to_csv(args.output_dir / "prediction_spearman_by_date.csv", index=False)
    overlap.to_csv(args.output_dir / "holdings_overlap_by_date.csv", index=False)
    long_short.to_csv(args.output_dir / "long_short_comparison.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
