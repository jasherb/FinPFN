#!/usr/bin/env python3
"""Compare newly generated checkpoint predictions with the bundled CSI CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


MODEL_COLUMNS = {
    "FinPFN": "finpfn_median_pred",
    "TabPFN": "tabpfn_median_pred",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--bundled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundled = pd.read_csv(args.bundled, compression="infer", index_col=0)
    bundled["date"] = pd.to_datetime(bundled["date"])
    rows = []
    for path in args.new_predictions:
        predictions = pd.read_parquet(path)
        predictions["date"] = pd.to_datetime(predictions["date"])
        predictions = predictions.loc[
            (predictions["status"] == "ok") & predictions["prediction"].notna()
        ]
        for (model, seed), group in predictions.groupby(["model", "seed"]):
            bundled_column = MODEL_COLUMNS.get(model)
            if bundled_column is None:
                continue
            collapsed = (
                group.groupby(["date", "id"], as_index=False)["prediction"].mean()
            )
            matched = collapsed.merge(
                bundled[["date", "id", bundled_column]],
                on=["date", "id"],
                how="inner",
                validate="one_to_one",
            )
            if len(matched) >= 2:
                spearman = stats.spearmanr(
                    matched["prediction"], matched[bundled_column]
                ).statistic
                pearson = stats.pearsonr(
                    matched["prediction"], matched[bundled_column]
                ).statistic
            else:
                spearman = pearson = float("nan")
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "new_unique_asset_dates": len(collapsed),
                    "bundled_unique_asset_dates": len(bundled),
                    "overlap_asset_dates": len(matched),
                    "overlap_fraction_of_new": len(matched) / len(collapsed),
                    "prediction_spearman": spearman,
                    "prediction_pearson": pearson,
                    "prediction_mean_absolute_difference": (
                        matched["prediction"] - matched[bundled_column]
                    ).abs().mean(),
                }
            )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
