#!/usr/bin/env python3
"""Evaluate checkpoint IC exactly as the released notebook does.

This intentionally retains repeated asset rows created by the notebook's
with-replacement stock sampling. It is a diagnostic companion to the common
evaluator, which collapses repeated asset-date predictions and uses raw returns for
an apples-to-apples comparison across models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def notebook_spearman(group: pd.DataFrame) -> float:
    valid = np.isfinite(group["target_group_z"]) & np.isfinite(group["prediction"])
    if valid.sum() < 2:
        return float("nan")
    return float(
        stats.spearmanr(
            group.loc[valid, "target_group_z"],
            group.loc[valid, "prediction"],
        ).statistic
    )


def main() -> None:
    args = parse_args()
    frames = []
    for path in args.predictions:
        frame = pd.read_parquet(path)
        required = {"model", "seed", "date", "id", "prediction", "target_group_z", "status"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        frame = frame.loc[frame["status"].eq("ok")].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    ic = (
        predictions.groupby(["model", "seed", "date"], sort=True)
        .apply(notebook_spearman, include_groups=False)
        .rename("ic_task_preprocessed_target")
        .reset_index()
    )

    rows = []
    for (model, seed), group in predictions.groupby(["model", "seed"], sort=True):
        values = ic.loc[
            ic["model"].eq(model) & ic["seed"].eq(seed),
            "ic_task_preprocessed_target",
        ].dropna()
        standard_deviation = values.std(ddof=1)
        rows.append(
            {
                "model": model,
                "seed": seed,
                "prediction_rows_with_repetitions": len(group),
                "unique_asset_dates": group[["date", "id"]].drop_duplicates().shape[0],
                "repeated_asset_rows": int(group.duplicated(["date", "id"]).sum()),
                "n_dates": len(values),
                "mean_ic": values.mean(),
                "ic_std_ddof1": standard_deviation,
                "ir": (
                    values.mean() / standard_deviation
                    if np.isfinite(standard_deviation) and standard_deviation != 0
                    else float("nan")
                ),
            }
        )
    summary = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ic.to_csv(args.output_dir / "notebook_exact_ic_by_period.csv", index=False)
    summary.to_csv(args.output_dir / "notebook_exact_ic_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
