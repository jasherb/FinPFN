#!/usr/bin/env python3
"""Validate returned ensemble-member prediction artifacts without evaluating them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-dates", type=int, default=242)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    summaries = []
    for path in args.predictions:
        path = path.resolve()
        metadata_path = path.with_suffix(".metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing prediction/metadata pair for {path}")
        metadata = json.loads(metadata_path.read_text())
        frame = pd.read_parquet(path)
        member_mean = [f"member_{index}_mean" for index in range(8)]
        member_median = [f"member_{index}_median" for index in range(8)]
        numeric = [
            "prediction",
            "prediction_mean",
            "prediction_mode",
            "predictive_std",
            "predictive_q10",
            "predictive_q25",
            "predictive_q50",
            "predictive_q75",
            "predictive_q90",
        ] + member_mean + member_median
        required = {
            "model",
            "seed",
            "split",
            "date",
            "id",
            "status",
            *numeric,
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        frame["date"] = pd.to_datetime(frame["date"])
        dates = frame["date"].nunique()
        if dates != args.expected_dates:
            raise ValueError(f"{path.name}: expected {args.expected_dates} dates, got {dates}")
        if set(frame["split"].unique()) != {"validation"}:
            raise ValueError(f"{path.name}: not a validation-only artifact")
        failed = int((frame["status"] != "ok").sum())
        nonfinite = int((~np.isfinite(frame[numeric].to_numpy(dtype=float))).sum())
        quantiles = frame[
            [
                "predictive_q10",
                "predictive_q25",
                "predictive_q50",
                "predictive_q75",
                "predictive_q90",
            ]
        ].to_numpy()
        quantile_order_failures = int((np.diff(quantiles, axis=1) < -1e-10).sum())
        q50_difference = float(
            np.max(np.abs(frame["predictive_q50"] - frame["prediction"]))
        )
        observed_hash = sha256(path)
        if observed_hash != metadata.get("prediction_sha256"):
            raise ValueError(f"{path.name}: metadata prediction checksum mismatch")
        if failed or nonfinite or quantile_order_failures or q50_difference > 2e-6:
            raise ValueError(
                f"{path.name} failed validation: failed={failed}, nonfinite={nonfinite}, "
                f"quantile_order={quantile_order_failures}, q50_diff={q50_difference}"
            )
        if metadata.get("failed_groups") != 0:
            raise ValueError(f"{path.name}: metadata records failed groups")
        reference = metadata.get("reference_output_verification") or {}
        if reference.get("maximum_absolute_difference", np.inf) > 2e-6:
            raise ValueError(f"{path.name}: public predict equivalence was not verified")
        summaries.append(
            {
                "model": frame["model"].iat[0],
                "rows": len(frame),
                "dates": dates,
                "unique_asset_dates": frame[["date", "id"]].drop_duplicates().shape[0],
                "repeated_rows_beyond_first": (
                    len(frame) - frame[["date", "id"]].drop_duplicates().shape[0]
                ),
                "groups": metadata["successful_groups"],
                "runtime_seconds": metadata["runtime_seconds"],
                "failed_rows": failed,
                "nonfinite_outputs": nonfinite,
                "max_public_predict_difference": reference[
                    "maximum_absolute_difference"
                ],
                "sha256": observed_hash,
            }
        )
    print(pd.DataFrame(summaries).to_string(index=False))
    print("UNCERTAINTY ARTIFACT VALIDATION PASSED")


if __name__ == "__main__":
    main()
