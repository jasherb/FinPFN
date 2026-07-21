#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


MARKETS = {
    "csi500": {
        "features": 30,
        "split1": "2021-01-01",
        "split2": "2022-01-01",
    },
    "us": {
        "features": 90,
        "split1": "2000-01-01",
        "split2": "2010-01-01",
    },
}

EXCLUDED_FEATURE_COLUMNS = {"date", "id", "target", "__index_level_0__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--market", choices=["auto", "csi500", "us", "index"], default="auto"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--sample-rows", type=int, default=100_000)
    return parser.parse_args()


def iso_date(value: int, unit: str = "us") -> str:
    return np.datetime_as_string(np.datetime64(value, unit), unit="s")


def detect_market(path: Path, names: list[str], requested: str) -> str:
    if requested != "auto":
        return requested
    if {"time", "symbol", "open", "close"}.issubset(names):
        return "index"
    feature_count = len([name for name in names if name not in EXCLUDED_FEATURE_COLUMNS])
    if feature_count == 30:
        return "csi500"
    if feature_count == 90:
        return "us"
    raise ValueError(
        f"Unable to identify {path.name}: found {feature_count} model feature columns"
    )


def parquet_metadata(path: Path, parquet_file: pq.ParquetFile) -> dict[str, Any]:
    metadata = parquet_file.metadata
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "physical_columns": metadata.num_columns,
        "created_by": metadata.created_by,
        "columns": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in parquet_file.schema_arrow
        ],
    }


def inspect_index(path: Path, parquet_file: pq.ParquetFile) -> dict[str, Any]:
    started = time.time()
    frame = parquet_file.read().to_pandas()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    numeric = ["open", "close", "high", "low", "volume", "money"]
    summary = parquet_metadata(path, parquet_file)
    summary.update(
        {
            "dataset": "csi500_index_price",
            "time_min": frame["time"].min().isoformat(),
            "time_max": frame["time"].max().isoformat(),
            "unique_dates": int(frame["time"].nunique()),
            "unique_symbols": int(frame["symbol"].nunique()),
            "time_ordered_non_decreasing": bool(frame["time"].is_monotonic_increasing),
            "time_ordered_non_increasing": bool(frame["time"].is_monotonic_decreasing),
            "duplicate_symbol_time_rows": int(
                frame.duplicated(["symbol", "time"]).sum()
            ),
            "duplicate_full_rows": int(frame.duplicated().sum()),
            "missing_by_column": {
                column: int(frame[column].isna().sum()) for column in frame.columns
            },
            "non_finite_by_column": {
                column: int((~np.isfinite(frame[column].to_numpy())).sum())
                for column in numeric
            },
            "runtime_seconds": time.time() - started,
        }
    )
    return summary


def inspect_panel(
    path: Path,
    parquet_file: pq.ParquetFile,
    market: str,
    batch_size: int,
    sample_rows: int,
) -> dict[str, Any]:
    started = time.time()
    config = MARKETS[market]
    schema = parquet_file.schema_arrow
    names = schema.names
    required = {"date", "id", "target"}
    missing_required = sorted(required.difference(names))
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    features = [name for name in names if name not in EXCLUDED_FEATURE_COLUMNS]
    value_columns = ["target", *features]
    value_indices = [names.index(name) for name in value_columns]
    numeric_count = len(value_columns)
    column_missing = {
        name: {"null": 0, "nan": 0, "infinite": 0} for name in names
    }
    global_min = np.full(numeric_count, np.inf, dtype=np.float64)
    global_max = np.full(numeric_count, -np.inf, dtype=np.float64)

    total_rows = parquet_file.metadata.num_rows
    selected_rows = min(sample_rows, total_rows)
    sample_indices = np.linspace(0, total_rows - 1, selected_rows, dtype=np.int64)
    sample_chunks: list[np.ndarray] = []
    row_offset = 0

    date_sums: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(numeric_count, dtype=np.float64)
    )
    date_sumsq: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(numeric_count, dtype=np.float64)
    )
    date_value_counts: dict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(numeric_count, dtype=np.int64)
    )
    date_row_counts: dict[int, int] = defaultdict(int)
    asset_ids: set[Any] = set()
    current_date: int | None = None
    current_date_ids: set[Any] = set()
    duplicate_asset_date_rows = 0
    dates_ordered = True
    last_seen_date: int | None = None

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_rows = batch.num_rows
        date_array = batch.column(names.index("date"))
        id_array = batch.column(names.index("id"))
        if date_array.null_count:
            raise ValueError("Date column contains null values; ordered audit is undefined")
        if id_array.null_count:
            raise ValueError("ID column contains null values; uniqueness audit is undefined")

        date_values = date_array.cast(pa.int64()).to_numpy(zero_copy_only=False)
        if len(date_values) > 1 and np.any(date_values[1:] < date_values[:-1]):
            dates_ordered = False
        if last_seen_date is not None and int(date_values[0]) < last_seen_date:
            dates_ordered = False
        last_seen_date = int(date_values[-1])

        id_values = id_array.to_numpy(zero_copy_only=False)
        asset_ids.update(id_values.tolist())

        boundaries = np.r_[0, np.flatnonzero(date_values[1:] != date_values[:-1]) + 1, batch_rows]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            date_value = int(date_values[start])
            if current_date != date_value:
                current_date = date_value
                current_date_ids = set()
            segment_ids = id_values[start:end].tolist()
            duplicate_asset_date_rows += sum(
                identifier in current_date_ids for identifier in segment_ids
            )
            current_date_ids.update(segment_ids)
            date_row_counts[date_value] += end - start

        value_arrays = []
        for name, index in zip(value_columns, value_indices, strict=True):
            array = batch.column(index)
            values = array.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            null_count = array.null_count
            nan_count = int(np.isnan(values).sum())
            finite = np.isfinite(values)
            column_missing[name]["null"] += null_count
            column_missing[name]["nan"] += max(nan_count - null_count, 0)
            column_missing[name]["infinite"] += int(np.isinf(values).sum())
            if finite.any():
                global_min[len(value_arrays)] = min(
                    global_min[len(value_arrays)], float(values[finite].min())
                )
                global_max[len(value_arrays)] = max(
                    global_max[len(value_arrays)], float(values[finite].max())
                )
            value_arrays.append(values)
        values_matrix = np.column_stack(value_arrays)

        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            date_value = int(date_values[start])
            block = values_matrix[start:end]
            finite = np.isfinite(block)
            safe = np.where(finite, block, 0.0)
            date_sums[date_value] += safe.sum(axis=0)
            date_sumsq[date_value] += np.square(safe).sum(axis=0)
            date_value_counts[date_value] += finite.sum(axis=0)

        sample_start = np.searchsorted(sample_indices, row_offset, side="left")
        sample_end = np.searchsorted(
            sample_indices, row_offset + batch_rows, side="left"
        )
        if sample_end > sample_start:
            local_indices = sample_indices[sample_start:sample_end] - row_offset
            sample_chunks.append(values_matrix[local_indices])
        row_offset += batch_rows

        for name in names:
            if name in value_columns:
                continue
            array = batch.column(names.index(name))
            column_missing[name]["null"] += array.null_count

    if not dates_ordered:
        duplicate_asset_date_rows = -1

    ordered_dates = sorted(date_row_counts)
    per_date_means = []
    per_date_stds = []
    for date_value in ordered_dates:
        counts = date_value_counts[date_value]
        sums = date_sums[date_value]
        sumsq = date_sumsq[date_value]
        means = np.divide(
            sums,
            counts,
            out=np.full(numeric_count, np.nan),
            where=counts > 0,
        )
        variance_numerator = sumsq - np.divide(
            np.square(sums),
            counts,
            out=np.zeros(numeric_count),
            where=counts > 0,
        )
        variances = np.divide(
            variance_numerator,
            counts - 1,
            out=np.full(numeric_count, np.nan),
            where=counts > 1,
        )
        per_date_means.append(means)
        per_date_stds.append(np.sqrt(np.maximum(variances, 0.0)))
    means_matrix = np.vstack(per_date_means)
    stds_matrix = np.vstack(per_date_stds)
    samples = np.vstack(sample_chunks)

    feature_diagnostics = {}
    for index, name in enumerate(value_columns):
        sample = samples[:, index]
        sample = sample[np.isfinite(sample)]
        date_means = means_matrix[:, index]
        date_stds = stds_matrix[:, index]
        feature_diagnostics[name] = {
            "global_min": None if math.isinf(global_min[index]) else global_min[index],
            "sample_p01": None if not len(sample) else float(np.quantile(sample, 0.01)),
            "sample_median": None if not len(sample) else float(np.quantile(sample, 0.5)),
            "sample_p99": None if not len(sample) else float(np.quantile(sample, 0.99)),
            "global_max": None if math.isinf(global_max[index]) else global_max[index],
            "median_abs_cross_sectional_mean": float(
                np.nanmedian(np.abs(date_means))
            ),
            "median_cross_sectional_std_ddof1": float(np.nanmedian(date_stds)),
            "fraction_dates_mean_abs_below_1e_6": float(
                np.nanmean(np.abs(date_means) < 1e-6)
            ),
            "fraction_dates_std_within_0_05_of_1": float(
                np.nanmean(np.abs(date_stds - 1.0) < 0.05)
            ),
        }

    split1 = np.datetime64(config["split1"], "us").astype(np.int64)
    split2 = np.datetime64(config["split2"], "us").astype(np.int64)
    split_predicates = {
        "train": lambda value: value < split1,
        "validation": lambda value: split1 <= value < split2,
        "test": lambda value: value >= split2,
    }
    splits = {}
    for name, predicate in split_predicates.items():
        dates = [value for value in ordered_dates if predicate(value)]
        splits[name] = {
            "rows": int(sum(date_row_counts[value] for value in dates)),
            "dates": len(dates),
            "date_min": None if not dates else iso_date(dates[0]),
            "date_max": None if not dates else iso_date(dates[-1]),
        }

    rows_per_date = np.array(list(date_row_counts.values()), dtype=np.int64)
    feature_mean_diagnostics = np.array(
        [
            feature_diagnostics[name]["median_abs_cross_sectional_mean"]
            for name in features
        ]
    )
    feature_std_diagnostics = np.array(
        [
            feature_diagnostics[name]["median_cross_sectional_std_ddof1"]
            for name in features
        ]
    )
    summary = parquet_metadata(path, parquet_file)
    summary.update(
        {
            "dataset": market,
            "target_column": "target",
            "feature_count": len(features),
            "expected_feature_count": config["features"],
            "model_features": features,
            "saved_non_model_columns": [
                name for name in names if name == "__index_level_0__"
            ],
            "date_min": iso_date(ordered_dates[0]),
            "date_max": iso_date(ordered_dates[-1]),
            "unique_dates": len(ordered_dates),
            "unique_assets": len(asset_ids),
            "dates_ordered_non_decreasing": dates_ordered,
            "duplicate_asset_date_rows": duplicate_asset_date_rows,
            "duplicate_full_rows": 0 if duplicate_asset_date_rows == 0 else None,
            "rows_per_date": {
                "min": int(rows_per_date.min()),
                "median": float(np.median(rows_per_date)),
                "max": int(rows_per_date.max()),
            },
            "missingness_by_column": column_missing,
            "missingness_totals": {
                key: int(sum(item[key] for item in column_missing.values()))
                for key in ["null", "nan", "infinite"]
            },
            "splits": splits,
            "preprocessing_diagnostics": {
                "median_across_features_of_median_abs_cross_sectional_mean": float(
                    np.nanmedian(feature_mean_diagnostics)
                ),
                "median_across_features_of_median_cross_sectional_std": float(
                    np.nanmedian(feature_std_diagnostics)
                ),
                "target": feature_diagnostics["target"],
                "by_column": feature_diagnostics,
                "sample_rows": int(len(samples)),
            },
            "runtime_seconds": time.time() - started,
            "limitations": [
                "Forward-return timing cannot be verified from the parquet panel alone.",
                "Point-in-time feature availability cannot be verified without source construction code and source timestamps.",
                "Winsorization is inferred only from observed distributions; the original transform cannot be proven from final values.",
            ],
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    parquet_file = pq.ParquetFile(args.input)
    market = detect_market(args.input, parquet_file.schema_arrow.names, args.market)
    if market == "index":
        summary = inspect_index(args.input, parquet_file)
    else:
        summary = inspect_panel(
            args.input,
            parquet_file,
            market,
            args.batch_size,
            args.sample_rows,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
