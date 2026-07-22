#!/usr/bin/env python3
"""Generate new predictions from released TabPFN/FinPFN checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

for variable_name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(variable_name, "4")

import numpy as np
import pandas as pd
import tabpfn
import torch
from tabpfn import TabPFNRegressor

from reproduction_common import (
    MARKET_CONFIG,
    PREDICTION_COLUMNS,
    load_panel,
    validate_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--market", choices=sorted(MARKET_CONFIG), required=True)
    parser.add_argument(
        "--models", nargs="+", choices=["FinPFN", "TabPFN"], default=["FinPFN", "TabPFN"]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--sampling-mode",
        choices=["artifact_unique500", "notebook_with_replacement", "all_common"],
        default="artifact_unique500",
    )
    parser.add_argument("--stocks-per-group", type=int, default=50)
    parser.add_argument("--sample-assets", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument(
        "--estimator-random-state",
        type=int,
        default=0,
        help=(
            "TabPFN ensemble/preprocessing seed. The released notebook omits this "
            "argument, so tabpfn==2.0.8 uses its default value of 0. This is "
            "intentionally separate from --seeds, which controls stock sampling."
        ),
    )
    parser.add_argument(
        "--estimator-n-jobs",
        type=int,
        default=4,
        help="CPU workers available to TabPFN preprocessing/inference.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-date-pairs", type=int)
    parser.add_argument("--max-groups-per-date", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing prediction/metadata pair",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(repository: Path, market: str, model: str) -> Path:
    relative = (
        MARKET_CONFIG[market]["checkpoint"]
        if model == "FinPFN"
        else "models/tabpfn-v2-regressor.ckpt"
    )
    path = repository / relative
    if not path.is_file():
        raise FileNotFoundError(f"Released checkpoint is absent: {relative}")
    return path


def target_zscore(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return values * 0.0
    return (values - values.mean()) / standard_deviation


def sample_groups(
    common_ids: np.ndarray,
    *,
    mode: str,
    rng: np.random.RandomState,
    stocks_per_group: int,
    sample_assets: int,
) -> list[np.ndarray]:
    if mode == "notebook_with_replacement":
        count = int(np.ceil(len(common_ids) / stocks_per_group))
        sampled = rng.choice(
            common_ids,
            size=(count, stocks_per_group),
            replace=True,
        )
        # The released notebook sorts each sampled two-date group by date and ID
        # before splitting context/query rows. The dates are already separated in
        # this implementation, so sorting each identifier row is equivalent.
        return [np.sort(identifiers) for identifiers in sampled]

    shuffled = rng.permutation(common_ids)
    if mode == "artifact_unique500":
        usable = min(sample_assets, len(shuffled))
        usable -= usable % stocks_per_group
        shuffled = shuffled[:usable]
    if len(shuffled) == 0:
        return []
    return [
        shuffled[start : start + stocks_per_group]
        for start in range(0, len(shuffled), stocks_per_group)
    ]


def make_group(
    context: pd.DataFrame,
    query: pd.DataFrame,
    identifiers: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    context_group = context.set_index("id").loc[identifiers].reset_index()
    query_group = query.set_index("id").loc[identifiers].reset_index()
    context_group["target_group_z"] = target_zscore(context_group["target"])
    query_group["target_group_z"] = target_zscore(query_group["target"])
    return (
        context_group[features].to_numpy(dtype=np.float32),
        context_group["target_group_z"].to_numpy(dtype=np.float32),
        query_group[features].to_numpy(dtype=np.float32),
        query_group[["date", "id", "target_group_z"]].copy(),
    )


def hardware_record(device: str) -> dict[str, object]:
    record: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "requested_device": device,
        "torch_version": torch.__version__,
        "tabpfn_version": tabpfn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        record["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return record


def run_model(
    *,
    frame_by_date: dict[pd.Timestamp, pd.DataFrame],
    dates: list[pd.Timestamp],
    features: list[str],
    market: str,
    model: str,
    checkpoint: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.RandomState(seed)
    output_rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    started = time.perf_counter()
    attempted_groups = 0
    successful_groups = 0
    date_pairs = list(zip(dates[:-1], dates[1:], strict=True))
    if args.max_date_pairs is not None:
        date_pairs = date_pairs[: args.max_date_pairs]

    for context_date, query_date in date_pairs:
        context = frame_by_date[context_date]
        query = frame_by_date[query_date]
        common_ids = np.intersect1d(
            context["id"].to_numpy(), query["id"].to_numpy(), assume_unique=True
        )
        groups = sample_groups(
            common_ids,
            mode=args.sampling_mode,
            rng=rng,
            stocks_per_group=args.stocks_per_group,
            sample_assets=args.sample_assets,
        )
        if args.max_groups_per_date is not None:
            groups = groups[: args.max_groups_per_date]

        for group_id, identifiers in enumerate(groups):
            attempted_groups += 1
            group_started = time.perf_counter()
            try:
                context_x, context_y, query_x, result = make_group(
                    context, query, identifiers, features
                )
                estimator = TabPFNRegressor(
                    model_path=checkpoint,
                    device=args.device,
                    n_estimators=args.n_estimators,
                    random_state=args.estimator_random_state,
                    n_jobs=args.estimator_n_jobs,
                ).fit(context_x, context_y)
                predictions = estimator.predict(query_x, output_type="full")
                result["market"] = market
                result["model"] = model
                result["seed"] = seed
                result["context_date"] = context_date
                result["group_id"] = group_id
                result["prediction"] = np.asarray(predictions["median"])
                result["prediction_mean"] = np.asarray(predictions["mean"])
                result["status"] = "ok"
                result["inference_seconds"] = time.perf_counter() - group_started
                output_rows.append(result[PREDICTION_COLUMNS])
                successful_groups += 1
            except Exception as error:  # recorded and optionally continued for coverage audit
                failure_rows = pd.DataFrame(
                    {
                        "market": market,
                        "model": model,
                        "seed": seed,
                        "date": query_date,
                        "id": identifiers,
                        "context_date": context_date,
                        "group_id": group_id,
                        "prediction": np.nan,
                        "prediction_mean": np.nan,
                        "target_group_z": np.nan,
                        "status": f"error:{type(error).__name__}",
                        "inference_seconds": time.perf_counter() - group_started,
                    }
                )
                output_rows.append(failure_rows[PREDICTION_COLUMNS])
                failure = {
                    "context_date": context_date.isoformat(),
                    "query_date": query_date.isoformat(),
                    "group_id": group_id,
                    "samples": int(len(identifiers)),
                    "error_type": type(error).__name__,
                }
                failures.append(failure)
                if not args.continue_on_error:
                    raise

    if output_rows:
        predictions = pd.concat(output_rows, ignore_index=True)
    else:
        predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)
    metadata = {
        "market": market,
        "model": model,
        "seed": seed,
        "sampling_mode": args.sampling_mode,
        "stocks_per_group": args.stocks_per_group,
        "sample_assets": args.sample_assets,
        "n_estimators": args.n_estimators,
        "estimator_random_state": args.estimator_random_state,
        "estimator_n_jobs": args.estimator_n_jobs,
        "checkpoint_sha256": sha256(checkpoint),
        "features": features,
        "date_pairs_attempted": len(date_pairs),
        "attempted_groups": attempted_groups,
        "successful_groups": successful_groups,
        "failed_groups": len(failures),
        "failed_samples": int(sum(item["samples"] for item in failures)),
        "failures": failures,
        "prediction_rows": int(len(predictions)),
        "successful_prediction_rows": int((predictions["status"] == "ok").sum()),
        "runtime_seconds": time.perf_counter() - started,
        "hardware": hardware_record(args.device),
    }
    return predictions, metadata


def main() -> None:
    args = parse_args()
    if args.stocks_per_group < 2:
        raise ValueError("--stocks-per-group must be at least 2")
    if args.sample_assets < args.stocks_per_group:
        raise ValueError("--sample-assets must be at least --stocks-per-group")
    if args.estimator_n_jobs == 0 or args.estimator_n_jobs < -1:
        raise ValueError("--estimator-n-jobs must be -1 or a positive integer")

    repository = Path(__file__).resolve().parents[2]
    dataset = args.dataset.resolve()
    frame = load_panel(dataset, args.market, split="test")
    features = validate_panel(frame, args.market)
    if frame.duplicated(["date", "id"]).any():
        raise ValueError("Dataset contains duplicate asset-date rows")
    dates = frame["date"].drop_duplicates().sort_values().tolist()
    frame_by_date = {
        date: part.reset_index(drop=True) for date, part in frame.groupby("date", sort=True)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for model in args.models:
            checkpoint = checkpoint_path(repository, args.market, model)
            slug = f"{args.market}_{model.lower()}_seed{seed}_{args.sampling_mode}"
            prediction_path = args.output_dir / f"{slug}.parquet"
            metadata_path = args.output_dir / f"{slug}.metadata.json"
            existing = [path for path in [prediction_path, metadata_path] if path.exists()]
            if existing and not args.overwrite:
                names = ", ".join(path.name for path in existing)
                raise FileExistsError(
                    f"Refusing to replace existing artifacts: {names}. "
                    "Move them aside or pass --overwrite explicitly."
                )
            predictions, metadata = run_model(
                frame_by_date=frame_by_date,
                dates=dates,
                features=features,
                market=args.market,
                model=model,
                checkpoint=checkpoint,
                seed=seed,
                args=args,
            )
            predictions.to_parquet(prediction_path, index=False)
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            print(
                f"{model} sampling_seed={seed} "
                f"estimator_random_state={args.estimator_random_state}: "
                f"{len(predictions)} predictions, "
                f"{metadata['failed_groups']} failed groups, "
                f"{metadata['runtime_seconds']:.2f}s"
            )


if __name__ == "__main__":
    main()
