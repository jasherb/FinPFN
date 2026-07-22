#!/usr/bin/env python3
"""Artifact-faithful CSI checkpoint inference with ensemble-member summaries."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

for variable_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable_name, "4")

import numpy as np
import pandas as pd
import tabpfn
import torch
from tabpfn import TabPFNRegressor


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPTS = REPOSITORY / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from reproduction_common import MARKET_CONFIG, load_panel, validate_panel  # noqa: E402
from run_checkpoint_inference import (  # noqa: E402
    checkpoint_path,
    make_group,
    sample_groups,
    sha256,
)

from predict_with_members import (  # noqa: E402
    DEFAULT_QUANTILES,
    PINNED_TABPFN_VERSION,
    predict_with_members,
    verify_against_public_predict,
)


BASE_COLUMNS = [
    "market",
    "model",
    "seed",
    "split",
    "date",
    "id",
    "context_date",
    "group_id",
    "prediction",
    "prediction_mean",
    "prediction_mode",
    "predictive_std",
    "predictive_q10",
    "predictive_q25",
    "predictive_q50",
    "predictive_q75",
    "predictive_q90",
    "target_group_z",
    "status",
    "inference_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--market", choices=["csi500"], default="csi500")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument(
        "--models", nargs="+", choices=["FinPFN", "TabPFN"], required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--sampling-mode",
        choices=["notebook_with_replacement"],
        default="notebook_with_replacement",
    )
    parser.add_argument("--stocks-per-group", type=int, default=50)
    parser.add_argument("--sample-assets", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--estimator-random-state", type=int, default=0)
    parser.add_argument("--estimator-n-jobs", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-date-pairs", type=int)
    parser.add_argument("--max-groups-per-date", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--verify-reference-output",
        action="store_true",
        help="On the first successful group, rerun public predict and compare outputs",
    )
    return parser.parse_args()


def hardware_record(device: str) -> dict[str, object]:
    record: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "requested_device": device,
        "torch_version": torch.__version__,
        "tabpfn_version": tabpfn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        record["visible_cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    return record


def output_columns(n_estimators: int) -> list[str]:
    member_columns = [f"member_{index}_mean" for index in range(n_estimators)]
    member_columns += [f"member_{index}_median" for index in range(n_estimators)]
    return BASE_COLUMNS + member_columns


def run_model(
    *,
    frame_by_date: dict[pd.Timestamp, pd.DataFrame],
    dates: list[pd.Timestamp],
    features: list[str],
    model: str,
    checkpoint: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.RandomState(seed)
    rows: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    started = time.perf_counter()
    attempted_groups = 0
    successful_groups = 0
    reference_comparison: dict[str, float] | None = None
    first_config_descriptions: list[str] | None = None
    columns = output_columns(args.n_estimators)

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
                predictions = predict_with_members(estimator, query_x)
                if args.verify_reference_output and reference_comparison is None:
                    reference_comparison = verify_against_public_predict(
                        estimator, query_x, predictions
                    )
                if first_config_descriptions is None:
                    first_config_descriptions = predictions[
                        "member_config_descriptions"
                    ]

                result["market"] = args.market
                result["model"] = model
                result["seed"] = seed
                result["split"] = args.split
                result["context_date"] = context_date
                result["group_id"] = group_id
                result["prediction"] = predictions["median"]
                result["prediction_mean"] = predictions["mean"]
                result["prediction_mode"] = predictions["mode"]
                result["predictive_std"] = predictions["predictive_std"]
                for quantile, suffix, values in zip(
                    DEFAULT_QUANTILES,
                    ("q10", "q25", "q50", "q75", "q90"),
                    predictions["quantiles"],
                    strict=True,
                ):
                    if not np.isfinite(quantile):
                        raise ValueError("Non-finite quantile declaration")
                    result[f"predictive_{suffix}"] = values
                for index in range(args.n_estimators):
                    result[f"member_{index}_mean"] = predictions["member_means"][:, index]
                    result[f"member_{index}_median"] = predictions[
                        "member_medians"
                    ][:, index]
                result["status"] = "ok"
                result["inference_seconds"] = time.perf_counter() - group_started
                rows.append(result[columns])
                successful_groups += 1
            except Exception as error:
                failure = pd.DataFrame(
                    {
                        "market": args.market,
                        "model": model,
                        "seed": seed,
                        "split": args.split,
                        "date": query_date,
                        "id": identifiers,
                        "context_date": context_date,
                        "group_id": group_id,
                        "target_group_z": np.nan,
                        "status": f"error:{type(error).__name__}",
                        "inference_seconds": time.perf_counter() - group_started,
                    }
                )
                for column in columns:
                    if column not in failure:
                        failure[column] = np.nan
                rows.append(failure[columns])
                failures.append(
                    {
                        "context_date": context_date.isoformat(),
                        "query_date": query_date.isoformat(),
                        "group_id": group_id,
                        "samples": int(len(identifiers)),
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:500],
                    }
                )
                if not args.continue_on_error:
                    raise

    predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    numeric_outputs = [
        "prediction",
        "prediction_mean",
        "prediction_mode",
        "predictive_std",
        "predictive_q10",
        "predictive_q25",
        "predictive_q50",
        "predictive_q75",
        "predictive_q90",
    ] + [f"member_{index}_{kind}" for kind in ("mean", "median") for index in range(args.n_estimators)]
    finite_counts = {
        column: int(np.isfinite(pd.to_numeric(predictions[column], errors="coerce")).sum())
        for column in numeric_outputs
    }
    metadata: dict[str, object] = {
        "market": args.market,
        "model": model,
        "split": args.split,
        "seed": seed,
        "sampling_mode": args.sampling_mode,
        "stocks_per_group": args.stocks_per_group,
        "sample_assets": args.sample_assets,
        "n_estimators": args.n_estimators,
        "estimator_random_state": args.estimator_random_state,
        "estimator_n_jobs": args.estimator_n_jobs,
        "features": features,
        "checkpoint_sha256": sha256(checkpoint),
        "attempted_groups": attempted_groups,
        "successful_groups": successful_groups,
        "failed_groups": attempted_groups - successful_groups,
        "prediction_rows": int(len(predictions)),
        "successful_prediction_rows": int((predictions["status"] == "ok").sum()),
        "date_pairs_attempted": len(date_pairs),
        "runtime_seconds": time.perf_counter() - started,
        "finite_output_counts": finite_counts,
        "reference_output_verification": reference_comparison,
        "first_group_member_config_descriptions": first_config_descriptions,
        "member_order_note": (
            "Columns follow the fitted TabPFN 2.0.8 executor iterator order; "
            "downstream dispersion measures are permutation invariant."
        ),
        "distribution_note": (
            "predictive_std and quantiles describe the aggregate bar distribution; "
            "member columns describe individual ensemble forward passes after border "
            "translation. Raw high-dimensional logits are intentionally not stored."
        ),
        "failures": failures,
        "hardware": hardware_record(args.device),
    }
    return predictions, metadata


def main() -> None:
    args = parse_args()
    if tabpfn.__version__ != PINNED_TABPFN_VERSION:
        raise RuntimeError(
            f"Expected tabpfn=={PINNED_TABPFN_VERSION}, found {tabpfn.__version__}"
        )
    if args.n_estimators != 8:
        raise ValueError("Artifact-faithful uncertainty run requires 8 estimators")
    if args.stocks_per_group != 50:
        raise ValueError("Artifact-faithful uncertainty run requires 50 stocks/group")
    if args.estimator_random_state != 0:
        raise ValueError("Notebook-exact estimator random state must be 0")
    if args.estimator_n_jobs != 4:
        raise ValueError("Prepared resource policy requires 4 CPU workers")

    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    uncertainty_root = (
        REPOSITORY / "reproduction/next_phase/uncertainty/artifacts"
    ).resolve()
    try:
        output_dir.relative_to(uncertainty_root)
    except ValueError as error:
        raise ValueError(f"Output must remain under {uncertainty_root}") from error
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_panel(dataset, args.market, split=args.split)
    features = validate_panel(frame, args.market)
    if frame.duplicated(["date", "id"]).any():
        raise ValueError("Dataset contains duplicate asset-date rows")
    dates = frame["date"].drop_duplicates().sort_values().tolist()
    frame_by_date = {
        date: part.reset_index(drop=True) for date, part in frame.groupby("date", sort=True)
    }

    for seed in args.seeds:
        for model in args.models:
            checkpoint = checkpoint_path(REPOSITORY, args.market, model)
            slug = (
                f"{args.market}_{model.lower()}_seed{seed}_{args.split}_"
                f"{args.sampling_mode}_members"
            )
            prediction_path = output_dir / f"{slug}.parquet"
            metadata_path = output_dir / f"{slug}.metadata.json"
            existing = [path for path in (prediction_path, metadata_path) if path.exists()]
            if existing:
                raise FileExistsError(
                    "Refusing to overwrite uncertainty artifacts: "
                    + ", ".join(path.name for path in existing)
                )
            predictions, metadata = run_model(
                frame_by_date=frame_by_date,
                dates=dates,
                features=features,
                model=model,
                checkpoint=checkpoint,
                seed=seed,
                args=args,
            )
            predictions.to_parquet(prediction_path, index=False)
            metadata["prediction_sha256"] = sha256(prediction_path)
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"{model} {args.split} seed={seed}: {len(predictions)} rows, "
                f"{metadata['successful_groups']}/{metadata['attempted_groups']} groups, "
                f"{metadata['runtime_seconds']:.2f}s"
            )


if __name__ == "__main__":
    main()
