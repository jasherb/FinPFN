#!/usr/bin/env python3
"""Temporal validation and final fitting shared by reconstructed baselines."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

for variable_name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(variable_name, "4")

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats

from reproduction_common import MARKET_CONFIG, PREDICTION_COLUMNS, load_panel, validate_panel


def parser_for(model: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Fit the independently reconstructed {model} baseline"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--market", choices=sorted(MARKET_CONFIG), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument(
        "--smoke-rows-per-split",
        type=int,
        help="Deterministic spread sample for compatibility testing only",
    )
    return parser


def spread_sample(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame
    positions = np.linspace(0, len(frame) - 1, maximum, dtype=np.int64)
    return frame.iloc[positions].reset_index(drop=True)


def mean_daily_ic(dates: pd.Series, target: np.ndarray, prediction: np.ndarray) -> float:
    frame = pd.DataFrame({"date": dates.to_numpy(), "target": target, "prediction": prediction})

    def correlation(group: pd.DataFrame) -> float:
        if group["target"].nunique() < 2 or group["prediction"].nunique() < 2:
            return np.nan
        return float(stats.spearmanr(group["target"], group["prediction"]).statistic)

    return float(frame.groupby("date", sort=True).apply(correlation, include_groups=False).mean())


def load_candidates(model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "configs/baseline_search.json"
    configuration = json.loads(path.read_text())
    return configuration[model], configuration["protocol"]


def hardware_record() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }


def run_baseline(
    *,
    args: argparse.Namespace,
    model_name: str,
    make_estimator: Callable[[dict[str, Any], int], Any],
    package_versions: dict[str, str],
) -> None:
    candidates, protocol = load_candidates(model_name)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ValueError("No hyperparameter candidates selected")

    train = spread_sample(
        load_panel(args.dataset, args.market, split="train"),
        args.smoke_rows_per_split,
    )
    validation = spread_sample(
        load_panel(args.dataset, args.market, split="validation"),
        args.smoke_rows_per_split,
    )
    test = spread_sample(
        load_panel(args.dataset, args.market, split="test"),
        args.smoke_rows_per_split,
    )
    features = validate_panel(train, args.market)
    for name, frame in {"validation": validation, "test": test}.items():
        other_features = validate_panel(frame, args.market)
        if other_features != features:
            raise ValueError(f"{name} feature order differs from training")

    scale = MARKET_CONFIG[args.market]["return_to_percentage_points"]
    train_x = train[features].to_numpy(dtype=np.float32)
    validation_x = validation[features].to_numpy(dtype=np.float32)
    train_y = train["target"].to_numpy(dtype=np.float64) * scale
    validation_y = validation["target"].to_numpy(dtype=np.float64) * scale

    validation_results = []
    selection_started = time.perf_counter()
    best_score = -np.inf
    best_index = 0
    for index, parameters in enumerate(candidates):
        started = time.perf_counter()
        estimator = make_estimator(parameters, args.seed)
        estimator.fit(train_x, train_y)
        prediction = estimator.predict(validation_x)
        score = mean_daily_ic(validation["date"], validation_y, prediction)
        validation_results.append(
            {
                "candidate_index": index,
                "parameters": parameters,
                "validation_mean_daily_spearman_ic": score,
                "fit_and_validation_seconds": time.perf_counter() - started,
            }
        )
        print(f"candidate={index} validation_mean_ic={score:.8f} parameters={parameters}")
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_index = index

    selection_seconds = time.perf_counter() - selection_started

    selected = candidates[best_index]
    combined = pd.concat([train, validation], ignore_index=True)
    combined_x = combined[features].to_numpy(dtype=np.float32)
    combined_y = combined["target"].to_numpy(dtype=np.float64) * scale
    final_estimator = make_estimator(selected, args.seed)
    fit_started = time.perf_counter()
    final_estimator.fit(combined_x, combined_y)
    final_fit_seconds = time.perf_counter() - fit_started
    test_x = test[features].to_numpy(dtype=np.float32)
    prediction_started = time.perf_counter()
    test_prediction = np.asarray(final_estimator.predict(test_x))
    prediction_seconds = time.perf_counter() - prediction_started

    output = pd.DataFrame(
        {
            "market": args.market,
            "model": model_name,
            "seed": args.seed,
            "date": test["date"].to_numpy(),
            "id": test["id"].to_numpy(),
            "context_date": pd.NaT,
            "group_id": -1,
            "prediction": test_prediction,
            "prediction_mean": test_prediction,
            "target_group_z": np.nan,
            "status": "ok",
            "inference_seconds": prediction_seconds,
        }
    )[PREDICTION_COLUMNS]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{args.market}_{model_name.lower()}_seed{args.seed}"
    if args.smoke_rows_per_split is not None:
        slug += "_smoke"
    output.to_parquet(args.output_dir / f"{slug}.parquet", index=False)
    joblib.dump(final_estimator, args.output_dir / f"{slug}.joblib")
    metadata = {
        "market": args.market,
        "model": model_name,
        "seed": args.seed,
        "protocol": protocol,
        "smoke_rows_per_split": args.smoke_rows_per_split,
        "row_counts": {
            "train": len(train),
            "validation": len(validation),
            "final_train": len(combined),
            "test": len(test),
        },
        "features": features,
        "target_scale_from_stored_value": scale,
        "validation_results": validation_results,
        "selected_candidate_index": best_index,
        "selected_parameters": selected,
        "selected_validation_mean_daily_spearman_ic": best_score,
        "selection_seconds": selection_seconds,
        "final_fit_seconds": final_fit_seconds,
        "test_prediction_seconds": prediction_seconds,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            **package_versions,
        },
        "hardware": hardware_record(),
    }
    (args.output_dir / f"{slug}.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    search_rows = []
    for result in validation_results:
        search_rows.append(
            {
                "candidate_index": result["candidate_index"],
                "selected": result["candidate_index"] == best_index,
                **result["parameters"],
                "validation_mean_daily_spearman_ic": result[
                    "validation_mean_daily_spearman_ic"
                ],
                "fit_and_validation_seconds": result[
                    "fit_and_validation_seconds"
                ],
            }
        )
    pd.DataFrame(search_rows).to_csv(
        args.output_dir / f"{slug}.validation_search.csv", index=False
    )
    print(f"selected={selected} validation_mean_ic={best_score:.8f}")
    print(f"wrote {len(output)} test predictions to {args.output_dir / f'{slug}.parquet'}")
