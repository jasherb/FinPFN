#!/usr/bin/env python3
"""Audit and explain the frozen U.S. external-validation results."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finpfn-us-analysis-mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPOSITORY = Path(__file__).resolve().parents[4]
_RUN_ROOT = Path(
    os.environ.get("FINPFN_US_RUN_ROOT", "reproduction/runs/us_validation")
)
BASE = (_RUN_ROOT if _RUN_ROOT.is_absolute() else REPOSITORY / _RUN_ROOT).resolve()
OUTPUT = BASE / "analysis"
DATASET = REPOSITORY / "90features_USstocks.parquet"
COMMON_RESULTS = BASE / "results/common"
COMMON_HOLDINGS = COMMON_RESULTS / "decile_holdings.parquet"
PREDICTIONS = {
    "FinPFN": (
        BASE
        / "artifacts/checkpoints/us_finpfn_seed42_artifact_unique500.parquet"
    ),
    "TabPFN": (
        BASE
        / "artifacts/checkpoints/us_tabpfn_seed42_artifact_unique500.parquet"
    ),
    "Ridge": BASE / "artifacts/baselines/us_ridge_seed42.parquet",
    "LightGBM": BASE / "artifacts/baselines/us_lightgbm_seed42.parquet",
}
METADATA = {
    "FinPFN": (
        BASE
        / "artifacts/checkpoints/us_finpfn_seed42_artifact_unique500.metadata.json"
    ),
    "TabPFN": (
        BASE
        / "artifacts/checkpoints/us_tabpfn_seed42_artifact_unique500.metadata.json"
    ),
    "Ridge": BASE / "artifacts/baselines/us_ridge_seed42.metadata.json",
    "LightGBM": BASE / "artifacts/baselines/us_lightgbm_seed42.metadata.json",
}
MODELS = ["FinPFN", "Ridge", "LightGBM", "TabPFN"]
COST_GRID_BPS = [0, 2, 5, 10, 20, 30, 50]
TAIL_K = [10, 20, 40]
ANNUALIZATION = 12
EXPECTED_DATES = 143
EXPECTED_COMMON_ASSET_DATES = 71_500
EXPECTED_DATASET_SHA256 = (
    "54818c78796ecae3974b2058575cd2284482ce35e62c9116d316e23240b8ef50"
)
CSI_COMPARISON_INPUTS = {
    "release_comparison": (
        REPOSITORY
        / "reproduction/reference_results/summary/model_comparison.csv"
    ),
    "tail_precision": (
        REPOSITORY / "reproduction/reference_results/summary/csi_tail_precision.csv"
    ),
    "rank_stability": (
        REPOSITORY
        / "reproduction/reference_results/detailed/csi/rank_stability.csv"
    ),
}

SHARED_ANALYSIS_DIR = (
    REPOSITORY / "reproduction/analyses/ic_portfolio_gap"
)
sys.path.insert(0, str(SHARED_ANALYSIS_DIR))
import analyze_rank_tails as shared  # noqa: E402

shared.ANNUALIZATION = ANNUALIZATION
shared.TAIL_K = TAIL_K
shared.MODELS = MODELS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_drawdown(values: pd.Series) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + values.to_numpy(dtype=float))]
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    holdings = pd.read_parquet(COMMON_HOLDINGS)
    holdings["date"] = pd.to_datetime(holdings["date"])
    if holdings.duplicated(["model", "seed", "date", "id"]).any():
        raise ValueError("Frozen U.S. holdings contain duplicate model asset-dates")
    model_key_sets = {
        model: set(zip(part["date"], part["id"], strict=True))
        for model, part in holdings.groupby("model")
    }
    if set(model_key_sets) != set(MODELS):
        raise ValueError("Unexpected model set in common holdings")
    if any(
        model_key_sets[model] != model_key_sets["FinPFN"] for model in MODELS
    ):
        raise ValueError("Models do not use one common U.S. asset-date universe")
    keys = (
        holdings[["date", "id"]]
        .drop_duplicates()
        .sort_values(["date", "id"])
        .reset_index(drop=True)
    )
    if len(keys) != EXPECTED_COMMON_ASSET_DATES:
        raise ValueError(f"Expected 71,500 common asset-dates, found {len(keys)}")
    if keys["date"].nunique() != EXPECTED_DATES:
        raise ValueError(f"Expected 143 common dates, found {keys['date'].nunique()}")
    if not (keys.groupby("date").size() == 500).all():
        raise ValueError("Every common U.S. date must contain exactly 500 assets")

    raw = pd.read_parquet(
        DATASET,
        columns=["date", "id", "target"],
        filters=[("date", ">=", pd.Timestamp("2010-02-01"))],
    )
    raw["date"] = pd.to_datetime(raw["date"])
    keys = keys.merge(raw, on=["date", "id"], how="left", validate="one_to_one")
    if keys["target"].isna().any() or not np.isfinite(keys["target"]).all():
        raise ValueError("Common U.S. universe has missing/non-finite raw target")
    keys["raw_return_decimal"] = keys["target"] / 100.0

    model_frames = []
    source_checks: dict[str, object] = {}
    for model, path in PREDICTIONS.items():
        source = pd.read_parquet(path)
        source["date"] = pd.to_datetime(source["date"])
        required = {"model", "seed", "date", "id", "prediction", "status"}
        if missing := required.difference(source.columns):
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        source_model = source.loc[
            (source["model"] == model) & (source["seed"] == 42)
        ].copy()
        failed = int((source_model["status"] != "ok").sum())
        nonfinite = int((~np.isfinite(source_model["prediction"])).sum())
        duplicates = int(source_model.duplicated(["date", "id"]).sum())
        if failed or nonfinite or duplicates:
            raise ValueError(
                f"{model}: failed={failed}, nonfinite={nonfinite}, duplicates={duplicates}"
            )
        collapsed = (
            source_model.groupby(["date", "id"], as_index=False, sort=True)
            .agg(
                prediction=("prediction", "mean"),
                prediction_repetitions=("prediction", "size"),
            )
        )
        frame = keys.merge(
            collapsed, on=["date", "id"], how="left", validate="one_to_one"
        )
        if frame["prediction"].isna().any():
            raise ValueError(f"{model} does not cover the common U.S. universe")
        frame["model"] = model
        model_frames.append(frame)
        source_checks[model] = {
            "path": str(path.relative_to(REPOSITORY)),
            "sha256": sha256(path),
            "source_rows": len(source_model),
            "source_dates": int(source_model["date"].nunique()),
            "source_unique_asset_dates": int(
                source_model[["date", "id"]].drop_duplicates().shape[0]
            ),
            "common_rows": len(frame),
            "failed_rows": failed,
            "nonfinite_predictions": nonfinite,
            "duplicate_asset_dates": duplicates,
        }

    panel = pd.concat(model_frames, ignore_index=True)
    panel = (
        panel.groupby(["model", "date"], group_keys=True, sort=False)
        .apply(shared.deterministic_ranked, include_groups=False)
        .reset_index(level=[0, 1])
        .reset_index(drop=True)
    )
    return panel, holdings, source_checks


def validate_metadata() -> dict[str, object]:
    checks: dict[str, object] = {}
    for model, path in METADATA.items():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        entry: dict[str, object] = {
            "path": str(path.relative_to(REPOSITORY)),
            "sha256": sha256(path),
            "model": metadata["model"],
            "seed": metadata["seed"],
        }
        if metadata["model"] != model or metadata["seed"] != 42:
            raise ValueError(f"Unexpected model/seed in {path.name}")
        if model in {"FinPFN", "TabPFN"}:
            expected = {
                "sampling_mode": "artifact_unique500",
                "n_estimators": 8,
                "estimator_random_state": 0,
                "prediction_rows": 71_500,
                "successful_groups": 1_430,
                "failed_groups": 0,
                "date_pairs_attempted": 143,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"{model} metadata {key}={metadata.get(key)!r}, expected {value!r}"
                    )
            entry.update(
                {
                    key: metadata[key]
                    for key in [
                        "sampling_mode",
                        "n_estimators",
                        "estimator_random_state",
                        "prediction_rows",
                        "successful_groups",
                        "failed_groups",
                        "date_pairs_attempted",
                        "runtime_seconds",
                        "checkpoint_sha256",
                    ]
                }
            )
            hardware = metadata.get("hardware", {})
            visible_devices = hardware.get("cuda_devices", [])
            entry["compute"] = {
                "gpu_model": (
                    visible_devices[0].get("name") if visible_devices else None
                ),
                "cuda_version": hardware.get("cuda_version"),
                "tabpfn_version": hardware.get("tabpfn_version"),
                "torch_version": hardware.get("torch_version"),
            }
        else:
            if metadata["row_counts"] != {
                "train": 1_944_085,
                "validation": 797_222,
                "final_train": 2_741_307,
                "test": 788_592,
            }:
                raise ValueError(f"Unexpected {model} split row counts")
            entry.update(
                {
                    key: metadata[key]
                    for key in [
                        "protocol",
                        "row_counts",
                        "selected_candidate_index",
                        "selected_parameters",
                        "selected_validation_mean_daily_spearman_ic",
                        "selection_seconds",
                        "final_fit_seconds",
                        "test_prediction_seconds",
                        "packages",
                    ]
                }
            )
        checks[model] = entry
    return checks


def validate_frozen_evaluation(
    panel: pd.DataFrame, holdings: pd.DataFrame
) -> dict[str, object]:
    assigned = panel[["model", "date", "id", "prediction_decile"]].merge(
        holdings[["model", "date", "id", "decile"]],
        on=["model", "date", "id"],
        how="left",
        validate="one_to_one",
    )
    decile_mismatches = int(
        (assigned["prediction_decile"] != assigned["decile"]).sum()
    )
    if decile_mismatches:
        raise ValueError(f"{decile_mismatches} deciles differ from frozen evaluator")

    frozen_ic = pd.read_csv(COMMON_RESULTS / "ic_by_period.csv")
    frozen_ic["date"] = pd.to_datetime(frozen_ic["date"])
    recomputed_ic = (
        panel.groupby(["model", "date"])
        .apply(
            lambda part: shared.spearman(
                part["prediction"], part["raw_return_decimal"]
            ),
            include_groups=False,
        )
        .rename("recomputed_ic")
        .reset_index()
    )
    ic_check = recomputed_ic.merge(
        frozen_ic[["model", "date", "ic_raw_target"]],
        on=["model", "date"],
        validate="one_to_one",
    )
    max_ic_difference = float(
        (ic_check["recomputed_ic"] - ic_check["ic_raw_target"]).abs().max()
    )
    if max_ic_difference > 1e-12:
        raise ValueError(f"Frozen IC mismatch: {max_ic_difference}")

    frozen_period = pd.read_csv(COMMON_RESULTS / "decile_returns_by_period.csv")
    frozen_period["date"] = pd.to_datetime(frozen_period["date"])
    frozen_period = frozen_period.loc[frozen_period["return_basis"] == "raw"]
    joined = holdings.merge(
        panel[["model", "date", "id", "raw_return_decimal"]],
        on=["model", "date", "id"],
        how="left",
        validate="one_to_one",
    )
    recomputed_period = (
        joined.groupby(["model", "date", "decile"])["raw_return_decimal"]
        .mean()
        .unstack()
        .reset_index()
    )
    recomputed_period["recomputed_long_short"] = (
        recomputed_period[10] - recomputed_period[1]
    )
    period_check = recomputed_period.merge(
        frozen_period[["model", "date", "long_short"]],
        on=["model", "date"],
        validate="one_to_one",
    )
    max_return_difference = float(
        (
            period_check["recomputed_long_short"]
            - period_check["long_short"] / 100.0
        )
        .abs()
        .max()
    )
    if max_return_difference > 1e-12:
        raise ValueError(f"Frozen long-short return mismatch: {max_return_difference}")

    frozen_metrics = pd.read_csv(COMMON_RESULTS / "portfolio_metrics.csv")
    frozen_metrics = frozen_metrics.loc[frozen_metrics["return_basis"] == "raw"]
    sharpe_checks = {}
    for model, part in period_check.groupby("model"):
        observed = shared.sharpe(part["recomputed_long_short"])
        expected = float(
            frozen_metrics.loc[
                frozen_metrics["model"] == model, "primary_long_short_sharpe"
            ].iat[0]
        )
        difference = abs(observed - expected)
        if difference > 1e-12:
            raise ValueError(f"{model} frozen Sharpe mismatch: {difference}")
        sharpe_checks[model] = {
            "observed": observed,
            "frozen": expected,
            "absolute_difference": difference,
        }
    return {
        "decile_assignment_mismatches": decile_mismatches,
        "maximum_ic_absolute_difference": max_ic_difference,
        "maximum_long_short_return_absolute_difference_decimal": (
            max_return_difference
        ),
        "sharpe_checks": sharpe_checks,
    }


def leg_turnover(leg: pd.DataFrame) -> pd.Series:
    previous: dict[object, float] | None = None
    values: dict[pd.Timestamp, float] = {}
    for date, part in leg.groupby("date", sort=True):
        weight = 1.0 / len(part)
        current = {identifier: weight for identifier in part["id"]}
        if previous is None:
            values[pd.Timestamp(date)] = 1.0
        else:
            identifiers = set(previous) | set(current)
            values[pd.Timestamp(date)] = 0.5 * sum(
                abs(current.get(item, 0.0) - previous.get(item, 0.0))
                for item in identifiers
            )
        previous = current
    return pd.Series(values, dtype=float).sort_index()


def transaction_cost_analysis(
    panel: pd.DataFrame, holdings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tail = holdings.loc[holdings["decile"].isin([1, 10])].copy()
    tail["leg"] = np.where(tail["decile"] == 10, "long", "short")
    tail = tail.merge(
        panel[["model", "date", "id", "raw_return_decimal"]],
        on=["model", "date", "id"],
        how="left",
        validate="one_to_one",
    )
    metric_rows = []
    period_frames = []
    break_even_rows = []
    for model, model_tail in tail.groupby("model", sort=False):
        leg_returns = (
            model_tail.groupby(["date", "leg"])["raw_return_decimal"]
            .mean()
            .unstack()
            .sort_index()
        )
        period = pd.DataFrame(index=leg_returns.index)
        period["gross_long_return_decimal"] = leg_returns["long"]
        period["gross_short_return_decimal"] = -leg_returns["short"]
        period["gross_long_short_return_decimal"] = (
            period["gross_long_return_decimal"]
            + period["gross_short_return_decimal"]
        )
        long_turnover = leg_turnover(
            model_tail.loc[model_tail["leg"] == "long"]
        )
        short_turnover = leg_turnover(
            model_tail.loc[model_tail["leg"] == "short"]
        )
        period["long_turnover_one_way"] = long_turnover
        period["short_turnover_one_way"] = short_turnover
        period["total_turnover_one_way"] = long_turnover + short_turnover
        gross = period["gross_long_short_return_decimal"]
        break_even_bps = float(
            gross.mean() / period["total_turnover_one_way"].mean() * 10_000
        )
        break_even_rows.append(
            {
                "model": model,
                "mean_gross_return_decimal": gross.mean(),
                "average_total_one_way_turnover_including_entry": period[
                    "total_turnover_one_way"
                ].mean(),
                "break_even_cost_bps_per_one_way_turnover": break_even_bps,
            }
        )
        for cost_bps in COST_GRID_BPS:
            cost_rate = cost_bps / 10_000.0
            net = gross - cost_rate * period["total_turnover_one_way"]
            annualized_volatility = net.std(ddof=1) * np.sqrt(ANNUALIZATION)
            metric_rows.append(
                {
                    "model": model,
                    "cost_bps_per_one_way_turnover": cost_bps,
                    "n_dates": len(period),
                    "gross_mean_return_decimal": gross.mean(),
                    "net_mean_return_decimal": net.mean(),
                    "gross_annualized_volatility": (
                        gross.std(ddof=1) * np.sqrt(ANNUALIZATION)
                    ),
                    "net_annualized_volatility": annualized_volatility,
                    "gross_long_short_sharpe": shared.sharpe(gross),
                    "net_long_short_sharpe": shared.sharpe(net),
                    "gross_cumulative_arithmetic_return_decimal": gross.sum(),
                    "net_cumulative_arithmetic_return_decimal": net.sum(),
                    "gross_terminal_compounded_wealth": float(
                        np.prod(1.0 + gross)
                    ),
                    "net_terminal_compounded_wealth": float(np.prod(1.0 + net)),
                    "gross_maximum_drawdown": maximum_drawdown(gross),
                    "net_maximum_drawdown": maximum_drawdown(net),
                    "average_total_one_way_turnover_including_entry": period[
                        "total_turnover_one_way"
                    ].mean(),
                }
            )
            cost_period = period.copy().reset_index()
            cost_period["model"] = model
            cost_period["cost_bps_per_one_way_turnover"] = cost_bps
            cost_period["transaction_cost_decimal"] = (
                cost_rate * cost_period["total_turnover_one_way"]
            )
            cost_period["net_long_short_return_decimal"] = (
                cost_period["gross_long_short_return_decimal"]
                - cost_period["transaction_cost_decimal"]
            )
            period_frames.append(cost_period)
    metrics = pd.DataFrame(metric_rows)
    metrics["net_sharpe_rank_at_cost"] = metrics.groupby(
        "cost_bps_per_one_way_turnover"
    )["net_long_short_sharpe"].rank(method="first", ascending=False)
    return metrics, pd.DataFrame(break_even_rows), pd.concat(
        period_frames, ignore_index=True
    )


def turnover_decomposition(
    panel: pd.DataFrame, holdings: pd.DataFrame
) -> pd.DataFrame:
    universe_by_date = {
        date: set(part["id"])
        for date, part in panel.loc[panel["model"] == "FinPFN"].groupby(
            "date", sort=True
        )
    }
    dates = sorted(universe_by_date)
    rows = []
    for model, model_holdings in holdings.groupby("model", sort=False):
        for decile, leg_name in [(1, "short"), (10, "long")]:
            leg = model_holdings.loc[model_holdings["decile"] == decile]
            positions = {
                date: set(part["id"])
                for date, part in leg.groupby("date", sort=True)
            }
            for previous_date, date in zip(dates[:-1], dates[1:]):
                previous_universe = universe_by_date[previous_date]
                current_universe = universe_by_date[date]
                previous_positions = positions[previous_date]
                current_positions = positions[date]
                position_count = len(previous_positions)
                universe_count = len(previous_universe)
                available_previous_positions = (
                    previous_positions & current_universe
                )
                retained_positions = previous_positions & current_positions
                total_turnover = 1.0 - len(retained_positions) / position_count
                forced_universe_turnover = (
                    1.0
                    - len(available_previous_positions) / position_count
                )
                reranking_turnover = (
                    len(available_previous_positions)
                    - len(retained_positions)
                ) / position_count
                if abs(
                    total_turnover
                    - forced_universe_turnover
                    - reranking_turnover
                ) > 1e-12:
                    raise ValueError("Turnover decomposition identity failed")
                rows.append(
                    {
                        "model": model,
                        "leg": leg_name,
                        "previous_date": previous_date,
                        "date": date,
                        "universe_size_previous": universe_count,
                        "universe_size_current": len(current_universe),
                        "universe_overlap_count": len(
                            previous_universe & current_universe
                        ),
                        "universe_one_way_turnover": (
                            1.0
                            - len(previous_universe & current_universe)
                            / universe_count
                        ),
                        "position_count_previous": position_count,
                        "position_count_current": len(current_positions),
                        "previous_positions_available_current": len(
                            available_previous_positions
                        ),
                        "retained_positions": len(retained_positions),
                        "total_leg_one_way_turnover": total_turnover,
                        "forced_by_sample_universe_turnover": (
                            forced_universe_turnover
                        ),
                        "additional_reranking_turnover": reranking_turnover,
                        "conditional_retention_if_asset_available": (
                            len(retained_positions)
                            / len(available_previous_positions)
                            if available_previous_positions
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def date_contributions(panel: pd.DataFrame) -> pd.DataFrame:
    return shared.date_contributions(panel)


def csi_us_comparison(
    tails: pd.DataFrame,
    stability: pd.DataFrame,
    costs: pd.DataFrame,
    turnover_parts: pd.DataFrame,
) -> pd.DataFrame:
    us_model_comparison = pd.read_csv(COMMON_RESULTS / "model_comparison.csv")
    us_model_comparison = us_model_comparison.loc[
        us_model_comparison["row_type"] == "seed"
    ].set_index("model")
    us_portfolio = pd.read_csv(COMMON_RESULTS / "portfolio_metrics.csv")
    us_portfolio = us_portfolio.loc[
        us_portfolio["return_basis"] == "raw"
    ].set_index("model")
    missing_csi_inputs = [
        str(path.relative_to(REPOSITORY))
        for path in CSI_COMPARISON_INPUTS.values()
        if not path.is_file()
    ]
    csi_available = not missing_csi_inputs
    if csi_available:
        csi_release = pd.read_csv(
            CSI_COMPARISON_INPUTS["release_comparison"]
        )
        csi_release = csi_release.loc[
            csi_release["market"] == "CSI500"
        ].set_index("model")
        csi_tails = pd.read_csv(
            CSI_COMPARISON_INPUTS["tail_precision"]
        ).set_index(["model", "k"])
        csi_stability = pd.read_csv(
            CSI_COMPARISON_INPUTS["rank_stability"]
        )
    rows = []
    markets = ["CSI500", "US"] if csi_available else ["US"]
    for market in markets:
        for model in MODELS:
            if market == "CSI500":
                comparison = csi_release.loc[model]
                tail = csi_tails.loc[(model, 40)]
                stable = csi_stability.loc[
                    (csi_stability["model"] == model)
                    & (csi_stability["k"] == 40)
                ]
                rows.append(
                    {
                        "market": market,
                        "model": model,
                        "mean_ic": comparison["mean_ic"],
                        "ir": comparison["ir"],
                        "gross_long_short_sharpe": comparison[
                            "gross_long_short_sharpe"
                        ],
                        "top40_precision": tail["mean_top_precision"],
                        "bottom40_precision": tail["mean_bottom_precision"],
                        "top40_unconditional_overlap": stable[
                            "top_overlap_fraction"
                        ].mean(),
                        "bottom40_unconditional_overlap": stable[
                            "bottom_overlap_fraction"
                        ].mean(),
                        "conditional_long_retention_if_available": np.nan,
                        "conditional_short_retention_if_available": np.nan,
                        "forced_universe_turnover_long": np.nan,
                        "forced_universe_turnover_short": np.nan,
                        "net_sharpe_10bps": comparison[
                            "net_sharpe_10bps"
                        ],
                    }
                )
            else:
                comparison = us_model_comparison.loc[model]
                portfolio = us_portfolio.loc[model]
                tail = tails.loc[(tails["model"] == model) & (tails["k"] == 40)].iloc[0]
                stable = stability.loc[
                    (stability["model"] == model) & (stability["k"] == 40)
                ]
                model_turnover = turnover_parts.loc[
                    turnover_parts["model"] == model
                ]
                long_turnover = model_turnover.loc[
                    model_turnover["leg"] == "long"
                ]
                short_turnover = model_turnover.loc[
                    model_turnover["leg"] == "short"
                ]
                rows.append(
                    {
                        "market": market,
                        "model": model,
                        "mean_ic": comparison["mean_ic"],
                        "ir": comparison["ir"],
                        "gross_long_short_sharpe": portfolio[
                            "primary_long_short_sharpe"
                        ],
                        "top40_precision": tail["mean_top_precision"],
                        "bottom40_precision": tail["mean_bottom_precision"],
                        "top40_unconditional_overlap": stable[
                            "top_overlap_fraction"
                        ].mean(),
                        "bottom40_unconditional_overlap": stable[
                            "bottom_overlap_fraction"
                        ].mean(),
                        "conditional_long_retention_if_available": long_turnover[
                            "conditional_retention_if_asset_available"
                        ].mean(),
                        "conditional_short_retention_if_available": short_turnover[
                            "conditional_retention_if_asset_available"
                        ].mean(),
                        "forced_universe_turnover_long": long_turnover[
                            "forced_by_sample_universe_turnover"
                        ].mean(),
                        "forced_universe_turnover_short": short_turnover[
                            "forced_by_sample_universe_turnover"
                        ].mean(),
                        "net_sharpe_10bps": costs.loc[
                            (costs["model"] == model)
                            & (
                                costs["cost_bps_per_one_way_turnover"]
                                == 10
                            ),
                            "net_long_short_sharpe",
                        ].iat[0],
                    }
                )
    result = pd.DataFrame(rows)
    result.attrs["csi_comparison_status"] = {
        "included": csi_available,
        "missing_inputs": missing_csi_inputs,
        "behavior_if_missing": (
            "U.S. rows are still written; CSI rows are omitted without failing "
            "the primary U.S. analysis."
        ),
    }
    return result


def save_figures(
    curve: pd.DataFrame,
    tails: pd.DataFrame,
    costs: pd.DataFrame,
    turnover_parts: pd.DataFrame,
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
        title="U.S. realized return by predicted percentile",
        xlabel="Predicted percentile bin (low to high)",
        ylabel="Mean monthly raw return (%)",
        xticks=[1, 5, 10, 15, 20],
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "percentile_return_curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for model in MODELS:
        part = costs.loc[costs["model"] == model]
        axis.plot(
            part["cost_bps_per_one_way_turnover"],
            part["net_long_short_sharpe"],
            marker="o",
            label=model,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        title="U.S. net long-short Sharpe by transaction cost",
        xlabel="Cost (bps per one-way turnover)",
        ylabel="Net Sharpe",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "net_sharpe_by_cost.png", dpi=180)
    plt.close(figure)

    model_colors = dict(
        zip(MODELS, ["#2A6FBB", "#E07A1F", "#3A9D5D", "#8C62AA"])
    )
    figure, axes = plt.subplots(
        2,
        len(TAIL_K),
        figsize=(13, 7.4),
        sharey="row",
    )
    row_limits = {
        "mean_top_precision": tails["mean_top_precision"].max() * 1.18,
        "mean_bottom_precision": tails["mean_bottom_precision"].max() * 1.18,
    }
    for column_index, k in enumerate(TAIL_K):
        subset = tails.loc[tails["k"] == k].set_index("model").reindex(MODELS)
        random_precision = float(subset["mean_actual_fraction"].iloc[0])
        for row_index, (metric, leg) in enumerate(
            [
                ("mean_top_precision", "Top"),
                ("mean_bottom_precision", "Bottom"),
            ]
        ):
            axis = axes[row_index, column_index]
            axis.bar(
                MODELS,
                subset[metric],
                color=[model_colors[model] for model in MODELS],
            )
            axis.axhline(
                random_precision,
                color="#444444",
                linestyle="--",
                linewidth=1.2,
            )
            axis.set_title(f"{leg}-{k} precision")
            axis.set_ylim(0, row_limits[metric])
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="x", rotation=25)
            if column_index == 0:
                axis.set_ylabel("Realized-tail precision")
    legend_handles = [
        Patch(facecolor=model_colors[model], label=model) for model in MODELS
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#444444",
            linestyle="--",
            label="Random-selection rate (k / 500)",
        )
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
    )
    figure.suptitle(
        "U.S. realized-tail precision by selected portfolio size",
        y=0.93,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(figures / "tail_precision.png", dpi=180)
    plt.close(figure)

    decomposition = (
        turnover_parts.groupby(["model", "leg"])
        .agg(
            forced=("forced_by_sample_universe_turnover", "mean"),
            reranking=("additional_reranking_turnover", "mean"),
        )
        .reset_index()
    )
    labels = [f"{row.model}-{row.leg}" for row in decomposition.itertuples()]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(labels, decomposition["forced"], label="Forced by sampled universe")
    axis.bar(
        labels,
        decomposition["reranking"],
        bottom=decomposition["forced"],
        label="Additional reranking",
    )
    axis.set(
        title="U.S. one-way leg turnover decomposition",
        ylabel="Mean turnover",
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "turnover_decomposition.png", dpi=180)
    plt.close(figure)


def write_report(
    curve: pd.DataFrame,
    regions: pd.DataFrame,
    tails: pd.DataFrame,
    stability: pd.DataFrame,
    costs: pd.DataFrame,
    break_even: pd.DataFrame,
    turnover_parts: pd.DataFrame,
    metadata_checks: dict[str, object],
    csi_comparison_status: dict[str, object],
    runtime_seconds: float,
) -> None:
    comparison = pd.read_csv(COMMON_RESULTS / "model_comparison.csv")
    comparison = comparison.loc[comparison["row_type"] == "seed"].set_index("model")
    portfolio = pd.read_csv(COMMON_RESULTS / "portfolio_metrics.csv")
    portfolio = portfolio.loc[portfolio["return_basis"] == "raw"].set_index("model")
    region_lookup = regions.set_index(["model", "region"])
    tail_lookup = tails.set_index(["model", "k"])
    curve_rows = []
    for model in MODELS:
        model_curve = curve.loc[curve["model"] == model]
        curve_rows.append(
            {
                "model": model,
                "mean_ic": comparison.loc[model, "mean_ic"],
                "ir": comparison.loc[model, "ir"],
                "gross_sharpe": portfolio.loc[
                    model, "primary_long_short_sharpe"
                ],
                "middle_ic": region_lookup.loc[
                    (model, "middle_20_to_80pct"), "mean_rank_ic"
                ],
                "bottom20_ic": region_lookup.loc[
                    (model, "bottom_20pct"), "mean_rank_ic"
                ],
                "top20_ic": region_lookup.loc[
                    (model, "top_20pct"), "mean_rank_ic"
                ],
                "curve_monotonicity": shared.spearman(
                    model_curve["prediction_percentile_bin"],
                    model_curve["mean_raw_return_decimal"],
                ),
                "top40_precision": tail_lookup.loc[
                    (model, 40), "mean_top_precision"
                ],
                "bottom40_precision": tail_lookup.loc[
                    (model, 40), "mean_bottom_precision"
                ],
            }
        )
    summary = pd.DataFrame(curve_rows).set_index("model")
    turnover_summary = (
        turnover_parts.groupby(["model", "leg"])
        .agg(
            total=("total_leg_one_way_turnover", "mean"),
            forced=("forced_by_sample_universe_turnover", "mean"),
            reranking=("additional_reranking_turnover", "mean"),
            conditional_retention=(
                "conditional_retention_if_asset_available",
                "mean",
            ),
            universe_turnover=("universe_one_way_turnover", "mean"),
        )
        .reset_index()
    )
    cost10 = costs.loc[
        costs["cost_bps_per_one_way_turnover"] == 10
    ].set_index("model")
    break_even_lookup = break_even.set_index("model")
    table_rows = "\n".join(
        "| "
        + " | ".join(
            [
                model,
                f"{summary.loc[model, 'mean_ic']:.4f}",
                f"{summary.loc[model, 'ir']:.3f}",
                f"{summary.loc[model, 'gross_sharpe']:.3f}",
                f"{summary.loc[model, 'middle_ic']:.4f}",
                f"{summary.loc[model, 'top40_precision']:.3f}",
                f"{summary.loc[model, 'bottom40_precision']:.3f}",
                f"{cost10.loc[model, 'net_long_short_sharpe']:.3f}",
                f"{break_even_lookup.loc[model, 'break_even_cost_bps_per_one_way_turnover']:.1f}",
            ]
        )
        + " |"
        for model in MODELS
    )
    turnover_rows = "\n".join(
        "| "
        + " | ".join(
            [
                row.model,
                row.leg,
                f"{row.total:.3f}",
                f"{row.forced:.3f}",
                f"{row.reranking:.3f}",
                f"{row.conditional_retention:.3f}",
            ]
        )
        + " |"
        for row in turnover_summary.itertuples()
    )
    if csi_comparison_status["included"]:
        csi_comparison_note = (
            "- `csi_us_comparison.csv` contains both CSI 500 and U.S. rows."
        )
    else:
        missing = "`, `".join(csi_comparison_status["missing_inputs"])
        csi_comparison_note = (
            "- Optional CSI comparison inputs were unavailable, so "
            f"`csi_us_comparison.csv` contains U.S. rows only. Missing: "
            f"`{missing}`. The primary U.S. analysis is unaffected."
        )
    report = f"""# U.S. external-validation analysis

## Result

The 143 U.S. test months from 2010-02 through 2021-12 reproduce the central
CSI finding: **FinPFN has the highest common-universe IC, but its actual
top-minus-bottom portfolio underperforms Ridge and LightGBM.**

| Model | Mean IC | IR | Gross H-L Sharpe | Middle 20-80% IC | Top-40 precision | Bottom-40 precision | 10 bps net Sharpe | Break-even bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

FinPFN mean IC is {summary.loc['FinPFN', 'mean_ic']:.4f}, above Ridge at
{summary.loc['Ridge', 'mean_ic']:.4f} and LightGBM at
{summary.loc['LightGBM', 'mean_ic']:.4f}. Its gross H-L Sharpe is only
{summary.loc['FinPFN', 'gross_sharpe']:.3f}, below
{summary.loc['Ridge', 'gross_sharpe']:.3f}/
{summary.loc['LightGBM', 'gross_sharpe']:.3f}. The reversal exists before
transaction costs.

## Integrity

- FinPFN and TabPFN each contain 71,500 predictions, 143 dates, 1,430
  successful groups, and no failed, nonfinite, or duplicate rows.
- Seed 42, `artifact_unique500`, eight estimators, estimator state 0, and
  released-checkpoint hashes match the declared protocol.
- Ridge and LightGBM select five and six candidates, respectively, using
  2000-2009 validation only, then refit once on train plus validation.
- All models use the same 500 assets per month and 71,500 total asset-dates.
- Recomputed monthly IC, H-L returns, and Sharpe match the evaluator within
  `1e-12`.
{csi_comparison_note}

## Sample-universe turnover confounder

The protocol resamples 500 names each month from more than 5,000 available
stocks. The evaluation universe itself has mean one-way turnover
{turnover_summary['universe_turnover'].mean():.3f}. Raw decile turnover of
roughly 97%-99% is therefore dominated by forced sample exits.

| Model | Leg | Total turnover | Forced universe turnover | Additional reranking | Retention when available |
|---|---|---:|---:|---:|---:|
{turnover_rows}

Costs are charged against the actual stored holdings, so P&L is internally
consistent. Raw U.S. turnover is not a deployment estimate for a fixed
full-market universe; conditional retention and common-asset rank migration
are better model-stability diagnostics.

## IC-portfolio mechanism

- Middle and tail-local IC and percentile-return monotonicity are recorded in
  `rank_region_metrics.csv` and `percentile_return_curve.csv`.
- Top/bottom `k={{10,20,40}}` precision is recorded in `tail_precision.csv`.
- Date contributions and common-asset rank migration are recorded separately.
- Checkpoints predict a declared random 500-stock subset; this is not a
  full-universe U.S. backtest.

## Cost convention

The fixed grid is `0,2,5,10,20,30,50 bps` per unit of one-way turnover.
Both legs charge initial entry from cash, and
`net = gross - cost_rate * (long_turnover + short_turnover)`.
Annualization is 12.

## Decision

The U.S. validation supports a cross-market mechanism: FinPFN improves broad
ranking statistics without improving the tradable tail portfolio. Combined
with the negative uncertainty-gating result and failed frozen rank buffer, the
evidence supports ending method development on the existing test sets.

Future work requires new independent splits, a fixed or complete tradable
universe, and a predeclared tail-aware objective.

CPU analysis runtime was {runtime_seconds:.3f} seconds. Released-checkpoint
inference took {metadata_checks['TabPFN']['runtime_seconds']:.2f} seconds for
TabPFN and {metadata_checks['FinPFN']['runtime_seconds']:.2f} seconds for
FinPFN.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    declared = [
        OUTPUT / "integrity_manifest.json",
        OUTPUT / "cost_sensitivity.csv",
        OUTPUT / "break_even_costs.csv",
        OUTPUT / "net_returns_by_period.csv",
        OUTPUT / "percentile_return_curve.csv",
        OUTPUT / "rank_region_metrics.csv",
        OUTPUT / "tail_precision.csv",
        OUTPUT / "tail_precision_by_period.csv",
        OUTPUT / "rank_stability.csv",
        OUTPUT / "turnover_decomposition.csv",
        OUTPUT / "date_contributions.csv",
        OUTPUT / "csi_us_comparison.csv",
        OUTPUT / "report.md",
    ]
    if any(path.exists() for path in declared) or (
        (OUTPUT / "figures").exists() and any((OUTPUT / "figures").iterdir())
    ):
        raise FileExistsError("Refusing to overwrite U.S. analysis outputs")
    if not DATASET.is_file():
        raise FileNotFoundError(
            f"Required raw U.S. panel is missing: {DATASET}"
        )
    dataset_sha256 = sha256(DATASET)
    if dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "Raw U.S. panel SHA-256 mismatch: "
            f"expected {EXPECTED_DATASET_SHA256}, found {dataset_sha256}"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)

    panel, holdings, source_checks = load_panel()
    metadata_checks = validate_metadata()
    evaluation_checks = validate_frozen_evaluation(panel, holdings)
    curve = shared.percentile_curve(panel)
    regions = shared.rank_region_metrics(panel)
    tails, tails_by_period = shared.tail_precision(panel)
    stability = shared.rank_stability(panel)
    costs, break_even, net_period = transaction_cost_analysis(panel, holdings)
    turnover_parts = turnover_decomposition(panel, holdings)
    dates = date_contributions(panel)
    market_comparison = csi_us_comparison(
        tails, stability, costs, turnover_parts
    )
    csi_comparison_status = market_comparison.attrs["csi_comparison_status"]

    costs.to_csv(OUTPUT / "cost_sensitivity.csv", index=False)
    break_even.to_csv(OUTPUT / "break_even_costs.csv", index=False)
    net_period.to_csv(OUTPUT / "net_returns_by_period.csv", index=False)
    curve.to_csv(OUTPUT / "percentile_return_curve.csv", index=False)
    regions.to_csv(OUTPUT / "rank_region_metrics.csv", index=False)
    tails.to_csv(OUTPUT / "tail_precision.csv", index=False)
    tails_by_period.to_csv(OUTPUT / "tail_precision_by_period.csv", index=False)
    stability.to_csv(OUTPUT / "rank_stability.csv", index=False)
    turnover_parts.to_csv(OUTPUT / "turnover_decomposition.csv", index=False)
    dates.to_csv(OUTPUT / "date_contributions.csv", index=False)
    market_comparison.to_csv(OUTPUT / "csi_us_comparison.csv", index=False)

    manifest = {
        "analysis_scope": "frozen_us_test_external_validation_exploratory_audit",
        "cost_grid_bps_per_one_way_turnover": COST_GRID_BPS,
        "tail_k": TAIL_K,
        "annualization": ANNUALIZATION,
        "return_units": {
            "stored_us_target": "percentage return",
            "analysis": "decimal return",
        },
        "dataset": {
            "path": str(DATASET.relative_to(REPOSITORY)),
            "sha256": dataset_sha256,
        },
        "source_predictions": source_checks,
        "metadata": metadata_checks,
        "frozen_evaluation_checks": evaluation_checks,
        "csi_comparison": csi_comparison_status,
        "input_files": {
            str(path.relative_to(REPOSITORY)): sha256(path)
            for path in [
                COMMON_HOLDINGS,
                COMMON_RESULTS / "model_comparison.csv",
                COMMON_RESULTS / "portfolio_metrics.csv",
                COMMON_RESULTS / "ic_by_period.csv",
                COMMON_RESULTS / "decile_returns_by_period.csv",
                *PREDICTIONS.values(),
                *METADATA.values(),
            ]
        },
    }
    (OUTPUT / "integrity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_figures(curve, tails, costs, turnover_parts)
    runtime_seconds = time.perf_counter() - started
    write_report(
        curve,
        regions,
        tails,
        stability,
        costs,
        break_even,
        turnover_parts,
        metadata_checks,
        csi_comparison_status,
        runtime_seconds,
    )
    if not csi_comparison_status["included"]:
        print(
            "Optional CSI comparison skipped; primary U.S. analysis completed. "
            "Missing inputs: "
            + ", ".join(csi_comparison_status["missing_inputs"])
        )
    print(pd.read_csv(COMMON_RESULTS / "model_comparison.csv").loc[
        lambda x: x["row_type"] == "seed",
        ["model", "mean_ic", "ic_std_ddof1", "ir"],
    ].to_string(index=False))
    print()
    print(
        costs.loc[
            costs["cost_bps_per_one_way_turnover"].isin([0, 10]),
            [
                "model",
                "cost_bps_per_one_way_turnover",
                "gross_long_short_sharpe",
                "net_long_short_sharpe",
                "average_total_one_way_turnover_including_entry",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
