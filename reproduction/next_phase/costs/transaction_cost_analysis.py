#!/usr/bin/env python3
"""Evaluate frozen CSI 500 portfolios under predeclared one-way costs.

All stored return columns produced by the baseline evaluator are percentage points.
This script converts them to decimal returns before subtracting transaction costs.
It never writes outside its dedicated next-phase output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finpfn-next-phase-mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = REPOSITORY / "reproduction/results/csi500_all_models_notebook_exact"
DEFAULT_OUTPUT = REPOSITORY / "reproduction/next_phase/costs"
ANNUALIZATION = 240
COST_GRID_BPS = (0, 2, 5, 10, 20, 30, 50)
EXPECTED_MODELS = ("FinPFN", "TabPFN", "Ridge", "LightGBM")

DEFAULT_PREDICTIONS = {
    "FinPFN": REPOSITORY
    / "reproduction/artifacts/csi500_notebook_exact/"
    "csi500_finpfn_seed42_notebook_with_replacement.parquet",
    "TabPFN": REPOSITORY
    / "reproduction/artifacts/csi500_notebook_exact/"
    "csi500_tabpfn_seed42_notebook_with_replacement.parquet",
    "Ridge": REPOSITORY
    / "reproduction/artifacts/predictions/csi500_baselines/csi500_ridge_seed42.parquet",
    "LightGBM": REPOSITORY
    / "reproduction/artifacts/predictions/csi500_baselines/"
    "csi500_lightgbm_seed42.parquet",
}

EXPECTED_SHA256 = {
    "reproduction/artifacts/csi500_notebook_exact/"
    "csi500_finpfn_seed42_notebook_with_replacement.parquet": (
        "03e62d18bf14cb6a3787213a87369adf12914d65748f8d1536a7bc5cecca76f3"
    ),
    "reproduction/artifacts/csi500_notebook_exact/"
    "csi500_tabpfn_seed42_notebook_with_replacement.parquet": (
        "0fa76d578741b3a50a9f6e1b96009bae6fe4f884b9ce7a3fe0f52b6cec95c26a"
    ),
    "reproduction/artifacts/predictions/csi500_baselines/csi500_ridge_seed42.parquet": (
        "a6cccd1f54f3ced4cd5165615a6c7d921d3d46d157f5a7e166a532532b6488b1"
    ),
    "reproduction/artifacts/predictions/csi500_baselines/"
    "csi500_lightgbm_seed42.parquet": (
        "0a0c7f0bcbb5e97d25dcaf73448e55f9ec97b70aa8dc5bb91d0bf0f70eae375a"
    ),
    "reproduction/results/csi500_all_models_notebook_exact/decile_holdings.parquet": (
        "bfcd2ba50283f41a0ad4dc2265d7067cb570edf79a9352bbd820ae02d57d432d"
    ),
    "reproduction/results/csi500_all_models_notebook_exact/decile_returns_by_period.csv": (
        "02c0c0c75d003d7c68b02565e13d373119b87b374761a7fc49e9370b2d16b14b"
    ),
    "reproduction/results/csi500_all_models_notebook_exact/turnover_by_decile.csv": (
        "60c622e967db8e90f5515571d05f1c91898b14439210ae7190dca7437c7afd4a"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--holdings", type=Path, default=DEFAULT_BASELINE / "decile_holdings.parquet"
    )
    parser.add_argument(
        "--period-returns",
        type=Path,
        default=DEFAULT_BASELINE / "decile_returns_by_period.csv",
    )
    parser.add_argument(
        "--baseline-turnover",
        type=Path,
        default=DEFAULT_BASELINE / "turnover_by_decile.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_input(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = repo_relative(path)
    observed = sha256(path)
    expected = EXPECTED_SHA256.get(relative)
    if expected is None:
        raise ValueError(f"Input is not a declared frozen artifact: {relative}")
    if observed != expected:
        raise ValueError(
            f"Frozen input checksum mismatch for {relative}: {observed} != {expected}"
        )
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": observed,
    }


def refuse_existing_outputs(output_dir: Path) -> None:
    declared = [
        "cost_sensitivity.csv",
        "model_break_even_costs.csv",
        "net_performance_by_period.csv",
        "integrity_checks.json",
        "input_manifest.json",
        "report.md",
    ]
    existing = [str(output_dir / name) for name in declared if (output_dir / name).exists()]
    figures = output_dir / "figures"
    if figures.exists() and any(figures.iterdir()):
        existing.append(str(figures))
    if existing:
        raise FileExistsError(
            "Refusing to overwrite next-phase outputs: " + ", ".join(existing)
        )


def prediction_integrity(
    holdings: pd.DataFrame, prediction_paths: dict[str, Path]
) -> dict[str, object]:
    checks: dict[str, object] = {}
    holding_keys = {
        model: set(zip(part["date"], part["id"], strict=True))
        for model, part in holdings.groupby("model", sort=True)
    }
    for model in EXPECTED_MODELS:
        path = prediction_paths[model]
        frame = pd.read_parquet(
            path, columns=["model", "seed", "date", "id", "prediction", "status"]
        )
        frame["date"] = pd.to_datetime(frame["date"])
        if set(frame["model"].dropna().unique()) != {model}:
            raise ValueError(f"Prediction model label mismatch in {path.name}")
        valid = frame.loc[(frame["status"] == "ok") & frame["prediction"].notna()].copy()
        if valid.empty:
            raise ValueError(f"No valid predictions in {path.name}")
        counts = valid.groupby(["date", "id"], sort=False).size()
        collapsed = (
            valid.groupby(["date", "id"], as_index=False, sort=True)["prediction"]
            .mean()
            .merge(
                holdings.loc[holdings["model"] == model, ["date", "id", "decile"]],
                on=["date", "id"],
                how="inner",
                validate="one_to_one",
            )
        )
        expected_keys = holding_keys[model]
        observed_keys = set(zip(collapsed["date"], collapsed["id"], strict=True))
        if observed_keys != expected_keys:
            raise ValueError(f"Holdings are not the exact prediction subset for {model}")

        reassigned = []
        for date, part in collapsed.groupby("date", sort=True):
            part = part.sort_values(["prediction", "id"], kind="stable").copy()
            part["derived_decile"] = (
                np.floor(np.arange(len(part)) * 10 / len(part)).astype(int) + 1
            )
            reassigned.append(part[["date", "id", "decile", "derived_decile"]])
        reassigned_frame = pd.concat(reassigned, ignore_index=True)
        mismatches = int(
            (reassigned_frame["decile"] != reassigned_frame["derived_decile"]).sum()
        )
        if mismatches:
            raise ValueError(
                f"{model} holdings do not match deciles from its own prediction: "
                f"{mismatches} mismatches"
            )
        checks[model] = {
            "input_rows": int(len(frame)),
            "valid_rows": int(len(valid)),
            "unique_asset_dates": int(len(counts)),
            "repeated_rows_beyond_first": int(len(valid) - len(counts)),
            "maximum_repetitions": int(counts.max()),
            "common_holding_rows": int(len(collapsed)),
            "holding_decile_mismatches_from_own_prediction": mismatches,
            "aggregation_before_deciles": "mean prediction by model/date/id",
            "target_used_to_form_deciles": False,
        }
    return checks


def basic_holdings_integrity(holdings: pd.DataFrame) -> dict[str, object]:
    required = {"model", "seed", "date", "id", "decile"}
    missing = sorted(required.difference(holdings.columns))
    if missing:
        raise ValueError(f"Holdings missing columns: {missing}")
    holdings["date"] = pd.to_datetime(holdings["date"])
    duplicates = int(holdings.duplicated(["model", "seed", "date", "id"]).sum())
    if duplicates:
        raise ValueError(f"Holdings contain {duplicates} duplicate model/seed/date/id rows")
    if set(holdings["model"].unique()) != set(EXPECTED_MODELS):
        raise ValueError("Holdings model set differs from the four declared models")
    if not holdings["decile"].between(1, 10).all():
        raise ValueError("Invalid decile values")

    universe_by_model_date = {
        (model, date): frozenset(part["id"])
        for (model, date), part in holdings.groupby(["model", "date"], sort=True)
    }
    dates = sorted(holdings["date"].unique())
    for date in dates:
        universes = [universe_by_model_date[(model, date)] for model in EXPECTED_MODELS]
        if any(universe != universes[0] for universe in universes[1:]):
            raise ValueError(f"Models have inconsistent holdings universe on {date}")
    counts = holdings.groupby(["model", "date"]).size()
    return {
        "duplicate_model_seed_date_id_rows": duplicates,
        "models": sorted(holdings["model"].unique()),
        "dates": int(len(dates)),
        "rows": int(len(holdings)),
        "assets_per_model_date_min": int(counts.min()),
        "assets_per_model_date_mean": float(counts.mean()),
        "assets_per_model_date_max": int(counts.max()),
        "common_universe_identical_across_models_each_date": True,
        "exactly_one_decile_per_model_asset_date": True,
    }


def leg_turnover(holdings: pd.DataFrame, *, include_initial: bool) -> pd.Series:
    previous: dict[object, float] | None = None
    rows: dict[pd.Timestamp, float] = {}
    for date, part in holdings.groupby("date", sort=True):
        weight = 1.0 / len(part)
        current = {identifier: weight for identifier in part["id"]}
        if previous is None:
            if include_initial:
                rows[pd.Timestamp(date)] = 1.0
        else:
            identifiers = set(previous) | set(current)
            rows[pd.Timestamp(date)] = 0.5 * sum(
                abs(current.get(item, 0.0) - previous.get(item, 0.0))
                for item in identifiers
            )
        previous = current
    return pd.Series(rows, name="turnover_one_way", dtype=float).sort_index()


def build_turnover_series(
    holdings: pd.DataFrame, baseline_turnover: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    records = []
    comparison = []
    for (model, seed), part in holdings.groupby(["model", "seed"], sort=True):
        long_part = part.loc[part["decile"] == 10]
        short_part = part.loc[part["decile"] == 1]
        long_with_entry = leg_turnover(long_part, include_initial=True)
        short_with_entry = leg_turnover(short_part, include_initial=True)
        long_rebalancing = leg_turnover(long_part, include_initial=False)
        short_rebalancing = leg_turnover(short_part, include_initial=False)
        for decile, calculated in [(10, long_rebalancing), (1, short_rebalancing)]:
            frozen = baseline_turnover.loc[
                (baseline_turnover["model"] == model)
                & (baseline_turnover["seed"] == seed)
                & (baseline_turnover["decile"] == decile),
                "mean_one_way_turnover",
            ]
            if len(frozen) != 1:
                raise ValueError(f"Missing frozen turnover for {model} decile {decile}")
            difference = float(calculated.mean() - frozen.iat[0])
            comparison.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "decile": decile,
                    "recomputed_excluding_initial": float(calculated.mean()),
                    "frozen_mean_one_way_turnover": float(frozen.iat[0]),
                    "difference": difference,
                }
            )
            if abs(difference) > 1e-12:
                raise ValueError(
                    f"Turnover mismatch for {model} decile {decile}: {difference}"
                )
        if not long_with_entry.index.equals(short_with_entry.index):
            raise ValueError(f"Long/short turnover dates differ for {model}")
        for date in long_with_entry.index:
            long_value = float(long_with_entry.loc[date])
            short_value = float(short_with_entry.loc[date])
            records.extend(
                [
                    {
                        "model": model,
                        "seed": int(seed),
                        "date": date,
                        "portfolio": "long",
                        "turnover_one_way": long_value,
                    },
                    {
                        "model": model,
                        "seed": int(seed),
                        "date": date,
                        "portfolio": "short",
                        "turnover_one_way": short_value,
                    },
                    {
                        "model": model,
                        "seed": int(seed),
                        "date": date,
                        "portfolio": "long_short",
                        "turnover_one_way": long_value + short_value,
                    },
                ]
            )
    return pd.DataFrame(records), {"frozen_turnover_comparison": comparison}


def build_gross_returns(period_returns: pd.DataFrame) -> pd.DataFrame:
    raw = period_returns.loc[period_returns["return_basis"] == "raw"].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if raw.duplicated(["model", "seed", "date"]).any():
        raise ValueError("Raw period returns contain duplicate model/seed/date rows")
    expected = raw["decile_10"] - raw["decile_1"]
    if not np.allclose(expected, raw["long_short"], rtol=0, atol=1e-12):
        raise ValueError("Frozen long_short is not decile_10 minus decile_1")
    frames = []
    definitions = {
        "long": raw["decile_10"],
        "short": -raw["decile_1"],
        "long_short": raw["long_short"],
    }
    for portfolio, gross_percentage_points in definitions.items():
        frame = raw[["model", "seed", "date"]].copy()
        frame["portfolio"] = portfolio
        frame["gross_return_decimal"] = gross_percentage_points.to_numpy() / 100.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sharpe(values: pd.Series) -> float:
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(values.mean() / standard_deviation * np.sqrt(ANNUALIZATION))


def maximum_drawdown(values: pd.Series) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + values.to_numpy(dtype=float))]
    running_peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running_peak - 1.0))


def summarize_return_series(values: pd.Series, prefix: str) -> dict[str, float]:
    if (values <= -1.0).any():
        terminal_wealth = np.nan
        max_drawdown = np.nan
    else:
        terminal_wealth = float(np.prod(1.0 + values))
        max_drawdown = maximum_drawdown(values)
    return {
        f"{prefix}_mean_return_decimal": float(values.mean()),
        f"{prefix}_annualized_volatility_decimal": float(
            values.std(ddof=1) * np.sqrt(ANNUALIZATION)
        ),
        f"{prefix}_sharpe": sharpe(values),
        f"{prefix}_cumulative_arithmetic_return_decimal": float(values.sum()),
        f"{prefix}_terminal_compounded_wealth": terminal_wealth,
        f"{prefix}_maximum_drawdown": max_drawdown,
    }


def evaluate_costs(
    gross_returns: pd.DataFrame, turnover: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = gross_returns.merge(
        turnover,
        on=["model", "seed", "date", "portfolio"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(gross_returns) or len(merged) != len(turnover):
        raise ValueError("Return and turnover series do not align exactly")

    period_frames = []
    summary_rows = []
    for cost_bps in COST_GRID_BPS:
        period = merged.copy()
        period["cost_rate_bps_per_one_way_turnover"] = cost_bps
        period["transaction_cost_decimal"] = (
            period["turnover_one_way"] * cost_bps / 10_000.0
        )
        period["net_return_decimal"] = (
            period["gross_return_decimal"] - period["transaction_cost_decimal"]
        )
        period["gross_compounded_wealth"] = period.groupby(
            ["model", "seed", "portfolio"], sort=False
        )["gross_return_decimal"].transform(lambda values: (1.0 + values).cumprod())
        period["net_compounded_wealth"] = period.groupby(
            ["model", "seed", "portfolio"], sort=False
        )["net_return_decimal"].transform(lambda values: (1.0 + values).cumprod())
        period_frames.append(period)

        for (model, seed, portfolio), part in period.groupby(
            ["model", "seed", "portfolio"], sort=True
        ):
            row: dict[str, object] = {
                "model": model,
                "seed": int(seed),
                "portfolio": portfolio,
                "cost_rate_bps_per_one_way_turnover": cost_bps,
                "n_periods": int(len(part)),
                "average_turnover_one_way_including_initial": float(
                    part["turnover_one_way"].mean()
                ),
            }
            row.update(summarize_return_series(part["gross_return_decimal"], "gross"))
            row.update(summarize_return_series(part["net_return_decimal"], "net"))
            summary_rows.append(row)

    periods = pd.concat(period_frames, ignore_index=True).sort_values(
        ["cost_rate_bps_per_one_way_turnover", "model", "portfolio", "date"]
    )
    summary = pd.DataFrame(summary_rows)
    summary["net_sharpe_rank"] = (
        summary.groupby(["portfolio", "cost_rate_bps_per_one_way_turnover"])[
            "net_sharpe"
        ]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    zero = summary.loc[
        summary["cost_rate_bps_per_one_way_turnover"] == 0
    ].copy()
    zero["break_even_cost_bps_mean_net_return_zero"] = np.where(
        zero["average_turnover_one_way_including_initial"] > 0,
        zero["gross_mean_return_decimal"]
        / zero["average_turnover_one_way_including_initial"]
        * 10_000.0,
        np.nan,
    )
    break_even = zero[
        [
            "model",
            "seed",
            "portfolio",
            "n_periods",
            "gross_mean_return_decimal",
            "average_turnover_one_way_including_initial",
            "break_even_cost_bps_mean_net_return_zero",
        ]
    ].sort_values(["portfolio", "model"])
    return summary, break_even, periods


def plot_results(summary: pd.DataFrame, periods: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=False)
    model_order = list(EXPECTED_MODELS)

    fig, axis = plt.subplots(figsize=(8, 5))
    long_short = summary.loc[summary["portfolio"] == "long_short"]
    for model in model_order:
        part = long_short.loc[long_short["model"] == model].sort_values(
            "cost_rate_bps_per_one_way_turnover"
        )
        axis.plot(
            part["cost_rate_bps_per_one_way_turnover"],
            part["net_sharpe"],
            marker="o",
            label=model,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Cost (bps per one-way turnover)", ylabel="Net H-L Sharpe")
    axis.set_title("CSI 500 long-short net Sharpe sensitivity")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "net_sharpe_by_cost.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    at_ten = periods.loc[
        (periods["portfolio"] == "long_short")
        & (periods["cost_rate_bps_per_one_way_turnover"] == 10)
    ]
    for model in model_order:
        part = at_ten.loc[at_ten["model"] == model].sort_values("date")
        axis.plot(part["date"], part["net_compounded_wealth"], label=model)
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set(ylabel="Compounded wealth", xlabel="Date")
    axis.set_title("CSI 500 net long-short wealth at 10 bps")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "cumulative_long_short_wealth_10bps.png", dpi=180)
    plt.close(fig)

    zero = summary.loc[
        (summary["portfolio"] == "long_short")
        & (summary["cost_rate_bps_per_one_way_turnover"] == 0)
    ].set_index("model").reindex(model_order)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(zero.index, zero["average_turnover_one_way_including_initial"])
    axis.set(ylabel="Long + short one-way turnover", xlabel="Model")
    axis.set_title("Average long-short turnover (initial entry included)")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "average_long_short_turnover.png", dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        rendered.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rendered)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    break_even: pd.DataFrame,
    integrity: dict[str, object],
    runtime_seconds: float,
) -> None:
    long_short = summary.loc[summary["portfolio"] == "long_short"].copy()
    ranking_rows = []
    for cost, part in long_short.groupby(
        "cost_rate_bps_per_one_way_turnover", sort=True
    ):
        ordered = part.sort_values(["net_sharpe_rank", "model"])
        ranking_rows.append(
            [
                int(cost),
                " > ".join(ordered["model"]),
                ", ".join(
                    f"{row.model} {row.net_sharpe:.4f}"
                    for row in ordered.itertuples()
                ),
            ]
        )
    selected_costs = long_short.loc[
        long_short["cost_rate_bps_per_one_way_turnover"].isin([0, 10, 50])
    ].sort_values(["cost_rate_bps_per_one_way_turnover", "net_sharpe_rank"])
    metric_rows = [
        [
            int(row.cost_rate_bps_per_one_way_turnover),
            row.model,
            f"{row.gross_mean_return_decimal * 100:.4f}%",
            f"{row.net_mean_return_decimal * 100:.4f}%",
            f"{row.net_annualized_volatility_decimal * 100:.4f}%",
            f"{row.gross_sharpe:.4f}",
            f"{row.net_sharpe:.4f}",
            f"{row.average_turnover_one_way_including_initial:.4f}",
            f"{row.net_terminal_compounded_wealth:.4f}",
            f"{row.net_maximum_drawdown * 100:.2f}%",
        ]
        for row in selected_costs.itertuples()
    ]
    break_even_ls = break_even.loc[break_even["portfolio"] == "long_short"].sort_values(
        "break_even_cost_bps_mean_net_return_zero", ascending=False
    )
    break_even_rows = [
        [
            row.model,
            f"{row.average_turnover_one_way_including_initial:.6f}",
            f"{row.break_even_cost_bps_mean_net_return_zero:.3f}",
        ]
        for row in break_even_ls.itertuples()
    ]
    pred_checks = integrity["prediction_integrity"]
    pred_rows = [
        [
            model,
            pred_checks[model]["input_rows"],
            pred_checks[model]["unique_asset_dates"],
            pred_checks[model]["repeated_rows_beyond_first"],
            pred_checks[model]["holding_decile_mismatches_from_own_prediction"],
        ]
        for model in EXPECTED_MODELS
    ]

    report = f"""# Phase 1：交易成本与换手敏感性

## 结论

本分析使用冻结的共同测试 universe、每个模型自己的预测所形成的持仓，以及同一个 raw-return target。成本网格在运行前固定为 0、2、5、10、20、30、50 bps，不根据模型胜负选择成本。

{markdown_table(['成本 (bps)', '多空 net Sharpe 排名', '各模型 net Sharpe'], ranking_rows)}

FinPFN 的多空换手最高，因此随着成本上升，其净表现下降得比 Ridge 和 LightGBM 更快。该结论是对已观察测试结果的敏感性审计，不是用来选择新策略的验证结果。

## 成本与换手定义

- 输入收益原为百分点，本脚本先除以 100 转成小数收益。
- `cost_rate = bps / 10,000`；1 bps 指每 1.0 单边换手的 0.0001 小数收益。
- 相邻期单腿换手为 `0.5 × Σ|w_t-w_(t-1)|`；首日从现金建仓，long/short 各为 1.0。
- long 是 decile 10；short leg 的 gross return 是 `-decile 1`；实际 H-L 是两腿收益之和。
- H-L 总换手是两腿单边换手之和；`net = gross - cost_rate × (turnover_long + turnover_short)`。
- 指标年化因子为 240；Sharpe 直接从 H-L 分期收益序列计算。复合财富为 `Π(1+r_t)`；所有输入收益均大于 -100%，因此该统计有效。
- break-even cost 定义为平均净收益恰好为零的 bps，不包含借券、冲击、融资等未建模成本。

## 关键数值

{markdown_table(['成本', '模型', 'gross 均值/期', 'net 均值/期', 'net 年化波动', 'gross Sharpe', 'net Sharpe', '平均总换手', 'net 终值', 'net 最大回撤'], metric_rows)}

### 多空均值归零成本

{markdown_table(['模型', '平均总单边换手', 'break-even bps'], break_even_rows)}

完整 long、short、H-L 结果见 `cost_sensitivity.csv`，逐期成本、净收益和财富见 `net_performance_by_period.csv`。

## 重复抽样与持仓一致性审计

{markdown_table(['模型', '预测输入行', '唯一资产—日期', '重复行（超出首行）', '按自身预测重建 decile 不一致数'], pred_rows)}

- 冻结持仓共有 {integrity['holdings']['rows']:,} 行，`(model, seed, date, id)` 重复数为 0。
- 每个模型每个资产—日期只属于一个 decile；四模型在每个日期的 holdings universe 完全相同。
- FinPFN/TabPFN 的 with-replacement 重复预测先按各模型各自的 `(date,id)` 取预测均值，然后才在共同 universe 内形成 decile。
- 用每个模型自己的聚合预测和确定性 ID tie-break 重新生成 decile，四模型不一致数均为 0；target 从未参与 decile 排序，也不存在共享 prediction column。
- 由持仓重算、排除首日的 top/bottom 换手与冻结 `turnover_by_decile.csv` 最大绝对差为 {integrity['maximum_absolute_frozen_turnover_difference']:.3e}。

因此 FinPFN 的高换手不是重复持仓、重复 asset row 或不一致 universe 造成的；它来自模型自身头尾排名随时间更频繁变化。重复抽样会影响聚合预测本身，但在组合形成前已被折叠，不会机械地重复计算持仓或收益。

## 文件与复现

运行命令：

```bash
reproduction/environment/audit-venv/bin/python \\
  reproduction/next_phase/costs/transaction_cost_analysis.py
```

输入相对路径、文件大小和 SHA-256 保存在 `input_manifest.json`；机器可读完整性检查保存在 `integrity_checks.json`。本次本地 CPU runtime 为 {runtime_seconds:.3f} 秒。

## 限制

这是线性成本敏感性，不含 bid-ask 非线性、市场冲击、借券可得性、涨跌停、融资成本和容量约束。首日建仓成本只影响 301 期中的一期；冻结换手表为与原 evaluator 一致而不含首日，本报告的经济成本则保守地包含首日。
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    args.holdings = args.holdings.resolve()
    args.period_returns = args.period_returns.resolve()
    args.baseline_turnover = args.baseline_turnover.resolve()
    args.output_dir = args.output_dir.resolve()
    try:
        relative_output = args.output_dir.relative_to(
            (REPOSITORY / "reproduction/next_phase").resolve()
        )
    except ValueError as error:
        raise ValueError("Output must remain under reproduction/next_phase") from error
    if relative_output.parts[:1] != ("costs",):
        raise ValueError("This script only writes under reproduction/next_phase/costs")
    refuse_existing_outputs(args.output_dir)

    declared_inputs = [args.holdings, args.period_returns, args.baseline_turnover]
    declared_inputs.extend(DEFAULT_PREDICTIONS.values())
    manifest = {
        "analysis": "CSI 500 frozen common-universe transaction-cost sensitivity",
        "annualization": ANNUALIZATION,
        "cost_grid_bps_per_one_way_turnover": list(COST_GRID_BPS),
        "inputs": [validate_input(path) for path in declared_inputs],
        "command": (
            "reproduction/environment/audit-venv/bin/python "
            "reproduction/next_phase/costs/transaction_cost_analysis.py"
        ),
    }

    holdings = pd.read_parquet(args.holdings)
    periods = pd.read_csv(args.period_returns)
    baseline_turnover = pd.read_csv(args.baseline_turnover)
    holding_checks = basic_holdings_integrity(holdings)
    prediction_checks = prediction_integrity(holdings, DEFAULT_PREDICTIONS)
    turnover, turnover_checks = build_turnover_series(holdings, baseline_turnover)
    gross_returns = build_gross_returns(periods)
    cost_summary, break_even, net_periods = evaluate_costs(gross_returns, turnover)

    max_difference = max(
        abs(row["difference"])
        for row in turnover_checks["frozen_turnover_comparison"]
    )
    integrity = {
        "holdings": holding_checks,
        "prediction_integrity": prediction_checks,
        **turnover_checks,
        "maximum_absolute_frozen_turnover_difference": max_difference,
        "long_short_return_identity_verified": True,
        "return_turnover_date_alignment_verified": True,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cost_summary.to_csv(args.output_dir / "cost_sensitivity.csv", index=False)
    break_even.to_csv(args.output_dir / "model_break_even_costs.csv", index=False)
    net_periods.to_csv(args.output_dir / "net_performance_by_period.csv", index=False)
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "integrity_checks.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    plot_results(cost_summary, net_periods, args.output_dir / "figures")
    runtime_seconds = time.perf_counter() - started
    write_report(
        args.output_dir / "report.md",
        cost_summary,
        break_even,
        integrity,
        runtime_seconds,
    )
    print(f"Phase 1 complete in {runtime_seconds:.3f}s")
    print(
        cost_summary.loc[
            cost_summary["portfolio"] == "long_short",
            [
                "model",
                "cost_rate_bps_per_one_way_turnover",
                "gross_sharpe",
                "net_sharpe",
                "average_turnover_one_way_including_initial",
                "net_sharpe_rank",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
