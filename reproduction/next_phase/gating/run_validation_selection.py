#!/usr/bin/env python3
"""Select a small FinPFN gating/turnover-control overlay on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finpfn-next-phase-mpl")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPOSITORY / "reproduction/next_phase/uncertainty/asset_date_uncertainty.parquet"
)
DEFAULT_CONFIG = (
    REPOSITORY
    / "reproduction/next_phase/gating/configs/validation_grid.json"
)
DEFAULT_OUTPUT = REPOSITORY / "reproduction/next_phase/gating"
VALIDATION_START = pd.Timestamp("2021-01-01")
VALIDATION_END = pd.Timestamp("2022-01-01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-date-uncertainty", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(pair["x"], pair["y"]).statistic)


def sharpe(values: pd.Series, annualization: int) -> float:
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return np.nan
    return float(values.mean() / standard_deviation * np.sqrt(annualization))


def maximum_drawdown(values: pd.Series) -> float:
    wealth = np.r_[1.0, np.cumprod(1.0 + values.to_numpy(dtype=float))]
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def deterministic_ranks(part: pd.DataFrame) -> pd.DataFrame:
    ranked = part.sort_values(["prediction", "id"], kind="stable").copy()
    ranked["prediction_rank_deterministic"] = (
        np.arange(1, len(ranked) + 1) / len(ranked)
    )
    ranked["decile"] = (
        np.floor(np.arange(len(ranked)) * 10 / len(ranked)).astype(int) + 1
    )
    return ranked


def choose_extremes(ranked: pd.DataFrame) -> tuple[list[object], list[object]]:
    short = ranked.loc[ranked["decile"] == 1, "id"].tolist()
    long = ranked.loc[ranked["decile"] == 10, "id"].tolist()
    return long, short


def fill_by_prediction(
    ranked: pd.DataFrame,
    retained: list[object],
    target_count: int,
    *,
    long_side: bool,
    excluded: set[object] | None = None,
) -> list[object]:
    retained_set = set(retained)
    excluded = set() if excluded is None else excluded
    order = ranked.sort_values(
        ["prediction", "id"], ascending=[not long_side, True], kind="stable"
    )["id"]
    result = list(retained)
    for identifier in order:
        if len(result) >= target_count:
            break
        if identifier not in retained_set and identifier not in excluded:
            result.append(identifier)
            retained_set.add(identifier)
    return result


def generate_holdings(frame: pd.DataFrame, candidate: dict[str, object]) -> pd.DataFrame:
    rows = []
    previous_long: list[object] = []
    previous_short: list[object] = []
    long_entry_index: dict[object, int] = {}
    short_entry_index: dict[object, int] = {}
    kind = str(candidate["kind"])

    for date_index, (date, part) in enumerate(frame.groupby("date", sort=True)):
        ranked = deterministic_ranks(part)
        base_long, base_short = choose_extremes(ranked)
        target_long = len(base_long)
        target_short = len(base_short)
        available = set(ranked["id"])

        if kind == "unmodified":
            long_ids, short_ids = base_long, base_short
        elif kind == "confidence_gate":
            signal = str(candidate["uncertainty_signal"])
            fraction = float(candidate["fraction"])
            long_candidates = ranked.loc[ranked["id"].isin(base_long)].sort_values(
                [signal, "id"], kind="stable"
            )
            short_candidates = ranked.loc[ranked["id"].isin(base_short)].sort_values(
                [signal, "id"], kind="stable"
            )
            long_count = max(1, int(np.ceil(len(long_candidates) * fraction)))
            short_count = max(1, int(np.ceil(len(short_candidates) * fraction)))
            long_ids = long_candidates.head(long_count)["id"].tolist()
            short_ids = short_candidates.head(short_count)["id"].tolist()
        elif kind == "uncertainty_adjusted":
            signal = str(candidate["uncertainty_signal"])
            penalty = float(candidate["lambda"])
            ranked["uncertainty_rank"] = ranked[signal].rank(method="first", pct=True)
            upper = ranked.loc[
                ranked["prediction_rank_deterministic"] > 0.5
            ].copy()
            lower = ranked.loc[
                ranked["prediction_rank_deterministic"] <= 0.5
            ].copy()
            upper["priority"] = (
                upper["prediction_rank_deterministic"]
                - penalty * upper["uncertainty_rank"]
            )
            lower["priority"] = (
                1.0
                - lower["prediction_rank_deterministic"]
                - penalty * lower["uncertainty_rank"]
            )
            long_ids = (
                upper.sort_values(["priority", "id"], ascending=[False, True])
                .head(target_long)["id"]
                .tolist()
            )
            short_ids = (
                lower.sort_values(["priority", "id"], ascending=[False, True])
                .head(target_short)["id"]
                .tolist()
            )
        elif kind == "rank_buffer":
            exit_fraction = float(candidate["exit_fraction"])
            rank_map = ranked.set_index("id")["prediction_rank_deterministic"]
            retained_long = [
                identifier
                for identifier in previous_long
                if identifier in available and rank_map.loc[identifier] > 1 - exit_fraction
            ]
            retained_short = [
                identifier
                for identifier in previous_short
                if identifier in available and rank_map.loc[identifier] <= exit_fraction
            ]
            long_ids = fill_by_prediction(
                ranked,
                retained_long,
                target_long,
                long_side=True,
                excluded=set(retained_short),
            )
            short_ids = fill_by_prediction(
                ranked,
                retained_short,
                target_short,
                long_side=False,
                excluded=set(long_ids),
            )
        elif kind == "minimum_holding":
            holding_periods = int(candidate["holding_periods"])
            retained_long = [
                identifier
                for identifier in previous_long
                if identifier in available
                and date_index - long_entry_index[identifier] < holding_periods
            ]
            retained_short = [
                identifier
                for identifier in previous_short
                if identifier in available
                and date_index - short_entry_index[identifier] < holding_periods
            ]
            long_ids = fill_by_prediction(
                ranked,
                retained_long,
                target_long,
                long_side=True,
                excluded=set(retained_short),
            )
            short_ids = fill_by_prediction(
                ranked,
                retained_short,
                target_short,
                long_side=False,
                excluded=set(long_ids),
            )
            new_long = set(long_ids).difference(previous_long)
            new_short = set(short_ids).difference(previous_short)
            for identifier in new_long:
                long_entry_index[identifier] = date_index
            for identifier in new_short:
                short_entry_index[identifier] = date_index
            long_entry_index = {
                identifier: long_entry_index[identifier]
                for identifier in long_ids
            }
            short_entry_index = {
                identifier: short_entry_index[identifier]
                for identifier in short_ids
            }
        else:
            raise ValueError(f"Unknown candidate kind: {kind}")

        if set(long_ids).intersection(short_ids):
            raise ValueError(f"Long/short holdings overlap for {candidate['name']} on {date}")
        for leg, identifiers in (("long", long_ids), ("short", short_ids)):
            rows.extend(
                {
                    "candidate": candidate["name"],
                    "date": date,
                    "id": identifier,
                    "leg": leg,
                }
                for identifier in identifiers
            )
        previous_long = long_ids
        previous_short = short_ids
    return pd.DataFrame(rows)


def leg_turnover(holdings: pd.DataFrame) -> pd.Series:
    previous: dict[object, float] | None = None
    observations: dict[pd.Timestamp, float] = {}
    for date, part in holdings.groupby("date", sort=True):
        current = {identifier: 1.0 / len(part) for identifier in part["id"]}
        if previous is None:
            observations[pd.Timestamp(date)] = 1.0
        else:
            identifiers = set(previous) | set(current)
            observations[pd.Timestamp(date)] = 0.5 * sum(
                abs(current.get(item, 0.0) - previous.get(item, 0.0))
                for item in identifiers
            )
        previous = current
    return pd.Series(observations, dtype=float).sort_index()


def evaluate_candidate(
    frame: pd.DataFrame,
    holdings: pd.DataFrame,
    candidate: dict[str, object],
    *,
    annualization: int,
    cost_bps: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    joined = holdings.merge(
        frame[["date", "id", "prediction", "raw_return_decimal"]],
        on=["date", "id"],
        how="left",
        validate="many_to_one",
    )
    if joined[["prediction", "raw_return_decimal"]].isna().any().any():
        raise ValueError(f"Missing return/prediction for {candidate['name']}")
    period_leg = (
        joined.groupby(["date", "leg"])["raw_return_decimal"]
        .mean()
        .unstack()
        .sort_index()
    )
    period = pd.DataFrame(index=period_leg.index)
    period["gross_long_return_decimal"] = period_leg["long"]
    period["gross_short_return_decimal"] = -period_leg["short"]
    period["gross_long_short_return_decimal"] = (
        period["gross_long_return_decimal"] + period["gross_short_return_decimal"]
    )
    long_turnover = leg_turnover(holdings.loc[holdings["leg"] == "long"])
    short_turnover = leg_turnover(holdings.loc[holdings["leg"] == "short"])
    period["long_turnover_one_way"] = long_turnover.reindex(period.index)
    period["short_turnover_one_way"] = short_turnover.reindex(period.index)
    period["total_turnover_one_way"] = (
        period["long_turnover_one_way"] + period["short_turnover_one_way"]
    )
    period["transaction_cost_decimal"] = (
        period["total_turnover_one_way"] * cost_bps / 10_000.0
    )
    period["net_long_short_return_decimal"] = (
        period["gross_long_short_return_decimal"] - period["transaction_cost_decimal"]
    )
    period["candidate"] = candidate["name"]
    period = period.reset_index()

    full_ic = frame.groupby("date").apply(
        lambda part: spearman(part["prediction"], part["raw_return_decimal"]),
        include_groups=False,
    )
    traded_ic = joined.groupby("date").apply(
        lambda part: spearman(part["prediction"], part["raw_return_decimal"]),
        include_groups=False,
    )
    full_ic_std = full_ic.std(ddof=1)
    traded_ic_std = traded_ic.std(ddof=1)
    total_assets = frame.groupby("date").size()
    traded_assets = holdings.groupby("date").size()
    leg_counts = holdings.groupby(["date", "leg"]).size().unstack()
    gross = period["gross_long_short_return_decimal"]
    net = period["net_long_short_return_decimal"]
    result = {
        "candidate": candidate["name"],
        "kind": candidate["kind"],
        "declared_order": int(candidate["declared_order"]),
        "uncertainty_signal": candidate.get("uncertainty_signal"),
        "fraction": candidate.get("fraction"),
        "lambda": candidate.get("lambda"),
        "exit_fraction": candidate.get("exit_fraction"),
        "holding_periods": candidate.get("holding_periods"),
        "n_dates": len(period),
        "full_universe_mean_ic": float(full_ic.mean()),
        "full_universe_ic_std_ddof1": float(full_ic_std),
        "full_universe_ir": float(full_ic.mean() / full_ic_std),
        "traded_universe_mean_ic": float(traded_ic.mean()),
        "traded_universe_ic_std_ddof1": float(traded_ic_std),
        "traded_universe_ir": float(traded_ic.mean() / traded_ic_std),
        "gross_long_mean_return_decimal": float(
            period["gross_long_return_decimal"].mean()
        ),
        "gross_short_mean_return_decimal": float(
            period["gross_short_return_decimal"].mean()
        ),
        "gross_long_short_mean_return_decimal": float(gross.mean()),
        "gross_long_short_sharpe": sharpe(gross, annualization),
        "average_total_one_way_turnover": float(
            period["total_turnover_one_way"].mean()
        ),
        "fixed_cost_bps_per_one_way_turnover": cost_bps,
        "net_long_short_mean_return_decimal": float(net.mean()),
        "net_long_short_sharpe": sharpe(net, annualization),
        "net_terminal_compounded_wealth": float(np.prod(1.0 + net)),
        "net_maximum_drawdown": maximum_drawdown(net),
        "mean_asset_coverage": float((traded_assets / total_assets).mean()),
        "mean_long_positions": float(leg_counts["long"].mean()),
        "mean_short_positions": float(leg_counts["short"].mean()),
        "mean_long_hhi": float((1.0 / leg_counts["long"]).mean()),
        "mean_short_hhi": float((1.0 / leg_counts["short"]).mean()),
        "maximum_position_weight": float(
            max((1.0 / leg_counts["long"]).max(), (1.0 / leg_counts["short"]).max())
        ),
    }
    return result, period


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    output.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(output)


def write_report(
    output: Path,
    results: pd.DataFrame,
    selected: dict[str, object],
    runtime_seconds: float,
) -> None:
    ordered = results.sort_values("selection_rank")
    rows = [
        [
            int(row.selection_rank),
            row.candidate,
            row.kind,
            f"{row.gross_long_short_sharpe:.3f}",
            f"{row.net_long_short_sharpe:.3f}",
            f"{row.average_total_one_way_turnover:.3f}",
            f"{row.mean_asset_coverage:.3f}",
            f"{row.net_maximum_drawdown * 100:.2f}%",
        ]
        for row in ordered.itertuples()
    ]
    chosen = selected["validation_metrics"]
    baseline = results.loc[results["candidate"] == "finpfn_unmodified"].iloc[0]
    report = f"""# Phase 3：validation-only gating 与换手控制选择

## 选择结果

固定成本为每单位单边换手 10 bps。全部 15 个候选在读取结果前写入 `configs/validation_grid.json`，只使用 2021 validation；主目标为实际多空净 Sharpe。

冻结选择为 **`{selected['selected_candidate']['name']}`**（`{selected['selected_candidate']['kind']}`）：validation gross Sharpe {chosen['gross_long_short_sharpe']:.4f}，net Sharpe {chosen['net_long_short_sharpe']:.4f}，平均总单边换手 {chosen['average_total_one_way_turnover']:.4f}。未修改 FinPFN 的相应值为 gross {baseline.gross_long_short_sharpe:.4f}、net {baseline.net_long_short_sharpe:.4f}、换手 {baseline.average_total_one_way_turnover:.4f}。

{markdown_table(['排名', '候选', '类型', 'gross Sharpe', 'net Sharpe', '换手', '覆盖率', 'net MDD'], rows)}

## 解释边界

- full-universe IC/IR 对纯交易 overlay 保持不变；`validation_results.csv` 另报实际持仓 union 内的 IC/IR。
- confidence gate 只在原始 top/bottom decile 内保留低 uncertainty 的 75% 或 50%，每条腿仍等权且 gross exposure 不变，因此覆盖下降会提高集中度。
- uncertainty-adjusted 候选在上半区对 long priority、下半区对 short priority 对称扣除 uncertainty rank penalty，避免 long/short 重叠。
- rank buffer 和 minimum holding 完全不使用 uncertainty，是识别 uncertainty 增量价值所需的纯换手对照。
- 选择现已写入 `selected_config.json`；之后不得依据测试表现改动。test 只允许一次冻结评估。

本地 CPU runtime 为 {runtime_seconds:.3f} 秒。完整逐期 validation 收益在 `validation_performance_by_period.csv`，持仓在 ignored 的 `validation_holdings.parquet`。
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    input_path = args.asset_date_uncertainty.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    gating_root = (REPOSITORY / "reproduction/next_phase/gating").resolve()
    if output_dir != gating_root:
        raise ValueError(f"Output must be {gating_root}")
    declared_outputs = [
        output_dir / "validation_results.csv",
        output_dir / "selected_config.json",
        output_dir / "validation_performance_by_period.csv",
        output_dir / "validation_holdings.parquet",
        output_dir / "report.md",
    ]
    if any(path.exists() for path in declared_outputs) or (
        (output_dir / "figures").exists() and any((output_dir / "figures").iterdir())
    ):
        raise FileExistsError("Refusing to overwrite Phase 3 validation outputs")

    config = json.loads(config_path.read_text())
    protocol = config["protocol"]
    annualization = int(protocol["annualization"])
    cost_bps = float(protocol["fixed_cost_bps_per_one_way_turnover"])
    candidates = []
    for order, candidate in enumerate(config["candidates"]):
        candidate = dict(candidate)
        candidate["declared_order"] = order
        candidates.append(candidate)

    frame = pd.read_parquet(input_path)
    frame = frame.loc[frame["model"] == "FinPFN"].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.empty or not (
        (frame["date"] >= VALIDATION_START) & (frame["date"] < VALIDATION_END)
    ).all():
        raise ValueError("Input is not the declared FinPFN validation asset-date panel")
    if frame.duplicated(["date", "id"]).any():
        raise ValueError("Validation input contains duplicate asset-date rows")
    for signal in config["uncertainty_signals"]:
        if signal not in frame or not np.isfinite(frame[signal]).all():
            raise ValueError(f"Missing or non-finite uncertainty signal: {signal}")

    result_rows = []
    holdings_frames = []
    period_frames = []
    for candidate in candidates:
        holdings = generate_holdings(frame, candidate)
        result, periods = evaluate_candidate(
            frame,
            holdings,
            candidate,
            annualization=annualization,
            cost_bps=cost_bps,
        )
        result_rows.append(result)
        holdings_frames.append(holdings)
        period_frames.append(periods)
    results = pd.DataFrame(result_rows)
    results = results.sort_values(
        [
            "net_long_short_sharpe",
            "average_total_one_way_turnover",
            "gross_long_short_sharpe",
            "declared_order",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    results["selection_rank"] = np.arange(1, len(results) + 1)
    selected_row = results.iloc[0]
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate["name"] == selected_row["candidate"]
    )
    validation_metrics = {
        key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
        for key, value in selected_row.to_dict().items()
    }
    selected = {
        "selection_status": "frozen_before_test",
        "selected_candidate": selected_candidate,
        "validation_metrics": validation_metrics,
        "selection_protocol": protocol,
        "input_asset_date_uncertainty_sha256": sha256(input_path),
        "validation_grid_sha256": sha256(config_path),
        "input_rows": len(frame),
        "input_dates": frame["date"].nunique(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    pd.concat(period_frames, ignore_index=True).to_csv(
        output_dir / "validation_performance_by_period.csv", index=False
    )
    pd.concat(holdings_frames, ignore_index=True).to_parquet(
        output_dir / "validation_holdings.parquet", index=False
    )
    (output_dir / "selected_config.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for row in results.itertuples():
        axis.scatter(
            row.average_total_one_way_turnover,
            row.net_long_short_sharpe,
            marker="*" if row.selection_rank == 1 else "o",
            s=130 if row.selection_rank == 1 else 45,
        )
        axis.annotate(row.candidate, (row.average_total_one_way_turnover, row.net_long_short_sharpe), fontsize=7)
    axis.set(
        xlabel="Average long + short one-way turnover",
        ylabel="Validation net H-L Sharpe (10 bps)",
        title="Predeclared FinPFN overlays: validation selection",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "validation_net_sharpe_vs_turnover.png", dpi=180)
    plt.close(fig)

    runtime_seconds = time.perf_counter() - started
    write_report(output_dir / "report.md", results, selected, runtime_seconds)
    print(results.to_string(index=False))
    print(f"SELECTED {selected_candidate['name']}")


if __name__ == "__main__":
    main()
