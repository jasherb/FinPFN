#!/usr/bin/env python3
"""Evaluate the validation-frozen FinPFN overlay exactly once on the test split."""

from __future__ import annotations

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

from run_validation_selection import evaluate_candidate, generate_holdings, sha256


REPOSITORY = Path(__file__).resolve().parents[3]
GATING_DIR = REPOSITORY / "reproduction/next_phase/gating"
SELECTED_CONFIG = GATING_DIR / "selected_config.json"
DATASET = REPOSITORY / "30features_csi500.parquet"
COMMON_HOLDINGS = (
    REPOSITORY
    / "reproduction/results/csi500_all_models_notebook_exact/decile_holdings.parquet"
)
FROZEN_PORTFOLIO_METRICS = (
    REPOSITORY
    / "reproduction/results/csi500_all_models_notebook_exact/portfolio_metrics.csv"
)
PREDICTIONS = {
    "FinPFN": (
        REPOSITORY
        / "reproduction/artifacts/csi500_notebook_exact/"
        "csi500_finpfn_seed42_notebook_with_replacement.parquet"
    ),
    "Ridge": (
        REPOSITORY
        / "reproduction/artifacts/predictions/csi500_baselines/"
        "csi500_ridge_seed42.parquet"
    ),
    "LightGBM": (
        REPOSITORY
        / "reproduction/artifacts/predictions/csi500_baselines/"
        "csi500_lightgbm_seed42.parquet"
    ),
}
ANNUALIZATION = 240
EXPECTED_TEST_DATES = 301
EXPECTED_TEST_ASSET_DATES = 120_620
EXPECTED_FROZEN_SHARPE = {
    "FinPFN": 4.383558921159609,
    "Ridge": 4.888952290269178,
    "LightGBM": 4.810359613592669,
}


def hash_json_payload(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_common_keys() -> pd.DataFrame:
    holdings = pd.read_parquet(COMMON_HOLDINGS, columns=["date", "id"])
    holdings["date"] = pd.to_datetime(holdings["date"])
    keys = holdings.drop_duplicates(["date", "id"]).sort_values(["date", "id"])
    if keys.duplicated(["date", "id"]).any():
        raise ValueError("Common universe contains duplicate asset-date keys")
    if keys["date"].nunique() != EXPECTED_TEST_DATES or len(keys) != EXPECTED_TEST_ASSET_DATES:
        raise ValueError(
            f"Unexpected common universe: {keys['date'].nunique()} dates, {len(keys)} rows"
        )
    return keys


def load_raw_returns(keys: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(DATASET, columns=["date", "id", "target"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = keys.merge(raw, on=["date", "id"], how="left", validate="one_to_one")
    if raw["target"].isna().any() or not np.isfinite(raw["target"]).all():
        raise ValueError("Common test universe has missing or non-finite raw returns")
    return raw.rename(columns={"target": "raw_return_decimal"})


def load_model_frame(
    model: str, path: Path, keys_and_returns: pd.DataFrame
) -> pd.DataFrame:
    predictions = pd.read_parquet(
        path, columns=["model", "seed", "date", "id", "prediction", "status"]
    )
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = predictions.loc[
        (predictions["model"] == model)
        & (predictions["seed"] == 42)
        & (predictions["status"] == "ok")
        & predictions["prediction"].notna()
    ].copy()
    collapsed = (
        predictions.groupby(["date", "id"], as_index=False, sort=True)["prediction"]
        .mean()
    )
    frame = keys_and_returns.merge(
        collapsed, on=["date", "id"], how="left", validate="one_to_one"
    )
    if frame["prediction"].isna().any() or not np.isfinite(frame["prediction"]).all():
        raise ValueError(f"{model} does not fully cover the frozen common universe")
    frame["model"] = model
    return frame


def frozen_sharpe_check(results: pd.DataFrame) -> dict[str, object]:
    stored = pd.read_csv(FROZEN_PORTFOLIO_METRICS)
    stored = stored.loc[
        (stored["return_basis"] == "raw") & stored["model"].isin(EXPECTED_FROZEN_SHARPE)
    ]
    checks: dict[str, object] = {}
    for model, expected in EXPECTED_FROZEN_SHARPE.items():
        observed = float(
            results.loc[
                results["comparison"] == f"{model}_unmodified",
                "gross_long_short_sharpe",
            ].iat[0]
        )
        stored_value = float(
            stored.loc[stored["model"] == model, "primary_long_short_sharpe"].iat[0]
        )
        maximum_difference = max(abs(observed - expected), abs(observed - stored_value))
        if maximum_difference > 1e-10:
            raise ValueError(
                f"{model} gross Sharpe does not reproduce frozen evaluator: "
                f"observed={observed}, expected={expected}, stored={stored_value}"
            )
        checks[model] = {
            "observed": observed,
            "expected_constant": expected,
            "stored_portfolio_metrics": stored_value,
            "maximum_absolute_difference": maximum_difference,
        }
    return checks


def write_report(
    results: pd.DataFrame,
    selected: dict[str, object],
    checks: dict[str, object],
    runtime_seconds: float,
) -> None:
    rows = []
    for row in results.itertuples():
        rows.append(
            "| "
            + " | ".join(
                [
                    row.comparison,
                    f"{row.gross_long_short_sharpe:.4f}",
                    f"{row.net_long_short_sharpe:.4f}",
                    f"{row.average_total_one_way_turnover:.4f}",
                    f"{row.net_long_short_mean_return_decimal * 100:.4f}%",
                    f"{row.net_terminal_compounded_wealth:.4f}",
                    f"{row.net_maximum_drawdown * 100:.2f}%",
                ]
            )
            + " |"
        )
    selected_name = selected["selected_candidate"]["name"]
    selected_row = results.loc[results["comparison"] == selected_name].iloc[0]
    base_row = results.loc[results["comparison"] == "FinPFN_unmodified"].iloc[0]
    report = f"""# Phase 3：冻结配置的唯一一次测试期评估

## 结论

验证期冻结配置 **`{selected_name}`** 已在 301 个共同测试日期上执行一次。固定成本仍为每单位单边换手 10 bps，未根据测试结果修改参数。

该 overlay 的 gross Sharpe 为 {selected_row.gross_long_short_sharpe:.4f}、net Sharpe 为 {selected_row.net_long_short_sharpe:.4f}、平均总单边换手为 {selected_row.average_total_one_way_turnover:.4f}。未修改 FinPFN 分别为 {base_row.gross_long_short_sharpe:.4f}、{base_row.net_long_short_sharpe:.4f}、{base_row.average_total_one_way_turnover:.4f}。

| 比较项 | gross Sharpe | net Sharpe | 总单边换手 | net 平均期收益 | net 期末财富 | net MDD |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 完整性与解释边界

- `selected_config.json` 在本脚本读取测试预测前已经标记为 `frozen_before_test`。
- FinPFN、Ridge、LightGBM 的未修改 gross Sharpe 均以不超过 `1e-10` 的误差复现冻结 evaluator；核对详情写入 `test_evaluation_manifest.json`。
- 所有模型使用同一 120,620 个资产—日期、同一 raw-return target、同一确定性 tie-break 和实际多空收益序列 Sharpe。
- 换手是 long 与 short 两条等权腿的单边换手之和，首日从现金建仓各记 1.0；因此这里的平均值会略高于冻结 baseline 中“不含首日”的摘要。
- 验证胜出方法不使用 uncertainty。验证中的 uncertainty gating 既未胜出，也没有被带到测试期再次调参；这是一项负面增量价值结果。
- 这是预先冻结策略的唯一测试评估；不得因为本结果改变 exit fraction 后重跑。

本地 CPU runtime 为 {runtime_seconds:.3f} 秒。
"""
    (GATING_DIR / "test_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    outputs = [
        GATING_DIR / "test_results.csv",
        GATING_DIR / "test_performance_by_period.csv",
        GATING_DIR / "test_holdings.parquet",
        GATING_DIR / "test_evaluation_manifest.json",
        GATING_DIR / "test_report.md",
        GATING_DIR / "figures/test_net_wealth.png",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "Refusing to overwrite the one-time frozen Phase 3 test evaluation"
        )

    selected = json.loads(SELECTED_CONFIG.read_text(encoding="utf-8"))
    if selected.get("selection_status") != "frozen_before_test":
        raise ValueError("Selected configuration is not frozen_before_test")
    candidate = dict(selected["selected_candidate"])
    if candidate.get("kind") != "rank_buffer":
        raise ValueError(
            "This local evaluator is authorized only for the selected turnover-only "
            "rank-buffer configuration; uncertainty candidates require test member outputs"
        )
    cost_bps = float(
        selected["selection_protocol"]["fixed_cost_bps_per_one_way_turnover"]
    )
    if cost_bps != 10.0:
        raise ValueError(f"Unexpected frozen selection cost: {cost_bps}")

    common_keys = load_common_keys()
    keys_and_returns = load_raw_returns(common_keys)
    frames = {
        model: load_model_frame(model, path, keys_and_returns)
        for model, path in PREDICTIONS.items()
    }

    comparisons: list[tuple[str, pd.DataFrame, dict[str, object]]] = []
    for model in ("FinPFN", "Ridge", "LightGBM"):
        baseline_candidate = {
            "name": f"{model}_unmodified",
            "kind": "unmodified",
            "declared_order": 0,
        }
        comparisons.append((model, frames[model], baseline_candidate))
    comparisons.append(("FinPFN", frames["FinPFN"], candidate))

    result_rows = []
    holding_frames = []
    period_frames = []
    for model, frame, comparison_candidate in comparisons:
        holdings = generate_holdings(frame, comparison_candidate)
        result, period = evaluate_candidate(
            frame,
            holdings,
            comparison_candidate,
            annualization=ANNUALIZATION,
            cost_bps=cost_bps,
        )
        result["comparison"] = comparison_candidate["name"]
        result["source_model"] = model
        result_rows.append(result)
        holdings["source_model"] = model
        holding_frames.append(holdings)
        period["source_model"] = model
        period_frames.append(period)

    results = pd.DataFrame(result_rows)
    checks = frozen_sharpe_check(results)
    periods = pd.concat(period_frames, ignore_index=True)
    holdings = pd.concat(holding_frames, ignore_index=True)

    results.to_csv(GATING_DIR / "test_results.csv", index=False)
    periods.to_csv(GATING_DIR / "test_performance_by_period.csv", index=False)
    holdings.to_parquet(GATING_DIR / "test_holdings.parquet", index=False)

    selected_name = candidate["name"]
    figure_period = periods.loc[
        periods["candidate"].isin(["FinPFN_unmodified", selected_name])
    ].copy()
    figure_period["net_wealth"] = figure_period.groupby("candidate")[
        "net_long_short_return_decimal"
    ].transform(lambda values: (1.0 + values).cumprod())
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for name, part in figure_period.groupby("candidate", sort=False):
        axis.plot(part["date"], part["net_wealth"], label=name)
    axis.set(
        title="Frozen test: FinPFN net wealth at 10 bps",
        xlabel="Date",
        ylabel="Compounded wealth",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(GATING_DIR / "figures/test_net_wealth.png", dpi=180)
    plt.close(figure)

    runtime_seconds = time.perf_counter() - started
    manifest = {
        "evaluation_status": "completed_once",
        "selection_payload_sha256": hash_json_payload(selected),
        "selected_config_file_sha256": sha256(SELECTED_CONFIG),
        "selected_candidate": candidate,
        "cost_bps_per_one_way_turnover": cost_bps,
        "annualization": ANNUALIZATION,
        "test_dates": int(common_keys["date"].nunique()),
        "test_asset_dates": int(len(common_keys)),
        "input_sha256": {
            "dataset": sha256(DATASET),
            "common_holdings": sha256(COMMON_HOLDINGS),
            "frozen_portfolio_metrics": sha256(FROZEN_PORTFOLIO_METRICS),
            **{f"{model}_predictions": sha256(path) for model, path in PREDICTIONS.items()},
        },
        "frozen_gross_sharpe_checks": checks,
        "runtime_seconds": runtime_seconds,
    }
    (GATING_DIR / "test_evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(results, selected, checks, runtime_seconds)
    print(
        results[
            [
                "comparison",
                "gross_long_short_sharpe",
                "net_long_short_sharpe",
                "average_total_one_way_turnover",
                "net_terminal_compounded_wealth",
                "net_maximum_drawdown",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
