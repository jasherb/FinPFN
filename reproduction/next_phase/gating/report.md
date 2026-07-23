# Phase 3：validation-only gating 与换手控制选择

## 选择结果

固定成本为每单位单边换手 10 bps。全部 15 个候选在读取结果前写入 `configs/validation_grid.json`，只使用 2021 validation；主目标为实际多空净 Sharpe。

冻结选择为 **`rank_buffer_exit20pct`**（`rank_buffer`）：validation gross Sharpe 5.7632，net Sharpe 1.9380，平均总单边换手 1.6242。未修改 FinPFN 的相应值为 gross 5.8014、net 1.5507、换手 1.7895。

| 排名 | 候选 | 类型 | gross Sharpe | net Sharpe | 换手 | 覆盖率 | net MDD |
|---|---|---|---|---|---|---|---|
| 1 | rank_buffer_exit20pct | rank_buffer | 5.763 | 1.938 | 1.624 | 0.200 | -6.04% |
| 2 | rank_buffer_exit15pct | rank_buffer | 5.823 | 1.811 | 1.702 | 0.200 | -6.03% |
| 3 | finpfn_unmodified | unmodified | 5.801 | 1.551 | 1.790 | 0.200 | -7.19% |
| 4 | minimum_holding_3 | minimum_holding | 3.823 | 1.527 | 0.956 | 0.200 | -3.64% |
| 5 | gate_std_75pct | confidence_gate | 5.280 | 1.237 | 1.847 | 0.151 | -7.52% |
| 6 | gate_interval_75pct | confidence_gate | 5.126 | 1.014 | 1.844 | 0.151 | -7.57% |
| 7 | minimum_holding_2 | minimum_holding | 3.685 | 0.930 | 1.172 | 0.200 | -5.49% |
| 8 | gate_interval_50pct | confidence_gate | 4.913 | 0.723 | 1.872 | 0.101 | -12.71% |
| 9 | gate_std_50pct | confidence_gate | 4.936 | 0.716 | 1.870 | 0.101 | -13.39% |
| 10 | adjust_std_lambda025 | uncertainty_adjusted | 5.179 | 0.221 | 1.777 | 0.200 | -6.53% |
| 11 | adjust_interval_lambda025 | uncertainty_adjusted | 4.050 | -1.020 | 1.774 | 0.200 | -9.91% |
| 12 | adjust_interval_lambda050 | uncertainty_adjusted | 4.740 | -1.934 | 1.688 | 0.200 | -11.49% |
| 13 | adjust_std_lambda050 | uncertainty_adjusted | 4.632 | -2.065 | 1.691 | 0.200 | -12.04% |
| 14 | adjust_std_lambda100 | uncertainty_adjusted | 2.893 | -4.072 | 1.590 | 0.200 | -20.55% |
| 15 | adjust_interval_lambda100 | uncertainty_adjusted | 2.496 | -4.532 | 1.588 | 0.200 | -22.23% |

## 解释边界

- full-universe IC/IR 对纯交易 overlay 保持不变；`validation_results.csv` 另报实际持仓 union 内的 IC/IR。
- confidence gate 只在原始 top/bottom decile 内保留低 uncertainty 的 75% 或 50%，每条腿仍等权且 gross exposure 不变，因此覆盖下降会提高集中度。
- uncertainty-adjusted 候选在上半区对 long priority、下半区对 short priority 对称扣除 uncertainty rank penalty，避免 long/short 重叠。
- rank buffer 和 minimum holding 完全不使用 uncertainty，是识别 uncertainty 增量价值所需的纯换手对照。
- 选择现已写入 `selected_config.json`；之后不得依据测试表现改动。test 只允许一次冻结评估。

本地 CPU runtime 为 10.628 秒。完整逐期 validation 收益在 `validation_performance_by_period.csv`，持仓在 ignored 的 `validation_holdings.parquet`。

## 冻结配置的唯一一次 test

`rank_buffer_exit20pct` 在读取测试预测前已冻结。测试 evaluator 先要求 FinPFN、Ridge、LightGBM 的未修改 gross Sharpe 精确复现冻结 evaluator，三者最大差不超过 `1e-10`，然后才评估 overlay。

| test 比较 | gross Sharpe | 10 bps net Sharpe | 总单边换手 | net 期末财富 | net MDD |
|---|---:|---:|---:|---:|---:|
| FinPFN unmodified | 4.3836 | -0.9000 | 1.7830 | 0.9089 | -10.68% |
| rank_buffer_exit20pct | 3.7599 | -1.0397 | 1.6175 | 0.8962 | -11.77% |
| Ridge unmodified | 4.8890 | 1.0350 | 1.4788 | 1.1209 | -6.64% |
| LightGBM unmodified | 4.8104 | 0.6857 | 1.5516 | 1.0751 | -7.69% |

rank buffer 确实降低换手，但 gross signal 损失更大，validation 改善没有样本外延续。不得依据该 test 结果更换 20% exit boundary 或追加 test-driven search。完整结果见 `test_results.csv`、`test_performance_by_period.csv`、`test_evaluation_manifest.json` 和 `test_report.md`。

实现 QA：第一次 validation 运行在写任何结果前发现 minimum-holding 候选可能让锁定的 long/short 重叠并安全中止；修复只是在补位时排除对侧锁定/已选 ID，未改变候选网格、成本或选择目标。第一次 test 运行也在写结果前因旧文本中的截断 Sharpe 常数与冻结 CSV 相差约 `1.2e-7` 而安全中止；随后仅用冻结 CSV 精确常数替换核对值，策略和参数未改变。
