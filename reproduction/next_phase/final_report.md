# CSI 500 FinPFN：不确定性、换手与 IC–组合差距研究报告

日期：2026-07-23
状态：Phase 0–5 的全部本地可执行工作完成；CSI 冻结基线未改动；未重训 FinPFN；未启动 U.S. 外部验证。

## 1. 执行摘要

FinPFN 在 301 个共同测试日期、120,620 个共同资产—日期上的全截面 IC/IR 确实优于 Ridge 和 LightGBM，但这种统计优势没有转换成更好的极端组合选择：

- FinPFN mean IC / IR 为 0.04560 / 0.7120，高于 Ridge 0.03741 / 0.5392 和 LightGBM 0.03643 / 0.5666。
- FinPFN gross H-L Sharpe 为 4.3836，低于 Ridge 4.8890 和 LightGBM 4.8104。
- FinPFN 总单边换手 1.7830，为三者最高；在 10 bps/单边换手下 net Sharpe 为 -0.9000，而 Ridge/LightGBM 为 1.0350/0.6857。
- FinPFN predictive dispersion 能弱至中等地预测 asset-level error/rank instability，但 uncertainty gating 在 validation 没有增量价值。
- validation 选出的纯 rank buffer 在唯一一次 test 上失败：虽降低换手，却同时损失 gross return，net Sharpe 从 -0.9000 降至 -1.0397。
- 机制审计显示 FinPFN 的优势更多来自全截面中部排序；其 top-40 命中率和相邻期头尾留存显著低于基线，解释了“IC 更高但组合更差、换手更高”。

研究建议是 **B：先做一次严格预声明的 U.S. 外部验证**，不继续在 CSI test 上开发或调参。如果同一机制在 U.S. 重现，应停止方法开发并将项目定型为复现和模型风险审计。

## 2. 冻结基线

以下结果来自既有 notebook-exact predictions、共同 raw-return target 和共同 asset-date universe；本阶段未重新推断或修改。

| 模型 | mean IC | IC SD (ddof=1) | IR | gross H-L Sharpe | 总单边换手（含首日） | break-even bps | 10 bps net Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinPFN | 0.045597 | 0.064040 | 0.712002 | 4.383559 | 1.782981 | 8.296 | -0.900017 |
| Ridge | 0.037409 | 0.069378 | 0.539210 | 4.888952 | 1.478813 | 12.683 | 1.035046 |
| LightGBM | 0.036434 | 0.064305 | 0.566578 | 4.810360 | 1.551627 | 11.660 | 0.685659 |
| TabPFN | -0.037758 | 0.072213 | -0.522875 | -5.1927 | 1.827160 | -10.622 | -10.0965 |

Sharpe 主指标始终由实际 top-minus-bottom 分期收益序列计算；`Sharpe(top)-Sharpe(bottom)` 只在冻结复现报告中保留为 notebook-style 辅助值。

一致性审计确认：

- FinPFN/TabPFN 重复 group-query rows 在形成 decile 前按各自 `(date,id)` 取预测均值；
- 四模型使用同一 universe 和同一 raw target；
- 每个模型由自己的预测排序，没有共享 prediction column，也没有用 target 形成 decile；
- 冻结持仓没有重复资产—日期，重新生成 decile 的不一致数为 0；
- FinPFN 高换手不是重复持仓或 file reuse 造成的。

完整原始复现见 `reproduction/notes/csi500_reproduction_report_zh.md`。

## 3. Phase 1：成本敏感性

固定成本网格为每单位单边换手 0、2、5、10、20、30、50 bps：

| 成本 bps | Ridge net Sharpe | LightGBM | FinPFN | TabPFN |
|---:|---:|---:|---:|---:|
| 0 | 4.8890 | 4.8104 | 4.3836 | -5.1927 |
| 2 | 4.1187 | 3.9862 | 3.3268 | -6.1723 |
| 5 | 2.9628 | 2.7491 | 1.7417 | -7.6428 |
| 10 | 1.0350 | 0.6857 | -0.9000 | -10.0965 |
| 20 | -2.8231 | -3.4463 | -6.1818 | -15.0134 |
| 50 | -14.3792 | -15.8475 | -21.9892 | -29.8255 |

定义：

- 单腿换手为 `0.5 × Σ|w_t-w_(t-1)|`；
- 首日从现金建仓，long/short 各记 1.0；
- H-L 总换手为两腿单边换手之和；
- `net_return_t = gross_return_t - (bps/10,000) × total_turnover_t`。

FinPFN 的 mean gross return 较低且换手较高，使其 break-even cost 只有 8.296 bps。这里尚未计入借券、冲击、融资、涨跌停和容量约束，因此不是保守成本上界。

详细结果：`costs/report.md`、`cost_sensitivity.csv`、`net_performance_by_period.csv`。

## 4. Phase 2：不确定性审计

### 4.1 输入完整性

研究者返回的 validation artifacts 已全部通过校验：

| 模型 | group-query rows | 唯一资产—日期 | 重复 rows | groups | dates | failed/nonfinite |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 157,200 | 96,859 | 60,341 | 3,144 | 242 | 0 |
| FinPFN | 157,200 | 96,859 | 60,341 | 3,144 | 242 | 0 |

TabPFN / FinPFN GPU runtime 为 1,888.75 / 1,908.48 秒。公开 `predict(full)` 与保存的 aggregate mean/median/mode/quantiles 最大差为 0。

### 4.2 FinPFN calibration diagnostics

| signal | 绝对截面 z error | 绝对 rank error | 下一期 rank instability | 日期级 IC deterioration |
|---|---:|---:|---:|---:|
| predictive interval width | 0.2282 | 0.1048 | 0.0995 | 0.0186 |
| predictive SD | 0.2198 | 0.1042 | 0.1007 | 0.0618 |
| mean–median disagreement | 0.0700 | 0.0349 | 0.0313 | -0.0397 |
| total member SD | 0.0074 | 0.0023 | 0.0061 | -0.0404 |

数值是 pooled Spearman。predictive interval/SD 对个体误差有信息，但不能显著预测日期级 IC deterioration。

关键混杂是 prediction extremeness：FinPFN predictive-SD quintile 1→5 时，绝对 rank error 从 0.276 增至 0.377，但 realized-tail precision 也从 3.45% 增至 24.51%。所以高 dispersion 同时包含更多真正尾部资产，不能简单解释为“应剔除的不可靠样本”，也不能称为 calibrated posterior uncertainty。

详细结果：`uncertainty/report.md`、`calibration_metrics.csv`、`uncertainty_quantile_metrics.csv`。

## 5. Phase 3：validation 选择与唯一 test

固定 10 bps 成本、15 个预声明候选，只用 2021 validation 选择：

| 候选 | 类型 | validation gross Sharpe | validation net Sharpe | 换手 |
|---|---|---:|---:|---:|
| rank_buffer_exit20pct | turnover-only | 5.7632 | 1.9380 | 1.6242 |
| rank_buffer_exit15pct | turnover-only | 5.8230 | 1.8114 | 1.7024 |
| FinPFN unmodified | baseline | 5.8014 | 1.5507 | 1.7895 |
| gate_std_75pct | uncertainty | 5.2798 | 1.2374 | 1.8473 |
| gate_interval_75pct | uncertainty | 5.1260 | 1.0141 | 1.8438 |

所有 uncertainty-adjusted score 候选表现更差。冻结 `rank_buffer_exit20pct` 后，只进行一次 test：

| test 比较 | gross Sharpe | 10 bps net Sharpe | 换手 | net 期末财富 | net MDD |
|---|---:|---:|---:|---:|---:|
| FinPFN unmodified | 4.3836 | -0.9000 | 1.7830 | 0.9089 | -10.68% |
| rank_buffer_exit20pct | 3.7599 | -1.0397 | 1.6175 | 0.8962 | -11.77% |
| Ridge unmodified | 4.8890 | 1.0350 | 1.4788 | 1.1209 | -6.64% |
| LightGBM unmodified | 4.8104 | 0.6857 | 1.5516 | 1.0751 | -7.69% |

buffer 的换手下降没有抵消 gross signal 损失，validation 改善未样本外延续。不能根据该 test 结果更换 exit fraction 或追加阈值搜索。

详细结果：`gating/report.md`、`test_report.md`、`validation_results.csv`、`test_results.csv`。

## 6. Phase 4：IC–组合差距机制

| 模型 | 全体 IC | 中部 20–80% IC | bottom 20% IC | top 20% IC | 20-bin 单调性 | top-40 precision | bottom-40 precision | top-40 留存 | bottom-40 留存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FinPFN | 0.0456 | 0.0240 | 0.0381 | 0.0033 | 0.844 | 0.090 | 0.202 | 0.084 | 0.137 |
| Ridge | 0.0374 | 0.0153 | 0.0521 | -0.0062 | 0.953 | 0.149 | 0.190 | 0.276 | 0.252 |
| LightGBM | 0.0364 | 0.0187 | 0.0506 | -0.0000 | 0.895 | 0.156 | 0.203 | 0.238 | 0.215 |

解释：

1. FinPFN 在截面中部有更多小排序改善，足以抬高全部资产的 Spearman IC。
2. 组合只交易约 40 个 top 和 40 个 bottom 资产。FinPFN top-40 precision 明显弱于基线，bottom precision 只与基线接近；其 20-bin prediction–return 曲线也较不单调。
3. FinPFN 头尾成员相邻期留存仅 8.4%/13.7%，远低于 Ridge 和 LightGBM，直接导致更高换手。
4. FinPFN 日期级 IC 与 H-L return Spearman 为 0.713，说明二者相关但不等价；全体排序改善不能保证极端收益。
5. FinPFN 同一资产—日期平均出现于 1.621 个抽样 groups。group-composition prediction SD 与绝对 rank error 相关仅 -0.032，不能把 gap 全部归因于重复分组；组内 z-score 和跨组尺度仍是无法排除的构造混杂。

因此最有证据的机制是：**FinPFN 改善广泛但幅度较小的全截面排序，同时极端多头识别和时间稳定性不足；高换手进一步侵蚀本已较弱的尾部组合优势。**

详细结果：`ic_portfolio_gap/report.md`、`percentile_return_curve.csv`、`tail_precision.csv`、`rank_stability.csv`、`date_contributions.csv`。

## 7. 局限

- Phase 1 和 Phase 4 是冻结测试结果的事后敏感性/机制审计，不是新策略的无偏估计。
- 主要 checkpoint 只有一个 sampling seed；不能完全分离模型、抽样和时间序列不确定性。
- validation ensemble dispersion 混合了成员 preprocessing、预测分布、50-stock group composition 和 target z-score。
- raw parquet 缺少源时间戳和两条 forward-return 价格腿，point-in-time feature availability 与 target alignment 仍无法独立验证。
- 线性成本未包含市场冲击、借券可得性、融资、涨跌停和容量。
- CSI 数据没有文档化 sector/size/liquidity 标签，本分析没有从匿名或含义不清的 features 推断经济分类。
- FinPFN top-tail 问题已被量化，但当前观察性 artifact 不能证明某一个模型内部组件是因果来源。

## 8. 决策与下一步

决策为 **B：U.S. market external validation**，而不是继续在 CSI 500 上调 gating 或重训。

已准备的人工命令位于 `us_external_validation/manual_commands.md`：

- U.S. released checkpoints：单 A100 80 GB，预计两模型共 30–50 分钟；
- Ridge + 六候选 LightGBM：4 CPU threads，预计 10–45 分钟；
- common-universe evaluation：预计 <5 分钟；
- 无 FinPFN training/fine-tuning，无 CSI 参数回调。

若 U.S. 重现“更高 IC、较差尾部 precision/稳定性、成本后无优势”，最终选择 C 并结束方法开发。只有 U.S. 同时出现 IC、tail precision、实际 H-L 和净成本优势，才值得另写一个全新、与当前 test 隔离的原创研究计划。

当前精确下一动作：**等待研究者批准 U.S. 外部验证；在批准前不执行、commit 或 push。**

## 9. 本地执行命令

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/next_phase/uncertainty/evaluate_uncertainty.py \
  --predictions \
    reproduction/next_phase/uncertainty/artifacts/validation/csi500_tabpfn_seed42_validation_notebook_with_replacement_members.parquet \
    reproduction/next_phase/uncertainty/artifacts/validation/csi500_finpfn_seed42_validation_notebook_with_replacement_members.parquet \
  --dataset 30features_csi500.parquet \
  --output-dir reproduction/next_phase/uncertainty

reproduction/environment/audit-venv/bin/python \
  reproduction/next_phase/gating/run_validation_selection.py

reproduction/environment/audit-venv/bin/python \
  reproduction/next_phase/gating/evaluate_frozen_test.py

reproduction/environment/audit-venv/bin/python \
  reproduction/next_phase/ic_portfolio_gap/analyze_rank_tails.py
```

Phase 1 命令、输入 checksums 和公式见 `costs/report.md`。所有脚本都只写入 `reproduction/next_phase/`；关键阶段拒绝覆盖既有输出。
