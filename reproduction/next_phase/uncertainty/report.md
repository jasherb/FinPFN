# Phase 2：不确定性可用性与校准审计

## 结论

完整的 2021 validation 成员输出已经返回并通过校验。FinPFN 与 TabPFN 各有 157,200 个 group-query rows、96,859 个唯一资产—日期、60,341 个重复抽样 rows、3,144 个 groups 和 242 个日期；没有失败组或非有限预测。公开 `predict(full)` 与旁路保存的聚合 mean/median/mode/quantiles 最大绝对差均为 0。

FinPFN 的 aggregate predictive interval width 和 predictive SD 是可测的**误差/不稳定性相关信号**，但不是已校准的 posterior uncertainty：

| FinPFN signal | 绝对截面 z 误差 Spearman | 绝对 rank 误差 | 下一期 rank instability | 日期级 IC deterioration |
|---|---:|---:|---:|---:|
| predictive interval width | 0.2282 | 0.1048 | 0.0995 | 0.0186 |
| predictive SD | 0.2198 | 0.1042 | 0.1007 | 0.0618 |
| mean–median disagreement | 0.0700 | 0.0349 | 0.0313 | -0.0397 |
| total member SD | 0.0074 | 0.0023 | 0.0061 | -0.0404 |

前两项在 pooled asset-date 层面与误差显著正相关，但它们不能可靠预测下一日期的总体 IC deterioration：242 个日期上的相关系数很小且不显著。逐成员离散度本身几乎没有可用关系；member rank disagreement 和 group-composition SD 与误差反而弱负相关，因此不能按“越大越不可靠”的方向直接解释。

## 关键混杂：uncertainty 同时刻画预测极端程度

FinPFN predictive SD 从最低到最高 quintile 时：

- 平均绝对截面 z 误差从 0.783 增至 1.525；
- 平均绝对 rank error 从 0.276 增至 0.377；
- 下一期 rank instability 从 0.266 增至 0.353；
- 但预测尾部的 realized-tail precision 也从 3.45% 增至 24.51%。

predictive interval width 有同样模式，最高 quintile 的 tail precision 为 24.58%。这说明高 dispersion 与预测幅度/尾部候选强烈纠缠：它既对应更大误差，也包含更多真正的 realized tail。因而“剔除高 uncertainty”不等于提高选股质量，很可能同时删掉最有信息的极端预测。

`coverage_performance.csv` 是预先声明的 coverage 诊断，不作为单独的测试期选择依据。完整逐信号统计见 `calibration_metrics.csv` 和 `uncertainty_quantile_metrics.csv`。

## 输出来源与可复现性

输入：

- TabPFN validation members SHA-256：`3a11b02dff6c97e70d2416aadb6af9a6423a7da7a5b9c14b08601d058bf92255`
- FinPFN validation members SHA-256：`c3ea66ecf46cba10dfed4d943c79c7a84d80d7f681dd68b070de5c0cd6e9d58e`
- CSI parquet SHA-256：`9e0d61f5d70151d4f2f7b40918a8ddb79f86fb54a0fe86759f5c1f2869fe1b3e`

服务器记录的推断 runtime 为 TabPFN 1,888.75 秒、FinPFN 1,908.48 秒。原始 validation parquet 位于 `artifacts/validation/`；机器可读参数、split、coverage grid 与 checksums 见 `evaluation_manifest.json`。

## 解释边界

- 这里测得的是 ensemble/predictive dispersion，不是经过 coverage calibration 证明的概率型后验不确定性。
- 重复的 50-stock group 抽样、组内 target z-score、成员 preprocessing 和预测幅度都会进入 dispersion。
- validation 不包含官方 2022 CSI shock windows，也没有独立 CIMV 序列，因此没有虚构 shock/non-shock 标签。
- 是否具有交易增量价值必须由预声明的 Phase 3 策略验证；相关性本身不能证明经济价值。

## Phase 3 衔接结果

Phase 3 的 15 个预声明候选只用 validation 选择。最佳 uncertainty 方案是 `gate_std_75pct`，但其 10 bps net Sharpe 仅 1.237，低于未修改 FinPFN 的 1.551；所有 uncertainty-adjusted score 方案更差。胜出配置是完全不使用 uncertainty 的 `rank_buffer_exit20pct`，validation net Sharpe 为 1.938。

因此 Phase 2 的最终判断是：**存在统计误差相关信号，但没有观察到 uncertainty 对交易策略的增量价值。** 由于 uncertainty 方案未在 validation 胜出，不生成也不需要测试期成员 GPU artifact。
