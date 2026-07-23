# Phase 5 决策门

## 建议：B — 进行一次严格预声明的美国市场外部验证

不建议现在继续扩大 CSI 500 方法开发或重训 FinPFN。建议冻结当前结论，下一步只做一次 U.S. 2010–2021 released-checkpoint 外部验证；若机制重复出现，则转为 C（结束方法开发，将项目定位为复现与模型风险审计）。

## 证据

| 决策条件 | 结果 | 判断 |
|---|---|---|
| uncertainty 预测 error/rank failure | predictive interval width 与绝对 z/rank error Spearman 0.228/0.105；与下一期 rank instability 0.099 | 有统计信号，但较弱且与 prediction extremeness 混杂 |
| uncertainty gating 改善 validation net performance | 最佳 uncertainty 候选 net Sharpe 1.237，低于未修改 FinPFN 1.551 | 否 |
| turnover control 保留 IC 并改善净表现 | rank buffer validation 1.938，但 test -1.040，低于未修改 -0.900 | 否；样本外消失 |
| IC–portfolio gap 有清晰机制 | FinPFN 中部 IC 较强，但 top-40 precision 仅 0.090，头尾留存显著较低 | 是 |
| plausible costs 下有经济优势 | 10 bps net Sharpe FinPFN -0.900、Ridge 1.035、LightGBM 0.686；FinPFN break-even 8.296 bps | 否 |
| tail performance 优于简单模型 | FinPFN top-40 precision 0.090，Ridge/LightGBM 0.149/0.156；gross Sharpe 也更低 | 否 |
| further progress 是否需要 test-driven tuning | CSI validation 胜出的 buffer 已在 test 失败；继续调阈值会构成 test-driven tuning | 是 |

正面证据只支持一个可外部检验的机制假设：FinPFN 可能提高大量中部排序，却因极端选择不准和排名不稳定而无法转换为更好的组合收益。负面证据足以否定“继续在 CSI test 上调 gating/turnover 参数”的做法。

## 为什么选 B，而不是 A 或立即 C

- 不选 A：uncertainty 没有验证期交易增量价值，turnover overlay 样本外失败，成本后 FinPFN 明显弱于 Ridge/LightGBM。此时重训或扩大搜索的 test-overfitting 风险太高。
- 暂不直接选 C：仓库已经包含独立的 U.S. monthly panel 和 released U.S. checkpoint，外部验证不要求 FinPFN 重训；它能区分“CSI/日频特有现象”和“模型任务构造的一般性风险”。
- B 不是继续调方法。U.S. sampling、seed、baseline search、universe 和指标必须在查看 U.S. test 前冻结，只运行一次。

## U.S. 验证后的最终门槛

完成 U.S. 外部验证后：

- 若再次出现“较高全体 IC、较低 tail precision/稳定性、成本后无优势”，选择 C，停止方法开发。
- 若 FinPFN 在 U.S. 同时表现出更好的共同-universe IC、tail precision、实际 H-L Sharpe，并在预声明成本下保持优势，才考虑 A；届时必须另写训练/验证计划，不能回到 CSI test 调参。
- 若结果混合或 checkpoint reproduction 本身失败，仍优先选择 C 或将结论限制为市场依赖，不用额外 test 搜索消除歧义。

## 已准备但未执行

人工命令、资源估算和防覆盖 wrappers 位于：

`reproduction/next_phase/us_external_validation/manual_commands.md`

主要估算：

- U.S. parquet 1.5 GB；3,529,899 rows；test 144 months。
- 单张 A100 80 GB checkpoint inference：两模型顺序执行约 30–50 分钟。
- Ridge + 六候选 LightGBM：4 CPU threads，预计约 10–45 分钟，8–16 GB RAM。
- evaluation：预计 <5 分钟。

## 当前停止点

未执行 U.S. baseline、GPU inference 或 evaluation；未重训/微调 FinPFN；未 commit 或 push。下一步必须由研究者明确批准。
