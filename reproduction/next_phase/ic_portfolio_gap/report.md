# Phase 4：FinPFN 的 IC–组合表现差距

## 执行口径

本分析只读取冻结的 301 个测试日期、120,620 个共同资产—日期和统一 raw-return target。预测先按 `(model,date,id)` 对重复 group rows 取均值，与官方 evaluator 一致。20 个 predicted-percentile bins、`k={10,20,40}`、20%/80% 局部区域均在分析前固定；结果是对已知测试表现的**探索性机制审计**，不用于回调 Phase 3。

| 模型 | 全体 IC | 中部 20–80% IC | bottom 20% IC | top 20% IC | 20-bin 单调性 | top-40 precision | bottom-40 precision | top-40 相邻留存 | bottom-40 相邻留存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FinPFN | 0.0456 | 0.0240 | 0.0381 | 0.0033 | 0.844 | 0.090 | 0.202 | 0.084 | 0.137 |
| Ridge | 0.0374 | 0.0153 | 0.0521 | -0.0062 | 0.953 | 0.149 | 0.190 | 0.276 | 0.252 |
| LightGBM | 0.0364 | 0.0187 | 0.0506 | -0.0000 | 0.895 | 0.156 | 0.203 | 0.238 | 0.215 |
| TabPFN | -0.0378 | -0.0209 | -0.0013 | -0.0410 | -0.902 | 0.142 | 0.130 | 0.091 | 0.084 |

## 核心解释

1. **FinPFN 的优势并不等于更好的尾部资产识别。** 全体平均 IC 为 0.0456，高于 Ridge 的 0.0374 和 LightGBM 的 0.0364；但 top/bottom 40 precision 分别为 0.090/0.202，须与两条基线在上表直接比较。局部 IC 也表明总体 rank 改善和每一侧尾部内部的精细排序不是同一个任务。
2. **FinPFN 的头尾成员更不稳定。** 相邻日期 top/bottom 40 留存为 0.084/0.137；Ridge 为 0.276/0.252，LightGBM 为 0.238/0.215。这与 Phase 1 的 FinPFN 最高换手一致。
3. **IC 与尾部组合收益只有不完全对应。** FinPFN 日期级 IC 与同日 long-short return 的 Spearman 为 0.713。IC 使用全部约 401 个资产的排序信息，而 decile 组合只使用约 40+40 个极端资产；许多中小排序改善可以提高 IC，却不会进入持仓。
4. **简单降换手不能自动保留 IC 的经济收益。** Phase 3 的 20% rank buffer 在 validation 改善净 Sharpe，但唯一一次 test 将 gross/net Sharpe 降至 3.760/-1.040；说明不稳定性是症状之一，却不能单靠 buffer 修复尾部收益。
5. **50-stock group 重复抽样是可见混杂，但不是充分解释。** FinPFN 同一资产—日期平均重复 1.621 次；group-composition prediction SD 与绝对 rank error 的 Spearman 为 -0.032，尾部 SD 为 0.0114、中部为 0.0151。这说明 group composition 会改变预测，但仅凭相关性不能断言它造成全部 tail gap。

## 最大 FinPFN 日期贡献

| 日期 | IC | long-short return |
|---|---:|---:|
| 2022-01-28 | 0.1474 | 1.950% |
| 2022-11-01 | 0.1359 | 1.687% |
| 2022-06-10 | 0.0903 | 1.392% |
| 2022-06-21 | 0.1306 | 1.301% |
| 2022-06-30 | -0.0665 | -1.298% |

逐日期的 leave-one-out IC 与 long-short Sharpe、收益占比见 `date_contributions.csv`；逐资产尾部贡献见 `asset_contributions.csv`。FinPFN 绝对资产贡献最大的 5 个 ID 占全部绝对资产贡献的 3.52%，因此报告同时保留日期和资产集中度，避免把总结果误解为均匀分布。

## 对预声明假设的判断

- **“FinPFN 改善中部排序但不改善极端选择”**：由中部/尾部局部 IC、20-bin 曲线和 top/bottom precision 联合判断；总体上得到支持，但不是“优势只存在于中部”的强结论。
- **“FinPFN 极端预测较不稳定”**：得到支持；相邻成员留存更低、rank migration 更大，并与较高换手一致。
- **“FinPFN 通过许多小排序改善获得更高 IC”**：得到支持；全体 IC 优势大于尾部 precision/组合优势。
- **“高换手移除或反转统计优势”**：成本侵蚀得到支持；但 Phase 3 的 buffer 失败说明高换手并非唯一因果机制。
- **“group-wise task construction 影响 global tail ranking”**：存在可测 composition dispersion，属于合理混杂；当前 artifact 不足以给出因果证明。

数据中只有匿名技术/财务 features 和资产 ID，没有文档化 sector、size、volatility 或 liquidity 标签，因此没有从含义不明确的列名推断经济分组。完整数值见本目录 CSV，图见 `figures/`。本地 CPU runtime 为 13.757 秒。
