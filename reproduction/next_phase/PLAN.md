# CSI 500 下一阶段研究计划

状态：Phase 0–1 完成，Phase 2 本地部分完成，等待研究者人工 GPU

制定日期：2026-07-22
研究边界：不重训或微调 FinPFN；不覆盖既有复现文件；所有新增产物仅写入 `reproduction/next_phase/`。

## 1. 研究问题与可检验假设

核心问题是：为什么 FinPFN 在统一资产—日期样本上的横截面 IC/IR 高于 Ridge 和 LightGBM，却有更差的多空组合表现和更高换手；不确定性或交易稳定性信息能否改善其经济用途？

预先声明的假设如下。

1. **中部排序假设**：FinPFN 的 IC 优势主要来自截面中部的大量小幅排序改善，而非顶部/底部尾部选择。
2. **尾部误差假设**：FinPFN 的极端预测误差、错误入选率或尾部非单调性高于基线，导致 IC 优势不能转换为组合收益。
3. **排序不稳定假设**：FinPFN 的头尾成员和相邻期排名更不稳定，产生较高换手。
4. **成本侵蚀假设**：在统一、预先声明的单边交易成本下，FinPFN 的净多空表现比 Ridge/LightGBM 更快恶化。
5. **不确定性可用性假设**：若八成员预测离散度确实可提取，它与未来绝对误差、排名误差、尾部误选或 IC 恶化正相关。
6. **增量价值假设**：若不确定性有效，基于验证期选择的简单置信度控制在同等或相近换手下降幅度下，应优于仅用 rank buffer/持有期的机械降频方案。
7. **任务构造假设**：50 股票分组及重复抽样可能提升组内/总体 IC，却削弱把所有资产合并后的全截面尾部可比性。

每项假设都允许得到否定结果；不会为了获得正结论而根据测试期反复调参。

## 2. 已有输入、冻结边界与缺失输入

### 2.1 冻结输入

共同测试集为 301 个查询日、120,620 个唯一资产—日期（平均 400.731 个/日），组合收益统一使用 CSI parquet 中的原始小数收益并在报告时换算为百分点。四个模型都在相同日期和资产集合上比较。

| 类型 | 冻结路径 | SHA-256 |
|---|---|---|
| 完整中文复现报告 | `reproduction/notes/csi500_reproduction_report_zh.md` | `79fe5684dc4cf59155a0943aa844c964ee9f8496eb67fd90001bf7fae5b015a6` |
| FinPFN notebook-exact 预测 | `reproduction/artifacts/csi500_notebook_exact/csi500_finpfn_seed42_notebook_with_replacement.parquet` | `03e62d18bf14cb6a3787213a87369adf12914d65748f8d1536a7bc5cecca76f3` |
| TabPFN notebook-exact 预测 | `reproduction/artifacts/csi500_notebook_exact/csi500_tabpfn_seed42_notebook_with_replacement.parquet` | `0fa76d578741b3a50a9f6e1b96009bae6fe4f884b9ce7a3fe0f52b6cec95c26a` |
| Ridge 预测 | `reproduction/artifacts/predictions/csi500_baselines/csi500_ridge_seed42.parquet` | `a6cccd1f54f3ced4cd5165615a6c7d921d3d46d157f5a7e166a532532b6488b1` |
| LightGBM 预测 | `reproduction/artifacts/predictions/csi500_baselines/csi500_lightgbm_seed42.parquet` | `0a0c7f0bcbb5e97d25dcaf73448e55f9ec97b70aa8dc5bb91d0bf0f70eae375a` |
| 共同持仓 | `reproduction/results/csi500_all_models_notebook_exact/decile_holdings.parquet` | `bfcd2ba50283f41a0ad4dc2265d7067cb570edf79a9352bbd820ae02d57d432d` |
| 分期 decile 收益 | `reproduction/results/csi500_all_models_notebook_exact/decile_returns_by_period.csv` | `02c0c0c75d003d7c68b02565e13d373119b87b374761a7fc49e9370b2d16b14b` |
| 分期 IC | `reproduction/results/csi500_all_models_notebook_exact/ic_by_period.csv` | `a0078994a059eaacd0b1167cd02c982d72651d0f724ce92fbc3f77d7e820aba4` |
| decile 换手 | `reproduction/results/csi500_all_models_notebook_exact/turnover_by_decile.csv` | `60c622e967db8e90f5515571d05f1c91898b14439210ae7190dca7437c7afd4a` |
| 组合指标 | `reproduction/results/csi500_all_models_notebook_exact/portfolio_metrics.csv` | `88aac0f9c11d3981632d2599f5c54d31ab30233f8767d81c8b4a1b9c4cf7a1de` |
| 评估范围 | `reproduction/results/csi500_all_models_notebook_exact/evaluation_scope.json` | `fa63634019af046c37eac5cd9f03fbe1bf912ef2d0b49565c069ac506c6d4fbc` |

CSI 原始 parquet、released checkpoints、checkpoint metadata、环境记录和官方评估脚本也只读使用。脚本每次执行前校验输入摘要，不允许把输出目录指向冻结路径。

### 2.2 当前可用信息

- 四模型聚合预测、共同 raw-return target、共同持仓、逐期 IC/decile/多空收益和 decile 换手均已存在。
- FinPFN/TabPFN 文件保存了 `prediction`（八估计器输出的聚合中位数）和 `prediction_mean`，但没有成员预测、预测标准差、分位数、logits 或区间。
- 精确 checkpoint metadata 记录八估计器、抽样 seed 42、estimator random state 0、with-replacement 分组和 checkpoint 校验和。
- 2021 验证期和 2022-01 至 2023-04 测试期边界明确；当前 released-checkpoint 预测只覆盖测试期。

### 2.3 缺失信息

- FinPFN/TabPFN 在验证期的逐成员预测；测试期逐成员预测也未保存。
- 可以直接用于验证选择的 FinPFN 不确定性分数。
- sector、size、liquidity 等经济标签未确认存在，不会从模糊 feature 名推断。
- 原始特征时间戳和 forward-return 两条价格腿缺失，因此点时可用性与 target 对齐仍是数据来源层面的不可验证假设。

## 3. 阶段、依赖、指标与输出

### Phase 1 — 交易成本与换手敏感性（本地 CPU）

依赖：仅冻结的共同持仓和 raw decile return series，不重新推断模型。

固定成本网格为每单位**单边换手** `0, 2, 5, 10, 20, 30, 50 bps`；该网格不因模型排名改变。收益先换成小数，`cost_rate = bps / 10,000`。

- 每个等权单腿在相邻期的换手为 `0.5 × Σ|w_t - w_{t-1}|`。
- 首个组合日从现金建仓，long 和 short 各记 1.0 单边换手。
- 多空组合是 +1 long、-1 short，总 gross exposure 为 2；多空总换手为 long 与 short 单边换手之和，范围通常为 0–2。
- `net_return_t = gross_return_t - cost_rate × turnover_t`。买入、卖出、退出和再平衡都由权重变化涵盖；借券费、冲击成本和融资成本不在本阶段假定中。

每模型/成本报告 gross/net mean、年化波动、实际多空收益序列 Sharpe、算术累计收益、复合财富、最大回撤、平均换手和均值归零的 break-even bps，并按每个成本层级的 net Sharpe 排名。辅助报告 long 和 short 两条腿。高换手审计须验证：共同持仓没有重复 `(model,date,id)`；每个资产每期只属于一个 decile；四模型每期评估 universe 相同；重复抽样预测在形成 decile 前已按资产—日期取均值；由持仓重算的换手与冻结摘要一致（冻结摘要不含首日）。

输出：

- `reproduction/next_phase/costs/transaction_cost_analysis.py`
- `reproduction/next_phase/costs/cost_sensitivity.csv`
- `reproduction/next_phase/costs/model_break_even_costs.csv`
- `reproduction/next_phase/costs/net_performance_by_period.csv`
- `reproduction/next_phase/costs/figures/`
- `reproduction/next_phase/costs/report.md`

### Phase 2 — 不确定性可用性与校准审计（先本地；缺失部分需人工 GPU）

先对 schema、metadata 和聚合列做只读审计，确认哪些 uncertainty output 真正存在。`prediction_mean - prediction` 仅作为“聚合均值—中位数分歧”，不能冒充 ensemble dispersion。

当前已确认逐成员输出缺失，因此本阶段将：

1. 新增独立 inference 脚本/参数，保存每个 ensemble member 对每个 group-query row 的预测；
2. 明确支持 `validation` 和 `test` split，分别使用其内部相邻日期，因此验证查询日从 2021 年第二个交易日开始；
3. 保持 checkpoint、预处理、抽样模式、seed、估计器数量和聚合预测完全不变；
4. 新目录存在即拒绝写入，不提供默认覆盖；
5. 在本地做静态检查和无需 GPU 的输入/命令检查，然后暂停，由研究者手动执行单 GPU wrapper。

成员输出返回后，先只用验证期评估：ensemble SD、MAD、IQR、mean-median disagreement 和 cross-sectional rank disagreement（仅保留 TabPFN 2.0.8 实际可导出的量）。指标包括 uncertainty 与绝对预测误差/绝对 rank error 的 Spearman、错误尾部入选率、按 uncertainty quantile 的误差与尾部 precision、适用时的 coverage/reliability curve、shock/non-shock 差异、下一期 IC 与换手/排名不稳定的关系。所有结论称为 ensemble disagreement，除非覆盖率和校准证据足以支持更强表述。

输出：

- `reproduction/next_phase/uncertainty/audit_uncertainty_outputs.py`
- `reproduction/next_phase/uncertainty/evaluate_uncertainty.py`
- `reproduction/next_phase/uncertainty/calibration_metrics.csv`
- `reproduction/next_phase/uncertainty/coverage_performance.csv`
- `reproduction/next_phase/uncertainty/figures/`
- `reproduction/next_phase/uncertainty/report.md`
- `reproduction/next_phase/uncertainty/artifacts/validation/`（人工 GPU，新产物）
- `reproduction/next_phase/uncertainty/artifacts/test/`（仅冻结策略后人工 GPU，新产物）
- `reproduction/next_phase/uncertainty/scripts/`（人工执行 wrappers）

### Phase 3 — 验证期选择的 gating 与换手控制（成员预测返回后，本地 CPU）

进入条件：Phase 2 至少发现一个有可测验证信息的 uncertainty/stability signal。固定选择成本为 **10 bps / 单边换手**；不会根据测试结果改变。

候选网格在读取测试期成员预测前固定为：

- 原始 FinPFN：无控制；
- confidence coverage：最低 uncertainty 的 `{100%, 75%, 50%}` 候选头尾资产；
- 对称 uncertainty-adjusted rank score：多头 `rank(pred) - λ·rank(uncertainty)`、空头 `rank(pred) + λ·rank(uncertainty)`，`λ ∈ {0.25, 0.5, 1.0}`；
- 不含 uncertainty 的 rank buffer：进入 top/bottom 10%，退出边界 `{15%, 20%}`；
- 不含 uncertainty 的最短持有期 `{2, 3}` 个期间。

候选过多或某方法无法严格对称实现时，优先删去而不是扩大搜索。验证主目标是 10 bps 下 actual long-short net Sharpe，同分依次选择更低换手、更高 gross Sharpe、声明顺序靠前者。并报告 gross/net Sharpe、IC/IR、换手、coverage、最大回撤、两腿表现、持仓数量及 HHI/最大权重。冻结一个配置后，测试期只执行一次。测试对照必须包含 unmodified FinPFN、Ridge、LightGBM，以及验证期选定的无 uncertainty 换手控制。

若验证期 uncertainty 无用，停止 uncertainty gating，不查看测试期来挽救结果，并将负面结论带入 Phase 4。

输出：

- `reproduction/next_phase/gating/configs/`
- `reproduction/next_phase/gating/run_validation_selection.py`
- `reproduction/next_phase/gating/evaluate_frozen_test.py`
- `reproduction/next_phase/gating/validation_results.csv`
- `reproduction/next_phase/gating/selected_config.json`
- `reproduction/next_phase/gating/test_results.csv`
- `reproduction/next_phase/gating/figures/`
- `reproduction/next_phase/gating/report.md`

### Phase 4 — 解释 IC–组合差距（本地 CPU）

依赖：冻结共同测试预测；本阶段是机制/归因分析，不用于回调 Phase 3 参数。分析预先固定为 20 个 predicted-percentile bins；top/bottom `k ∈ {10, 20, 40}`（约 2.5%、5%、10%，同时报告实际比例）；相邻日期成员留存、percentile rank migration、预测幅度—收益关系、尾部误差，以及逐日期 IC 与多空收益贡献和 leave-one-date-out 影响。

主要指标：percentile 平均/中位 realized return 及单调性 Spearman；中部 20%–80% 与两端的局部 rank IC；top/bottom precision（相对 realized-return 同规模尾部）；Jaccard/overlap、rank migration；按日期和资产的集中度；累计结果对最大贡献日期的敏感性。仅在数据中存在清晰且文档化标签时才做 subgroup 分析。

输出：

- `reproduction/next_phase/ic_portfolio_gap/analyze_rank_tails.py`
- `reproduction/next_phase/ic_portfolio_gap/percentile_return_curve.csv`
- `reproduction/next_phase/ic_portfolio_gap/tail_precision.csv`
- `reproduction/next_phase/ic_portfolio_gap/rank_stability.csv`
- `reproduction/next_phase/ic_portfolio_gap/date_contributions.csv`
- `reproduction/next_phase/ic_portfolio_gap/figures/`
- `reproduction/next_phase/ic_portfolio_gap/report.md`

### Phase 5 — 决策门（本地写作，不自动启动新研究）

整合 Phase 1–4，形成 `reproduction/next_phase/DECISION.md` 与 `reproduction/next_phase/final_report.md`，在 A（扩大原创研究）、B（美国市场外部验证）、C（停止方法开发并定位为复现/模型风险审计）中给出一项建议。判断严格使用用户列出的正向/停止条件。不会自动重训、启动美国市场或执行服务器任务；只准备所需命令和计算量估计并等待批准。

## 4. 验证/测试隔离

- **验证期**：2021-01-01（含）至 2022-01-01（不含）；只在这里定义/选择 uncertainty score、coverage、lambda、buffer、holding period 和最终方法。
- **测试期**：2022-01-01（含）至 2023-04-03；冻结基线已经测试过，因此 Phase 1 和 Phase 4 明确属于对冻结测试结果的解释性/敏感性分析，而非新策略的确认性选择。
- Phase 3 的 test artifact 在 `selected_config.json` 写入并校验后才读取；冻结配置只测试一次。
- 不会使用论文数字作为目标，也不会依据测试期更改成本假设、阈值或方法。

## 5. 预计计算需求

| 工作 | 位置 | 预计资源/时间 |
|---|---|---|
| Phase 1 成本分析 | 本地 CPU | 1–4 线程，<2 GB RAM，通常 <2 分钟 |
| Phase 2 schema/校准代码和静态检查 | 本地 CPU | 1–4 线程，<4 GB RAM，数分钟 |
| Phase 2 验证期成员推断 | 研究者人工 GPU | 单张 A100 80GB；FinPFN 与 TabPFN 各约 20–30 分钟，实际以日志为准 |
| Phase 2 测试期成员推断 | 仅冻结选择后、研究者人工 GPU | 单张 A100 80GB；每模型约 35 分钟，新增 8 成员列使文件略增大 |
| Phase 2 校准 / Phase 3 选择 | 本地 CPU | 1–4 线程，<8 GB RAM，数分钟至约 15 分钟 |
| Phase 4 归因 | 本地 CPU | 1–4 线程，<8 GB RAM，预计 <15 分钟 |

不申请多 GPU，不下载新数据。GPU 时间是估算，不作为成功标准。

## 6. 停止标准

- 输入校验和、共同 universe 或 raw-return join 不一致：停止并报告，不修补冻结输入。
- Phase 1 发现持仓重复、共享预测列、目标分组或换手计算错误：隔离为新结果并报告；未获批准不修改冻结 evaluator。
- TabPFN 2.0.8 API 不能无模型变化地导出逐成员预测：停止该路径，不用伪造 proxy 代替。
- Phase 2 没有验证期有效 signal：不进入 uncertainty gating；仅允许预声明的无 uncertainty 换手对照和 Phase 4 机制分析。
- Phase 3 配置一旦冻结，测试只运行一次；任何实现 bug 必须先证明与经济结果无关，并保留失败运行记录。
- 任一动作需要 GPU、服务器、较大下载、重训、昂贵搜索、覆盖冻结文件、commit 或 push：暂停等待明确批准。

## 7. 风险与混杂因素

- notebook-with-replacement 造成 195,550 行但只有 120,620 个唯一资产—日期；成员离散度可能同时反映 group composition、label z-score 和 estimator ensemble，不能自动解释为后验不确定性。
- 每组独立 target z-score 会改变跨组预测尺度；全市场尾部排名可能受任务分组影响。
- FinPFN/TabPFN 的验证期成员推断将使用同一可见 notebook 规则，但 bundle 的原始抽样状态不可恢复。
- 测试期成本和机制分析属于事后审计；它们可解释既有结果，却不是新策略的无偏性能估计。
- 等权 decile 的历史收益未含可交易性、涨跌停、借券、滑点、融资和容量限制；成本网格只是敏感性分析。
- 每日约 400 个共同资产，固定数量 top-k 的百分比会随日期略变，因此同时报告 k 与覆盖率。
- shock 日期来自 notebook 的硬编码窗口；缺少独立 CIMV 原始序列。
- 只有一个 released checkpoint / sampling seed 的主要结果，不能把模型随机性与时间序列不确定性完全分离。

## 8. 执行记录

| 阶段 | 状态 | 实际结果/依赖更新 |
|---|---|---|
| Phase 0 | 完成 | 已冻结并校验上述输入；确认无逐成员预测。 |
| Phase 1 | 完成 | 本地 CPU 4.306 秒。FinPFN / Ridge / LightGBM 多空总换手为 1.782981 / 1.478813 / 1.551627；break-even cost 为 8.296 / 12.683 / 11.660 bps；10 bps net Sharpe 为 -0.900 / 1.035 / 0.686。持仓、模型自有预测、共同 universe、重复折叠和冻结换手均通过一致性检查。 |
| Phase 2 | 等待人工 GPU | 现有聚合文件无逐成员/分布摘要。已核对官方 PyPI TabPFN 2.0.8 source（source SHA-256 `ffb739fc898c7144cd974a50a4185050cd5cae09405be2c81a8b6a3c7421fcbc`），新增只读旁路成员输出。两模型各一组 CPU smoke 全部有限，且聚合均值/中位数/众数/五个分位数与公共 API 最大绝对差均为 0。验证期评估脚本已 smoke 贯通；需人工运行 242 个 validation query dates。 |
| Phase 3 | 阻塞 | 等待 Phase 2 验证信号。 |
| Phase 4 | 待执行 | 仅依赖冻结测试预测，但按顺序在 Phase 3 后执行。 |
| Phase 5 | 待执行 | 依赖所有可执行阶段。 |

每一阶段完成后，本表会追加实际 runtime、命令、关键校验与是否触发停止条件；历史内容不删除。
