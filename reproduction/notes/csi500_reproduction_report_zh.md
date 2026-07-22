# FinPFN CSI 500 复现报告

报告日期：2026-07-22

复现代码主分支：`master`

notebook-exact 代码提交：`d5c3cc9801a9fbc4b183c6e2d9ea8ac678117002`

上游仓库审计基准提交：`99b2a0e`
复现状态：**CSI 500 主要复现工作完成；整篇论文复现尚未包含美国市场。**

## 1. 执行摘要

本报告覆盖 FinPFN 论文在 CSI 500 数据上的主要可验证结论，包括：

- 官方数据、切分、特征和目标检查；
- Ridge 和 LightGBM baseline；
- vanilla TabPFN 和发布的 FinPFN checkpoint；
- 横截面 Spearman IC、IC 标准差和 IR；
- 十分位组合、top decile、top-minus-bottom 多空组合；
- gross Sharpe、累计收益和 turnover；
- 服务器环境、硬件、运行时间、输入文件校验和；
- 目标口径、随机抽样、重复股票、组合指标和潜在泄漏核查。

主要结论如下。

1. **CSI 500 的主要复现阶段已经完成。** 四个目标模型均已产生实际预测并用统一 evaluator 计算指标；没有训练或微调 FinPFN，没有交易成本或额外特征，也没有根据测试结果选择参数或随机种子。
2. **FinPFN 的横截面排序优势得到部分复现。** 在 notebook-exact 配置下，保留重复股票的官方 notebook IC 得到 FinPFN IR `0.797333`，接近论文的 `0.85`，但没有精确达到。
3. **在同一 raw-return、同一资产日期宇宙下，FinPFN 的 IR 超过重建的 Ridge 和 LightGBM。** FinPFN 为 `0.712002`，LightGBM 为 `0.566578`，Ridge 为 `0.539210`。
4. **FinPFN 的组合全面领先没有复现。** 同一宇宙下，FinPFN 的真实 gross long-short Sharpe 为 `4.383559`，低于 LightGBM 的 `4.810360` 和 Ridge 的 `4.888952`；top-decile 累计收益和多空累计收益也没有领先。
5. **论文的 headline 比较存在目标口径不对称。** 官方 notebook 对 FinPFN/TabPFN 使用每个 50 股票任务内处理后的 `target`，而对 Ridge/LightGBM 使用全日期横截面的 `return`。在作者发布 CSV 上，这一差异把 FinPFN IR 从同口径 raw-return 的约 `0.683862`提高到论文口径的 `0.855546`。
6. **发布 CSV 无法由可见 notebook 精确生成。** CSV 每天恰好有 500 个不重复股票，而 notebook 明确使用有放回抽样。notebook-exact 新预测与发布预测的 FinPFN Spearman 相关仅为 `0.463426`，说明作者生成发布 CSV 时使用的精确抽样/分组状态没有公开保存。

因此，本项目目前最严谨的表述是：

> CSI 500 上部分复现了 FinPFN 的横截面排序优势，但没有精确恢复作者发布预测，也没有复现 FinPFN 在投资组合指标上全面优于 Ridge 和 LightGBM 的结论。

## 2. 范围、完成状态与边界

### 2.1 已完成

| 工作项 | 状态 |
|---|---|
| 仓库、README、脚本、notebook、环境文件审计 | 完成 |
| 官方 CSI 500 parquet 完整性与切分审计 | 完成 |
| Ridge 五候选 validation-only 选择 | 完成 |
| LightGBM 六候选 validation-only 选择 | 完成 |
| vanilla TabPFN 发布 checkpoint 完整推理 | 完成 |
| FinPFN 发布 checkpoint 完整推理 | 完成 |
| artifact-shape 500-unique 运行 | 完成并冻结 |
| visible-notebook-exact 有放回运行 | 完成并冻结 |
| IC、SD、IR、分位数组合、Sharpe、turnover | 完成 |
| Ridge/LightGBM 近似 Sharpe 独立性检查 | 完成 |
| 发布预测 CSV 对照 | 完成 |
| 子期与官方 CSI shock windows | 完成 |
| 图形与 per-period 序列 | 完成 |

### 2.2 尚未完成或不属于当前阶段

- 美国股票市场的完整 checkpoint 和 baseline 运行尚未执行，因此不能称为整篇论文全部复现完成。
- U.S. 官方 VIX regime 日期缺少原始来源，不能静默构造。
- FinPFN 优化、重新训练、替代特征、交易成本和不确定性 gating 均属于下一阶段，不应回写或替换本报告的冻结基线。
- 作者未发布生成最终 parquet 的原始数据代码、完整 baseline fitting code、精确抽样状态或完整环境锁文件，相关 provenance 无法独立证明。

## 3. 仓库与官方产物审计

主要入口和产物为：

- `scripts/main.py`：底层 FinPFN/TabPFN fine-tuning 训练循环；
- `scripts/training_utils/data_utils.py`：数据切分、相邻日期任务、股票抽样和目标标准化；
- `finpfn.ipynb`：唯一发布的完整预测与评估流程；
- `models/finpfn_30feats_csi500.ckpt`：CSI 500 FinPFN checkpoint；
- `models/tabpfn-v2-regressor.ckpt`：vanilla TabPFN v2 regressor checkpoint；
- `results/finpfn_perf_csi500.csv.gz`：作者发布的 CSI 500 预测结果。

关键文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `30features_csi500.parquet` | `9e0d61f5d70151d4f2f7b40918a8ddb79f86fb54a0fe86759f5c1f2869fe1b3e` |
| `models/finpfn_30feats_csi500.ckpt` | `c035f2a79c74ab7f38b023fa98624d078b6389c3d096ac1a1270b04361dd0214` |
| `models/tabpfn-v2-regressor.ckpt` | `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736` |
| `results/finpfn_perf_csi500.csv.gz` | `a39f91cc7f967982d8ee81471a75539d0d2f47d07feede86f7bf5e3097a10172` |

作者发布 CSV 有 150,500 行、301 个 query 日期、每天恰好 500 个不重复资产日期预测。所有资产日期均可匹配官方 parquet。

## 4. 数据审计

### 4.1 CSI 500 数据概况

| 项目 | 数值 |
|---|---:|
| 文件大小 | 263,138,068 bytes |
| 总行数 | 1,039,372 |
| 日期数 | 1,762 |
| 股票 ID 数 | 1,057 |
| 模型特征数 | 30 |
| 日期范围 | 2016-01-05 至 2023-04-03 |
| 每日股票数 | 最小 464；中位数 607；最大 626 |

数据检查结果：

- 物理行顺序按日期非递减；
- 资产日期键唯一；
- 无 null、NaN 或无穷值；
- 30 个特征已在每个日期横截面上近似零均值、样本标准差 1；
- `target` 是未标准化的 decimal return，范围约 `-4.97%` 至 `6.81%`；
- `target` 的每日横截面标准差中位数约 `1.81%`；
- 极值与抽样的 1%/99% 分位点相符，但最终 parquet 无法证明上游 winsorization 的精确实现。

### 4.2 官方代码切分

| Split | 行数 | 日期数 | 实际日期 |
|---|---:|---:|---|
| Train | 703,709 | 1,217 | 2016-01-05 至 2020-12-31 |
| Validation | 149,450 | 243 | 2021-01-04 至 2021-12-31 |
| Test | 186,213 | 302 | 2022-01-04 至 2023-04-03 |

代码使用半开区间。checkpoint 推理用相邻日期，因此第一个 test 日期只作为 context，实际 query 从 2022-01-05 开始，共 301 个日期。

### 4.3 特征、目标和任务构造

每个相邻日期任务为：

- context：`(X_{t-1}, y_{t-1})`；
- query features：`X_t`；
- query target：`y_t`，只用于事后评估；
- 只保留在相邻两个日期都存在的股票；
- 每个任务 50 个 context 股票和相同 50 个 query 股票；
- context/query 目标分别在该 50 股票组内按样本标准差进行 z-score；
- 模型输出使用八个 ensemble members 的 median prediction。

CSI parquet 特征已经预处理，因此 baseline 和 checkpoint 推理直接使用存储值，没有再次 winsorize 或标准化特征。

## 5. 时间对齐与泄漏核查

没有发现 query target 被输入模型：checkpoint inference 的 `.fit()` 只接收上一日期的 context feature/label，`.predict()` 只接收 query features。

但仍有不能由最终产物证明的时间对齐假设：

- 论文称中国特征仅使用 `t-1` close 之前的信息；
- target 为 `t` 至 `t+1` 的 Barra-adjusted open-to-open return；
- parquet 没有原始数据时间戳、原始 return legs 或构造代码。

因此没有观察到直接 target leakage，但 point-in-time feature availability 和 forward-return alignment 只能作为作者数据 provenance 假设，不能由本仓库独立验证。

## 6. 运行环境与硬件

### 6.1 Checkpoint GPU 环境

| 组件 | 版本/配置 |
|---|---|
| OS | Linux 5.15, x86_64, glibc 2.35 |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA runtime | 12.1 |
| NVIDIA driver | 535.230.02 |
| TabPFN | 2.0.8 |
| pandas | 2.3.3 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| PyArrow | 18.1.0 |
| GPU | 单张 NVIDIA A100 80GB PCIe |
| CPU 限制 | 4 workers；主机记录 64 logical CPUs |

GPU 运行由研究者手动执行；Codex 没有登录或向服务器发出命令。服务器保存的环境记录最初在提交 `1155bf3` 捕获；notebook-exact wrappers 来自提交 `d5c3cc9`，运行环境未改变。

### 6.2 本地 baseline 环境

Ridge/LightGBM 在 macOS arm64 主机上使用 4 CPU threads。metadata 记录 10 logical CPUs、NumPy 1.26.4、pandas 2.2.3、scikit-learn 1.6.1；LightGBM 为 4.6.0。

## 7. 模型与选择协议

### 7.1 Ridge

Ridge 是独立重建 baseline，因为仓库没有发布作者的 fitted model 或 baseline 训练代码。

- seed：42；
- 候选 alpha：`0.001, 0.01, 0.1, 1, 10`；
- selection data：仅 validation；
- selection metric：每日 Spearman IC 的平均值；
- 最终 fit：选择后在 train + validation 上一次性 refit；
- test：只预测和评估一次，未用于调参。

| Alpha | Validation mean IC | Fit + validation (s) |
|---:|---:|---:|
| **0.001** | **0.037600339** | 0.363 |
| 0.01 | 0.037600282 | 0.312 |
| 0.1 | 0.037600207 | 0.326 |
| 1 | 0.037599822 | 0.227 |
| 10 | 0.037600131 | 0.224 |

选择时间 `1.453 s`，最终 refit `0.466 s`，test prediction `0.004 s`。

### 7.2 LightGBM

LightGBM 同样是独立重建 baseline。六个候选在运行前固定，只使用 validation 选择。

| Candidate | learning rate | Trees | Leaves | Depth | Min child | L1 | L2 | Validation mean IC | 时间 (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.05 | 500 | 31 | 4 | 50 | 0 | 0 | 0.042854663 | 6.009 |
| **1** | **0.05** | **500** | **63** | **6** | **50** | **0** | **0** | **0.043467032** | **8.050** |
| 2 | 0.03 | 800 | 31 | 6 | 100 | 0 | 0.1 | 0.042712057 | 10.815 |
| 3 | 0.03 | 800 | 63 | 6 | 100 | 0.1 | 0.1 | 0.042843866 | 12.495 |
| 4 | 0.02 | 1,200 | 31 | -1 | 100 | 0.1 | 1 | 0.043287852 | 16.068 |
| 5 | 0.02 | 1,200 | 63 | -1 | 100 | 1 | 1 | 0.043393256 | 21.500 |

Candidate 1 仅因 validation mean IC 最高而被选择。六候选选择 `74.937 s`，最终 refit `9.844 s`，test prediction `0.913 s`，总 modeling runtime `85.694 s`。

### 7.3 vanilla TabPFN 与 FinPFN

共同设置：

- 发布 checkpoint，不重新训练；
- TabPFN 2.0.8；
- 30 个 CSI 特征；
- 相邻日期 context/query；
- 50 context + 50 query；
- 八个 estimators；
- median prediction 为主预测；
- 单 GPU；
- 无 transaction costs；
- 所有测试指标只作为报告，不用于选择配置。

执行了两个预先区分的配置。

#### A. artifact-shape primary

- 股票抽样 seed 42；
- estimator random state 42；
- 每日期对抽 500 个不重复共同股票；
- 分为十个 50 股票组；
- 目的是匹配发布 CSV 可观察到的“每天 500 unique assets”形状。

该运行不能称为 literal notebook run，因为可见 notebook 使用有放回抽样且 estimator default random state 为 0。结果被保留并冻结，没有因表现较差而删除。

#### B. notebook-exact follow-up

- NumPy 股票抽样 seed 42；
- 有放回抽样；
- 每组抽样后按 ID 排序，与 notebook 的 `sort_values([date, id])` 一致；
- estimator random state 0，即 TabPFN 2.0.8 默认值；
- `n_jobs=4`，用于资源合规。固定输入 smoke 表明 `n_jobs=4` 与 notebook 默认 `-1` 的输出逐元素相同。

该配置是本报告主要 checkpoint 结论的基础。

## 8. 指标定义与评估口径

### 8.1 IC 和 IR

- IC：每日期、跨股票的 Spearman rank correlation；
- IC SD：日期 IC 的样本标准差，`ddof=1`；
- IR：`mean(IC) / std(IC)`，不年化。

存在两种必须分开的目标口径：

1. **paper/notebook target**：FinPFN 和 TabPFN 使用每个 50 股票任务内预处理的 `target_group_z`，并在将多个组拼回同一日期后计算 IC；notebook-exact 还保留有放回抽样产生的重复股票行。
2. **common raw-return target**：所有模型统一使用 parquet 原始 `target` 排名；重复预测先按 model/seed/date/id 平均，再使用共同资产日期宇宙。

paper 口径用于历史忠实复现；common raw-return 口径用于模型间可比结论。

### 8.2 Portfolio

- 每日期按各模型自己的 `prediction` 排序；
- `id` 仅作 deterministic tie-breaker；
- 分十个尽可能等大的 deciles；
- 每组等权；
- bottom = decile 1，top = decile 10；
- long-short = top - bottom；
- 使用 parquet 原始 return；CSI decimal return 乘 100 后以 percentage points 报告；
- 累计收益使用 arithmetic cumulative sum，不复利；
- CSI Sharpe 年化因子为 `sqrt(240)`；
- 不含交易成本。

报告同时保留：

- `notebook_sharpe_spread = Sharpe(top) - Sharpe(bottom)`；
- `primary_long_short_sharpe = Sharpe(top_return - bottom_return)`。

论文所谓 H-L Sharpe 是前者，不是实际多空收益序列的 Sharpe。本文以后者作为主组合指标。

## 9. 作者发布结果与论文基准

作者发布 CSV 能精确恢复论文 headline IC/IR：

| Model | 发布 CSV mean IC | IC SD | IR | 论文 IR |
|---|---:|---:|---:|---:|
| FinPFN | 0.042006 | 0.049099 | 0.855546 | 0.85 |
| TabPFN | -0.028794 | 0.065050 | -0.442639 | -0.44 |
| Ridge | 0.040702 | 0.068034 | 0.598258 | 0.60 |
| LightGBM | 0.044120 | 0.063827 | 0.691239 | 0.70 |

但 FinPFN/TabPFN 与 baselines 使用不同目标。将发布 FinPFN 预测改用 common raw return 后，FinPFN IR 约为 `0.683862`，略低于发布 LightGBM 的 `0.691239`。因此论文 headline 中对 LightGBM 的领先至少部分来自目标定义不对称。

论文报告的 CSI portfolio 摘要包括：FinPFN top-decile Sharpe `5.1`、top cumulative `30.5%`、H-L reported Sharpe `9.8`、H-L cumulative `69.0%`；LightGBM 对应为 `3.1`、`23.5%`、`7.3` 和 `0.3%`。最后一个 `0.3%` 与发布 notebook 的 decile 端点相差约 60.3 percentage points，极可能是论文排版或录入错误。保存的 nonlinear Sharpe 输出也无法从 raw parquet 完全重算。

## 10. 独立重建 baseline 结果

这些是各模型在完整 302 日期 test 宇宙上的 standalone 结果，不是后续 checkpoint-sampled common universe。

| Model | 日期 | 预测行 | Mean IC | IC SD | IR | Top Sharpe | True H-L Sharpe | Top cumulative (pp) | H-L cumulative (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 302 | 186,213 | 0.040454 | 0.065650 | 0.616215 | 3.115775 | 6.176169 | 21.515341 | 66.370689 |
| LightGBM | 302 | 186,213 | 0.038657 | 0.058421 | 0.661700 | 3.135077 | 6.176134 | 19.218125 | 62.432569 |

Turnover：

| Model | Bottom one-way turnover | Top one-way turnover |
|---|---:|---:|
| Ridge | 0.613221 | 0.576384 |
| LightGBM | 0.664703 | 0.631120 |

### 10.1 Ridge/LightGBM 近似相同 Sharpe 的独立检查

两者真实 H-L Sharpe 分别为 `6.176169` 和 `6.176134`，但已排除文件复用或共同排序错误：

- prediction Pearson：`0.548496`；
- 全行 prediction Spearman：`0.630361`；
- 每日期 prediction Spearman 平均：`0.632606`；
- top-decile 平均持仓重合：`44.9376%`；
- bottom-decile 平均持仓重合：`50.3071%`；
- H-L return Pearson：`0.601474`；
- H-L series 最大绝对差：`1.688325 pp`；
- 两条保存收益序列重算误差均不超过 `2.22e-16`。

两个模型只是恰好具有近似相同的年化均值/波动率比，没有更改 evaluator。

## 11. Checkpoint 运行完整性

### 11.1 Artifact-shape primary

| Model | 行数 | 日期 | 每日行数 | 成功 groups | 失败 groups | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 150,500 | 301 | 500 | 3,010 / 3,010 | 0 | 1,669.336 |
| FinPFN | 150,500 | 301 | 500 | 3,010 / 3,010 | 0 | 1,697.121 |

所有 `prediction`、`prediction_mean` 和 `target_group_z` 都是有限值。

### 11.2 Notebook-exact

| Model | 总行数 | Unique asset-dates | 重复行 | 成功 groups | 失败 groups | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 195,550 | 120,620 | 74,930 | 3,911 / 3,911 | 0 | 2,054.745 |
| FinPFN | 195,550 | 120,620 | 74,930 | 3,911 / 3,911 | 0 | 2,078.874 |

重复行是有放回抽样的预期结果。两模型均覆盖 301 个 query 日期，无 failed rows、duplicate-file reuse 或 non-finite values。

返回文件本地 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `csi500_finpfn_seed42_notebook_with_replacement.parquet` | `03e62d18bf14cb6a3787213a87369adf12914d65748f8d1536a7bc5cecca76f3` |
| `csi500_finpfn_seed42_notebook_with_replacement.metadata.json` | `ff3cff21aa9a3902af7bfc2ca21d1bfd42168325a72919b719e0ebef3431a9d0` |
| `csi500_tabpfn_seed42_notebook_with_replacement.parquet` | `0fa76d578741b3a50a9f6e1b96009bae6fe4f884b9ce7a3fe0f52b6cec95c26a` |
| `csi500_tabpfn_seed42_notebook_with_replacement.metadata.json` | `cafcd2ec91e981a7f9015a866baa6184bb0d067a094c6f5abae91641be9e4530` |

## 12. IC/IR 复现结果

### 12.1 Artifact-shape primary：paper task target

| Model | Mean IC | IC SD | IR | 论文 IR |
|---|---:|---:|---:|---:|
| FinPFN | 0.031801 | 0.049099 | 0.647677 | 0.85 |
| TabPFN | -0.030197 | 0.065235 | -0.462894 | -0.44 |

该运行匹配发布 CSV 的 500-unique 形状，但没有匹配 visible notebook 的抽样和 estimator state，因此不是最终的 notebook-faithful 结果。

在该 301 日期、500 股票/日宇宙中统一使用 raw return 后：

| Model | Raw mean IC | IC SD | IR | True H-L Sharpe |
|---|---:|---:|---:|---:|
| FinPFN | 0.033254 | 0.057335 | 0.579996 | 5.124263 |
| TabPFN | -0.029691 | 0.065904 | -0.450515 | -3.549967 |
| Ridge | 0.042790 | 0.068989 | 0.620244 | 6.272572 |
| LightGBM | 0.039938 | 0.061976 | 0.644403 | 6.316055 |

这解释了为什么第一次运行没有显示 FinPFN 超过 baselines，也说明后续 notebook-exact 修正是实质性配置修正，而不是简单重新格式化结果。

### 12.2 Notebook-exact：保留重复行的 paper target

| Model | Mean IC | IC SD | IR | 发布 CSV IR | 论文 IR |
|---|---:|---:|---:|---:|---:|
| FinPFN | 0.043864 | 0.055013 | 0.797333 | 0.855546 | 0.85 |
| TabPFN | -0.034296 | 0.068915 | -0.497656 | -0.442639 | -0.44 |

相对第一次 FinPFN IR `0.647677`，notebook-exact 提高到 `0.797333`，证明有放回抽样、组内排序和 estimator seed 对结果具有实质影响。

FinPFN mean IC `0.043864` 实际高于发布 CSV 的 `0.042006`；没有达到相同 IR 的主要原因是日期 IC 标准差更高：`0.055013` 对 `0.049099`。即平均排序信号已经恢复，但跨期稳定性没有完全恢复。

### 12.3 Notebook-exact common raw-return comparison

重复预测先对每个 asset-date 平均。FinPFN/TabPFN 共同覆盖 120,620 asset-dates，301 日期，平均每天 `400.730897` 个股票；Ridge/LightGBM 被限制到同一宇宙。

| Model | Mean IC | IC SD | IR | Paper IR（不同口径） |
|---|---:|---:|---:|---:|
| **FinPFN** | **0.045597** | 0.064040 | **0.712002** | 0.85 |
| LightGBM | 0.036434 | 0.064305 | 0.566578 | 0.70 |
| Ridge | 0.037409 | 0.069378 | 0.539210 | 0.60 |
| TabPFN | -0.037758 | 0.072213 | -0.522875 | -0.44 |

这是本报告最公平的横截面预测比较。FinPFN 在相同目标和相同资产日期下领先两个重建 baselines。

## 13. Portfolio 结果

Notebook-exact common universe，raw gross return，无交易成本：

| Model | Bottom Sharpe | Top Sharpe | Notebook Sharpe spread | True H-L Sharpe | Bottom cumulative (pp) | Top cumulative (pp) | H-L cumulative (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinPFN | -3.729370 | 2.366336 | 6.095706 | 4.383559 | -30.548438 | 13.976832 | 44.525270 |
| LightGBM | -4.220336 | 2.574612 | 6.794947 | 4.810360 | -34.702081 | 19.756541 | 54.458622 |
| Ridge | -4.800512 | 2.144884 | 6.945396 | 4.888952 | -39.767048 | 16.689462 | 56.456510 |
| TabPFN | 2.932325 | -4.496578 | -7.428904 | -5.192668 | 21.751054 | -36.666937 | -58.417991 |

结论：

- FinPFN top Sharpe 高于 Ridge、低于 LightGBM；
- FinPFN true H-L Sharpe 低于 Ridge 和 LightGBM；
- FinPFN top cumulative 和 H-L cumulative 都低于两个 baselines；
- 因此 IC/IR 优势没有转化为本样本上的全面 portfolio 优势。

Turnover：

| Model | Bottom turnover | Top turnover |
|---|---:|---:|
| FinPFN | 0.864380 | 0.917878 |
| LightGBM | 0.785946 | 0.764186 |
| Ridge | 0.748795 | 0.728281 |
| TabPFN | 0.916045 | 0.910539 |

FinPFN top-decile turnover 约 `91.8%`，明显高于两个 baseline。由于本阶段按论文假设不含交易成本，这一高 turnover 尚未扣减收益；它是下一阶段经济可实现性评估的重要风险。

## 14. 子期与 regime 结果

Common raw-return IC：

| Model | 2022 IR（241 日期） | 2023 IR（60 日期） |
|---|---:|---:|
| FinPFN | 0.715507 | 0.697697 |
| LightGBM | 0.570012 | 0.548119 |
| Ridge | 0.516280 | 0.626552 |
| TabPFN | -0.552868 | -0.408633 |

FinPFN 在 2022 和 2023 两个子期均保持 common-universe IR 领先。

Notebook 提供的七个 CSI shock windows 合计覆盖 11 个交易日：

| Model | Shock IR（11 日期） | 非 shock IR（290 日期） |
|---|---:|---:|
| FinPFN | 0.725175 | 0.710203 |
| LightGBM | 0.176473 | 0.579117 |
| Ridge | 0.619931 | 0.536013 |
| TabPFN | -0.477733 | -0.524230 |

这些 regime 指标是测试后诊断，不用于模型选择。由于只有 11 个 shock 日期，不应过度解释。

## 15. 与作者发布预测的直接比较

Notebook-exact 重复预测先按 asset-date 平均，再与发布 CSV 匹配：

| Model | New unique asset-dates | Overlap | Overlap / new | Spearman | Pearson | Mean absolute difference |
|---|---:|---:|---:|---:|---:|---:|
| FinPFN | 120,620 | 97,664 | 0.809683 | 0.463426 | 0.533319 | 0.045996 |
| TabPFN | 120,620 | 97,664 | 0.809683 | 0.359216 | 0.431168 | 0.249491 |

相对 artifact-shape primary，FinPFN 与发布 bundle 的相关性有所提高，但仍远未达到同一预测。剩余差异的最直接解释是：发布 CSV 的 500 unique universe 与 visible notebook 的有放回抽样不兼容，精确 bundle generation state 没有公开。

## 16. 主要偏差、歧义与实现问题

1. **Bundle 与 notebook 抽样冲突。** notebook 用 `replace=True`，bundle 每天恰好 500 unique assets。
2. **模型间 IC target 不一致。** FinPFN/TabPFN 用 50 股票组内 target；baseline 用全日期 return。
3. **组内 target 处理改变跨组排名。** 每组分别 z-score 后再合并，不保证全日期排名保持不变。
4. **Portfolio H-L Sharpe 定义不标准。** 论文使用 `Sharpe(top)-Sharpe(bottom)`，本文同时报告真实 H-L series Sharpe。
5. **保存的 portfolio Sharpe 输出不完全可重算。** LightGBM 累计端点可从 raw parquet 恢复，但部分 nonlinear Sharpe 和论文表值不一致。
6. **Baseline 官方实现缺失。** 本文 Ridge/LightGBM 是预先声明、validation-only 的独立重建，不是作者模型二进制复现。
7. **环境未完整锁定。** requirements 不足以重建作者原始环境；Python 3.10/3.12 metadata 也不一致。
8. **论文/代码训练设置存在差异。** 包括 attention heads、learning rate 和 optimizer 描述与 checkpoint/code metadata 不一致。
9. **Point-in-time provenance 缺失。** 无原始 timestamps 和 return construction，不能彻底排除上游数据对齐问题。
10. **高 turnover 未计成本。** 本阶段按论文假设保持 gross/no-cost；不代表策略净收益可实现。

## 17. 可复现命令摘要

### 17.1 Ridge

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/train_ridge.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --output-dir reproduction/artifacts/predictions/csi500_baselines \
  --seed 42
```

### 17.2 LightGBM

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/train_lightgbm.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --output-dir reproduction/artifacts/predictions/csi500_baselines \
  --seed 42
```

### 17.3 Notebook-exact checkpoint inference

以下 wrappers 由研究者在批准的单 GPU 主机手动执行：

```bash
bash reproduction/scripts/run_csi_tabpfn_notebook_exact.sh
bash reproduction/scripts/run_csi_finpfn_notebook_exact.sh
```

### 17.4 Literal notebook IC

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/evaluate_notebook_checkpoint_ic.py \
  --predictions \
    reproduction/artifacts/csi500_notebook_exact/csi500_tabpfn_seed42_notebook_with_replacement.parquet \
    reproduction/artifacts/csi500_notebook_exact/csi500_finpfn_seed42_notebook_with_replacement.parquet \
  --output-dir reproduction/results/csi500_notebook_exact
```

### 17.5 Common raw-return evaluation

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/evaluate_predictions.py \
  --predictions \
    reproduction/artifacts/predictions/csi500_baselines/csi500_ridge_seed42.parquet \
    reproduction/artifacts/predictions/csi500_baselines/csi500_lightgbm_seed42.parquet \
    reproduction/artifacts/csi500_notebook_exact/csi500_tabpfn_seed42_notebook_with_replacement.parquet \
    reproduction/artifacts/csi500_notebook_exact/csi500_finpfn_seed42_notebook_with_replacement.parquet \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --output-dir reproduction/results/csi500_all_models_notebook_exact \
  --figures-dir reproduction/figures/csi500_all_models_notebook_exact
```

完整命令历史见 `reproduction/notes/commands.md`，服务器手动运行流程见 `reproduction/notes/manual_checkpoint_runbook.md`。

## 18. 输出文件索引

### 18.1 核心结果

- `reproduction/results/csi500_notebook_exact/notebook_exact_ic_summary.csv`
- `reproduction/results/csi500_notebook_exact/notebook_exact_ic_by_period.csv`
- `reproduction/results/csi500_all_models_notebook_exact/model_comparison.csv`
- `reproduction/results/csi500_all_models_notebook_exact/ic_by_period.csv`
- `reproduction/results/csi500_all_models_notebook_exact/ic_by_subperiod.csv`
- `reproduction/results/csi500_all_models_notebook_exact/portfolio_metrics.csv`
- `reproduction/results/csi500_all_models_notebook_exact/decile_returns_by_period.csv`
- `reproduction/results/csi500_all_models_notebook_exact/decile_holdings.parquet`
- `reproduction/results/csi500_all_models_notebook_exact/turnover_by_decile.csv`
- `reproduction/results/csi500_all_models_notebook_exact/prediction_coverage.csv`
- `reproduction/results/csi500_all_models_notebook_exact/regime_metrics.csv`
- `reproduction/results/bundled_prediction_comparison_notebook_exact.csv`

### 18.2 图形

- `reproduction/figures/csi500_all_models_notebook_exact/ic_timeseries_csi500.png`
- `reproduction/figures/csi500_all_models_notebook_exact/cumulative_deciles_csi500_finpfn.png`
- `reproduction/figures/csi500_all_models_notebook_exact/cumulative_deciles_csi500_lightgbm.png`
- `reproduction/figures/csi500_all_models_notebook_exact/cumulative_deciles_csi500_ridge.png`
- `reproduction/figures/csi500_all_models_notebook_exact/cumulative_deciles_csi500_tabpfn.png`
- `reproduction/figures/csi500_all_models_notebook_exact/cumulative_long_short_csi500_finpfn.png`
- `reproduction/figures/csi500_all_models_notebook_exact/cumulative_long_short_csi500_lightgbm.png`
- `reproduction/figures/csi500_all_models_notebook_exact/cumulative_long_short_csi500_ridge.png`
- `reproduction/figures/csi500_all_models_notebook_exact/cumulative_long_short_csi500_tabpfn.png`

所有 prediction、fitted model、result CSV、holdings、logs、figures 和 environment captures 均位于 gitignored 的 `reproduction/` 子目录中，不会进入源码提交。

## 19. 对下一阶段 FinPFN 优化的边界

本报告应作为优化阶段前冻结的 CSI baseline。下一阶段若优化 FinPFN，应遵守：

1. 不覆盖当前 checkpoint predictions、metrics 或 figures；
2. 创建新配置名和新输出目录；
3. 只使用 train/validation 进行模型和超参数选择；
4. test 只在最终配置冻结后评估一次；
5. 同时报告 common raw-return IC、真实 H-L Sharpe、turnover 和含成本结果；
6. 不以接近论文数字为选择标准；
7. 清楚区分“修复发布歧义”“方法优化”和“经济可实现性扩展”。

## 20. 最终结论

就 CSI 500 而言，本项目已经完成进入方法优化前所需的主要复现工作：数据和切分已审计，四个模型均有真实运行结果，baseline 选择没有使用 test，checkpoint 推理有完整环境与 checksum，主要统计和 portfolio 路径已生成，关键实现歧义已量化。

FinPFN 的 notebook-exact mean IC 已达到 `0.043864`，IR 为 `0.797333`；在统一 raw-return common universe 上，IR 为 `0.712002`，高于 Ridge 和 LightGBM。这支持 FinPFN 在横截面排序上的部分优势。与此同时，FinPFN true H-L Sharpe `4.383559` 低于 Ridge `4.888952` 和 LightGBM `4.810360`，且 turnover 更高。因此论文所暗示的全面投资组合优势没有被复现。

本阶段的正确终点不是继续调整 seed 直到命中论文数值，而是冻结并如实报告：**排序优势部分复现，精确 bundle 不可复现，组合全面领先未复现。** 后续 FinPFN 优化应作为独立的新研究阶段开展。
