# Phase 2：不确定性可用性与校准审计（等待验证期 GPU 产物）

## 当前结论

冻结的 FinPFN 与 TabPFN notebook-exact prediction artifacts **不足以进行 ensemble uncertainty 校准**。两者各有 195,550 行，只有：

- 聚合分布中位数 `prediction`；
- 聚合分布均值 `prediction_mean`；
- task-local target 与运行标识。

它们没有 8 个 estimator predictions、预测标准差、分位数、区间或 logits。`abs(prediction_mean - prediction)` 可以称为聚合分布的均值—中位数分歧，但不能代表成员离散度，也不能称为 calibrated posterior uncertainty。

| 模型 | 行数 | 聚合中位数 | 聚合均值 | 成员预测 | predictive SD | 分位数 | logits/分布 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinPFN | 195,550 | 有 | 有 | 无 | 无 | 无 | 无 |
| TabPFN | 195,550 | 有 | 有 | 无 | 无 | 无 | 无 |

机器可读审计见 `availability_audit.csv` 和 `availability_audit.json`。

## TabPFN 2.0.8 API 审计

官方 PyPI 的 `tabpfn-2.0.8.tar.gz` 只有 139.9 kB，SHA-256 为 `ffb739fc898c7144cd974a50a4185050cd5cae09405be2c81a8b6a3c7421fcbc`。源码确认：

1. `TabPFNRegressor.predict(output_type="full")` 返回聚合后的 mean、median、mode、quantiles、criterion 和 logits；
2. 8 个成员经各自预处理/target transform 前向传播，映射到共同 bar borders 后才合并；
3. 公共 full output 的 logits 已是成员聚合后的分布，不保留成员维度；
4. fitted executor 的 `iter_outputs` 可以在不改变模型的情况下取得同一批成员 forward outputs。

因此新增 `predict_with_members.py` 严格复写 2.0.8 的 border translation、temperature 和 aggregate-probability 逻辑，同时额外保存：

- 八个 member mean；
- 八个 member median；
- 聚合 predictive SD；
- 聚合 q10/q25/q50/q75/q90；
- 原有 aggregate median/mean/mode。

高维 raw logits 不落盘，因为它们体积很大，且所需 predictive interval、variance 与成员点预测已经在同一 inference pass 中无损计算。此选择没有改 checkpoint、权重、抽样、预处理、成员数或最终聚合预测。

## 本地 CPU smoke

在 validation split 的首个 date pair、首个 with-replacement 50-stock group 上执行：

| 模型 | 行 | group | runtime | 失败 | 非有限成员/分布输出 | 与公共 `predict(full)` 最大差 |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 50 | 1/1 | 4.260 s | 0 | 0 | 0.0 |
| FinPFN | 50 | 1/1 | 4.712 s | 0 | 0 | 0.0 |

比较覆盖 aggregate mean、median、mode 和 q10/q25/q50/q75/q90。验证脚本还确认 q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90，且 q50 与 baseline `prediction` 完全一致。

`smoke_evaluation/` 证明 validation-only collapse、raw-return join、误差统计、coverage/portfolio 和图表代码能够运行。该目录只有一个 query date，其数值**不是研究结果**，不得用于信号或阈值选择。

## 预定校准方法

完整验证期成员 artifact 返回后，`evaluate_uncertainty.py` 会在 2021 validation 数据上计算：

- total member SD、平均组内 ensemble SD、MAD、IQR；
- 同一 asset-date 多次抽到不同 50-stock group 的 group-composition SD；
- 8 个成员横截面 percentile-rank disagreement；
- mean–median disagreement；
- aggregate predictive interval width 与 predictive SD。

误差统一基于 raw-return target：主要使用同日 percentile absolute rank error；数值误差使用 prediction 与 raw target 各自的全截面 z-score 后的 absolute error，避免把 task-local z prediction 直接与 decimal return 相减。还会测下一相邻日期 rank instability、实现 forward IC deterioration、tail-selection precision、uncertainty quintile error 和 coverage–performance curve。

2021 validation 内没有官方硬编码的 2022 CSI shock windows，且没有 CIMV 原始序列，因此本阶段不会为验证期虚构 shock 标签。若以后在冻结测试上做 shock uncertainty 对照，只能标记为 exploratory。

## 当前停止点

Phase 2 现在需要完整 validation inference；这是单 GPU、服务器侧工作，所以 Codex 按约定停止，不执行 GPU、SSH 或服务器命令。人工命令见 `manual_gpu_commands.md`。

预计 TabPFN 与 FinPFN 各 20–35 分钟，总计约 45–70 分钟；单张 A100 80GB、4 CPU workers，无训练。目标目录已与冻结预测分离，而且已有目标文件时 runner 会直接拒绝覆盖。

完整 artifact 返回之前：

- `calibration_metrics.csv` 与 `coverage_performance.csv` 尚未生成；
- 不能判断 uncertainty 是否预测 error；
- 不能进入 Phase 3；
- 不能根据单组 smoke 选择任何 uncertainty score 或 gating threshold；
- 不应运行测试期成员 inference。
