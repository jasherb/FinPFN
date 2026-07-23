# U.S. external validation：待批准的人工命令

状态：**仅准备，未执行。** 本阶段不得在没有明确批准时启动。

## 预声明设计

- 数据：`90features_USstocks.parquet`，SHA-256 `54818c78796ecae3974b2058575cd2284482ce35e62c9116d316e23240b8ef50`。
- split：train 至 1999-12；validation 2000-01 至 2009-12；test 2010-01 至 2021-12。
- checkpoint inference：每个相邻月度 date pair 从共同资产中 seed-42 无放回抽取最多 500 个，分成 10 个 50-stock tasks；8 estimators；TabPFN estimator state 0；released checkpoints；不训练。
- Ridge/LightGBM：沿用 CSI 阶段已声明的 5/6 个候选，只用 validation mean daily Spearman IC 选择，之后 train+validation refit，test 只预测一次。
- headline evaluation：四模型在共同日期和共同资产—日期上，使用同一 raw target；actual H-L Sharpe 为主，notebook-style spread 仅作辅助。
- 这是外部验证，不会用 CSI test 或 U.S. test 改候选、sampling、seed、阈值或成本。

`artifact_unique500` 与 released CSI prediction artifact 的 500 unique assets/date 形状一致；它不同于 notebook 可见的全体资产 with-replacement 代码。后者在 U.S. 每月约 5,000 个资产时需要约十倍 groups，不作为本次预声明 primary，也不会在看到 primary 结果后自动追加。

## 运行前只读检查

从仓库根目录执行：

```bash
git status --short
sha256sum 90features_USstocks.parquet \
  models/finpfn_90feats_us.ckpt \
  models/tabpfn-v2-regressor.ckpt
```

预期 checkpoint SHA-256：

```text
493e2bd458618f2ddac97da754c3f23abc61a93baa95ae127636a918d3ba7a8f  models/finpfn_90feats_us.ckpt
2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736  models/tabpfn-v2-regressor.ckpt
```

## CPU baselines

本地或研究者人工登录的服务器均可运行：

```bash
bash reproduction/next_phase/us_external_validation/scripts/run_us_baselines.sh \
  2>&1 | tee reproduction/next_phase/us_external_validation/us_baselines.log
```

## 单 GPU smoke

只能由研究者在允许的服务器上人工运行。先确认 GPU 2 空闲，然后运行一个 date pair、一个 50-stock group、两个模型：

```bash
CUDA_VISIBLE_DEVICES=2 \
bash reproduction/next_phase/us_external_validation/scripts/run_us_checkpoint_smoke.sh \
  2>&1 | tee reproduction/next_phase/us_external_validation/us_checkpoint_smoke.log
```

预期每模型 50 行、1/1 successful group、0 failed group，且日志中的 `cuda_available` 应为 true。smoke 只验证兼容性，不是研究结果。

## 单 GPU released-checkpoint full inference

smoke 通过后，若 GPU 2 仍已确认空闲：

```bash
CUDA_VISIBLE_DEVICES=2 \
bash reproduction/next_phase/us_external_validation/scripts/run_us_checkpoints.sh \
  2>&1 | tee reproduction/next_phase/us_external_validation/us_checkpoints.log
```

## 共同 universe 评估

四份 prediction artifacts 都生成并验证后：

```bash
bash reproduction/next_phase/us_external_validation/scripts/evaluate_us_common.sh \
  2>&1 | tee reproduction/next_phase/us_external_validation/us_evaluation.log
```

此后再为 U.S. 结果运行与 CSI 完全相同的 cost、tail precision、rank stability 和 IC–portfolio-gap 审计；不在 test 上调参。

## 计算与存储估算

- 原始 parquet：1.5 GB，3,529,899 rows；test 144 months、788,592 asset-dates。
- checkpoint primary：143 query months × 500 assets = 71,500 prediction rows/model，约 1,430 groups/model。
- GPU smoke：每模型 1 group，通常几秒至约 1 分钟。
- 单张 A100 80 GB：预计每模型 15–25 分钟，两模型顺序执行约 30–50 分钟。
- Ridge：4 CPU threads，预计 <5 分钟。
- 六候选 LightGBM + final refit：4 CPU threads，预计 10–40 分钟；建议 8–16 GB RAM。
- evaluation：4 CPU threads，预计 <5 分钟。
- 数据已存在时，新增 artifacts/logs/results 预留 2 GB 足够；若服务器尚无 parquet，需另有约 1.5 GB。

估算来自已完成 CSI runtime 与 U.S. row/group 数量缩放，不是资源保证。wrapper 会拒绝覆盖主要 prediction/evaluation 输出。
