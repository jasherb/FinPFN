# Phase 3：冻结配置的唯一一次测试期评估

## 结论

验证期冻结配置 **`rank_buffer_exit20pct`** 已在 301 个共同测试日期上执行一次。固定成本仍为每单位单边换手 10 bps，未根据测试结果修改参数。

该 overlay 的 gross Sharpe 为 3.7599、net Sharpe 为 -1.0397、平均总单边换手为 1.6175。未修改 FinPFN 分别为 4.3836、-0.9000、1.7830。

| 比较项 | gross Sharpe | net Sharpe | 总单边换手 | net 平均期收益 | net 期末财富 | net MDD |
|---|---:|---:|---:|---:|---:|---:|
| FinPFN_unmodified | 4.3836 | -0.9000 | 1.7830 | -0.0304% | 0.9089 | -10.68% |
| Ridge_unmodified | 4.8890 | 1.0350 | 1.4788 | 0.0397% | 1.1209 | -6.64% |
| LightGBM_unmodified | 4.8104 | 0.6857 | 1.5516 | 0.0258% | 1.0751 | -7.69% |
| rank_buffer_exit20pct | 3.7599 | -1.0397 | 1.6175 | -0.0350% | 0.8962 | -11.76% |

## 完整性与解释边界

- `selected_config.json` 在本脚本读取测试预测前已经标记为 `frozen_before_test`。
- FinPFN、Ridge、LightGBM 的未修改 gross Sharpe 均以不超过 `1e-10` 的误差复现冻结 evaluator；核对详情写入 `test_evaluation_manifest.json`。
- 所有模型使用同一 120,620 个资产—日期、同一 raw-return target、同一确定性 tie-break 和实际多空收益序列 Sharpe。
- 换手是 long 与 short 两条等权腿的单边换手之和，首日从现金建仓各记 1.0；因此这里的平均值会略高于冻结 baseline 中“不含首日”的摘要。
- 验证胜出方法不使用 uncertainty。验证中的 uncertainty gating 既未胜出，也没有被带到测试期再次调参；这是一项负面增量价值结果。
- 这是预先冻结策略的唯一测试评估；不得因为本结果改变 exit fraction 后重跑。

本地 CPU runtime 为 3.764 秒。
