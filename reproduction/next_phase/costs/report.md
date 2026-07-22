# Phase 1：交易成本与换手敏感性

## 结论

本分析使用冻结的共同测试 universe、每个模型自己的预测所形成的持仓，以及同一个 raw-return target。成本网格在运行前固定为 0、2、5、10、20、30、50 bps，不根据模型胜负选择成本。

| 成本 (bps) | 多空 net Sharpe 排名 | 各模型 net Sharpe |
|---|---|---|
| 0 | Ridge > LightGBM > FinPFN > TabPFN | Ridge 4.8890, LightGBM 4.8104, FinPFN 4.3836, TabPFN -5.1927 |
| 2 | Ridge > LightGBM > FinPFN > TabPFN | Ridge 4.1187, LightGBM 3.9862, FinPFN 3.3268, TabPFN -6.1723 |
| 5 | Ridge > LightGBM > FinPFN > TabPFN | Ridge 2.9628, LightGBM 2.7491, FinPFN 1.7417, TabPFN -7.6428 |
| 10 | Ridge > LightGBM > FinPFN > TabPFN | Ridge 1.0350, LightGBM 0.6857, FinPFN -0.9000, TabPFN -10.0965 |
| 20 | Ridge > LightGBM > FinPFN > TabPFN | Ridge -2.8231, LightGBM -3.4463, FinPFN -6.1818, TabPFN -15.0134 |
| 30 | Ridge > LightGBM > FinPFN > TabPFN | Ridge -6.6811, LightGBM -7.5816, FinPFN -11.4590, TabPFN -19.9417 |
| 50 | Ridge > LightGBM > FinPFN > TabPFN | Ridge -14.3792, LightGBM -15.8475, FinPFN -21.9892, TabPFN -29.8255 |

FinPFN 的多空换手最高，因此随着成本上升，其净表现下降得比 Ridge 和 LightGBM 更快。该结论是对已观察测试结果的敏感性审计，不是用来选择新策略的验证结果。

## 成本与换手定义

- 输入收益原为百分点，本脚本先除以 100 转成小数收益。
- `cost_rate = bps / 10,000`；1 bps 指每 1.0 单边换手的 0.0001 小数收益。
- 相邻期单腿换手为 `0.5 × Σ|w_t-w_(t-1)|`；首日从现金建仓，long/short 各为 1.0。
- long 是 decile 10；short leg 的 gross return 是 `-decile 1`；实际 H-L 是两腿收益之和。
- H-L 总换手是两腿单边换手之和；`net = gross - cost_rate × (turnover_long + turnover_short)`。
- 指标年化因子为 240；Sharpe 直接从 H-L 分期收益序列计算。复合财富为 `Π(1+r_t)`；所有输入收益均大于 -100%，因此该统计有效。
- break-even cost 定义为平均净收益恰好为零的 bps，不包含借券、冲击、融资等未建模成本。

## 关键数值

| 成本 | 模型 | gross 均值/期 | net 均值/期 | net 年化波动 | gross Sharpe | net Sharpe | 平均总换手 | net 终值 | net 最大回撤 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | Ridge | 0.1876% | 0.1876% | 9.2075% | 4.8890 | 4.8890 | 1.4788 | 1.7485 | -5.03% |
| 0 | LightGBM | 0.1809% | 0.1809% | 9.0268% | 4.8104 | 4.8104 | 1.5516 | 1.7143 | -3.98% |
| 0 | FinPFN | 0.1479% | 0.1479% | 8.0989% | 4.3836 | 4.3836 | 1.7830 | 1.5540 | -3.39% |
| 0 | TabPFN | -0.1941% | -0.1941% | 8.9702% | -5.1927 | -5.1927 | 1.8272 | 0.5544 | -46.56% |
| 10 | Ridge | 0.1876% | 0.0397% | 9.2012% | 4.8890 | 1.0350 | 1.4788 | 1.1209 | -6.64% |
| 10 | LightGBM | 0.1809% | 0.0258% | 9.0178% | 4.8104 | 0.6857 | 1.5516 | 1.0751 | -7.69% |
| 10 | FinPFN | 0.1479% | -0.0304% | 8.0995% | 4.3836 | -0.9000 | 1.7830 | 0.9089 | -10.68% |
| 10 | TabPFN | -0.1941% | -0.3768% | 8.9567% | -5.1927 | -10.0965 | 1.8272 | 0.3194 | -68.48% |
| 50 | Ridge | 0.1876% | -0.5518% | 9.2107% | 4.8890 | -14.3792 | 1.4788 | 0.1881 | -81.23% |
| 50 | LightGBM | 0.1809% | -0.5949% | 9.0092% | 4.8104 | -15.8475 | 1.5516 | 0.1651 | -83.49% |
| 50 | FinPFN | 0.1479% | -0.7436% | 8.1156% | 4.3836 | -21.9892 | 1.7830 | 0.1053 | -89.47% |
| 50 | TabPFN | -0.1941% | -1.1077% | 8.9131% | -5.1927 | -29.8255 | 1.8272 | 0.0348 | -96.52% |

### 多空均值归零成本

| 模型 | 平均总单边换手 | break-even bps |
|---|---|---|
| Ridge | 1.478813 | 12.683 |
| LightGBM | 1.551627 | 11.660 |
| FinPFN | 1.782981 | 8.296 |
| TabPFN | 1.827160 | -10.622 |

完整 long、short、H-L 结果见 `cost_sensitivity.csv`，逐期成本、净收益和财富见 `net_performance_by_period.csv`。

## 重复抽样与持仓一致性审计

| 模型 | 预测输入行 | 唯一资产—日期 | 重复行（超出首行） | 按自身预测重建 decile 不一致数 |
|---|---|---|---|---|
| FinPFN | 195550 | 120620 | 74930 | 0 |
| TabPFN | 195550 | 120620 | 74930 | 0 |
| Ridge | 186213 | 186213 | 0 | 0 |
| LightGBM | 186213 | 186213 | 0 | 0 |

- 冻结持仓共有 482,480 行，`(model, seed, date, id)` 重复数为 0。
- 每个模型每个资产—日期只属于一个 decile；四模型在每个日期的 holdings universe 完全相同。
- FinPFN/TabPFN 的 with-replacement 重复预测先按各模型各自的 `(date,id)` 取预测均值，然后才在共同 universe 内形成 decile。
- 用每个模型自己的聚合预测和确定性 ID tie-break 重新生成 decile，四模型不一致数均为 0；target 从未参与 decile 排序，也不存在共享 prediction column。
- 由持仓重算、排除首日的 top/bottom 换手与冻结 `turnover_by_decile.csv` 最大绝对差为 1.110e-16。

因此 FinPFN 的高换手不是重复持仓、重复 asset row 或不一致 universe 造成的；它来自模型自身头尾排名随时间更频繁变化。重复抽样会影响聚合预测本身，但在组合形成前已被折叠，不会机械地重复计算持仓或收益。

## 文件与复现

运行命令：

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/next_phase/costs/transaction_cost_analysis.py
```

输入相对路径、文件大小和 SHA-256 保存在 `input_manifest.json`；机器可读完整性检查保存在 `integrity_checks.json`。本次本地 CPU runtime 为 4.306 秒。

## 限制

这是线性成本敏感性，不含 bid-ask 非线性、市场冲击、借券可得性、涨跌停、融资成本和容量约束。首日建仓成本只影响 301 期中的一期；冻结换手表为与原 evaluator 一致而不含首日，本报告的经济成本则保守地包含首日。
