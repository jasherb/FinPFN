# CSI 500 reconstructed LightGBM run

## Protocol

- Status: complete.
- Model role: independently reconstructed strong baseline, not the missing author
  estimator.
- Seed: 42.
- CPU threads: 4.
- Selection data: validation split only (2021 observations).
- Selection metric: mean date-wise Spearman IC.
- Final fit: selected configuration refit once on train plus validation.
- Test use: one prediction pass and one planned evaluation after selection; no test
  result was used for tuning.
- Rows: 703,709 train; 149,450 validation; 853,159 final fit; 186,213 test.
- Package: LightGBM 4.6.0.

## Fixed candidate results

| Candidate | learning_rate | n_estimators | num_leaves | max_depth | min_child_samples | reg_alpha | reg_lambda | Validation mean IC | Fit + validation seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.05 | 500 | 31 | 4 | 50 | 0.0 | 0.0 | 0.042854663 | 6.009 |
| **1** | **0.05** | **500** | **63** | **6** | **50** | **0.0** | **0.0** | **0.043467032** | **8.050** |
| 2 | 0.03 | 800 | 31 | 6 | 100 | 0.0 | 0.1 | 0.042712057 | 10.815 |
| 3 | 0.03 | 800 | 63 | 6 | 100 | 0.1 | 0.1 | 0.042843866 | 12.495 |
| 4 | 0.02 | 1,200 | 31 | -1 | 100 | 0.1 | 1.0 | 0.043287852 | 16.068 |
| 5 | 0.02 | 1,200 | 63 | -1 | 100 | 1.0 | 1.0 | 0.043393256 | 21.500 |

Candidate 1 was selected solely because it had the highest validation mean IC.

## Runtime

| Stage | Seconds |
|---|---:|
| Complete six-candidate selection | 74.937 |
| Final train+validation refit | 9.844 |
| Single test prediction pass | 0.913 |
| Total modeling runtime | 85.694 |

## Final test metrics

| Metric | Value |
|---|---:|
| Test dates | 302 |
| Prediction rows / coverage | 186,213 / 100% |
| Mean IC | 0.038657354 |
| IC standard deviation (`ddof=1`) | 0.058421257 |
| IR | 0.661700154 |
| Raw bottom-decile Sharpe | -5.843270 |
| Raw top-decile Sharpe | 3.135077 |
| Raw notebook Sharpe spread | 8.978347 |
| Primary true long-short Sharpe | 6.176134 |
| Long-short cumulative return | 62.432569 percentage points |
| Top-decile mean one-way turnover | 0.631120 |
| Bottom-decile mean one-way turnover | 0.664703 |

The artifact-style cross-sectional-excess top and bottom Sharpes are 3.807049 and
-5.781432, giving a notebook-style spread of 9.588481. The true long-short series is
unchanged by subtracting a common date-wise market return.

Across the notebook's 11 official CSI regime dates, mean IC was 0.013908, IC SD was
0.044150, and IR was 0.315029. Outside those dates, mean IC was 0.039593 and IR was
0.673959. These values are final diagnostics, not tuning inputs.

Machine-readable parameters and per-candidate timings are stored beside the ignored
prediction parquet in its `.metadata.json`; normalized test tables are under the
ignored `reproduction/results/csi500_lightgbm/` directory.
