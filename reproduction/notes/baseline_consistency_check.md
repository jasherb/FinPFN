# Ridge versus LightGBM portfolio consistency check

## Conclusion

The nearly identical true long-short Sharpe ratios are not caused by prediction-file
reuse, shared decile assignments, target-based ranking, or a reused long-short return
series. No evaluation change is warranted.

The two long-short series have different levels and paths but happen to have nearly
the same mean-to-standard-deviation ratio:

| Model | Mean period return | Period SD (`ddof=1`) | Annualized Sharpe | Cumulative return |
|---|---:|---:|---:|---:|
| Ridge | 0.219770 pp | 0.551259 pp | 6.176169 | 66.370689 pp |
| LightGBM | 0.206730 pp | 0.518553 pp | 6.176134 | 62.432569 pp |

## Prediction and holdings comparison

- All-row prediction Pearson correlation: 0.548496.
- All-row prediction Spearman correlation: 0.630361.
- Per-date prediction Spearman: mean 0.632606, median 0.637054, sample SD
  0.050854, minimum 0.498349, maximum 0.752348 across 302 dates.
- Top-decile mean membership overlap: 44.9376% of the smaller decile, or mean
  Jaccard 0.292209; mean intersection 27.533 stocks.
- Bottom-decile mean membership overlap: 50.3071% of the smaller decile, or mean
  Jaccard 0.338493; mean intersection 31.305 stocks.

## Long-short return comparison

- Pearson correlation: 0.601474.
- Spearman correlation: 0.563139.
- Mean absolute difference: 0.376026 percentage points.
- Maximum absolute difference: 1.688325 percentage points, on 2023-02-17; Ridge was
  0.300049 and LightGBM was -1.388276 percentage points.
- Recomputed versus previously saved evaluator series maximum difference:
  `2.22e-16` for each model (floating-point roundoff only).

## Exact prediction inputs and hashes

| Model | Evaluator input | SHA-256 |
|---|---|---|
| Ridge | `reproduction/artifacts/predictions/csi500_baselines/csi500_ridge_seed42.parquet` | `a6cccd1f54f3ced4cd5165615a6c7d921d3d46d157f5a7e166a532532b6488b1` |
| LightGBM | `reproduction/artifacts/predictions/csi500_baselines/csi500_lightgbm_seed42.parquet` | `0a0c7f0bcbb5e97d25dcaf73448e55f9ec97b70aa8dc5bb91d0bf0f70eae375a` |

The files are different and both contain 186,213 unique successful asset-date
predictions over 302 dates. The evaluator SHA-256 at verification was
`505a3c658f28a45a591337c85fd2e4e3b6d055aada0f6d27afa6ef988da5e3dd`.

## Decile independence confirmation

The unchanged evaluator groups the long-form prediction table by `model`, `seed`,
and `date`, calls `assign_deciles` independently for each group, sorts by that
group's own `prediction` with `id` only as a deterministic tie-breaker, and then
assigns deciles from the sorted row position. Raw targets enter only after ranking,
for calculating portfolio returns. They are not used for decile assignment.

Machine-readable outputs are under the ignored
`reproduction/results/baseline_consistency/` directory:

- `summary.json`;
- `prediction_spearman_by_date.csv`;
- `holdings_overlap_by_date.csv`;
- `long_short_comparison.csv`.
