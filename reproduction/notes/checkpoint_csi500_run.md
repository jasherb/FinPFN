# Full CSI 500 released-checkpoint inference

## Status and decision rule

The researcher manually ran the prepared single-GPU wrappers on the approved
compute host. Codex did not access the host. These are inference-only runs of the
released vanilla TabPFN and FinPFN checkpoints: no checkpoint was trained, no
configuration was selected on the test period, and no post-test tuning was done.

The primary configuration was fixed before execution: stock-sampling seed 42 and
estimator random state 42,
`artifact_unique500`, 500 common assets sampled without replacement for each
adjacent date pair, ten 50-stock tasks per date, eight estimators, median prediction,
and TabPFN 2.0.8 preprocessing defaults.

## Environment and integrity

- Repository commit: `1155bf3dd948acd6fd9eb1661f223bac8ef7577c`.
- Python 3.10.20; PyTorch 2.5.1+cu121; TabPFN 2.0.8; CUDA runtime 12.1.
- GPU visible to each process: NVIDIA A100 80GB PCIe; execution was single-GPU.
- FinPFN checkpoint SHA-256:
  `c035f2a79c74ab7f38b023fa98624d078b6389c3d096ac1a1270b04361dd0214`.
- TabPFN checkpoint SHA-256:
  `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`.
- The returned parquet/metadata pairs all match the SHA-256 checksums calculated on
  the compute host.

| Model | Rows | Dates | Rows/date | Groups succeeded | Failed groups | Non-finite primary predictions | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN | 150,500 | 301 | 500 | 3,010 / 3,010 | 0 | 0 | 1,669.336 |
| FinPFN | 150,500 | 301 | 500 | 3,010 / 3,010 | 0 | 0 | 1,697.121 |

The scikit-learn inverse-transform overflow warnings seen during TabPFN inference
did not produce non-finite values in `prediction`, `prediction_mean`, or
`target_group_z`.

## IC and IR: two target definitions must remain separate

The released notebook evaluates FinPFN and TabPFN against `target`, which was
preprocessed independently inside each 50-stock task. It evaluates the conventional
baselines against the whole-date standardized `return`. The task-local transform is
not rank-preserving after the ten groups are concatenated. Consequently, the
paper-faithful checkpoint IC and a common raw-return IC are different statistics.

Paper-faithful task-target results:

| Model | New mean IC | New IC SD | New IR | Bundled mean IC | Bundled IC SD | Bundled IR | Paper IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinPFN | 0.031801 | 0.049099 | 0.647677 | 0.042006 | 0.049099 | 0.855546 | 0.85 |
| TabPFN | -0.030197 | 0.065235 | -0.462894 | -0.028794 | 0.065050 | -0.442639 | -0.44 |

The bundled CSV exactly recovers the notebook/paper IC and IR, but the new
released-checkpoint run does not reproduce the bundled FinPFN predictions closely
enough to recover their result. TabPFN is close to the reported value. This is a
reported discrepancy, not a trigger for test-driven tuning.

Subsequent inspection identified one additional explicit difference: the released
notebook omits `TabPFNRegressor.random_state`, whose TabPFN 2.0.8 default is 0,
whereas this completed primary used 42. This primary remains frozen as the
artifact-shape run. A separate notebook-exact diagnostic with estimator random state
0 and with-replacement sampling was prepared and passed a one-group CPU smoke; it
must not overwrite or replace this result according to test performance.

With the smoke group's context and query rows held fixed, changing only estimator
state 0 to 42 yielded prediction Spearman correlations of 0.6794 for FinPFN and
0.9966 for TabPFN. This does not quantify full-period performance, but it shows that
the explicit seed deviation can materially affect FinPFN's cross-sectional ranks.

## Common raw-return evaluation

All four reproduced models were evaluated on the intersection of 301 dates and 500
asset-date rows per date, using each model's own predictions. IC uses raw parquet
targets; portfolios are equal-weight deciles of each model's predictions with
deterministic ID tie-breaking, raw gross returns, arithmetic cumulative sums, 240
period annualization, and no transaction costs.

| Model | Mean IC | IC SD | IR | Top-decile Sharpe | True H-L Sharpe | Top cumulative (pp) | H-L cumulative (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinPFN | 0.033254 | 0.057335 | 0.579996 | 3.760131 | 5.124263 | 21.384364 | 49.277119 |
| TabPFN | -0.029691 | 0.065904 | -0.450515 | -3.807750 | -3.549967 | -29.520194 | -37.854927 |
| Ridge | 0.042790 | 0.068989 | 0.620244 | 2.486935 | 6.272572 | 18.440509 | 69.577508 |
| LightGBM | 0.039938 | 0.061976 | 0.644403 | 2.784298 | 6.316055 | 18.941253 | 66.953164 |

| Model | Bottom-decile turnover | Top-decile turnover |
|---|---:|---:|
| FinPFN | 0.866733 | 0.894600 |
| TabPFN | 0.894000 | 0.895400 |
| Ridge | 0.687133 | 0.654933 |
| LightGBM | 0.726600 | 0.702267 |

The common-universe Ridge and LightGBM numbers differ slightly from their standalone
302-date/full-universe test results because this table intentionally restricts both
baselines to the checkpoint models' 301-date, 500-asset intersection.

## Comparison with the released prediction bundle

Each new checkpoint file contains 150,500 unique asset-date predictions. The new
and bundled files overlap on 122,241 asset-dates (81.2233%). On that overlap:

| Model | Spearman | Pearson | Mean absolute prediction difference |
|---|---:|---:|---:|
| FinPFN | 0.362211 | 0.418921 | 0.053539 |
| TabPFN | 0.311887 | 0.388540 | 0.253990 |

The released notebook does not preserve the exact random sampling/grouping state
used to create the bundled CSV. Because task composition changes both context data
and the task-local label transform, the released checkpoints and nominal seed are
not sufficient to reconstruct the bundled predictions exactly. This is the leading
implementation ambiguity; it has not been resolved by changing the method.

## Notebook-exact follow-up

After the primary discrepancy, a second predeclared run followed the visible
notebook choices: stock-sampling seed 42, sampling with replacement, 50-stock groups
sorted by ID after sampling, estimator random state 0, eight estimators, and four
resource-capped CPU workers. It wrote to a separate ignored directory and did not
overwrite the primary result.

| Model | Rows | Unique asset-dates | Repeated rows | Groups succeeded | Failed groups | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| TabPFN | 195,550 | 120,620 | 74,930 | 3,911 / 3,911 | 0 | 2,054.745 |
| FinPFN | 195,550 | 120,620 | 74,930 | 3,911 / 3,911 | 0 | 2,078.874 |

All primary/mean predictions and task-preprocessed targets are finite. The returned
files passed local schema and metadata validation. Local SHA-256 values are recorded
with the ignored artifacts:

| Returned file | Local SHA-256 |
|---|---|
| `csi500_finpfn_seed42_notebook_with_replacement.parquet` | `03e62d18bf14cb6a3787213a87369adf12914d65748f8d1536a7bc5cecca76f3` |
| `csi500_finpfn_seed42_notebook_with_replacement.metadata.json` | `ff3cff21aa9a3902af7bfc2ca21d1bfd42168325a72919b719e0ebef3431a9d0` |
| `csi500_tabpfn_seed42_notebook_with_replacement.parquet` | `0fa76d578741b3a50a9f6e1b96009bae6fe4f884b9ce7a3fe0f52b6cec95c26a` |
| `csi500_tabpfn_seed42_notebook_with_replacement.metadata.json` | `cafcd2ec91e981a7f9015a866baa6184bb0d067a094c6f5abae91641be9e4530` |

The literal notebook IC retains repeated asset rows, matching the notebook's
`groupby(date)` calculation:

| Model | Mean IC | IC SD | IR | Bundled IR | Paper IR |
|---|---:|---:|---:|---:|---:|
| FinPFN | 0.043864 | 0.055013 | 0.797333 | 0.855546 | 0.85 |
| TabPFN | -0.034296 | 0.068915 | -0.497656 | -0.442639 | -0.44 |

This materially closes the FinPFN gap relative to the artifact-shape primary IR of
0.647677, confirming that visible notebook sampling, sorting, and estimator state
matter. It does not exactly recover the bundle because the bundle has 500 unique
assets per date, which is incompatible with the visible with-replacement notebook
run.

For an apples-to-apples evaluation, repeated predictions were averaged per
asset-date and all four models were restricted to the resulting 120,620 common
asset-dates: 301 dates and an average 400.731 assets per date. IC and portfolios use
the raw parquet return target.

| Model | Mean IC | IC SD | IR | Top-decile Sharpe | True H-L Sharpe | Top cumulative (pp) | H-L cumulative (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FinPFN | 0.045597 | 0.064040 | 0.712002 | 2.366336 | 4.383559 | 13.976832 | 44.525270 |
| TabPFN | -0.037758 | 0.072213 | -0.522875 | -4.496578 | -5.192668 | -36.666937 | -58.417991 |
| Ridge | 0.037409 | 0.069378 | 0.539210 | 2.144884 | 4.888952 | 16.689462 | 56.456510 |
| LightGBM | 0.036434 | 0.064305 | 0.566578 | 2.574612 | 4.810360 | 19.756541 | 54.458622 |

| Model | Bottom-decile turnover | Top-decile turnover |
|---|---:|---:|
| FinPFN | 0.864380 | 0.917878 |
| TabPFN | 0.916045 | 0.910539 |
| Ridge | 0.748795 | 0.728281 |
| LightGBM | 0.785946 | 0.764186 |

Under this common raw-return IC definition, FinPFN outperforms both reconstructed
baselines in IR. It does not outperform them in the true gross long-short Sharpe,
top-decile cumulative return, or long-short cumulative return. The appropriate
conclusion is therefore partial recovery of the paper's prediction-ranking result,
not reproduction of comprehensive portfolio dominance.

Direct comparison with the released bundle improved but remained far from exact.
After collapsing repeated asset-date predictions, FinPFN had 0.463426 Spearman and
0.533319 Pearson correlation with the bundle on 97,664 overlapping asset-dates;
TabPFN had 0.359216 and 0.431168. The remaining mismatch is consistent with the
unpublished bundle-generation sampling/grouping state.

## Generated artifacts

Ignored outputs are under:

- `reproduction/results/csi500_all_models_primary/` for comparison tables,
  per-period IC, subperiod and regime metrics, decile returns, holdings, turnover,
  coverage, and evaluation scope;
- `reproduction/figures/csi500_all_models_primary/` for IC, cumulative decile, and
  cumulative long-short figures;
- `reproduction/results/bundled_prediction_comparison.csv` for direct comparison
  with the released prediction bundle;
- `reproduction/results/csi500_notebook_exact/` for literal notebook IC tables;
- `reproduction/results/csi500_all_models_notebook_exact/` and
  `reproduction/figures/csi500_all_models_notebook_exact/` for the common raw-return
  follow-up;
- `reproduction/results/bundled_prediction_comparison_notebook_exact.csv` for the
  notebook-exact/bundle prediction comparison.

The CSI seed-42 checkpoint stage is complete. The notebook-exact follow-up partially
recovers the FinPFN ranking result but does not exactly recover the published bundle
or its portfolio-dominance claim. No claim of successful full reproduction is made.
