# Repository and paper audit

## Upstream entry points and artifacts

- `scripts/main.py`: low-level TabPFN fine-tuning loop.
- `scripts/training_utils/data_utils.py`: parquet split, adjacent-date task creation,
  sampled-stock batching, and cross-sectional target standardization.
- `finpfn.ipynb`: the only released prediction/evaluation workflow.
- Bundled checkpoints: CSI 500 FinPFN, U.S. FinPFN, TabPFN v2 regressor, and an
  unused classifier checkpoint.
- Bundled result: one CSI 500 prediction CSV with 150,500 rows, 301 dates, exactly
  500 rows per date, and 619 unique stock identifiers across the period.
- The official parquet datasets are now present and audited. Still missing are raw
  source-construction code, fitted Ridge/LightGBM models, baseline fitting code,
  U.S. predictions, CIMV/VIX series, and a complete environment lock.
- All bundled binary artifacts have SHA-256 values recorded in
  `configs/checksums.sha256`.

## Official task construction

The paper defines the row at date `t` as features `X_t` paired with a forward return
target `y_t` over `[t, t+1]`. Each meta-task uses the intersection of stocks present
at adjacent observed dates:

- context: `(X_{t-1}, y_{t-1})`;
- query features: `X_t`;
- query label used only for training/validation: `y_t`.

The released loader sorts dates, constructs consecutive pairs, keeps identifiers
present on both dates, and uses 50 context plus 50 query stocks per task. During
training it samples stocks with replacement. It standardizes targets separately by
date within each sampled 50-stock group.

Paper splits:

| Market | Meta-training | Validation | Out of sample |
|---|---|---|---|
| CSI 500 | 2016-01-05 to 2021-01-01 | 2021-01-01 to 2022-01-01 | 2022-01-02 to 2023-04-03 |
| U.S. | 1962-06-29 to 1999-12-31 | 2000-01-31 to 2009-12-31 | 2010-01-31 to 2021-12-31 |

The code implements half-open cuts (`date < split1`, `split1 <= date < split2`,
`date >= split2`). Holiday/month-end availability appears to reconcile most printed
boundary differences, but the downloaded parquet dates must be checked explicitly.

## Preprocessing

The paper says missing feature values are imputed by the cross-sectional median at
each date; numerical features are winsorized at the 1st/99th percentiles and
standardized cross-sectionally. Targets are also clipped and standardized
cross-sectionally. CSI targets are multiplied by 100 when loaded; the notebook says
the U.S. targets are already percentage returns.

None of the feature imputation, winsorization, or standardization is implemented in
the repository. The downloaded parquets are final preprocessed artifacts. CSI
features are exactly cross-sectionally centered and sample-standardized by date; its
target is a clipped-looking raw decimal return. U.S. continuous features are mostly
rank transforms in `[-1, 1]`, eight fields are categorical, and its target is already
in percentage units. Upstream imputation/winsorization provenance remains unverified.

## Baseline settings

The paper says Ridge and LightGBM are tuned with 10-fold cross-validation on the
training set and then refit once on all training observations. Ridge searches alpha
over `{0.001, 0.01, 0.1, 1, 10}`. LightGBM fixes 500 trees and searches max depth
`{2, 4, 6}`, number of leaves `{31, 63, 127}`, column fraction `{0.7, 1}`, and
minimum child samples `{20, 50, 100}`. The splitter, shuffle policy, scoring metric,
seed, and chosen settings are not specified. Although the caption says chosen values
are bold, the supplied PDF contains no bold candidate values. The repository has no
baseline fitting or CV implementation, so these choices cannot be reconstructed
exactly. The reproduction therefore labels its baselines independently reconstructed:
the small grids in `configs/baseline_search.json` are selected by mean date-wise
Spearman IC on the official validation period only, then refit on train plus
validation and evaluated once on test.

## Metrics and portfolios

- IC: per-date Spearman rank correlation.
- IC standard deviation: pandas sample standard deviation (`ddof=1`).
- IR: mean IC divided by that sample standard deviation; it is not annualized.
- Portfolio: predictions are split into ten date-wise quantiles, equally weighted,
  and rebalanced every period.
- Return: the notebook subtracts the full cross-sectional mean return each date.
- Cumulative return: arithmetic cumulative sum, not compounding.
- CSI annualization: square root of 240.
- No transaction costs.

The notebook's reported H-L Sharpe is `Sharpe(decile 10) - Sharpe(decile 1)`, not
the Sharpe ratio of the period-by-period H-L return series. The evaluator retains
that notebook-style field only for comparison; the Sharpe of the actual period-wise
top-minus-bottom series is the primary portfolio metric. It reports both raw and
notebook-style cross-sectional-excess decile returns; their long-short series is
identical.

## Leakage and alignment review

No direct query-label input to the PFN was found: query targets are used for the
training/validation loss but are not passed to inference. The preceding-period
target is a forward return ending at the current feature date, so it is available at
prediction time only if date labels have exactly the timing described by the paper.
That alignment cannot be proven from the repository because raw return construction
code and source timestamps are missing.

Potential implementation concerns:

1. Inference samples stocks with replacement and then deduplicates results. One pass
   cannot normally yield the precomputed file's 500 unique rows per date, so the
   released cell source and stored artifact do not describe the same execution.
2. FinPFN/TabPFN IC uses targets standardized within random 50-stock groups, then
   correlates across the combined date. Different group transformations can alter
   the full-cross-section ranking.
3. A no-preprocessing TabPFN inference configuration is constructed but commented
   out at estimator creation, so estimator preprocessing may differ from fine-tuning.
4. The U.S. data have eight categorical features, but the notebook does not pass
   categorical feature indices to fine-tuning.
5. The paper describes 4 attention heads; bundled checkpoints report 6. The paper
   specifies learning rate `1e-5`; the notebook uses `3e-5`. The paper describes
   Adam with warmup/cosine; code uses Schedule-Free AdamW.
6. The paper says IR uncertainty uses 75% of dates without replacement. The notebook
   uses 200 draws with replacement (out of 301 CSI dates).
7. The current notebook calls a previous-return helper before the return column is
   merged and has baseline prediction code commented out, indicating stale,
   non-linear notebook state.
8. README/kernel naming says Python 3.10, while notebook language metadata records
   Python 3.12.9. The released dependency file is not sufficient to resolve which
   package versions produced the stored outputs.
9. The bundled CSV contains 500 unique identifiers per date, contradicting the live
   cell's with-replacement sampling. Primary inference uses seeded 500-ID sampling
   without replacement because it matches the observable artifact shape; literal
   with-replacement and all-common-ID modes remain labelled sensitivities.

## Verified bundled CSI 500 metrics

The precomputed CSV exactly recovers the paper's headline IC statistics:

| Model | Mean IC | IC std. | IR | Reported IR |
|---|---:|---:|---:|---:|
| FinPFN | 0.042006 | 0.049099 | 0.855546 | 0.85 |
| TabPFN | -0.028794 | 0.065050 | -0.442639 | -0.44 |
| Ridge | 0.040702 | 0.068034 | 0.598258 | 0.60 |
| LightGBM | 0.044120 | 0.063827 | 0.691239 | 0.70 |

The CSV's `return` column is standardized and unsuitable for portfolio P&L. Merging
raw parquet returns reproduces the stored LightGBM cumulative endpoints (D1 about
-36.84, D10 about 23.51, H-L about 60.36 percentage points), but stored nonlinear
Sharpe outputs still differ from clean recomputation. The published LightGBM H-L
cumulative return is printed as 0.3%, while the stored endpoints imply about 60.3%;
this appears to be a paper typographical error.

## Lightweight checks completed

- All released Python files parse successfully.
- All bundled artifact checksums verify.
- A synthetic three-date, 60-stock panel passes the released adjacent-date loader:
  context/query tensors contain 50 stocks and targets are centered per sampled date.
- The local evaluator generates 301-date IC and portfolio series, holdings, turnover,
  comparison tables, and readable plots from the bundled CSI 500 CSV.
- `tabpfn==2.0.8` loads both released regressor checkpoints locally. A one-date,
  one-group, eight-estimator CPU smoke produced 50 predictions with zero failures
  for each model (about 3.3 seconds FinPFN and 1.9 seconds TabPFN on this machine).
  Only 41 IDs overlapped the bundled first date; prediction Spearman correlations
  were about 0.48 and 0.14 respectively, showing that unreleased group/sampling
  state materially affects exact prediction comparison.
- The reconstructed Ridge loader, temporal validation selection, final refit,
  prediction schema, and common evaluator pass a 10,000-row-per-split smoke. Smoke
  metrics are diagnostic only and are not reproduction results.
- The full reconstructed CSI Ridge baseline selected alpha 0.001 on validation and
  produced 186,213 test predictions with full coverage. Clean test statistics are
  mean IC 0.040454, IC standard deviation 0.065650, and IR 0.616215. Its actual
  long-short Sharpe is 6.176169; the artifact-excess top/bottom Sharpe difference is
  9.911426. This is a newly reconstructed baseline, not the absent author model.
- The approved full six-candidate CSI LightGBM search selected candidate 1 using
  validation mean IC only: learning rate 0.05, 500 trees, 63 leaves, depth 6,
  minimum child samples 50, and zero L1/L2. After one train+validation refit and one
  test prediction/evaluation pass, it produced 186,213 predictions with full
  coverage: mean IC 0.038657, IC SD 0.058421, IR 0.661700, and true long-short
  Sharpe 6.176134. No test-driven retuning occurred. Complete timings and all six
  validation results are in `notes/lightgbm_csi500_run.md`.
- The nearly equal Ridge and LightGBM true long-short Sharpes were independently
  checked and are not a file/column reuse error. Prediction Pearson correlation is
  0.5485; mean date-wise rank correlation is 0.6326; top/bottom decile membership
  overlaps are about 44.94%/50.31%; and the long-short series correlation is 0.6015
  with a 1.6883 percentage-point maximum difference. Recalculation matches both
  saved evaluator series to `2.22e-16`. See `notes/baseline_consistency_check.md`.
- Full researcher-operated CSI checkpoint inference completed at commit `1155bf3`
  on a single visible A100 80GB GPU. TabPFN and FinPFN each produced 150,500 finite
  predictions over 301 dates with all 3,010 groups successful. Runtime was 1,669.336
  and 1,697.121 seconds, respectively. Returned artifacts passed the compute-host
  SHA-256 checks.
- Under the notebook's task-preprocessed-target IC definition, the new TabPFN run
  produced mean IC -0.030197, IC SD 0.065235, and IR -0.462894; FinPFN produced
  0.031801, 0.049099, and 0.647677. The bundled file still exactly recovers FinPFN
  IR 0.855546. Direct overlapping-prediction correlations are only 0.362 Spearman
  for FinPFN and 0.312 for TabPFN, consistent with the missing exact random
  sampling/grouping state being material. See `notes/checkpoint_csi500_run.md`.
- Post-result code comparison found that the completed primary explicitly used
  estimator random state 42, while the notebook omits that argument and TabPFN 2.0.8
  defaults to 0. This is now recorded as a primary-run deviation. Separate
  notebook-exact wrappers use sampling seed 42, with-replacement groups, the
  notebook's post-sampling ID sort, and estimator state 0; a one-date, one-group CPU
  smoke passed for both checkpoints.
- The notebook's default `n_jobs=-1` is capped at 4 in all wrappers for resource
  compliance. On fixed one-group inputs, predictions at 4 and `-1` workers were
  elementwise identical for both checkpoints.
- Holding the one-group context/query rows fixed, changing only estimator state 0
  to 42 gave prediction-rank correlations of 0.6794 for FinPFN and 0.9966 for
  TabPFN. This single-group smoke is not a performance result, but it confirms that
  the estimator-state mismatch can materially change FinPFN ranks and is not merely
  a metadata difference.
- The full researcher-operated notebook-exact run completed with 195,550 finite
  rows, 120,620 unique asset-dates, 74,930 expected repetitions, and 3,911/3,911
  successful groups per model. Runtime was 2,054.745 seconds for TabPFN and
  2,078.874 seconds for FinPFN. Literal repeated-row notebook IR was -0.497656 and
  0.797333, materially closing but not eliminating the paper gap.
- In the common raw-return evaluation after collapsing repetitions, FinPFN IR was
  0.712002 versus 0.566578 for LightGBM and 0.539210 for Ridge. FinPFN true gross
  H-L Sharpe was 4.383559 versus 4.810360 and 4.888952, respectively. The ranking
  advantage is partially recovered; portfolio dominance is not.
