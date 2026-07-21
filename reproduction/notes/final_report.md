# Artifact-faithful FinPFN reproduction report

> Interim report — full released-checkpoint inference is still pending. This file
> must not be read as a successful reproduction claim.

## Scope and decision rule

Released code, checkpoints, notebook behavior, downloaded parquets, and checkpoint
metadata govern where they conflict with the paper. No FinPFN retraining,
transaction costs, uncertainty gating, alternative features, or extra models have
been introduced.

## Dataset status

The official CSI 500 and U.S. panels have been audited in place and pass schema,
date-order, uniqueness, missingness, and non-finite checks. Exact counts, splits,
preprocessing diagnostics, and timing limitations are in `data_audit.md`. Raw source
timestamps are absent, so forward-return timing and point-in-time feature
availability cannot be independently proven.

## Artifact configuration

Checkpoint inference uses TabPFN 2.0.8, released checkpoint architecture metadata,
50 context and 50 query stocks on adjacent dates, group-wise sample-standardized
context labels, estimator-default preprocessing, median prediction, eight ensemble
members, and seed 42 for the primary run. The primary 500-without-replacement mode
matches the bundled CSV's observable 500-unique-assets-per-date shape. The live
notebook's with-replacement code is retained only as a sensitivity because it cannot
produce that shape in one pass.

## Completed checks and result

- Both released CSI checkpoints pass one-date, one-group CPU inference with no
  failures. This confirms compatibility only; it is not a performance result.
- A newly reconstructed Ridge baseline selected alpha 0.001 using validation-only
  mean date-wise Spearman IC. On all 302 CSI test dates it achieved mean IC
  0.040454, IC SD 0.065650, and IR 0.616215. Its primary actual-spread Sharpe was
  6.176169. It covered every test asset-date.
- The full independently reconstructed six-candidate LightGBM search selected
  `learning_rate=0.05`, `n_estimators=500`, `num_leaves=63`, `max_depth=6`,
  `min_child_samples=50`, and zero L1/L2 regularization using validation mean IC
  only. On the 302-date test it achieved mean IC 0.038657, IC SD 0.058421, IR
  0.661700, and primary true long-short Sharpe 6.176134 with full coverage. The
  complete candidate table and runtimes are in `lightgbm_csi500_run.md`.
- The normalized evaluator uses raw parquet returns for portfolios, deterministic
  tie-breaking by asset ID, equal-weight deciles, 240 CSI or 12 U.S. annualization,
  arithmetic cumulative sums, and no transaction costs. It clearly separates top
  and bottom Sharpe, notebook Sharpe spread, and true long-short Sharpe.
- Ridge and LightGBM's nearly identical true long-short Sharpes are verified as a
  coincidence of their mean/volatility ratios, not file reuse. Their prediction
  Pearson correlation is 0.5485, top-decile overlap is 44.94%, and long-short return
  correlation is 0.6015 with a 1.6883 percentage-point maximum difference. Full
  evidence is in `baseline_consistency_check.md`; no evaluation change was made.

## Pending before qualitative conclusions

1. Full released-checkpoint inference for FinPFN and vanilla TabPFN on an approved
   GPU route.
2. Common-date combined evaluation, official CSI regime windows, stability seeds,
   and the requested figures/tables.
3. Equivalent U.S. runs; official U.S. regime dates remain unavailable without the
   missing VIX episode source.

No claim about FinPFN outperforming TabPFN, Ridge, or LightGBM will be made until
the newly generated checkpoint predictions and combined metrics exist.
