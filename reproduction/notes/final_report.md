# Artifact-faithful FinPFN reproduction report

The consolidated, data-rich Chinese CSI report is
`reproduction/notes/csi500_reproduction_report_zh.md`.

> CSI seed-42 checkpoint inference and common evaluation are complete. The released
> FinPFN checkpoint did not recover the bundled/paper FinPFN ICIR, so this file must
> not be read as a successful exact reproduction claim.

## Scope and decision rule

Released code, checkpoints, notebook behavior, downloaded parquets, and checkpoint
metadata govern where they conflict with the paper. No FinPFN retraining,
transaction costs, uncertainty gating, alternative features, or extra models have
been introduced.

The completed CSI primary is an artifact-shape reconstruction, not a literal
execution of the visible notebook: it used 500 unique assets and estimator random
state 42. The notebook uses with-replacement stock sampling and the TabPFN 2.0.8
default estimator random state 0. A separately named notebook-exact diagnostic is
complete; it tested this ambiguity without overwriting the primary result.

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
members, sampling seed 42, and estimator random state 42 for the primary run. The
primary 500-without-replacement mode matches the bundled CSV's observable
500-unique-assets-per-date shape. The completed notebook-exact follow-up instead
uses the visible notebook's with-replacement sampling, post-sampling ID sort, and
estimator state 0.

## Completed checks and result

- Full released-checkpoint inference completed for both models with seed 42 on one
  NVIDIA A100 80GB PCIe. Each produced 150,500 finite predictions across 301 dates,
  500 assets per date, with all 3,010 groups successful and no duplicate asset-date
  rows. TabPFN took 1,669.336 seconds and FinPFN took 1,697.121 seconds.
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
- Under the paper-faithful task-preprocessed-target IC definition, the new TabPFN
  run achieved mean IC -0.030197, IC SD 0.065235, and IR -0.462894 versus the paper's
  -0.44. FinPFN achieved 0.031801, 0.049099, and 0.647677 versus the paper's 0.85.
  The bundled prediction CSV itself exactly recovers FinPFN IR 0.855546 and TabPFN
  IR -0.442639, so the FinPFN discrepancy lies between newly generated checkpoint
  predictions and the unreleased sampling/grouping state behind the bundle.
- On the common raw-return universe of 301 dates by 500 assets, FinPFN, TabPFN,
  Ridge, and LightGBM achieved IRs of 0.579996, -0.450515, 0.620244, and 0.644403.
  Their true gross long-short Sharpes were 5.124263, -3.549967, 6.272572, and
  6.316055, respectively. Full IC, portfolio, turnover, runtime, and comparison
  details are in `checkpoint_csi500_run.md`.
- The researcher-operated notebook-exact follow-up completed with 195,550 finite
  rows per model, 120,620 unique asset-dates, 74,930 expected repeated rows, and all
  3,911 groups successful. Literal notebook IC retained repetitions: FinPFN achieved
  mean IC 0.043864, IC SD 0.055013, and IR 0.797333; TabPFN achieved -0.034296,
  0.068915, and -0.497656. This is materially closer to the paper but not exact.
- After collapsing repetitions and restricting all models to the same 120,620
  asset-dates, raw-return IR was 0.712002 for FinPFN, 0.566578 for LightGBM,
  0.539210 for Ridge, and -0.522875 for TabPFN. FinPFN therefore recovered the
  common-universe IC/IR lead. Its true gross long-short Sharpe was 4.383559 versus
  4.810360 for LightGBM and 4.888952 for Ridge, so portfolio dominance was not
  reproduced.

## Remaining scope and limitations

1. The exact random sampling/grouping state used for the bundled checkpoint
   predictions is not published. Additional seeds would be sensitivity analysis,
   not recovery of that missing state, and must not be selected using test results.
2. Equivalent U.S. runs remain outstanding; official U.S. regime dates remain
   unavailable without the missing VIX episode source.

The generated CSI predictions and combined metrics now exist. They support a
partial reproduction of FinPFN's ranking advantage under the notebook-exact
configuration, but not an exact paper match or comprehensive portfolio advantage.
No post-test tuning or FinPFN retraining has been performed.
