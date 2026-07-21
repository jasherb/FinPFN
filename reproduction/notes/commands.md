# Command log and approved runbook

Commands are relative to the repository root. No SSH command completed and no
remote job was submitted. Generated predictions, fitted models, logs, figures, and
result tables are ignored by Git.

## Integrity and data audit

```bash
shasum -a 256 -c reproduction/configs/checksums.sha256
shasum -a 256 -c reproduction/configs/data_checksums.sha256

reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/inspect_dataset.py \
  --input 30features_csi500.parquet --market csi500 \
  --output reproduction/results/data_audit/csi500.json

reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/inspect_dataset.py \
  --input 90features_USstocks.parquet --market us \
  --output reproduction/results/data_audit/us.json
```

## Completed local checkpoint smoke

This is a compatibility test, not a reproduction result. It uses one date pair and
one 50-stock group but retains the released estimator default of eight ensembles.

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/run_checkpoint_inference.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --models FinPFN TabPFN \
  --output-dir reproduction/artifacts/predictions/smoke_n8 \
  --seeds 42 --sampling-mode artifact_unique500 \
  --n-estimators 8 --device cpu \
  --max-date-pairs 1 --max-groups-per-date 1
```

## Completed local Ridge smoke

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/train_ridge.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/artifacts/predictions/baseline_smoke \
  --seed 42 --max-candidates 2 --smoke-rows-per-split 10000
```

The same script was then completed on the full CSI splits with all five declared
alphas:

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/train_ridge.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/artifacts/predictions/csi500_baselines --seed 42
```

## Completed local LightGBM smoke

The isolated environment uses `lightgbm==4.6.0`. On macOS, its wheel also required
the OpenMP runtime. Only two candidates and deterministic spread samples were used:

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/train_lightgbm.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/artifacts/predictions/baseline_smoke \
  --seed 42 --max-candidates 2 --smoke-rows-per-split 10000
```

## Completed full CSI LightGBM baseline

The user approved the fixed six-candidate search. It ran locally with four CPU
threads, selected on validation only, refit once, and predicted/evaluated test once:

```bash
env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/train_lightgbm.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/artifacts/predictions/csi500_baselines --seed 42

env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/evaluate_predictions.py \
  --predictions reproduction/artifacts/predictions/csi500_baselines/csi500_lightgbm_seed42.parquet \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/results/csi500_lightgbm \
  --figures-dir reproduction/figures/csi500_lightgbm
```

All candidates, validation metrics, runtimes, selection, seed, and test metrics are
recorded in `notes/lightgbm_csi500_run.md` and the ignored run metadata.

## Completed Ridge/LightGBM independence check

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/check_baseline_consistency.py \
  --ridge reproduction/artifacts/predictions/csi500_baselines/csi500_ridge_seed42.parquet \
  --lightgbm reproduction/artifacts/predictions/csi500_baselines/csi500_lightgbm_seed42.parquet \
  --dataset 30features_csi500.parquet \
  --ridge-periods reproduction/results/csi500_ridge/decile_returns_by_period.csv \
  --lightgbm-periods reproduction/results/csi500_lightgbm/decile_returns_by_period.csv \
  --output-dir reproduction/results/baseline_consistency
```

This diagnostic imports the unchanged evaluator's `compute_portfolios` function and
reconstructs both sets of holdings and return series independently. Results are in
`notes/baseline_consistency_check.md`.

## Primary CSI checkpoint run (not yet launched)

Before running, select one permitted free GPU and set `CUDA_VISIBLE_DEVICES`. This
job is intentionally single-GPU. The wrapper refuses to run until that variable is
set.

```bash
bash reproduction/scripts/check_server.sh
bash reproduction/scripts/capture_environment.sh
bash reproduction/scripts/run_csi_checkpoint_primary.sh
```

Equivalent direct command:

```bash
python reproduction/scripts/run_checkpoint_inference.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --models TabPFN FinPFN \
  --output-dir reproduction/artifacts/predictions/csi500_primary \
  --seeds 42 --sampling-mode artifact_unique500 \
  --n-estimators 8 --device cuda
```

The equivalent U.S. wrapper is `run_us_checkpoint_primary.sh`; it is not launched
until the CSI primary and normalized evaluation succeed.

The full step-by-step researcher-only runbook, including separate vanilla TabPFN
and FinPFN wrappers, output verification, and manual artifact return, is in
`notes/manual_checkpoint_runbook.md`. Codex must not execute those server commands.

After the seed-42 primary completes, reasonable stability runs are seeds 4213 and
2025. Literal notebook-with-replacement and all-common modes are sensitivities and
must not replace the primary result.

## Full reconstructed baselines (completed for CSI)

```bash
python reproduction/scripts/train_ridge.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/artifacts/predictions/csi500_baselines --seed 42

python reproduction/scripts/train_lightgbm.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/artifacts/predictions/csi500_baselines --seed 42
```

Both scripts select only on the official validation period, refit on train plus
validation, and predict test once. Their candidates are fixed before execution in
`reproduction/configs/baseline_search.json`.

## Common corrected evaluation

Pass every reproduced prediction parquet to the same invocation:

```bash
python reproduction/scripts/evaluate_predictions.py \
  --predictions \
    reproduction/artifacts/predictions/csi500_primary/*.parquet \
    reproduction/artifacts/predictions/csi500_baselines/*.parquet \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/results \
  --figures-dir reproduction/figures
```

The evaluator defaults to the intersection of available dates and asset-date pairs
across model/seed runs so the headline comparison uses the same periods and
cross-sectional universe. It separately retains source and evaluated coverage. It
merges raw parquet targets, computes date-wise IC, yearly IC, failed rows, deciles,
turnover, notebook-style Sharpe spread, and the primary Sharpe of the actual
top-minus-bottom series.

New checkpoint predictions are compared with the bundled artifact using:

```bash
python reproduction/scripts/compare_bundled_predictions.py \
  --new-predictions reproduction/artifacts/predictions/csi500_primary/*.parquet \
  --bundled results/finpfn_perf_csi500.csv.gz \
  --output reproduction/results/bundled_prediction_comparison.csv
```

## Static validation

```bash
bash -n reproduction/scripts/check_server.sh \
  reproduction/scripts/capture_environment.sh \
  reproduction/scripts/run_csi_checkpoint_primary.sh \
  reproduction/scripts/run_us_checkpoint_primary.sh
python -m py_compile reproduction/scripts/*.py
python -m json.tool reproduction/configs/reported_targets.json
python -m json.tool reproduction/configs/baseline_search.json
```
