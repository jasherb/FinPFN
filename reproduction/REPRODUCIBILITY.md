# Reproducibility guide

Commands are run from the repository root. New outputs are written below the
ignored `reproduction/runs/` directory and are never written into
`reference_results/`.

## 1. Install

CPU-only analysis and baselines:

```bash
python3.10 -m venv .venv-cpu
. .venv-cpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproduction/environment/requirements-cpu-lock.txt
```

Released-checkpoint inference on a CUDA 12.1-compatible system:

```bash
python3.10 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproduction/environment/requirements-gpu-cu121-lock.txt
```

Place separately licensed data and checkpoints at the locations in
[ASSETS.md](ASSETS.md), then run:

```bash
python reproduction/scripts/preflight.py --mode full
```

Before checkpoint inference, activate the GPU environment and verify the exact
PyTorch/TabPFN stack plus CUDA visibility:

```bash
python reproduction/scripts/preflight.py --mode full --require-gpu
```

## 2. CSI 500

### CPU baselines

Model selection uses only the 2021 validation period. The frozen candidate
grids are in `configs/baseline_search.json`.

```bash
mkdir -p reproduction/runs/csi500/baselines

python reproduction/scripts/train_ridge.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --output-dir reproduction/runs/csi500/baselines \
  --seed 42

python reproduction/scripts/train_lightgbm.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --output-dir reproduction/runs/csi500/baselines \
  --seed 42
```

### Released checkpoints

Select one visible GPU. The literal-notebook diagnostic uses sampling with
replacement and TabPFN's estimator state 0:

```bash
export CUDA_VISIBLE_DEVICES=<GPU_INDEX>
export FINPFN_CSI_RUN_ROOT=reproduction/runs/csi500

bash reproduction/scripts/capture_environment.sh
bash reproduction/scripts/run_csi_checkpoint_notebook_exact.sh
```

The primary artifact-shape diagnostic uses 500 unique assets per date and
estimator state 42:

```bash
bash reproduction/scripts/run_csi_checkpoint_primary.sh
```

Each wrapper uses eight estimators, four CPU workers, one GPU, and refuses to
replace an existing prediction/metadata pair.

### Common evaluation

```bash
python reproduction/scripts/evaluate_predictions.py \
  --predictions \
    reproduction/runs/csi500/baselines/csi500_ridge_seed42.parquet \
    reproduction/runs/csi500/baselines/csi500_lightgbm_seed42.parquet \
    reproduction/runs/csi500/checkpoints/notebook_exact/csi500_tabpfn_seed42_notebook_with_replacement.parquet \
    reproduction/runs/csi500/checkpoints/notebook_exact/csi500_finpfn_seed42_notebook_with_replacement.parquet \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --date-policy intersection \
  --universe-policy intersection \
  --output-dir reproduction/runs/csi500/evaluation \
  --figures-dir reproduction/runs/csi500/figures
```

For the literal notebook repeated-row IC diagnostic:

```bash
python reproduction/scripts/evaluate_notebook_checkpoint_ic.py \
  --predictions \
    reproduction/runs/csi500/checkpoints/notebook_exact/csi500_tabpfn_seed42_notebook_with_replacement.parquet \
    reproduction/runs/csi500/checkpoints/notebook_exact/csi500_finpfn_seed42_notebook_with_replacement.parquet \
  --output-dir reproduction/runs/csi500/notebook_ic
```

## 3. U.S. external validation

The wrappers share one configurable run root:

```bash
export FINPFN_US_RUN_ROOT=reproduction/runs/us_validation

bash reproduction/analyses/us_validation/scripts/run_us_baselines.sh

export CUDA_VISIBLE_DEVICES=<GPU_INDEX>
bash reproduction/analyses/us_validation/scripts/run_us_checkpoint_smoke.sh
bash reproduction/analyses/us_validation/scripts/run_us_checkpoints.sh

bash reproduction/analyses/us_validation/scripts/evaluate_us_common.sh
bash reproduction/analyses/us_validation/scripts/run_us_full_analysis.sh
```

The checkpoint run uses released weights, seed 42, 500 unique assets per month,
eight estimators, estimator state 0, and no training. Ridge and LightGBM select
from the same predeclared candidate sets using 2000-2009 validation only.

## 4. Supporting analyses

The transaction-cost evaluator accepts the newly generated CSI holdings and
period returns:

```bash
python reproduction/analyses/transaction_costs/transaction_cost_analysis.py \
  --holdings reproduction/runs/csi500/evaluation/decile_holdings.parquet \
  --period-returns reproduction/runs/csi500/evaluation/decile_returns_by_period.csv \
  --baseline-turnover reproduction/runs/csi500/evaluation/turnover_by_decile.csv \
  --output-dir reproduction/runs/transaction_costs
```

The uncertainty, turnover-control, and tail scripts are retained as transparent
research code. Their input contracts are available with `--help` where
applicable. They write only below `reproduction/runs/`; the release does not
rerun model selection or evaluate new thresholds on the frozen test sets.

## 5. Validation/test separation

- CSI baseline selection: 2021 validation; test begins in 2022.
- U.S. baseline selection: 2000-2009 validation; test begins in 2010.
- The uncertainty and turnover-control grid was declared before validation
  selection. One configuration was frozen and evaluated once on CSI test.
- Cost and IC-portfolio-gap analyses of test data are explicitly exploratory.
- No FinPFN retraining, test-driven threshold replacement, or tuning to the
  paper's reported values is part of this release.

## 6. Data-free release verification

```bash
python3 reproduction/tests/verify_release.py
python3 reproduction/tests/check_public_tree.py
```

The checks validate headline metrics, English-only public text, local links,
file extensions, executable syntax, and common secret patterns. They do not
replace a dedicated security scanner before publication.
