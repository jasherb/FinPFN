#!/usr/bin/env bash
set -euo pipefail

run_root="${FINPFN_US_RUN_ROOT:-reproduction/runs/us_validation}"
output_dir="$run_root/evaluation"
figures_dir="$run_root/figures/common"
if [[ -e "$output_dir" || -e "$figures_dir" ]]; then
  echo "Refusing to overwrite existing U.S. common-universe evaluation" >&2
  exit 1
fi
mkdir -p "$output_dir" "$figures_dir"

python reproduction/scripts/evaluate_predictions.py \
  --dataset 90features_USstocks.parquet \
  --market us \
  --predictions \
    "$run_root/artifacts/baselines/us_ridge_seed42.parquet" \
    "$run_root/artifacts/baselines/us_lightgbm_seed42.parquet" \
    "$run_root/artifacts/checkpoints/us_tabpfn_seed42_artifact_unique500.parquet" \
    "$run_root/artifacts/checkpoints/us_finpfn_seed42_artifact_unique500.parquet" \
  --date-policy intersection \
  --universe-policy intersection \
  --output-dir "$output_dir" \
  --figures-dir "$figures_dir"
