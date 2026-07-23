#!/usr/bin/env bash
set -euo pipefail

output_dir="reproduction/next_phase/us_external_validation/results/common"
figures_dir="reproduction/next_phase/us_external_validation/figures/common"
if [[ -e "$output_dir" || -e "$figures_dir" ]]; then
  echo "Refusing to overwrite existing U.S. common-universe evaluation" >&2
  exit 1
fi
mkdir -p "$output_dir" "$figures_dir"

python reproduction/scripts/evaluate_predictions.py \
  --dataset 90features_USstocks.parquet \
  --market us \
  --predictions \
    reproduction/next_phase/us_external_validation/artifacts/baselines/us_ridge_seed42.parquet \
    reproduction/next_phase/us_external_validation/artifacts/baselines/us_lightgbm_seed42.parquet \
    reproduction/next_phase/us_external_validation/artifacts/checkpoints/us_tabpfn_seed42_artifact_unique500.parquet \
    reproduction/next_phase/us_external_validation/artifacts/checkpoints/us_finpfn_seed42_artifact_unique500.parquet \
  --date-policy intersection \
  --universe-policy intersection \
  --output-dir "$output_dir" \
  --figures-dir "$figures_dir"
