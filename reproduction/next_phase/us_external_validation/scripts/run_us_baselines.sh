#!/usr/bin/env bash
set -euo pipefail

for variable_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$variable_name=4"
done

output_dir="reproduction/next_phase/us_external_validation/artifacts/baselines"
if [[ -e "$output_dir/us_ridge_seed42.parquet" ||
      -e "$output_dir/us_lightgbm_seed42.parquet" ]]; then
  echo "Refusing to overwrite existing U.S. baseline predictions" >&2
  exit 1
fi
mkdir -p "$output_dir"

python reproduction/scripts/train_ridge.py \
  --dataset 90features_USstocks.parquet \
  --market us \
  --output-dir "$output_dir" \
  --seed 42

python reproduction/scripts/train_lightgbm.py \
  --dataset 90features_USstocks.parquet \
  --market us \
  --output-dir "$output_dir" \
  --seed 42
