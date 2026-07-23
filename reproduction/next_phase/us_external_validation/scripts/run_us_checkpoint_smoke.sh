#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one policy-approved free GPU}"

for variable_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$variable_name=4"
done

output_dir="reproduction/next_phase/us_external_validation/artifacts/checkpoint_smoke"
mkdir -p "$output_dir"

python reproduction/scripts/run_checkpoint_inference.py \
  --dataset 90features_USstocks.parquet \
  --market us \
  --models TabPFN FinPFN \
  --output-dir "$output_dir" \
  --seeds 42 \
  --sampling-mode artifact_unique500 \
  --n-estimators 8 \
  --estimator-random-state 0 \
  --estimator-n-jobs 4 \
  --device cuda \
  --max-date-pairs 1 \
  --max-groups-per-date 1
