#!/usr/bin/env bash
set -eu

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one policy-approved free GPU}"

for variable_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$variable_name=4"
done

python reproduction/scripts/run_checkpoint_inference.py \
  --dataset 90features_USstocks.parquet \
  --market us \
  --models FinPFN TabPFN \
  --output-dir reproduction/artifacts/predictions/us_primary \
  --seeds 42 \
  --sampling-mode artifact_unique500 \
  --n-estimators 8 \
  --device cuda
