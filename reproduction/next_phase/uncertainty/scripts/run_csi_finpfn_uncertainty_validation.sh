#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one policy-approved free GPU}"

for variable_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$variable_name=4"
done

python reproduction/next_phase/uncertainty/run_uncertainty_inference.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --split validation \
  --models FinPFN \
  --output-dir reproduction/next_phase/uncertainty/artifacts/validation \
  --seeds 42 \
  --sampling-mode notebook_with_replacement \
  --n-estimators 8 \
  --estimator-random-state 0 \
  --estimator-n-jobs 4 \
  --device cuda \
  --verify-reference-output
