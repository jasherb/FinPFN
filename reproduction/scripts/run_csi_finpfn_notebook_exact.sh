#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one policy-approved free GPU}"

for variable_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$variable_name=4"
done

python reproduction/scripts/run_checkpoint_inference.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --models FinPFN \
  --output-dir reproduction/artifacts/predictions/csi500_notebook_exact \
  --seeds 42 \
  --sampling-mode notebook_with_replacement \
  --n-estimators 8 \
  --estimator-random-state 0 \
  --estimator-n-jobs 4 \
  --device cuda
