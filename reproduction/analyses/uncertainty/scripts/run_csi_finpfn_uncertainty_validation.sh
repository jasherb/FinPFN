#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the selected GPU index}"

run_root="${FINPFN_UNCERTAINTY_RUN_ROOT:-reproduction/runs/uncertainty}"

for variable_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$variable_name=4"
done

python reproduction/analyses/uncertainty/run_uncertainty_inference.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --split validation \
  --models FinPFN \
  --output-dir "$run_root/artifacts/validation" \
  --seeds 42 \
  --sampling-mode notebook_with_replacement \
  --n-estimators 8 \
  --estimator-random-state 0 \
  --estimator-n-jobs 4 \
  --device cuda \
  --verify-reference-output
