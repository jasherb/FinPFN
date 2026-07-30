#!/usr/bin/env bash
set -euo pipefail

for variable_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$variable_name=4"
done

run_root="${FINPFN_UNCERTAINTY_RUN_ROOT:-reproduction/runs/uncertainty}"

python reproduction/analyses/uncertainty/run_uncertainty_inference.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --split validation \
  --models TabPFN FinPFN \
  --output-dir "$run_root/artifacts/smoke_validation" \
  --seeds 42 \
  --sampling-mode notebook_with_replacement \
  --n-estimators 8 \
  --estimator-random-state 0 \
  --estimator-n-jobs 4 \
  --device cpu \
  --max-date-pairs 1 \
  --max-groups-per-date 1 \
  --verify-reference-output
