#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
cd "$repository_root"

for variable_name in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$variable_name=4"
done

# This analysis is deliberately CPU-only. It reuses frozen predictions and does
# not train or run checkpoint inference.
export CUDA_VISIBLE_DEVICES=""
export FINPFN_US_RUN_ROOT="${FINPFN_US_RUN_ROOT:-reproduction/runs/us_validation}"

dataset="90features_USstocks.parquet"
expected_dataset_sha256="54818c78796ecae3974b2058575cd2284482ce35e62c9116d316e23240b8ef50"

if [[ ! -f "$dataset" ]]; then
  echo "Missing required raw U.S. panel: $repository_root/$dataset" >&2
  exit 1
fi

actual_dataset_sha256="$(sha256sum "$dataset" | awk '{print $1}')"
if [[ "$actual_dataset_sha256" != "$expected_dataset_sha256" ]]; then
  echo "Dataset SHA-256 mismatch; refusing to analyze a different panel." >&2
  echo "Expected: $expected_dataset_sha256" >&2
  echo "Actual:   $actual_dataset_sha256" >&2
  exit 1
fi

python reproduction/analyses/us_validation/analysis/analyze_us_results.py
