#!/usr/bin/env bash
set -euo pipefail

output_file="${FINPFN_ENVIRONMENT_RECORD:-reproduction/runs/environment.txt}"
if [[ -e "$output_file" ]]; then
  printf 'Refusing to overwrite environment record: %s\n' "$output_file" >&2
  exit 1
fi
mkdir -p "$(dirname "$output_file")"

{
  printf 'git_commit='
  git rev-parse HEAD
  python - <<'PY'
from importlib import metadata
import sys

print(f"python={sys.version.split()[0]}")
packages = [
    "torch",
    "tabpfn",
    "lightgbm",
    "scikit-learn",
    "pandas",
    "numpy",
    "scipy",
    "schedulefree",
    "wandb",
    "nvidia-ml-py",
    "seaborn",
    "pyarrow",
    "joblib",
    "matplotlib",
]
for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "not-installed"
    print(f"{package}={version}")

try:
    import torch
except ImportError:
    pass
else:
    print(f"torch_cuda_runtime={torch.version.cuda}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / (1024**3)
            print(f"gpu_{index}={properties.name};memory_gib={memory_gib:.1f}")
PY
} >"$output_file"

printf 'Wrote sanitized environment record to %s\n' "$output_file"
