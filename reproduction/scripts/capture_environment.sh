#!/usr/bin/env bash
set -eu

output_file='reproduction/environment/environment.txt'
mkdir -p "$(dirname "$output_file")"

{
  printf 'git_commit='
  git rev-parse HEAD
  printf 'kernel='
  uname -srvm
  python - <<'PY'
from importlib import metadata
import platform
import sys

print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
packages = [
    "torch", "tabpfn", "lightgbm", "scikit-learn", "pandas", "numpy",
    "scipy", "schedulefree", "wandb", "pynvml", "seaborn", "pyarrow",
    "joblib", "matplotlib",
]
for package in packages:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "not-installed"
    print(f"{package}={version}")
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total \
      --format=csv,noheader,nounits
  fi
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | awk '/release/ {print "nvcc=" $0}'
  fi
  printf 'packages_begin\n'
  python -m pip list --format=freeze
  printf 'packages_end\n'
} >"$output_file"

printf 'Wrote package-only environment record to %s\n' "$output_file"
