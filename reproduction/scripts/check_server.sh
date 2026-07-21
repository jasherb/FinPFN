#!/usr/bin/env bash
set -eu

printf 'scheduler='
found=''
for command_name in sbatch srun qsub bsub; do
  if command -v "$command_name" >/dev/null 2>&1; then
    found="$found $command_name"
  fi
done
if [ -n "$found" ]; then
  printf '%s\n' "$found"
else
  printf 'none-detected\n'
fi
printf 'gpus_begin\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
else
  printf 'nvidia-smi unavailable\n'
fi
printf 'gpus_end\n'

if command -v lscpu >/dev/null 2>&1; then
  lscpu | awk -F: \
    '/^CPU\(s\)/ || /^Model name/ || /^Socket\(s\)/ || /^Core\(s\) per socket/ || /^Thread\(s\) per core/ {
      gsub(/^[ \t]+/, "", $2); print $1 "=" $2
    }'
fi

if command -v free >/dev/null 2>&1; then
  free -h | awk '/^Mem:/ {print "ram_total=" $2, "ram_available=" $7}'
fi

df -Pk . | awk \
  'NR == 2 {print "current_fs_kb_total=" $2, "current_fs_kb_available=" $4, "current_fs_used_pct=" $5}'

printf 'quota_command='
if command -v quota >/dev/null 2>&1; then
  printf 'available\n'
else
  printf 'unavailable\n'
fi

printf 'configured_storage_envs='
storage_envs=''
for variable_name in SCRATCH PROJECT PROJECT_DIR WORK TMPDIR SLURM_TMPDIR; do
  eval "variable_value=\${$variable_name-}"
  if [ -n "$variable_value" ]; then
    storage_envs="$storage_envs $variable_name"
  fi
done
if [ -n "$storage_envs" ]; then
  printf '%s\n' "$storage_envs"
else
  printf 'none-set\n'
fi

printf 'environment_tools='
for command_name in python3 conda mamba micromamba uv module; do
  if command -v "$command_name" >/dev/null 2>&1; then
    printf ' %s' "$command_name"
  fi
done
printf '\n'

python3 --version 2>/dev/null || true
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | awk '/release/ {print "nvcc=" $0}'
else
  printf 'nvcc=unavailable\n'
fi
