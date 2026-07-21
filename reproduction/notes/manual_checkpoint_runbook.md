# Manual single-GPU released-checkpoint runbook

Codex must not connect to or issue commands on the compute server. These commands
are for the researcher to run manually after committing/pushing locally and pulling
the branch on the approved host.

## 0. Transfer the ignored dataset manually

Git does not contain the parquet datasets. For the CSI checkpoint run, only
`30features_csi500.parquet` is required; the released checkpoints arrive through
Git. First check whether the parquet already exists on the server and whether its
SHA-256 is correct. If it is absent, run the transfer yourself from a local terminal.
`rsync` over SSH is preferred because it reports progress and retains a partial file
if the connection drops:

```bash
cd <LOCAL_FINPFN_REPOSITORY>
rsync -avP 30features_csi500.parquet \
  <SSH_ALIAS>:<REMOTE_FINPFN_REPOSITORY>/
```

If `rsync` is unavailable, use:

```bash
scp 30features_csi500.parquet \
  <SSH_ALIAS>:<REMOTE_FINPFN_REPOSITORY>/
```

Do not put a password, key, or token in either command. Use the researcher's normal
SSH configuration/agent. Do not copy SSH private keys into the repository or server
workspace.

Verify both ends after transfer:

```bash
# Local macOS
shasum -a 256 30features_csi500.parquet

# Remote Linux, run manually from the repository root
sha256sum 30features_csi500.parquet
```

Both must equal:

```text
9e0d61f5d70151d4f2f7b40918a8ddb79f86fb54a0fe86759f5c1f2869fe1b3e
```

The U.S. parquet is not needed for the first CSI run. Transfer it later only when
the U.S. reproduction begins; its expected SHA-256 is recorded in
`reproduction/configs/data_checksums.sha256`.

## 1. Verify the environment and select one GPU

From the FinPFN repository root, activate the reproduction environment, then run:

```bash
bash reproduction/scripts/check_server.sh
python - <<'PY'
import torch
import tabpfn

print("torch", torch.__version__)
print("tabpfn", tabpfn.__version__)
print("cuda_available", torch.cuda.is_available())
assert tabpfn.__version__ == "2.0.8"
assert torch.cuda.is_available()
PY
nvidia-smi
```

Choose one policy-permitted free GPU from `nvidia-smi`, then substitute its integer
index below:

```bash
export CUDA_VISIBLE_DEVICES=<FREE_GPU_INDEX>
mkdir -p reproduction/logs
bash reproduction/scripts/capture_environment.sh
```

Both wrappers require `CUDA_VISIBLE_DEVICES` and set the four CPU thread limits.
They use only the released checkpoints, eight TabPFN ensembles, seed 42, adjacent
test dates, 50 context/query stocks, and the primary 500-unique-asset sampling mode.
They refuse to replace existing outputs unless `--overwrite` is explicitly passed
to the underlying Python command.

## 2. Run vanilla TabPFN first

```bash
bash reproduction/scripts/run_csi_tabpfn_primary.sh \
  > reproduction/logs/csi500_tabpfn_primary.log 2>&1
```

Equivalent direct command:

```bash
python reproduction/scripts/run_checkpoint_inference.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --models TabPFN \
  --output-dir reproduction/artifacts/predictions/csi500_primary \
  --seeds 42 \
  --sampling-mode artifact_unique500 \
  --n-estimators 8 \
  --device cuda
```

## 3. Run released FinPFN second

```bash
bash reproduction/scripts/run_csi_finpfn_primary.sh \
  > reproduction/logs/csi500_finpfn_primary.log 2>&1
```

Equivalent direct command:

```bash
python reproduction/scripts/run_checkpoint_inference.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --models FinPFN \
  --output-dir reproduction/artifacts/predictions/csi500_primary \
  --seeds 42 \
  --sampling-mode artifact_unique500 \
  --n-estimators 8 \
  --device cuda
```

To run both sequentially on the same selected GPU:

```bash
bash reproduction/scripts/run_csi_checkpoint_primary.sh
```

## 4. Verify and return the ignored artifacts

Expected output pairs:

```text
reproduction/artifacts/predictions/csi500_primary/
  csi500_tabpfn_seed42_artifact_unique500.parquet
  csi500_tabpfn_seed42_artifact_unique500.metadata.json
  csi500_finpfn_seed42_artifact_unique500.parquet
  csi500_finpfn_seed42_artifact_unique500.metadata.json
```

Inspect only non-sensitive run fields and calculate transfer checksums:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("reproduction/artifacts/predictions/csi500_primary")
for path in sorted(root.glob("*.metadata.json")):
    record = json.loads(path.read_text())
    print(
        path.name,
        "rows=", record["successful_prediction_rows"],
        "failed_groups=", record["failed_groups"],
        "seconds=", round(record["runtime_seconds"], 3),
    )
PY
sha256sum reproduction/artifacts/predictions/csi500_primary/*
```

Keep the outputs ignored. Transfer the two parquet/metadata pairs and environment
record back through the researcher's approved manual method; do not commit them.

## 5. Local comparison after manual transfer

Once the files are placed in the same ignored local artifact directory:

```bash
reproduction/environment/audit-venv/bin/python \
  reproduction/scripts/compare_bundled_predictions.py \
  --new-predictions \
    reproduction/artifacts/predictions/csi500_primary/csi500_tabpfn_seed42_artifact_unique500.parquet \
    reproduction/artifacts/predictions/csi500_primary/csi500_finpfn_seed42_artifact_unique500.parquet \
  --bundled results/finpfn_perf_csi500.csv.gz \
  --output reproduction/results/bundled_prediction_comparison.csv
```

The combined normalized evaluation will then use these two files plus the completed
Ridge and LightGBM prediction files. No FinPFN fine-tuning is authorized at this
stage.
