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

The completed primary run also used estimator `random_state=42`. This is now
explicit in its wrappers so the returned files remain reproducible. The released
notebook actually omits that argument, which means TabPFN 2.0.8 uses its default
estimator random state of 0. The separate notebook-exact diagnostic in section 6
corrects this without overwriting the completed primary run.

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
  --estimator-random-state 42 \
  --estimator-n-jobs 4 \
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
  --estimator-random-state 42 \
  --estimator-n-jobs 4 \
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

## 6. Notebook-exact diagnostic after the primary discrepancy

This is a separate, predeclared diagnostic. It follows the executable notebook's
visible inference choices: NumPy stock-sampling seed 42, sampling with replacement,
50 stocks per group sorted by ID after sampling, eight estimators, and the TabPFN
2.0.8 default estimator random state of 0. It writes to `csi500_notebook_exact` and
cannot overwrite the completed `csi500_primary` files.

The wrappers cap `n_jobs` at 4 instead of the notebook's default `-1` to respect the
resource policy. A fixed-input smoke produced elementwise-identical predictions at
4 and `-1` workers for both checkpoints, so this cap is operational rather than a
model-output change.

The deterministic loader audit predicts 301 date pairs, 3,911 groups, and 195,550
rows per model. Because sampling is with replacement, only about 120,620 unique
asset-dates are expected and about 74,930 rows are expected repetitions. Based on
the completed primary timings, allow roughly 36-37 minutes per model, or about 73
minutes sequentially on the same A100; actual time may vary.

After pulling the commit containing these wrappers, run both sequentially on the
same manually selected GPU:

```bash
export CUDA_VISIBLE_DEVICES=<FREE_GPU_INDEX>
mkdir -p reproduction/logs

bash reproduction/scripts/run_csi_tabpfn_notebook_exact.sh \
  > reproduction/logs/csi500_tabpfn_notebook_exact.log 2>&1

bash reproduction/scripts/run_csi_finpfn_notebook_exact.sh \
  > reproduction/logs/csi500_finpfn_notebook_exact.log 2>&1
```

The combined wrapper is equivalent:

```bash
bash reproduction/scripts/run_csi_checkpoint_notebook_exact.sh
```

Expected output names are:

```text
reproduction/artifacts/predictions/csi500_notebook_exact/
  csi500_tabpfn_seed42_notebook_with_replacement.parquet
  csi500_tabpfn_seed42_notebook_with_replacement.metadata.json
  csi500_finpfn_seed42_notebook_with_replacement.parquet
  csi500_finpfn_seed42_notebook_with_replacement.metadata.json
```

Because sampling is with replacement, repeated asset-date rows are expected and
must not be treated as inference failures. Evaluate the notebook-faithful IC while
retaining those repetitions with:

```bash
python reproduction/scripts/evaluate_notebook_checkpoint_ic.py \
  --predictions \
    reproduction/artifacts/predictions/csi500_notebook_exact/csi500_tabpfn_seed42_notebook_with_replacement.parquet \
    reproduction/artifacts/predictions/csi500_notebook_exact/csi500_finpfn_seed42_notebook_with_replacement.parquet \
  --output-dir reproduction/results/csi500_notebook_exact
```

For the corrected common raw-return comparison, pass the same files to
`evaluate_predictions.py`; that evaluator deliberately collapses repeated
asset-date rows before forming a common universe. Neither result may be selected or
discarded according to its test performance.
