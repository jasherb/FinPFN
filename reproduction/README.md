# FinPFN artifact-faithful reproduction

This directory isolates reproduction work from the upstream repository. Released
code, checkpoints, notebook behavior, and checkpoint metadata govern when they
conflict with the paper. Methodological extensions remain out of scope.

## Current status

- Repository commit audited: `99b2a0e`.
- Paper, training code, notebook, checkpoints, and bundled CSI 500 predictions audited.
- Official CSI 500, U.S., and CSI index-price parquets are present at repository
  root, hashed, ignored, and audited without modification.
- Full seed-42 released-checkpoint inference is complete for FinPFN and vanilla
  TabPFN on a researcher-operated single A100 80GB GPU. Each produced 150,500
  finite predictions with zero failed groups.
- Full reconstructed CSI Ridge and LightGBM baselines are complete. Their test IRs
  are 0.616215 and 0.661700, respectively; both have full source coverage.
- Codex-issued SSH or direct compute-server access is prohibited by the local policy
  and user instruction. Only the researcher may run the prepared manual wrappers.
- The common CSI evaluation, per-period IC, decile holdings/returns, turnover, and
  figures are complete. The new FinPFN run reached paper-faithful IR 0.647677 versus
  the paper's 0.85, while the bundled CSV exactly recovers 0.855546; this is not an
  exact reproduction. No FinPFN training or post-test tuning has been performed.

See `notes/audit.md` for the method trace and known discrepancies.

## Intended directory layout

```text
reproduction/
  configs/
  environment/
  figures/       # generated, ignored
  logs/          # generated, ignored
  notes/
  results/       # generated, ignored
  scripts/
```

## Expected human-run commands

Run these from the repository root on the approved compute host. They deliberately
contain no server name, account, credential, or absolute storage path.

```bash
bash reproduction/scripts/check_server.sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash reproduction/scripts/capture_environment.sh
```

The upstream requirements are not a complete lock file. Before model execution,
freeze the resolved environment and record Python, PyTorch, CUDA, driver, TabPFN,
LightGBM, pandas, NumPy, SciPy, and scikit-learn versions.

The official datasets are stored at the repository root and ignored by Git. Audit
them in place with the isolated PyArrow reader:

```bash
PYTHONPATH=reproduction/environment/audit-venv/lib/python3.11/site-packages \
python3 reproduction/scripts/inspect_dataset.py \
  --input 30features_csi500.parquet \
  --market csi500 \
  --output reproduction/results/csi500_dataset_summary.json

PYTHONPATH=reproduction/environment/audit-venv/lib/python3.11/site-packages \
python3 reproduction/scripts/inspect_dataset.py \
  --input 90features_USstocks.parquet \
  --market us \
  --output reproduction/results/us_dataset_summary.json
```

New checkpoint predictions use one common long schema and are written only beneath
the ignored `reproduction/artifacts/` directory. The primary sampling mode is
`artifact_unique500`: 500 common identifiers sampled without replacement per date
pair, split into ten 50-stock tasks. It matches the released CSV's observable shape;
the executable notebook cell instead samples with replacement, which remains an
explicitly labelled sensitivity mode.

Minimal checkpoint compatibility test:

```bash
python reproduction/scripts/run_checkpoint_inference.py \
  --dataset 30features_csi500.parquet \
  --market csi500 \
  --models FinPFN TabPFN \
  --output-dir reproduction/artifacts/predictions/smoke \
  --seeds 42 \
  --sampling-mode artifact_unique500 \
  --n-estimators 8 \
  --estimator-random-state 42 \
  --estimator-n-jobs 4 \
  --device cpu \
  --max-date-pairs 1 \
  --max-groups-per-date 1
```

See `notes/manual_checkpoint_runbook.md` for the exact human-run single-GPU commands
and `notes/commands.md` for the complete local command log and evaluation commands.
The released notebook does not fit Ridge or LightGBM. Reconstructed baselines use
only the validation period for selection and refit the selected model on train plus
validation before one test prediction pass.

After the CSI primary result, an estimator-seed discrepancy was identified: the
completed artifact-shape run used estimator random state 42, while the released
notebook omits the argument and TabPFN 2.0.8 therefore defaults to 0. Separate
`notebook_exact` wrappers now reproduce the visible notebook choices without
overwriting the primary files; see section 6 of `notes/manual_checkpoint_runbook.md`.

## Official execution order

1. Dataset/schema/split validation. (complete)
2. Released vanilla TabPFN and FinPFN checkpoint inference. (complete for CSI seed 42)
3. Ridge temporal-validation baseline. (complete for CSI)
4. LightGBM temporal-validation baseline. (complete for CSI)
5. IC series, IC mean, sample standard deviation, IR, decile portfolios, top decile,
   long-short, gross Sharpe, and turnover from reproduced holdings. (complete for CSI)
6. Fine-tuning only after checkpoint reproduction and only with explicit approval.
