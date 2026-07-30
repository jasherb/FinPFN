# Reproduction package

This directory contains the independent FinPFN reproduction and model-risk
audit. The upstream notebook and training utilities remain at the repository
root and under `scripts/`; root documentation and dependency metadata were
adapted for this public release.

## Public layout

```text
reproduction/
  analyses/           # cost, uncertainty, turnover, tail, and U.S. analysis code
  configs/            # predeclared searches, inference settings, and checksums
  environment/        # pinned CPU and CUDA 12.1 environments
  reference_results/  # immutable aggregate tables, diagnostics, and figures
  scripts/            # baseline, checkpoint, evaluation, and release utilities
  tests/              # data-free release and repository-hygiene checks
  runs/                # generated outputs; ignored by Git
```

The headline evidence is consolidated in [REPORT.md](REPORT.md). Exact commands
are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md), and externally obtained assets
are documented in [ASSETS.md](ASSETS.md).

## Reproducibility boundary

`reference_results/` is read-only evidence for this release. Analysis and
inference commands write to `reproduction/runs/` by default and refuse to
overwrite an existing run. Datasets, checkpoints, fitted models, generated
predictions, holdings, logs, and environment captures are not committed.

Run the data-free release verification from the repository root:

```bash
python3 reproduction/tests/verify_release.py
```
