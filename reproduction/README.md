# Reproduction and audit code

This directory contains the independent FinPFN reproduction and model-risk audit. It is intentionally separated from the upstream training and notebook code.

## Frozen scope

- Released-checkpoint inference for FinPFN and vanilla TabPFN; no FinPFN retraining.
- Validation-only model selection for independently reconstructed Ridge and LightGBM baselines.
- CSI 500 and U.S. common-universe evaluation on one raw-return target per market.
- Cross-sectional IC/IR, actual top-minus-bottom Sharpe, turnover, transaction costs, tail precision, rank stability, and uncertainty diagnostics.
- Final research decision: stop tuning on the existing test sets and retain the project as a reproduction and model-risk audit.

The consolidated English report is [AUDIT_REPORT.md](AUDIT_REPORT.md). Source assets and checksums are documented in [ASSETS.md](ASSETS.md).

## Directory map

```text
reproduction/
  configs/       # declared searches, inference settings, checksums
  environment/   # environment specifications and captured versions
  notes/         # CSI audit, command log, checkpoint runbook
  scripts/       # baseline, inference, evaluation, and integrity code
  next_phase/    # cost, uncertainty, gating, tail, and U.S. analyses
  public/        # README-figure generator and committed public figures
```

Large or sensitive runtime products are excluded:

```text
reproduction/artifacts/
reproduction/results/
reproduction/logs/
reproduction/figures/
reproduction/next_phase/**/artifacts/
```

For the complete human-run commands, use the [CSI checkpoint runbook](notes/manual_checkpoint_runbook.md), [command record](notes/commands.md), and [U.S. runbook](next_phase/us_external_validation/manual_commands.md). The scripts use repository-relative paths and refuse to overwrite the principal frozen outputs unless an explicit overwrite flag is passed.
