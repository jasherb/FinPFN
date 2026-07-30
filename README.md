# FinPFN Reproduction & Model-Risk Audit

**Research question:** Does FinPFN's stronger cross-sectional ranking accuracy translate into better tradable long–short portfolios across China and U.S. equities?

## Core results

All four models are evaluated on the same asset-date universe and the same raw-return target within each market. Sharpe is computed from the actual top-minus-bottom return series; CSI 500 is daily (301 test dates) and U.S. is monthly (143 test months).

| Market | Model | Mean IC | IR | Gross H–L Sharpe | Net H–L Sharpe at 10 bps |
|---|---|---:|---:|---:|---:|
| CSI 500 | **FinPFN** | **0.0456** | **0.712** | 4.384 | -0.900 |
| CSI 500 | Ridge | 0.0374 | 0.539 | **4.889** | **1.035** |
| CSI 500 | LightGBM | 0.0364 | 0.567 | 4.810 | 0.686 |
| CSI 500 | TabPFN | -0.0378 | -0.523 | -5.193 | -10.097 |
| U.S. | **FinPFN** | **0.0665** | **0.590** | 1.040 | 0.901 |
| U.S. | Ridge | 0.0439 | 0.540 | 1.592 | 1.426 |
| U.S. | LightGBM | 0.0432 | 0.566 | **1.620** | **1.473** |
| U.S. | TabPFN | 0.0021 | 0.019 | -0.009 | -0.148 |

![Mean cross-sectional IC and IR for all four models in both markets](reproduction/public/figures/ic_ir_overview.png)

![U.S. IC versus long-short Sharpe and realized-tail precision](reproduction/public/figures/ic_portfolio_gap.png)

## Three findings

1. **The statistical result is real but only partially reproduces the headline claim.** FinPFN has the highest common-universe IC/IR in both markets; the literal CSI notebook protocol yields IR 0.797 versus 0.85 reported in the paper.
2. **Higher IC did not produce the best portfolio.** Ridge and LightGBM have higher gross long–short Sharpe in both markets. FinPFN's U.S. top-40 precision is 6.1%, below the 8% random-selection rate, despite its strong bottom-tail precision.
3. **The gap is concentrated in tradable tails and stability.** FinPFN improves broad cross-sectional ordering but has weaker extreme-long selection and lower rank persistence. Uncertainty gating did not improve validation net Sharpe, and a validation-selected turnover buffer failed its one-shot CSI test.

## Reproduce

Datasets, released checkpoints, generated predictions, and logs are intentionally excluded. First place the verified assets at the paths in [ASSETS.md](reproduction/ASSETS.md), then install the top-level requirements.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

# CSI 500: validation-selected CPU baselines
python reproduction/scripts/train_ridge.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/artifacts/predictions/csi500_baselines --seed 42
python reproduction/scripts/train_lightgbm.py \
  --dataset 30features_csi500.parquet --market csi500 \
  --output-dir reproduction/artifacts/predictions/csi500_baselines --seed 42

# CSI 500: released-checkpoint inference on one GPU
CUDA_VISIBLE_DEVICES=0 \
  bash reproduction/scripts/run_csi_checkpoint_notebook_exact.sh

# U.S.: baselines, released-checkpoint inference, and frozen evaluation
bash reproduction/next_phase/us_external_validation/scripts/run_us_baselines.sh
CUDA_VISIBLE_DEVICES=0 \
  bash reproduction/next_phase/us_external_validation/scripts/run_us_checkpoints.sh
bash reproduction/next_phase/us_external_validation/scripts/evaluate_us_common.sh
bash reproduction/next_phase/us_external_validation/scripts/run_us_full_analysis.sh

# Regenerate the two README figures and corrected tail-precision figure
python reproduction/public/make_readme_figures.py
```

Exact environment details, checksums, evaluation commands, and non-overwrite runbooks are in the [full audit report](reproduction/AUDIT_REPORT.md), [CSI checkpoint runbook](reproduction/notes/manual_checkpoint_runbook.md), and [U.S. runbook](reproduction/next_phase/us_external_validation/manual_commands.md).

## Limitations

- This is an independent checkpoint reproduction and model-risk audit, not a complete retraining of FinPFN. The exact released CSI predictions cannot be regenerated from the visible notebook because its sampling behavior differs from the published artifact.
- Point-in-time feature construction and forward-return alignment cannot be independently verified from the final parquet files. The U.S. protocol samples a new 500-stock universe each month, so its raw turnover is not a live full-universe estimate.
- Linear transaction costs omit market impact, borrowing, financing, capacity, and execution constraints. Post-test mechanism analysis is diagnostic, not an unbiased strategy backtest.
- The upstream BSD-3-Clause license is preserved. Original FinPFN code and this audit's additions are identified in [UPSTREAM_ATTRIBUTION.md](UPSTREAM_ATTRIBUTION.md); restricted data and model files are obtained from their original providers and are not redistributed here.
