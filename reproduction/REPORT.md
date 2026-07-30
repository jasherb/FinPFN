# FinPFN Reproduction and Model-Risk Audit

## Executive summary

This project reproduces released FinPFN and vanilla TabPFN checkpoint inference,
independently reconstructs Ridge and LightGBM baselines, and evaluates all four
models on common raw-return universes in CSI 500 and U.S. equities.

FinPFN has the strongest full-cross-section IC/IR in both markets. That
statistical advantage does not translate into the strongest top-minus-bottom
portfolio: Ridge and LightGBM have higher gross Sharpe in both tests. The most
consistent mechanism is an IC-portfolio gap. FinPFN accumulates broad ranking
improvements, but its extreme-long selection and temporal rank stability are
weaker. Costs amplify the CSI shortfall. Ensemble dispersion predicts some
asset-level error, but uncertainty gating adds no validation value, and a
validation-selected turnover buffer fails its single frozen test.

The research decision is therefore to stop tuning on these test sets and retain
the work as a reproduction and model-risk audit.

## Experimental design

### Markets and splits

| Market | Train | Validation | Test | Common evaluation |
|---|---|---|---|---:|
| CSI 500 | 2016-2020 | 2021 | 2022 to 2023-04 | 301 dates, 120,620 asset-dates |
| U.S. | 1962-1999 | 2000-2009 | 2010-2021 | 143 months, 71,500 asset-dates |

Adjacent dates form context/query tasks. CSI uses 30 already standardized
features and decimal-return targets. The U.S. panel has 90 model features and
percentage-return targets; a stored index column is excluded.

### Models and selection

- FinPFN and TabPFN use released checkpoints, eight estimators, seed 42, and no
  retraining.
- Ridge has five predeclared alphas. LightGBM has six predeclared
  configurations. Selection uses validation mean date-wise Spearman IC only.
- The selected baseline is refit on train plus validation and evaluated once on
  test. Test results do not change candidates or parameters.
- Within each market, every model is compared on one common asset-date universe
  and one raw-return target. Repeated CSI task rows are averaged by
  model/date/asset before ranking.

### Metrics and portfolio construction

IC is date-wise Spearman correlation between predictions and the common raw
return. IC standard deviation uses `ddof=1`, and IR is mean IC divided by that
standard deviation.

Portfolios are equal-weight predicted top and bottom deciles. The primary
Sharpe is calculated from the actual top-minus-bottom return series. The
notebook's difference between separate leg Sharpes is retained only as a
secondary artifact-faithful diagnostic.

## Core results

| Market | Model | Mean IC | IC SD | IR | Gross H-L Sharpe | 10 bps net Sharpe |
|---|---|---:|---:|---:|---:|---:|
| CSI 500 | FinPFN | 0.04560 | 0.06404 | 0.7120 | 4.3836 | -0.9000 |
| CSI 500 | Ridge | 0.03741 | 0.06938 | 0.5392 | 4.8890 | 1.0350 |
| CSI 500 | LightGBM | 0.03643 | 0.06431 | 0.5666 | 4.8104 | 0.6857 |
| CSI 500 | TabPFN | -0.03776 | 0.07221 | -0.5229 | -5.1927 | -10.0965 |
| U.S. | FinPFN | 0.06654 | 0.11274 | 0.5902 | 1.0399 | 0.9013 |
| U.S. | Ridge | 0.04394 | 0.08139 | 0.5399 | 1.5916 | 1.4257 |
| U.S. | LightGBM | 0.04323 | 0.07633 | 0.5664 | 1.6195 | 1.4732 |
| U.S. | TabPFN | 0.00211 | 0.11212 | 0.0188 | -0.0087 | -0.1485 |

CSI Sharpe uses 240 daily periods for the release comparison; U.S. Sharpe uses
12 monthly periods. The cost column charges 10 basis points per unit of total
one-way long-plus-short turnover.

## Reproduction fidelity

The literal visible CSI notebook protocol produces 195,550 rows per checkpoint,
including 74,930 repeated task rows, with no failed groups or nonfinite
predictions. FinPFN reaches notebook-style IR 0.7973 versus approximately 0.85
reported by the paper.

An exact released-prediction match is not possible from the public notebook:

- the notebook samples stocks with replacement, while the released prediction
  file contains exactly 500 unique stocks per date;
- the released comparison mixes within-task standardized checkpoint targets
  with a full-date baseline target;
- the released artifact does not preserve the complete sampling and grouping
  state required to reconstruct each prediction.

The headline table therefore uses the stricter common-universe, common-target
evaluation. It reproduces FinPFN's statistical ranking advantage, but not a
portfolio-performance advantage over Ridge and LightGBM.

## Transaction costs and turnover

The predeclared grid is 0, 2, 5, 10, 20, 30, and 50 bps per unit of one-way
turnover. For each leg,

```text
turnover[t] = 0.5 * sum(abs(weight[t] - weight[t-1]))
net_return[t] = gross_return[t] - cost_rate * total_turnover[t]
```

The first portfolio date charges entry from cash; long and short legs are
added. CSI FinPFN average total one-way turnover is 1.7830, versus 1.4788 for
Ridge and 1.5516 for LightGBM. Its mean-return break-even cost is 8.30 bps,
below Ridge's 12.68 and LightGBM's 11.66 bps.

The U.S. protocol samples a new 500-stock universe each month. Forced universe
changes account for approximately 91% one-way turnover per month for every
model. Stored-holdings P&L is internally consistent, but raw U.S. turnover must
not be interpreted as deployment turnover on a fixed full-market universe.

## Uncertainty audit and turnover controls

Validation-only member outputs show that FinPFN predictive interval width has
information about asset-level error. Pooled Spearman correlation is 0.228 with
absolute cross-sectional z-error and 0.105 with absolute rank error. It does
not reliably predict date-level IC deterioration, and dispersion is strongly
confounded with prediction extremeness. It is therefore described as ensemble
or predictive dispersion, not calibrated posterior uncertainty.

Fifteen strategies were predeclared and selected on CSI validation at a fixed
10 bps cost:

- the best uncertainty strategy has validation net Sharpe 1.237, below
  unmodified FinPFN at 1.551;
- a turnover-only rank buffer reaches validation net Sharpe 1.938 and is frozen;
- on its only test evaluation, the buffer reduces turnover from 1.783 to 1.618
  but lowers net Sharpe from -0.900 to -1.040.

No replacement threshold or test-driven search was performed.

## Explaining the IC-portfolio gap

| Market | Model | Middle 20-80% IC | Top-40 precision | Bottom-40 precision |
|---|---|---:|---:|---:|
| CSI 500 | FinPFN | 0.0240 | 0.090 | 0.202 |
| CSI 500 | Ridge | 0.0153 | 0.149 | 0.190 |
| CSI 500 | LightGBM | 0.0187 | 0.156 | 0.203 |
| U.S. | FinPFN | 0.0400 | 0.061 | 0.224 |
| U.S. | Ridge | 0.0112 | 0.100 | 0.148 |
| U.S. | LightGBM | 0.0207 | 0.140 | 0.198 |

The evidence supports four linked observations:

1. FinPFN accumulates many small ordering improvements across the cross-section.
2. Decile portfolios depend on a much smaller extreme subset. FinPFN's
   extreme-long precision is materially lower than the baselines in both
   markets.
3. CSI top/bottom membership persistence is lower for FinPFN, creating
   additional turnover.
4. In the U.S. sample, FinPFN is strong in the bottom tail but has below-random
   top-40 precision, limiting the long-short portfolio.

This mechanism analysis is exploratory because it uses an already observed test
period. It explains the frozen result and does not justify additional tuning.

## Integrity checks

- Each model forms deciles from its own prediction column, never from targets or
  a shared ranking.
- Reconstructed holdings match frozen evaluator holdings with zero membership
  discrepancies.
- Checkpoint inference has zero failed groups and zero nonfinite predictions in
  both markets.
- Ridge and LightGBM's similar CSI Sharpe is not file reuse: prediction
  correlations, holdings overlap, and return paths are materially different.
- U.S. predictions contain 71,500 unique asset-date rows per model over the same
  143 months; recomputed IC and return-series metrics match the evaluator to
  numerical tolerance.
- Candidate grids, selected configurations, seeds, inputs, and hashes are stored
  in machine-readable files.

## Compute

Released-checkpoint inference used one NVIDIA A100 80 GB and four CPU workers.
The notebook-exact CSI runs took approximately 34 minutes per model; U.S. runs
took approximately 18 minutes per model. The verified GPU environment used
Python 3.10, PyTorch 2.5.1 with CUDA 12.1, and TabPFN 2.0.8. These figures are
benchmarks, not resource guarantees.

## Limitations

- Final parquet panels do not expose source timestamps or raw return legs, so
  point-in-time availability and exact forward-return alignment cannot be
  independently proven.
- CSI checkpoint outputs depend on sampling/group state missing from the
  original release.
- The paper's exact baseline fitting implementation is not public; Ridge and
  LightGBM are transparent validation-selected reconstructions.
- One principal checkpoint sampling seed is available per market.
- Linear costs omit impact, borrow availability, financing, price limits,
  capacity, and execution delay.
- Cost, tail, and mechanism analyses on test data are diagnostic unless a method
  was explicitly selected on validation and frozen before test.

## Research decision

Cross-market evidence supports ending method development on the existing CSI
and U.S. test sets. A future strategy study would require untouched validation
and test data, a fixed or complete tradable universe, and a predeclared
tail-aware objective.

Machine-readable results are indexed in
[reference_results/README.md](reference_results/README.md). Full commands and
the validation/test boundary are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
