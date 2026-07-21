# Official parquet data audit

The files were located automatically at the repository root and inspected in place.
They were not moved, renamed, or modified. SHA-256 values are recorded in
`reproduction/configs/data_checksums.sha256`; all parquet files are ignored by Git.

## Dataset summary

| Dataset | Size | Rows | Dates | Assets | Model features | Date range |
|---|---:|---:|---:|---:|---:|---|
| CSI 500 panel | 263,138,068 bytes | 1,039,372 | 1,762 | 1,057 | 30 | 2016-01-05 to 2023-04-03 |
| U.S. panel | 1,576,270,925 bytes | 3,529,899 | 715 | 28,160 | 90 | 1962-06-30 to 2021-12-31 |
| CSI 500 index price | 110,935 bytes | 2,118 | 2,118 | 1 symbol | 6 price/volume fields | 2016-01-04 to 2024-09-18 |

Both equity panels have chronologically non-decreasing physical row order, unique
asset-date pairs, zero duplicate full rows implied by that uniqueness, and no null,
NaN, or infinite values. The CSI index-price file is unique and complete but stored
in reverse chronological order.

## CSI 500 panel

- Columns: `date`, `id`, `target`, and the 30 paper features.
- Rows per date: minimum 464, median 607, maximum 626.
- The 30 features are already cross-sectionally centered and sample-standardized on
  every date: median cross-sectional mean is numerically zero and every feature has
  date-wise sample standard deviation 1.
- `target` is not standardized. It is a decimal return with observed range
  approximately -4.97% to 6.81% and median date-wise standard deviation 1.81%.
- The extrema coincide with sampled 1st/99th percentiles, consistent with clipping,
  but the exact upstream winsorization procedure cannot be proven from final data.

Official-code half-open splits:

| Split | Rows | Dates | Actual dates present |
|---|---:|---:|---|
| Train | 703,709 | 1,217 | 2016-01-05 to 2020-12-31 |
| Validation | 149,450 | 243 | 2021-01-04 to 2021-12-31 |
| Test | 186,213 | 302 | 2022-01-04 to 2023-04-03 |

Adjacent-date inference produces query dates beginning 2022-01-05, which explains
the bundled prediction file's 301 dates. That file contains exactly 500 predictions
per date, covering roughly 80.3%-81.8% of available test assets.

The bundled CSV's `return` column is not the raw parquet return; its standard
deviation is near one. All 150,500 prediction rows match a parquet asset-date pair.
Correct portfolio evaluation must merge `target` from this parquet and multiply by
100 only for percentage-point reporting. The CSV's group-standardized `target`
remains suitable only for reproducing the released notebook's historical IC logic.

## U.S. panel

- Physical columns: `date`, `id`, `target`, 90 model features, and the saved pandas
  column `__index_level_0__`. The saved index must not be used as a feature.
- Rows per date: minimum 502, median 5,395, maximum 8,739.
- `target` is already a percentage return, not a decimal return. Its observed range
  is -37.931 to 58.375 and its median date-wise standard deviation is 12.90.
- Continuous features are predominantly cross-sectional rank transforms bounded by
  -1 and 1, with standard deviation near `1/sqrt(3)` rather than z-scores.
- Eight categorical features are unscaled, matching the paper's categorical count:
  `convind`, `divi`, `divo`, `ms`, `nincr`, `rd`, `securedind`, and `sin`.
- This artifact preprocessing differs from the paper's generic statement that
  numerical features are standardized; artifact-faithful baselines will consume the
  stored feature values unchanged.

Official-code half-open splits:

| Split | Rows | Dates | Actual dates present |
|---|---:|---:|---|
| Train | 1,944,085 | 451 | 1962-06-30 to 1999-12-31 |
| Validation | 797,222 | 120 | 2000-01-31 to 2009-12-31 |
| Test | 788,592 | 144 | 2010-01-31 to 2021-12-31 |

The first stored date is 1962-06-30, whereas the paper reports 1962-06-29. The
released artifact date governs this reproduction.

## Timing and point-in-time limitations

The paper states that Chinese features use information through the close of `t-1`
and predict the Barra-adjusted open-to-open return from `t` to `t+1`. It states that
U.S. monthly, quarterly, and annual variables are lagged by one, four, and six months.
The final parquets contain no source timestamps or raw return legs, so neither claim
can be independently verified from these files. No direct target leakage is visible
in feature distributions, but forward-return alignment and point-in-time availability
remain unverified provenance assumptions.

## Regime-data availability

The official paper defines China shocks using a one-day CIMV increase greater than
2.5 with level above 20, followed by three trading days. It defines U.S. shocks using
a monthly VIX increase of at least 10 with level above 30, followed by three months.
Neither CIMV nor VIX is included in the downloaded data. The notebook does provide
seven hard-coded China evaluation windows, so those can be reproduced exactly. U.S.
official regime dates cannot be reconstructed without an additional volatility-index
source and will not be invented silently.
