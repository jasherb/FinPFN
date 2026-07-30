# Frozen reference results

These files are the compact, immutable evidence for the portfolio release.
They are not default output directories.

```text
reference_results/
  summary/      # headline cross-market, cost, uncertainty, and overlay tables
  detailed/     # per-period IC/returns and supporting diagnostics without raw features
  diagnostics/  # validation-only uncertainty calibration tables
  figures/      # public charts generated from committed aggregate tables
```

Raw feature panels, checkpoints, fitted models, predictions, holdings, logs, and
asset-level contribution files are excluded. New runs write to the ignored
`reproduction/runs/` directory.

The principal cross-market table is
[`summary/model_comparison.csv`](summary/model_comparison.csv). Figures can be
regenerated from the committed tables into the ignored `reproduction/runs/`
area with:

```bash
python3 reproduction/scripts/make_public_figures.py
```

Release verification checks the expected columns, row counts, headline values,
and figure presence without requiring proprietary assets:

```bash
python3 reproduction/tests/verify_release.py
```
