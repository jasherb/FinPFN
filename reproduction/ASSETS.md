# Data and checkpoint acquisition

This repository does not redistribute datasets, released model checkpoints, generated predictions, or the upstream result bundle. Obtain them from their original providers under the applicable access terms, place them at the paths below, and verify their SHA-256 hashes before running the audit.

The original FinPFN authors provide the data-access and checkpoint instructions in the [upstream repository](https://github.com/wangy8989/FinPFN). The TabPFN v2 checkpoint is distributed through the official [TabPFN project](https://github.com/PriorLabs/TabPFN). Access credentials are intentionally not copied into this repository.

| Asset | Expected local path | SHA-256 |
|---|---|---|
| CSI 500 index-price panel | `000905_XSHG_price.parquet` | `cca753a54ea557bb8a75ff11eb5a8b20ca29c7ed5512be33ff9432e7771a008e` |
| CSI 500 feature panel | `30features_csi500.parquet` | `9e0d61f5d70151d4f2f7b40918a8ddb79f86fb54a0fe86759f5c1f2869fe1b3e` |
| U.S. feature panel | `90features_USstocks.parquet` | `54818c78796ecae3974b2058575cd2284482ce35e62c9116d316e23240b8ef50` |
| CSI 500 FinPFN checkpoint | `models/finpfn_30feats_csi500.ckpt` | `c035f2a79c74ab7f38b023fa98624d078b6389c3d096ac1a1270b04361dd0214` |
| U.S. FinPFN checkpoint | `models/finpfn_90feats_us.ckpt` | `493e2bd458618f2ddac97da754c3f23abc61a93baa95ae127636a918d3ba7a8f` |
| TabPFN v2 regressor checkpoint | `models/tabpfn-v2-regressor.ckpt` | `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736` |
| TabPFN v2 classifier checkpoint | `models/tabpfn-v2-classifier.ckpt` | `f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f` |
| Optional upstream CSI result bundle | `results/finpfn_perf_csi500.csv.gz` | `a39f91cc7f967982d8ee81471a75539d0d2f47d07feede86f7bf5e3097a10172` |

Verify individual files on Linux:

```bash
sha256sum 30features_csi500.parquet
sha256sum 90features_USstocks.parquet
sha256sum models/*.ckpt
```

On macOS, replace `sha256sum` with `shasum -a 256`. The complete declared checksum files are [data_checksums.sha256](configs/data_checksums.sha256) and [checksums.sha256](configs/checksums.sha256).

Generated parquets, fitted baseline models, logs, and full evaluation
directories remain ignored. The compact aggregate CSVs and figures under
`reproduction/reference_results/` contain no source feature panel or model
weights.
