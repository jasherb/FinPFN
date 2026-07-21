#!/usr/bin/env bash
set -euo pipefail

bash reproduction/scripts/run_csi_tabpfn_primary.sh
bash reproduction/scripts/run_csi_finpfn_primary.sh
