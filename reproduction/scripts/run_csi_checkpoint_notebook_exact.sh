#!/usr/bin/env bash
set -euo pipefail

bash reproduction/scripts/run_csi_tabpfn_notebook_exact.sh
bash reproduction/scripts/run_csi_finpfn_notebook_exact.sh
