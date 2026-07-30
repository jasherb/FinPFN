#!/usr/bin/env bash
set -euo pipefail

bash reproduction/analyses/uncertainty/scripts/run_csi_tabpfn_uncertainty_validation.sh
bash reproduction/analyses/uncertainty/scripts/run_csi_finpfn_uncertainty_validation.sh
