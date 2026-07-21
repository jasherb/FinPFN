#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finpfn-matplotlib")
)

import lightgbm
from lightgbm import LGBMRegressor

from baseline_common import parser_for, run_baseline


def make_estimator(parameters: dict[str, Any], seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        **parameters,
        objective="regression",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def main() -> None:
    args = parser_for("LightGBM").parse_args()
    run_baseline(
        args=args,
        model_name="LightGBM",
        make_estimator=make_estimator,
        package_versions={"lightgbm": lightgbm.__version__},
    )


if __name__ == "__main__":
    main()
