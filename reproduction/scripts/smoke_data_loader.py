#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

for variable_name in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(variable_name, "4")

import numpy as np
import pandas as pd

repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))

from scripts.training_utils.data_utils import create_data


def synthetic_panel() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    rows = []
    for date in pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]):
        for stock_id in range(60):
            rows.append(
                {
                    "date": date,
                    "id": stock_id,
                    "feature_1": rng.standard_normal(),
                    "feature_2": rng.standard_normal(),
                    "target": rng.standard_normal(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    frame = synthetic_panel()
    expected_batches = {False: 2, True: 4}
    for training in [False, True]:
        context_x, query_x, context_y, query_y = create_data(
            data_set=frame,
            seq_len=100,
            train=training,
            date_style="consecutive",
        )
        batches = expected_batches[training]
        assert tuple(context_x.shape) == (50, batches, 2)
        assert tuple(query_x.shape) == (50, batches, 2)
        assert tuple(context_y.shape) == (50, batches, 1)
        assert tuple(query_y.shape) == (50, batches, 1)
        max_abs_mean = float(
            np.abs(context_y.squeeze(-1).mean(dim=0).numpy()).max()
        )
        assert max_abs_mean < 1e-6
        print(
            f"train_mode={training} batches={batches} "
            f"context_shape={tuple(context_x.shape)} "
            f"query_shape={tuple(query_x.shape)} "
            f"context_target_mean_max_abs={max_abs_mean:.3e}"
        )


if __name__ == "__main__":
    main()
