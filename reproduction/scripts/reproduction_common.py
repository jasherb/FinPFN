#!/usr/bin/env python3
"""Shared, artifact-faithful dataset conventions for reproduction scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MARKET_CONFIG = {
    "csi500": {
        "split1": pd.Timestamp("2021-01-01"),
        "split2": pd.Timestamp("2022-01-01"),
        "annualization": 240,
        "return_to_percentage_points": 100.0,
        "checkpoint": "models/finpfn_30feats_csi500.ckpt",
        "expected_features": 30,
    },
    "us": {
        "split1": pd.Timestamp("2000-01-01"),
        "split2": pd.Timestamp("2010-01-01"),
        "annualization": 12,
        "return_to_percentage_points": 1.0,
        "checkpoint": "models/finpfn_90feats_us.ckpt",
        "expected_features": 90,
    },
}

NON_FEATURE_COLUMNS = {"date", "id", "target", "__index_level_0__"}
PREDICTION_COLUMNS = [
    "market",
    "model",
    "seed",
    "date",
    "id",
    "context_date",
    "group_id",
    "prediction",
    "prediction_mean",
    "target_group_z",
    "status",
    "inference_seconds",
]


def feature_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column not in NON_FEATURE_COLUMNS]


def validate_panel(frame: pd.DataFrame, market: str) -> list[str]:
    required = {"date", "id", "target"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    features = feature_columns(frame.columns.tolist())
    expected = MARKET_CONFIG[market]["expected_features"]
    if len(features) != expected:
        raise ValueError(
            f"Expected {expected} {market} features, found {len(features)}"
        )
    return features


def load_panel(
    path: Path,
    market: str,
    *,
    split: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load an official parquet panel without changing stored values."""
    config = MARKET_CONFIG[market]
    filters = None
    if split == "train":
        filters = [("date", "<", config["split1"])]
    elif split == "validation":
        filters = [
            ("date", ">=", config["split1"]),
            ("date", "<", config["split2"]),
        ]
    elif split == "test":
        filters = [("date", ">=", config["split2"])]
    elif split not in {None, "train_validation"}:
        raise ValueError(f"Unknown split: {split}")
    elif split == "train_validation":
        filters = [("date", "<", config["split2"])]

    frame = pd.read_parquet(path, columns=columns, filters=filters)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "id"], kind="stable").reset_index(drop=True)
    return frame


def split_frame(frame: pd.DataFrame, market: str) -> dict[str, pd.DataFrame]:
    config = MARKET_CONFIG[market]
    return {
        "train": frame.loc[frame["date"] < config["split1"]],
        "validation": frame.loc[
            (frame["date"] >= config["split1"])
            & (frame["date"] < config["split2"])
        ],
        "test": frame.loc[frame["date"] >= config["split2"]],
    }
