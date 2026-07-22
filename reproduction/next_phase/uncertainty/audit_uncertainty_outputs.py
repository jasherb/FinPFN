#!/usr/bin/env python3
"""Audit saved FinPFN/TabPFN artifacts for genuine uncertainty information."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY / "reproduction/next_phase/uncertainty"
DEFAULT_INPUTS = (
    REPOSITORY
    / "reproduction/artifacts/csi500_notebook_exact/"
    "csi500_finpfn_seed42_notebook_with_replacement.parquet",
    REPOSITORY
    / "reproduction/artifacts/csi500_notebook_exact/"
    "csi500_tabpfn_seed42_notebook_with_replacement.parquet",
)
EXPECTED_SHA256 = {
    "csi500_finpfn_seed42_notebook_with_replacement.parquet": (
        "03e62d18bf14cb6a3787213a87369adf12914d65748f8d1536a7bc5cecca76f3"
    ),
    "csi500_tabpfn_seed42_notebook_with_replacement.parquet": (
        "0fa76d578741b3a50a9f6e1b96009bae6fe4f884b9ce7a3fe0f52b6cec95c26a"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    lower = {column: column.lower() for column in columns}
    return {
        "member_prediction_columns": [
            column
            for column, name in lower.items()
            if ("member" in name or "estimator" in name) and "prediction" in name
        ],
        "standard_deviation_columns": [
            column
            for column, name in lower.items()
            if "std" in name or "standard_deviation" in name
        ],
        "quantile_columns": [
            column
            for column, name in lower.items()
            if "quantile" in name or name.startswith("q") or "_q" in name
        ],
        "logit_columns": [column for column, name in lower.items() if "logit" in name],
        "distribution_columns": [
            column
            for column, name in lower.items()
            if "distribution" in name or "variance" in name or "interval" in name
        ],
        "aggregate_prediction_columns": [
            column for column in ("prediction", "prediction_mean") if column in columns
        ],
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    expected_root = (REPOSITORY / "reproduction/next_phase/uncertainty").resolve()
    if output_dir != expected_root:
        raise ValueError(f"Audit output must be {expected_root}")
    json_path = output_dir / "availability_audit.json"
    csv_path = output_dir / "availability_audit.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Refusing to overwrite an existing uncertainty audit")

    rows = []
    details = []
    for input_path in args.inputs:
        path = input_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256(path)
        expected_hash = EXPECTED_SHA256.get(path.name)
        if observed_hash != expected_hash:
            raise ValueError(f"Frozen checksum mismatch for {path.name}")
        frame = pd.read_parquet(path)
        categories = classify_columns(frame.columns.tolist())
        models = sorted(frame["model"].dropna().unique().tolist())
        if len(models) != 1:
            raise ValueError(f"Expected one model in {path.name}, got {models}")
        nonfinite = {
            column: int((~np.isfinite(pd.to_numeric(frame[column], errors="coerce"))).sum())
            for column in categories["aggregate_prediction_columns"]
        }
        metadata_path = path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        row = {
            "model": models[0],
            "rows": len(frame),
            "columns": len(frame.columns),
            "has_aggregate_median": "prediction" in frame,
            "has_aggregate_mean": "prediction_mean" in frame,
            "has_all_estimator_predictions": bool(
                categories["member_prediction_columns"]
            ),
            "has_predictive_standard_deviation": bool(
                categories["standard_deviation_columns"]
            ),
            "has_quantiles": bool(categories["quantile_columns"]),
            "has_logits": bool(categories["logit_columns"]),
            "has_distributional_outputs": bool(categories["distribution_columns"]),
            "metadata_n_estimators": metadata.get("n_estimators"),
        }
        rows.append(row)
        details.append(
            {
                **row,
                "path": path.relative_to(REPOSITORY).as_posix(),
                "sha256": observed_hash,
                "columns": frame.columns.tolist(),
                "classified_columns": categories,
                "aggregate_nonfinite_counts": nonfinite,
                "metadata_path": (
                    metadata_path.relative_to(REPOSITORY).as_posix()
                    if metadata_path.is_file()
                    else None
                ),
                "metadata_sampling_mode": metadata.get("sampling_mode"),
                "metadata_estimator_random_state": metadata.get(
                    "estimator_random_state"
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "conclusion": (
                    "Frozen artifacts contain only aggregate median and mean; they "
                    "do not contain member predictions or predictive-distribution "
                    "summaries sufficient for an uncertainty calibration audit."
                ),
                "artifacts": details,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
