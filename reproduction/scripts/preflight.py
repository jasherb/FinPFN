#!/usr/bin/env python3
"""Verify external assets and optional GPU compatibility before a full run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
ASSETS = {
    "csi500_dataset": (
        "30features_csi500.parquet",
        "9e0d61f5d70151d4f2f7b40918a8ddb79f86fb54a0fe86759f5c1f2869fe1b3e",
        {"csi", "full"},
    ),
    "us_dataset": (
        "90features_USstocks.parquet",
        "54818c78796ecae3974b2058575cd2284482ce35e62c9116d316e23240b8ef50",
        {"us", "full"},
    ),
    "csi500_finpfn_checkpoint": (
        "models/finpfn_30feats_csi500.ckpt",
        "c035f2a79c74ab7f38b023fa98624d078b6389c3d096ac1a1270b04361dd0214",
        {"csi", "full"},
    ),
    "us_finpfn_checkpoint": (
        "models/finpfn_90feats_us.ckpt",
        "493e2bd458618f2ddac97da754c3f23abc61a93baa95ae127636a918d3ba7a8f",
        {"us", "full"},
    ),
    "tabpfn_regressor_checkpoint": (
        "models/tabpfn-v2-regressor.ckpt",
        "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736",
        {"csi", "us", "full"},
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["csi", "us", "full"], default="full")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Also require the verified TabPFN/PyTorch versions and a CUDA device.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_gpu_environment() -> None:
    import tabpfn
    import torch

    if tabpfn.__version__ != "2.0.8":
        raise RuntimeError(f"Expected tabpfn 2.0.8, found {tabpfn.__version__}")
    if torch.__version__ != "2.5.1+cu121":
        raise RuntimeError(f"Expected torch 2.5.1+cu121, found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the current environment")
    print(f"GPU environment: torch={torch.__version__}, tabpfn={tabpfn.__version__}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")


def main() -> None:
    args = parse_args()
    checked = 0
    failures: list[str] = []
    for name, (relative, expected, modes) in ASSETS.items():
        if args.mode not in modes:
            continue
        path = REPOSITORY / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        observed = sha256(path)
        if observed != expected:
            failures.append(f"checksum mismatch for {relative}")
            continue
        checked += 1
        print(f"OK {name}: {relative}")

    if failures:
        formatted = "\n".join(f"- {failure}" for failure in failures)
        raise SystemExit(f"Preflight failed:\n{formatted}")

    if args.require_gpu:
        verify_gpu_environment()
    else:
        try:
            version = importlib.metadata.version("tabpfn")
        except importlib.metadata.PackageNotFoundError:
            version = "not installed in this environment"
        print(f"TabPFN: {version}; GPU check skipped")

    print(f"Preflight passed for mode={args.mode} ({checked} assets).")


if __name__ == "__main__":
    main()
