#!/usr/bin/env python3
"""Data-free regression checks for the frozen public release."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
REFERENCE = REPOSITORY / "reproduction/reference_results"
MODELS = {"FinPFN", "TabPFN", "Ridge", "LightGBM"}
EXPECTED = {
    ("CSI500", "FinPFN"): (0.0455968529139515, 0.7120020930739455, 4.383558921159609),
    ("CSI500", "Ridge"): (0.0374090905011318, 0.539210391258701, 4.888952290269178),
    ("CSI500", "LightGBM"): (0.0364338702779435, 0.5665780547013394, 4.810359613592669),
    ("CSI500", "TabPFN"): (-0.037758244755383, -0.5228748039641736, -5.192668051967355),
    ("US", "FinPFN"): (0.066537335321633, 0.5901714320619875, 1.0399494884616165),
    ("US", "Ridge"): (0.043944317490418, 0.5399171534904327, 1.5915836804534569),
    ("US", "LightGBM"): (0.0432337924700908, 0.5664131521527178, 1.6195312581021046),
    ("US", "TabPFN"): (0.0021131272782614, 0.0188475405944427, -0.008742581775104),
}
EXPECTED_COSTS = {
    ("CSI500", "FinPFN", 0): (4.383558921159607, 4.383558921159607, 1.7829807915323694),
    ("CSI500", "FinPFN", 10): (4.383558921159607, -0.9000171051438332, 1.7829807915323694),
    ("CSI500", "Ridge", 0): (4.888952290269178, 4.888952290269178, 1.4788131854046738),
    ("CSI500", "Ridge", 10): (4.888952290269178, 1.0350463756423836, 1.4788131854046738),
    ("CSI500", "LightGBM", 0): (4.81035961359267, 4.81035961359267, 1.5516270498384996),
    ("CSI500", "LightGBM", 10): (4.81035961359267, 0.6856585137160246, 1.5516270498384996),
    ("CSI500", "TabPFN", 0): (-5.192668051967354, -5.192668051967354, 1.8271595772286087),
    ("CSI500", "TabPFN", 10): (-5.192668051967354, -10.096502818085476, 1.8271595772286087),
    ("US", "FinPFN", 0): (1.039949488461616, 1.039949488461616, 1.951888111888113),
    ("US", "FinPFN", 10): (1.039949488461616, 0.9013267526782681, 1.951888111888113),
    ("US", "Ridge", 0): (1.5915836804534567, 1.5915836804534567, 1.9373426573426582),
    ("US", "Ridge", 10): (1.5915836804534567, 1.4257167150737535, 1.9373426573426582),
    ("US", "LightGBM", 0): (1.6195312581021044, 1.6195312581021044, 1.9411188811188822),
    ("US", "LightGBM", 10): (1.6195312581021044, 1.473174257632844, 1.9411188811188822),
    ("US", "TabPFN", 0): (-0.008742581775104093, -0.008742581775104093, 1.9678321678321689),
    ("US", "TabPFN", 10): (-0.008742581775104093, -0.14848491941126127, 1.9678321678321689),
}
EXPECTED_US_FINPFN_TAIL = {
    10: (0.03986013986013986, 0.1251748251748252),
    20: (0.0486013986013986, 0.1590909090909091),
    40: (0.06118881118881119, 0.22430069930069932),
}
EXPECTED_FIGURE_SHA256 = {
    "ic_ir_overview.png": "481ba9168e00ce18c296b94ac45270d5b23183240dbbfdc893e41aebf3a306d7",
    "ic_portfolio_gap.png": "dcc57eeafbd70ca4083650b61dddeb1ce7a07d0a891277a4a42c4fc93a919c91",
    "us_tail_precision.png": "cc12f563a83c85ad15d6e88f49b255760f13a33c7b2c7900f4e315662e49f3ed",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: {observed} != {expected}")


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"Invalid PNG signature: {path}")
    return struct.unpack(">II", signature[16:24])


def verify_model_comparison() -> None:
    path = REFERENCE / "summary/model_comparison.csv"
    rows = read_csv(path)
    if len(rows) != 8:
        raise AssertionError(f"Expected 8 comparison rows, found {len(rows)}")
    keys = {(row["market"], row["model"]) for row in rows}
    if keys != set(EXPECTED):
        raise AssertionError(f"Unexpected market/model keys: {sorted(keys)}")
    for row in rows:
        key = (row["market"], row["model"])
        mean_ic, ir, sharpe = EXPECTED[key]
        assert_close(float(row["mean_ic"]), mean_ic, f"{key} mean IC")
        assert_close(float(row["ir"]), ir, f"{key} IR")
        assert_close(
            float(row["gross_long_short_sharpe"]),
            sharpe,
            f"{key} gross H-L Sharpe",
        )


def verify_tail_tables() -> None:
    for market in ("csi", "us"):
        rows = read_csv(REFERENCE / f"summary/{market}_tail_precision.csv")
        keys = {(row["model"], int(row["k"])) for row in rows}
        expected = {(model, k) for model in MODELS for k in (10, 20, 40)}
        if keys != expected:
            raise AssertionError(f"Unexpected {market} tail keys")
        for row in rows:
            for column in (
                "mean_actual_fraction",
                "mean_top_precision",
                "mean_bottom_precision",
            ):
                value = float(row[column])
                if not 0.0 <= value <= 1.0:
                    raise AssertionError(f"{market} {column} outside [0, 1]")
        if market == "us":
            finpfn = {int(row["k"]): row for row in rows if row["model"] == "FinPFN"}
            for k, (top_precision, bottom_precision) in EXPECTED_US_FINPFN_TAIL.items():
                assert_close(
                    float(finpfn[k]["mean_top_precision"]),
                    top_precision,
                    f"US FinPFN top-{k} precision",
                )
                assert_close(
                    float(finpfn[k]["mean_bottom_precision"]),
                    bottom_precision,
                    f"US FinPFN bottom-{k} precision",
                )


def verify_cost_tables() -> None:
    observed: dict[tuple[str, str, int], tuple[float, float, float]] = {}

    for row in read_csv(REFERENCE / "summary/csi_cost_sensitivity.csv"):
        if row["portfolio"] != "long_short":
            continue
        cost = int(row["cost_rate_bps_per_one_way_turnover"])
        if cost not in (0, 10):
            continue
        observed[("CSI500", row["model"], cost)] = (
            float(row["gross_sharpe"]),
            float(row["net_sharpe"]),
            float(row["average_turnover_one_way_including_initial"]),
        )

    for row in read_csv(REFERENCE / "summary/us_cost_sensitivity.csv"):
        cost = int(row["cost_bps_per_one_way_turnover"])
        if cost not in (0, 10):
            continue
        observed[("US", row["model"], cost)] = (
            float(row["gross_long_short_sharpe"]),
            float(row["net_long_short_sharpe"]),
            float(row["average_total_one_way_turnover_including_entry"]),
        )

    if set(observed) != set(EXPECTED_COSTS):
        raise AssertionError("Unexpected market/model/cost keys")
    for key, expected in EXPECTED_COSTS.items():
        for label, value, target in zip(("gross Sharpe", "net Sharpe", "turnover"), observed[key], expected):
            assert_close(value, target, f"{key} {label}")


def verify_frozen_selection() -> None:
    path = REFERENCE / "summary/turnover_control_selected.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["selection_status"] != "frozen_before_test":
        raise AssertionError("Turnover-control selection is not frozen")
    if record["selected_candidate"]["name"] != "rank_buffer_exit20pct":
        raise AssertionError("Unexpected selected turnover-control candidate")


def verify_figures() -> None:
    for name, expected_sha256 in EXPECTED_FIGURE_SHA256.items():
        path = REFERENCE / "figures" / name
        width, height = png_dimensions(path)
        if width < 800 or height < 500:
            raise AssertionError(f"Figure is unexpectedly small: {name}")
        observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_sha256 != expected_sha256:
            raise AssertionError(f"Unexpected figure checksum: {name}")


def verify_required_documents() -> None:
    required = [
        REPOSITORY / "README.md",
        REPOSITORY / "LICENSE",
        REPOSITORY / "UPSTREAM_ATTRIBUTION.md",
        REPOSITORY / "reproduction/REPORT.md",
        REPOSITORY / "reproduction/REPRODUCIBILITY.md",
        REPOSITORY / "reproduction/ASSETS.md",
    ]
    missing = [path.relative_to(REPOSITORY).as_posix() for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"Missing release documents: {missing}")


def main() -> None:
    verify_required_documents()
    verify_model_comparison()
    verify_tail_tables()
    verify_cost_tables()
    verify_frozen_selection()
    verify_figures()
    print("Release metric verification passed.")


if __name__ == "__main__":
    main()
