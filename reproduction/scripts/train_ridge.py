#!/usr/bin/env python3
from __future__ import annotations

from sklearn.linear_model import Ridge

from baseline_common import parser_for, run_baseline


def make_estimator(parameters: dict[str, float], seed: int) -> Ridge:
    del seed
    return Ridge(alpha=parameters["alpha"], fit_intercept=True, solver="lsqr", tol=1e-6)


def main() -> None:
    args = parser_for("Ridge").parse_args()
    run_baseline(
        args=args,
        model_name="Ridge",
        make_estimator=make_estimator,
        package_versions={},
    )


if __name__ == "__main__":
    main()
