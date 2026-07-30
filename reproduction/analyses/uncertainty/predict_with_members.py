"""TabPFN 2.0.8 regression prediction with non-aggregated member summaries.

This module mirrors TabPFNRegressor.predict from the pinned 2.0.8 package. It uses
the already fitted estimator's executor and does not alter model weights,
preprocessing configurations, temperatures, borders, or ensemble aggregation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tabpfn
import torch
from sklearn.base import check_is_fitted
from tabpfn.preprocessing import RegressorEnsembleConfig
from tabpfn.regressor import TabPFNRegressor, _logits_to_output
from tabpfn.utils import (
    _fix_dtypes,
    _process_text_na_dataframe,
    _transform_borders_one,
    translate_probs_across_borders,
    validate_X_predict,
)


PINNED_TABPFN_VERSION = "2.0.8"
DEFAULT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def predict_with_members(
    estimator: TabPFNRegressor,
    X: np.ndarray,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[str, Any]:
    """Return standard aggregate outputs plus every member's mean and median.

    Member order is the order yielded by the fitted 2.0.8 inference executor.
    Downstream dispersion statistics are permutation invariant.
    """
    if tabpfn.__version__ != PINNED_TABPFN_VERSION:
        raise RuntimeError(
            f"This compatibility shim requires tabpfn=={PINNED_TABPFN_VERSION}; "
            f"found {tabpfn.__version__}"
        )
    check_is_fitted(estimator)
    if estimator.n_estimators < 2:
        raise ValueError("At least two estimators are required for member dispersion")

    validated = validate_X_predict(X, estimator)
    validated = _fix_dtypes(
        validated, cat_indices=estimator.categorical_features_indices
    )
    validated = _process_text_na_dataframe(
        validated, ord_encoder=estimator.preprocessor_
    )

    standard_borders = estimator.bardist_.borders.cpu().numpy()
    transformed_probabilities: list[torch.Tensor] = []
    config_descriptions: list[str] = []

    for output, config in estimator.executor_.iter_outputs(
        validated,
        device=estimator.device_,
        autocast=estimator.use_autocast_,
    ):
        if not isinstance(config, RegressorEnsembleConfig):
            raise TypeError(f"Unexpected ensemble configuration: {type(config)}")
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Unexpected estimator output: {type(output)}")
        if estimator.softmax_temperature != 1:
            output = output.float() / estimator.softmax_temperature

        if config.target_transform is None:
            transformed_borders = standard_borders.copy()
            logit_cancel_mask = None
            descending_borders = False
        else:
            logit_cancel_mask, descending_borders, transformed_borders = (
                _transform_borders_one(
                    standard_borders,
                    target_transform=config.target_transform,
                    repair_nan_borders_after_transform=(
                        estimator.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM
                    ),
                )
            )
            # This is the same branch as TabPFNRegressor.predict 2.0.8. The default
            # target transforms used in this reproduction have not triggered it.
            if descending_borders:
                transformed_borders = np.flip(transformed_borders).copy()

        if logit_cancel_mask is not None:
            output = output.clone()
            output[..., logit_cancel_mask] = float("-inf")

        probabilities = translate_probs_across_borders(
            output,
            frm=torch.as_tensor(transformed_borders, device=estimator.device_),
            to=estimator.bardist_.borders.to(estimator.device_),
        )
        transformed_probabilities.append(probabilities)
        config_descriptions.append(repr(config))

    if len(transformed_probabilities) != estimator.n_estimators:
        raise RuntimeError(
            f"Expected {estimator.n_estimators} members, got "
            f"{len(transformed_probabilities)}"
        )

    stacked_probabilities = torch.stack(transformed_probabilities, dim=0)
    if estimator.average_before_softmax:
        aggregate_probabilities = (
            stacked_probabilities.log().mean(dim=0).softmax(dim=-1)
        )
    else:
        aggregate_probabilities = stacked_probabilities.mean(dim=0)

    aggregate_logits = aggregate_probabilities.log()
    if aggregate_logits.dtype == torch.float16:
        aggregate_logits = aggregate_logits.float()
    aggregate_logits = aggregate_logits.cpu()

    criterion = estimator.renormalized_criterion_
    quantile_list = list(quantiles)
    aggregate_mean = _logits_to_output(
        output_type="mean",
        logits=aggregate_logits,
        criterion=criterion,
        quantiles=quantile_list,
    )
    aggregate_median = _logits_to_output(
        output_type="median",
        logits=aggregate_logits,
        criterion=criterion,
        quantiles=quantile_list,
    )
    aggregate_mode = _logits_to_output(
        output_type="mode",
        logits=aggregate_logits,
        criterion=criterion,
        quantiles=quantile_list,
    )
    aggregate_quantiles = _logits_to_output(
        output_type="quantiles",
        logits=aggregate_logits,
        criterion=criterion,
        quantiles=quantile_list,
    )
    predictive_variance = criterion.variance(aggregate_logits).clamp_min(0.0)

    member_means = []
    member_medians = []
    for probabilities in transformed_probabilities:
        member_logits = probabilities.log()
        if member_logits.dtype == torch.float16:
            member_logits = member_logits.float()
        member_logits = member_logits.cpu()
        member_means.append(_numpy(criterion.mean(member_logits)))
        member_medians.append(_numpy(criterion.median(member_logits)))

    return {
        "mean": np.asarray(aggregate_mean),
        "median": np.asarray(aggregate_median),
        "mode": np.asarray(aggregate_mode),
        "quantiles": [np.asarray(values) for values in aggregate_quantiles],
        "predictive_std": _numpy(predictive_variance.sqrt()),
        "member_means": np.column_stack(member_means),
        "member_medians": np.column_stack(member_medians),
        "member_config_descriptions": config_descriptions,
        "aggregate_logits_shape": list(aggregate_logits.shape),
    }


def verify_against_public_predict(
    estimator: TabPFNRegressor,
    X: np.ndarray,
    member_output: dict[str, Any],
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    absolute_tolerance: float = 2e-6,
) -> dict[str, float]:
    """Compare the compatibility output with TabPFN's public aggregate API."""
    reference = estimator.predict(X, output_type="full", quantiles=list(quantiles))
    comparisons = {
        "mean": float(np.max(np.abs(member_output["mean"] - reference["mean"]))),
        "median": float(
            np.max(np.abs(member_output["median"] - reference["median"]))
        ),
        "mode": float(np.max(np.abs(member_output["mode"] - reference["mode"]))),
    }
    for index, quantile in enumerate(quantiles):
        comparisons[f"quantile_{quantile:.2f}"] = float(
            np.max(
                np.abs(
                    member_output["quantiles"][index]
                    - np.asarray(reference["quantiles"][index])
                )
            )
        )
    maximum = max(comparisons.values())
    comparisons["maximum_absolute_difference"] = maximum
    if maximum > absolute_tolerance:
        raise AssertionError(
            f"Member shim differs from public predict by {maximum}, "
            f"above tolerance {absolute_tolerance}"
        )
    return comparisons
