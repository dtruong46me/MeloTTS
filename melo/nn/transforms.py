"""Normalizing flow transformation utilities for MeloTTS.

This module implements piecewise rational quadratic spline transforms used in
normalizing flow models. The core algorithm is based on the paper:

    "Neural Spline Flows" (Durkan et al., 2019)
    https://arxiv.org/abs/1906.04032

The transforms support both forward (density estimation) and inverse
(generation) directions and can handle unconstrained inputs via linear tails.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F


# Minimum bin width to prevent degenerate zero-width bins
DEFAULT_MIN_BIN_WIDTH = 1e-3

# Minimum bin height to prevent degenerate zero-height bins
DEFAULT_MIN_BIN_HEIGHT = 1e-3

# Minimum derivative value to ensure the spline remains monotone
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: Optional[str] = None,
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a piecewise rational quadratic spline transform.

    Dispatches to either the bounded ``rational_quadratic_spline`` or the
    unconstrained ``unconstrained_rational_quadratic_spline`` depending on
    whether ``tails`` is specified.

    Args:
        inputs: Input tensor to transform.
        unnormalized_widths: Unnormalized bin widths, shape ``[..., num_bins]``.
        unnormalized_heights: Unnormalized bin heights, shape ``[..., num_bins]``.
        unnormalized_derivatives: Unnormalized derivatives at bin knots,
            shape ``[..., num_bins + 1]``.
        inverse: If ``True``, apply the inverse transform.
        tails: Tail behavior outside the spline interval. Currently only
            ``"linear"`` is supported. If ``None``, inputs must lie within
            the unit interval ``[0, 1]``.
        tail_bound: Half-width of the interval ``[-tail_bound, tail_bound]``
            within which the spline is applied when ``tails`` is set.
        min_bin_width: Minimum width for each spline bin.
        min_bin_height: Minimum height for each spline bin.
        min_derivative: Minimum derivative value at bin knots.

    Returns:
        A tuple ``(outputs, logabsdet)`` where ``outputs`` is the transformed
        tensor and ``logabsdet`` is the log absolute determinant of the Jacobian.
    """
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet


def searchsorted(
    bin_locations: torch.Tensor,
    inputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Find bin indices for each input value via a differentiable sorted search.

    Adds a small epsilon to the last bin boundary to ensure inputs equal to the
    upper boundary are assigned to the last bin rather than going out of range.

    Args:
        bin_locations: Cumulative bin boundary tensor of shape ``[..., num_bins + 1]``.
            Modified in-place (last element incremented by ``eps``).
        inputs: Input values tensor of shape ``[...]``.
        eps: Small value added to the last bin boundary for numerical safety.

    Returns:
        Integer tensor of shape ``[...]`` with the 0-based bin index for each input.
    """
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: str = "linear",
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a rational quadratic spline with linear tails outside the interval.

    Points inside ``[-tail_bound, tail_bound]`` are transformed by the spline;
    points outside are passed through as-is (identity / linear tail).

    Args:
        inputs: Input tensor of any shape.
        unnormalized_widths: Unnormalized bin widths, shape ``[..., num_bins]``.
        unnormalized_heights: Unnormalized bin heights, shape ``[..., num_bins]``.
        unnormalized_derivatives: Unnormalized derivatives at bin knots,
            shape ``[..., num_bins - 1]``. Two boundary derivatives will be
            appended automatically.
        inverse: If ``True``, apply the inverse transform.
        tails: Tail type outside the spline interval. Only ``"linear"`` is
            currently supported.
        tail_bound: Half-width of the bounded interval ``[-tail_bound, tail_bound]``.
        min_bin_width: Minimum width for each spline bin.
        min_bin_height: Minimum height for each spline bin.
        min_derivative: Minimum derivative value at bin knots.

    Returns:
        A tuple ``(outputs, logabsdet)`` where ``outputs`` is the transformed
        tensor and ``logabsdet`` is the log absolute determinant of the Jacobian
        (zero for points in the linear tail region).

    Raises:
        RuntimeError: If ``tails`` is not ``"linear"``.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a monotone piecewise rational quadratic spline transform.

    Implements the rational quadratic spline bijection within the bounded
    domain ``[left, right] -> [bottom, top]``. Each bin of the spline is
    parameterized by its width, height, and the derivatives at the knots.

    Args:
        inputs: Input tensor. All values must lie within ``[left, right]``
            (forward) or ``[bottom, top]`` (inverse).
        unnormalized_widths: Pre-softmax bin widths, shape ``[..., num_bins]``.
        unnormalized_heights: Pre-softmax bin heights, shape ``[..., num_bins]``.
        unnormalized_derivatives: Pre-softplus derivatives at bin knots,
            shape ``[..., num_bins + 1]``.
        inverse: If ``True``, apply the inverse mapping
            (from ``[bottom, top]`` to ``[left, right]``).
        left: Left boundary of the input domain.
        right: Right boundary of the input domain.
        bottom: Bottom boundary of the output domain.
        top: Top boundary of the output domain.
        min_bin_width: Minimum width for each spline bin.
        min_bin_height: Minimum height for each spline bin.
        min_derivative: Minimum derivative value at bin knots.

    Returns:
        A tuple ``(outputs, logabsdet)`` where:
        - ``outputs``: Transformed values of the same shape as ``inputs``.
        - ``logabsdet``: Log absolute determinant of the Jacobian, same shape.
          Negative when ``inverse=True`` (consistent with change-of-variables).

    Raises:
        ValueError: If inputs are outside the valid domain, or if the minimum
            bin sizes are too large for the requested number of bins.
    """
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet
