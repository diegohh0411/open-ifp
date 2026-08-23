"""Differentiable exact-support SoftTopK masks for P6."""

from __future__ import annotations

import math

import mlx.core as mx


def realize_k(intermediate_size: int, fraction: float) -> int:
    """Convert an activation fraction to k using protocol round-half-up."""
    if intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    return math.floor(intermediate_size * fraction + 0.5)


def _validate_k(scores: mx.array, k: int) -> int:
    if scores.ndim == 0:
        raise ValueError("scores must have at least one dimension")
    width = scores.shape[-1]
    if not 1 <= k <= width:
        raise ValueError(f"k must satisfy 1 <= k <= {width}; got {k}")
    return width


def soft_topk_probabilities(
    scores: mx.array,
    k: int,
    *,
    iterations: int = 20,
    epsilon: float = 0.03,
    initial_epsilon: float = 4.0,
    epsilon_decay: float = 0.7,
) -> mx.array:
    """Normalize scores to [0, 1] weights whose final-axis sum is k.

    This is the entropy-regularized coordinate-descent normalization from
    Lei et al. (2023), including their epsilon-scaling defaults for text.
    Computation stays in float32 for numerical stability.
    """
    width = _validate_k(scores, k)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if epsilon <= 0.0 or initial_epsilon < epsilon:
        raise ValueError("epsilon must be positive and no larger than initial_epsilon")
    if not 0.0 < epsilon_decay < 1.0:
        raise ValueError("epsilon_decay must satisfy 0 < epsilon_decay < 1")

    scores = scores.astype(mx.float32)
    if k == width:
        return mx.ones_like(scores)

    dual_b = mx.zeros_like(scores)
    current_epsilon = initial_epsilon
    log_k = math.log(k)
    for _ in range(iterations):
        current_epsilon = max(epsilon, current_epsilon * epsilon_decay)
        dual_a = current_epsilon * (
            log_k
            - mx.logsumexp(
                (scores + dual_b) / current_epsilon,
                axis=-1,
                keepdims=True,
            )
        )
        dual_b = mx.minimum(-scores - dual_a, 0.0)

    probabilities = mx.exp((scores + dual_a + dual_b) / current_epsilon)

    # A finite coordinate-descent solve can end just below the equality
    # constraint after b clips values to one. Fill the small residual across
    # remaining capacity. This affine correction preserves ordering and bounds.
    residual = k - mx.sum(probabilities, axis=-1, keepdims=True)
    capacity = mx.sum(1.0 - probabilities, axis=-1, keepdims=True)
    return probabilities + residual * (1.0 - probabilities) / capacity


def soft_topk_mask(scores: mx.array, k: int) -> mx.array:
    """Return Hou's lambda * Top(lambda, k) mask with exactly k nonzeros."""
    width = _validate_k(scores, k)
    probabilities = soft_topk_probabilities(scores, k)
    if k == width:
        return probabilities

    selected = mx.argpartition(-probabilities, kth=k - 1, axis=-1)[..., :k]
    selected = mx.stop_gradient(selected)
    support = mx.any(
        selected[..., :, None] == mx.arange(width)[None, ...],
        axis=-2,
    )
    return probabilities * support.astype(probabilities.dtype)
