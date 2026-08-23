from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from open_ifp.soft_topk import realize_k, soft_topk_mask, soft_topk_probabilities


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.4, 1946), (0.6, 2918), (0.8, 3891), (1.0, 4864)],
)
def test_realize_k_uses_protocol_round_half_up(fraction: float, expected: int) -> None:
    assert realize_k(4864, fraction) == expected


def test_soft_topk_selects_exactly_the_highest_k_scores() -> None:
    scores = mx.array([[0.0, 1.0, 2.0, 3.0]], dtype=mx.float32)

    probabilities = soft_topk_probabilities(scores, k=2)
    mask = soft_topk_mask(scores, k=2)
    mx.eval(probabilities, mask)

    probabilities_np = np.asarray(probabilities)
    mask_np = np.asarray(mask)
    np.testing.assert_allclose(probabilities_np.sum(axis=-1), [2.0], atol=1e-5)
    assert np.all((probabilities_np >= 0.0) & (probabilities_np <= 1.0))
    np.testing.assert_array_equal(np.flatnonzero(mask_np[0]), [2, 3])
    np.testing.assert_allclose(mask_np[0, 2:], probabilities_np[0, 2:])


def test_soft_topk_full_budget_is_exactly_all_ones() -> None:
    scores = mx.array([[3.0, -2.0, 0.5, 9.0]], dtype=mx.bfloat16)

    probabilities = soft_topk_probabilities(scores, k=4)
    mask = soft_topk_mask(scores, k=4)
    mx.eval(probabilities, mask)

    np.testing.assert_array_equal(np.asarray(probabilities), np.ones((1, 4)))
    np.testing.assert_array_equal(np.asarray(mask), np.ones((1, 4)))
    assert probabilities.dtype == mx.float32
    assert mask.dtype == mx.float32


def test_soft_topk_rejects_out_of_range_k() -> None:
    scores = mx.zeros((2, 4))

    with pytest.raises(ValueError, match="1 <= k <= 4"):
        soft_topk_mask(scores, k=0)
    with pytest.raises(ValueError, match="1 <= k <= 4"):
        soft_topk_mask(scores, k=5)


def test_soft_topk_keeps_nonzero_gradients_for_selected_scores() -> None:
    coefficients = mx.array([[1.0, -2.0, 3.0, -4.0]], dtype=mx.float32)

    def loss(scores: mx.array) -> mx.array:
        return mx.sum(soft_topk_mask(scores, k=2) * coefficients)

    gradient = mx.grad(loss)(mx.array([[0.0, 1.0, 2.0, 3.0]], dtype=mx.float32))
    mx.eval(gradient)

    gradient_np = np.asarray(gradient)
    assert np.all(np.isfinite(gradient_np))
    assert np.count_nonzero(gradient_np) > 0
