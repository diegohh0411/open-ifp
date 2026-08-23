from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step
from mlx_lm.models.qwen2 import MLP, Model, ModelArgs

from open_ifp.qwen2_mask import MaskedQwen2, masked_qwen2_forward, masked_qwen2_mlp
from open_ifp.soft_topk import soft_topk_mask


def tiny_qwen2() -> Model:
    mx.random.seed(7)
    return Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=8,
            num_hidden_layers=2,
            intermediate_size=12,
            num_attention_heads=2,
            num_key_value_heads=1,
            rms_norm_eps=1e-6,
            vocab_size=16,
            max_position_embeddings=32,
        )
    )


def test_masked_qwen2_mlp_applies_mask_after_swiglu() -> None:
    mlp = MLP(dim=2, hidden_dim=3)
    gate_weight = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    up_weight = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]], dtype=np.float32)
    down_weight = np.array([[1.0, 2.0, 4.0], [-1.0, 1.0, 0.5]], dtype=np.float32)
    mlp.gate_proj.weight = mx.array(gate_weight)
    mlp.up_proj.weight = mx.array(up_weight)
    mlp.down_proj.weight = mx.array(down_weight)
    inputs = np.array([[[1.0, 2.0]]], dtype=np.float32)
    mask = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    actual = masked_qwen2_mlp(mlp, mx.array(inputs), mx.array(mask))
    mx.eval(actual)

    gate = inputs @ gate_weight.T
    up = inputs @ up_weight.T
    post_swiglu = (gate / (1.0 + np.exp(-gate))) * up
    expected = (post_swiglu * mask) @ down_weight.T
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_all_ones_masked_forward_matches_untouched_dense_model() -> None:
    model = tiny_qwen2()
    inputs = mx.array([[1, 2, 3, 4]])
    masks = mx.ones((2, 12), dtype=mx.float32)

    dense_logits = model(inputs)
    masked_logits = masked_qwen2_forward(model, inputs, masks)
    mx.eval(dense_logits, masked_logits)

    np.testing.assert_allclose(
        np.asarray(masked_logits),
        np.asarray(dense_logits),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("shape", [(1, 12), (2, 11), (2, 12, 1)])
def test_masked_forward_rejects_wrong_per_layer_mask_shape(shape: tuple[int, ...]) -> None:
    model = tiny_qwen2()

    with pytest.raises(ValueError, match=r"layer_masks must have shape \(2, 12\)"):
        masked_qwen2_forward(model, mx.array([[1, 2]]), mx.ones(shape))


def test_mask_scores_receive_gradients_while_backbone_stays_frozen() -> None:
    model = tiny_qwen2()
    model.freeze()
    assert tree_flatten(model.trainable_parameters()) == []
    inputs = mx.array([[1, 2, 3, 4]])
    initial_scores = mx.arange(24, dtype=mx.float32).reshape(2, 12) / 24.0

    def loss(scores: mx.array) -> mx.array:
        logits = masked_qwen2_forward(model, inputs, soft_topk_mask(scores, k=6))
        return mx.mean(mx.square(logits))

    gradient = mx.grad(loss)(initial_scores)
    mx.eval(gradient)

    gradient_np = np.asarray(gradient)
    assert np.all(np.isfinite(gradient_np))
    assert np.count_nonzero(gradient_np) > 0
    assert tree_flatten(model.trainable_parameters()) == []


def test_masked_adapter_supports_deterministic_cache_aware_generation() -> None:
    model = tiny_qwen2()
    adapter = MaskedQwen2(model, mx.ones((2, 12), dtype=mx.float32))
    prompt = mx.array([1, 2, 3, 4])

    first = [
        int(token)
        for token, _ in generate_step(prompt, adapter, max_tokens=4)
    ]
    second = [
        int(token)
        for token, _ in generate_step(prompt, adapter, max_tokens=4)
    ]

    assert len(first) == 4
    assert second == first
