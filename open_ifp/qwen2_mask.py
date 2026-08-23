"""Project-local masked forward path for MLX-LM Qwen2."""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask


def masked_qwen2_mlp(mlp: Any, inputs: mx.array, mask: mx.array) -> mx.array:
    """Apply one intermediate-dimension mask after SwiGLU and before down_proj."""
    width = mlp.down_proj.weight.shape[1]
    if mask.shape != (width,):
        raise ValueError(f"mask must have shape ({width},); got {mask.shape}")
    activation = swiglu(mlp.gate_proj(inputs), mlp.up_proj(inputs))
    return mlp.down_proj(activation * mask.astype(activation.dtype))


def masked_qwen2_forward(
    model: Any,
    inputs: mx.array,
    layer_masks: mx.array,
    cache: Optional[list[Any]] = None,
    input_embeddings: Optional[mx.array] = None,
) -> mx.array:
    """Run Qwen2 with an explicit [layer, intermediate] FFN mask matrix."""
    if getattr(model, "model_type", None) != "qwen2":
        raise ValueError("masked_qwen2_forward only supports model_type='qwen2'")

    backbone = model.model
    expected_shape = (backbone.num_hidden_layers, model.args.intermediate_size)
    if layer_masks.shape != expected_shape:
        raise ValueError(
            f"layer_masks must have shape {expected_shape}; got {layer_masks.shape}"
        )

    hidden = input_embeddings if input_embeddings is not None else backbone.embed_tokens(inputs)
    if cache is None:
        cache = [None] * len(backbone.layers)
    elif len(cache) != len(backbone.layers):
        raise ValueError(
            f"cache must contain {len(backbone.layers)} layer entries; got {len(cache)}"
        )
    attention_mask = create_attention_mask(hidden, cache[0])

    for layer_index, (layer, layer_cache) in enumerate(
        zip(backbone.layers, cache, strict=True)
    ):
        attention_output = layer.self_attn(
            layer.input_layernorm(hidden),
            attention_mask,
            layer_cache,
        )
        residual = hidden + attention_output
        mlp_output = masked_qwen2_mlp(
            layer.mlp,
            layer.post_attention_layernorm(residual),
            layer_masks[layer_index],
        )
        hidden = residual + mlp_output

    output = backbone.norm(hidden)
    if model.args.tie_word_embeddings:
        return backbone.embed_tokens.as_linear(output)
    return model.lm_head(output)


class MaskedQwen2:
    """Inference adapter compatible with mlx-lm generation helpers."""

    def __init__(self, dense_model: Any, layer_masks: mx.array):
        self.dense_model = dense_model
        self.layer_masks = mx.stop_gradient(layer_masks)
        self.args = dense_model.args
        self.model_type = dense_model.model_type

    @property
    def layers(self):
        return self.dense_model.layers

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        return masked_qwen2_forward(
            self.dense_model,
            inputs,
            self.layer_masks,
            cache,
            input_embeddings,
        )
