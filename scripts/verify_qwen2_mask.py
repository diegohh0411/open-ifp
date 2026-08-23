#!/usr/bin/env python3
"""Verify the Day 1B Qwen2 FFN SoftTopK mask path on the pinned model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import yaml
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.generate import generate_step

from open_ifp.qwen2_mask import MaskedQwen2, masked_qwen2_forward
from open_ifp.soft_topk import realize_k, soft_topk_mask


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocol/benchmark-v0.1.yaml"
DEFAULT_OUTPUT = ROOT / "results/day1b/qwen2-mask-verification.json"
DEFAULT_PROMPT = "In one sentence, what is instruction-following pruning?"
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2


def verification_spec(protocol: dict[str, Any]) -> dict[str, Any]:
    """Extract the frozen dimensions, seed, and realized budget grid."""
    layers = int(protocol["model"]["num_hidden_layers"])
    width = int(protocol["model"]["intermediate_size"])
    seed = int(protocol["training_data"]["seed"])
    p6 = protocol["p6"]
    if p6["rounding"] != "round_half_up":
        raise AssertionError("Day 1B requires protocol round_half_up")
    fractions = [float(fraction) for fraction in p6["activation_grid"]]
    realized = [int(k) for k in p6["realized_dimensions_for_qwen2_0_5b"]]
    dense_fraction = float(p6["dense_reference_fraction"])
    gradient_fraction = float(p6["day3_fixed_fraction"])
    computed = [realize_k(width, fraction) for fraction in fractions]
    if realized != computed:
        raise AssertionError(
            f"protocol realized dimensions {realized} do not match computed {computed}"
        )
    if dense_fraction != 1.0 or dense_fraction not in fractions:
        raise AssertionError(
            "protocol dense reference fraction must be the 1.0 activation budget"
        )
    if gradient_fraction not in fractions:
        raise AssertionError("protocol gradient fraction must be in activation_grid")
    return {
        "seed": seed,
        "num_hidden_layers": layers,
        "intermediate_size": width,
        "dense_fraction": dense_fraction,
        "gradient_fraction": gradient_fraction,
        "generation_fractions": [
            fraction for fraction in fractions if fraction != dense_fraction
        ],
        "budgets": [
            {"fraction": fraction, "k": k}
            for fraction, k in zip(fractions, realized, strict=True)
        ],
    }


def build_random_scores(
    *, seed: int, num_layers: int, intermediate_size: int
) -> np.ndarray:
    """Create the fixed random per-layer score matrix without global RNG state."""
    generator = np.random.default_rng(seed)
    return generator.standard_normal(
        (num_layers, intermediate_size), dtype=np.float32
    )


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return (encoded + "\n").encode("utf-8")


def canonical_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable UTF-8 JSON suitable for byte-for-byte rerun comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(payload))


def verify_or_write_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    """Create first-run evidence, or enforce exact committed evidence thereafter."""
    expected = _canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != expected:
            raise AssertionError(
                f"fresh verification does not match committed evidence at {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def as_float32_numpy(array: mx.array) -> np.ndarray:
    """Cross the MLX/NumPy boundary after normalizing unsupported BF16."""
    return np.asarray(array.astype(mx.float32))


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_snapshot_hashes(
    snapshot: Path, expected_hashes: dict[str, str]
) -> dict[str, str]:
    """Hash every protocol-pinned model/tokenizer payload before loading it."""
    actual: dict[str, str] = {}
    for filename, expected in sorted(expected_hashes.items()):
        path = snapshot / filename
        if not path.is_file():
            raise AssertionError(f"pinned snapshot file is missing: {filename}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise AssertionError(
                f"pinned snapshot hash mismatch for {filename}: {digest} != {expected}"
            )
        actual[filename] = digest
    return actual


def _mask_record(mask: mx.array, expected_k: int) -> dict[str, Any]:
    mask_np = np.asarray(mask, dtype=np.float32)
    support = (mask_np != 0).astype(np.uint8)
    per_layer = support.sum(axis=1).astype(int).tolist()
    return {
        "per_layer_nonzero_counts": per_layer,
        "all_layers_exact_k": all(count == expected_k for count in per_layer),
        "support_sha256": _sha256(support),
        "values_sha256": _sha256(mask_np),
    }


def _greedy_token_ids(
    model: MaskedQwen2, prompt_tokens: list[int], max_tokens: int
) -> list[int]:
    prompt = mx.array(prompt_tokens, dtype=mx.int32)
    return [
        int(token)
        for token, _ in generate_step(prompt, model, max_tokens=max_tokens)
    ]


def _dense_equivalence(
    model: Any, prompt_tokens: list[int], full_mask: mx.array
) -> dict[str, Any]:
    tokens = mx.array([prompt_tokens], dtype=mx.int32)
    dense_logits = model(tokens)
    masked_logits = masked_qwen2_forward(model, tokens, full_mask)
    mx.eval(dense_logits, masked_logits)
    dense_np = as_float32_numpy(dense_logits)
    masked_np = as_float32_numpy(masked_logits)
    difference = np.abs(masked_np - dense_np)
    denominator = np.maximum(np.abs(dense_np), 1e-6)
    passed = bool(
        np.allclose(masked_np, dense_np, atol=BF16_ATOL, rtol=BF16_RTOL)
    )
    if not passed:
        raise AssertionError("all-ones masked logits differ from the dense reference")
    return {
        "atol": BF16_ATOL,
        "rtol": BF16_RTOL,
        "passed": passed,
        "max_absolute_error": float(difference.max(initial=0.0)),
        "max_relative_error": float((difference / denominator).max(initial=0.0)),
    }


def _gradient_record(
    model: Any,
    prompt_tokens: list[int],
    scores: np.ndarray,
    k: int,
    fraction: float,
) -> dict[str, Any]:
    model.freeze()
    trainable_before = tree_flatten(model.trainable_parameters())
    gradient_tokens = mx.array([prompt_tokens[-4:]], dtype=mx.int32)

    def loss(score_array: mx.array) -> mx.array:
        masks = soft_topk_mask(score_array, k=k)
        logits = masked_qwen2_forward(model, gradient_tokens, masks)
        return mx.mean(mx.square(logits[:, -1, :].astype(mx.float32)))

    gradient = mx.grad(loss)(mx.array(scores))
    mx.eval(gradient)
    gradient_np = np.asarray(gradient, dtype=np.float32)
    per_layer_nonzero = np.count_nonzero(gradient_np, axis=1).astype(int).tolist()
    finite = bool(np.all(np.isfinite(gradient_np)))
    nonzero = int(np.count_nonzero(gradient_np))
    trainable_after = tree_flatten(model.trainable_parameters())
    passed = finite and nonzero > 0 and not trainable_before and not trainable_after
    if not passed:
        raise AssertionError("mask-score gradient or frozen-backbone check failed")
    return {
        "budget_fraction": fraction,
        "k": k,
        "finite": finite,
        "nonzero_count": nonzero,
        "per_layer_nonzero_counts": per_layer_nonzero,
        "l2_norm": float(np.linalg.norm(gradient_np)),
        "backbone_trainable_parameter_count_before": len(trainable_before),
        "backbone_trainable_parameter_count_after": len(trainable_after),
        "passed": passed,
    }


def run_verification(
    protocol_path: Path,
    output_path: Path,
    *,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    spec = verification_spec(protocol)
    model_protocol = protocol["model"]
    snapshot = Path(
        snapshot_download(
            repo_id=model_protocol["repository"],
            revision=model_protocol["revision"],
            local_files_only=True,
        )
    )
    if snapshot.name != model_protocol["revision"]:
        raise AssertionError("resolved model snapshot does not match pinned revision")
    snapshot_hashes = verify_snapshot_hashes(snapshot, model_protocol["files_sha256"])

    model, tokenizer = load(snapshot)
    if model.model_type != "qwen2":
        raise AssertionError(f"expected qwen2 model, got {model.model_type!r}")
    if len(model.layers) != spec["num_hidden_layers"]:
        raise AssertionError("loaded layer count does not match protocol")
    if model.args.intermediate_size != spec["intermediate_size"]:
        raise AssertionError("loaded intermediate size does not match protocol")

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if tokenizer.chat_template is not None
        else prompt
    )
    prompt_tokens = tokenizer.encode(formatted_prompt)
    scores_np = build_random_scores(
        seed=spec["seed"],
        num_layers=spec["num_hidden_layers"],
        intermediate_size=spec["intermediate_size"],
    )
    scores = mx.array(scores_np)

    masks: dict[float, mx.array] = {}
    budget_records: list[dict[str, Any]] = []
    for budget in spec["budgets"]:
        fraction = budget["fraction"]
        k = budget["k"]
        mask = soft_topk_mask(scores, k=k)
        repeat_mask = soft_topk_mask(mx.array(scores_np), k=k)
        mx.eval(mask, repeat_mask)
        record = {**budget, **_mask_record(mask, k)}
        repeat_record = _mask_record(repeat_mask, k)
        record["mask_deterministic"] = (
            record["support_sha256"] == repeat_record["support_sha256"]
            and record["values_sha256"] == repeat_record["values_sha256"]
        )
        if not record["all_layers_exact_k"] or not record["mask_deterministic"]:
            raise AssertionError(f"mask verification failed for fraction {fraction}")
        masks[fraction] = mask
        budget_records.append(record)

    budget_records_by_fraction = {
        record["fraction"]: record for record in budget_records
    }
    dense_fraction = spec["dense_fraction"]
    gradient_fraction = spec["gradient_fraction"]
    dense_equivalence = _dense_equivalence(
        model,
        prompt_tokens,
        masks[dense_fraction],
    )
    gradient = _gradient_record(
        model,
        prompt_tokens,
        scores_np,
        budget_records_by_fraction[gradient_fraction]["k"],
        gradient_fraction,
    )

    for fraction in spec["generation_fractions"]:
        adapter = MaskedQwen2(model, masks[fraction])
        first = _greedy_token_ids(adapter, prompt_tokens, max_tokens)
        second = _greedy_token_ids(adapter, prompt_tokens, max_tokens)
        deterministic = first == second
        if not deterministic:
            raise AssertionError(f"generation was nondeterministic at fraction {fraction}")
        record = budget_records_by_fraction[fraction]
        record["generation"] = {
            "deterministic": deterministic,
            "token_ids": first,
            "text": tokenizer.decode(first),
        }

    payload = {
        "artifact_version": "day1b-qwen2-mask-v1",
        "status": "pass",
        "model": {
            "repository": model_protocol["repository"],
            "revision": model_protocol["revision"],
            "cache_snapshot": model_protocol["cache_snapshot"],
            "model_type": model.model_type,
            "num_hidden_layers": spec["num_hidden_layers"],
            "intermediate_size": spec["intermediate_size"],
            "snapshot_integrity": {
                "verified": True,
                "files_sha256": snapshot_hashes,
            },
        },
        "runtime": {
            "mlx": importlib.metadata.version("mlx"),
            "mlx-lm": importlib.metadata.version("mlx-lm"),
        },
        "seed": spec["seed"],
        "prompt": prompt,
        "formatted_prompt_sha256": hashlib.sha256(
            formatted_prompt.encode("utf-8")
        ).hexdigest(),
        "prompt_token_ids": prompt_tokens,
        "max_generated_tokens": max_tokens,
        "soft_topk": {
            "algorithm": "entropy_regularized_coordinate_descent_with_exact_support",
            "iterations": 20,
            "initial_epsilon": 4.0,
            "epsilon": 0.03,
            "epsilon_decay": 0.7,
            "score_dtype": "float32",
            "mask_axis": "post_swiglu_pre_down_proj_intermediate",
        },
        "dense_equivalence": dense_equivalence,
        "gradient": gradient,
        "budgets": budget_records,
    }
    verify_or_write_canonical_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the pinned Qwen2 FFN SoftTopK mask path"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    payload = run_verification(
        args.protocol,
        args.output,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    print(f"verified: {payload['status']} -> {args.output}")


if __name__ == "__main__":
    main()
