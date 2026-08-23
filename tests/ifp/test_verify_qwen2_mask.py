from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from scripts.verify_qwen2_mask import (
    as_float32_numpy,
    build_random_scores,
    canonical_write_json,
    verify_or_write_canonical_json,
    verify_snapshot_hashes,
    verification_spec,
)


ROOT = Path(__file__).resolve().parents[2]


def test_verification_spec_uses_protocol_qwen_dimensions_and_budget_grid() -> None:
    protocol = {
        "model": {"num_hidden_layers": 24, "intermediate_size": 4864},
        "training_data": {"seed": 20260821},
        "p6": {
            "activation_grid": [0.4, 0.6, 0.8, 1.0],
            "rounding": "round_half_up",
            "realized_dimensions_for_qwen2_0_5b": [1946, 2918, 3891, 4864],
            "day3_fixed_fraction": 0.6,
            "dense_reference_fraction": 1.0,
        },
    }

    spec = verification_spec(protocol)

    assert spec == {
        "seed": 20260821,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "dense_fraction": 1.0,
        "gradient_fraction": 0.6,
        "generation_fractions": [0.4, 0.6, 0.8],
        "budgets": [
            {"fraction": 0.4, "k": 1946},
            {"fraction": 0.6, "k": 2918},
            {"fraction": 0.8, "k": 3891},
            {"fraction": 1.0, "k": 4864},
        ],
    }


def test_verification_spec_rejects_protocol_realized_dimension_drift() -> None:
    protocol = {
        "model": {"num_hidden_layers": 24, "intermediate_size": 4864},
        "training_data": {"seed": 20260821},
        "p6": {
            "activation_grid": [0.4, 0.6, 0.8, 1.0],
            "rounding": "round_half_up",
            "realized_dimensions_for_qwen2_0_5b": [1946, 2918, 3890, 4864],
            "day3_fixed_fraction": 0.6,
            "dense_reference_fraction": 1.0,
        },
    }

    with pytest.raises(AssertionError, match="realized dimensions"):
        verification_spec(protocol)


def test_verification_spec_derives_budget_roles_from_protocol() -> None:
    protocol = {
        "model": {"num_hidden_layers": 2, "intermediate_size": 20},
        "training_data": {"seed": 17},
        "p6": {
            "activation_grid": [0.5, 0.75, 1.0],
            "rounding": "round_half_up",
            "realized_dimensions_for_qwen2_0_5b": [10, 15, 20],
            "day3_fixed_fraction": 0.75,
            "dense_reference_fraction": 1.0,
        },
    }

    spec = verification_spec(protocol)

    assert spec["dense_fraction"] == 1.0
    assert spec["gradient_fraction"] == 0.75
    assert spec["generation_fractions"] == [0.5, 0.75]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dense_reference_fraction", 0.9, "dense reference fraction"),
        ("day3_fixed_fraction", 0.7, "gradient fraction"),
    ],
)
def test_verification_spec_rejects_budget_roles_outside_grid(
    field: str, value: float, message: str
) -> None:
    p6 = {
        "activation_grid": [0.4, 0.6, 0.8, 1.0],
        "rounding": "round_half_up",
        "realized_dimensions_for_qwen2_0_5b": [1946, 2918, 3891, 4864],
        "day3_fixed_fraction": 0.6,
        "dense_reference_fraction": 1.0,
    }
    p6[field] = value
    protocol = {
        "model": {"num_hidden_layers": 24, "intermediate_size": 4864},
        "training_data": {"seed": 20260821},
        "p6": p6,
    }

    with pytest.raises(AssertionError, match=message):
        verification_spec(protocol)


def test_random_scores_are_fixed_by_seed() -> None:
    first = build_random_scores(seed=17, num_layers=3, intermediate_size=5)
    second = build_random_scores(seed=17, num_layers=3, intermediate_size=5)
    different = build_random_scores(seed=18, num_layers=3, intermediate_size=5)

    assert first.shape == (3, 5)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(second, first)
    assert not np.array_equal(different, first)


def test_mlx_bfloat16_is_cast_before_numpy_conversion() -> None:
    values = mx.array([1.0, -2.0], dtype=mx.bfloat16)

    converted = as_float32_numpy(values)

    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, np.array([1.0, -2.0], dtype=np.float32))


def test_canonical_json_writer_is_byte_deterministic(tmp_path) -> None:
    path = tmp_path / "verification.json"
    payload = {"z": [2, 1], "a": "mask"}

    canonical_write_json(path, payload)
    first = path.read_bytes()
    canonical_write_json(path, payload)

    assert path.read_bytes() == first
    assert first.endswith(b"\n")
    assert json.loads(first) == payload


def test_existing_artifact_is_an_oracle_not_silently_overwritten(tmp_path) -> None:
    path = tmp_path / "verification.json"
    payload = {"status": "pass", "token_ids": [1, 2, 3]}
    canonical_write_json(path, payload)
    expected_bytes = path.read_bytes()

    verify_or_write_canonical_json(path, payload)
    with pytest.raises(AssertionError, match="does not match committed evidence"):
        verify_or_write_canonical_json(path, {**payload, "token_ids": [9]})

    assert path.read_bytes() == expected_bytes


def test_snapshot_payload_hashes_are_verified(tmp_path) -> None:
    model = tmp_path / "model.safetensors"
    tokenizer = tmp_path / "tokenizer.json"
    model.write_bytes(b"weights")
    tokenizer.write_bytes(b"tokens")
    expected = {
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        "tokenizer.json": hashlib.sha256(b"tokens").hexdigest(),
    }

    assert verify_snapshot_hashes(tmp_path, expected) == expected
    model.write_bytes(b"substituted")
    with pytest.raises(AssertionError, match=r"model\.safetensors"):
        verify_snapshot_hashes(tmp_path, expected)


def test_committed_artifact_pins_cross_run_oracles() -> None:
    artifact = json.loads(
        (ROOT / "results/day1b/qwen2-mask-verification.json").read_text(
            encoding="utf-8"
        )
    )

    sparse = artifact["budgets"][:3]
    assert [record["support_sha256"] for record in sparse] == [
        "49dc68f86fafa5171543018779d8e1d758b694f0a9a10f726124ab30f6f1b27d",
        "aa8afe857dd8f02ee2d4d15dc8c89f190df390b7bbc85d5f142c48cf72891f8f",
        "887e5df32bebd01ccfdcc66d9c71bf15794797784d96f884c1eb100536b9cce2",
    ]
    assert [record["generation"]["token_ids"] for record in sparse] == [
        [101065, 101065, 101065, 102357, 102357, 102357, 102357, 102357],
        [304, 279, 5290, 279, 304, 279, 5290, 151643],
        [54974, 92585, 287, 85192, 374, 264, 4647, 1483],
    ]


def test_readme_documents_the_day1b_verification_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m scripts.verify_qwen2_mask" in readme
    assert "results/day1b/qwen2-mask-verification.json" in readme
