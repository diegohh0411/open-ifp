from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mlx.core as mx

from scripts.verify_qwen2_mask import (
    as_float32_numpy,
    build_random_scores,
    canonical_write_json,
    verification_spec,
)


ROOT = Path(__file__).resolve().parents[2]


def test_verification_spec_uses_protocol_qwen_dimensions_and_budget_grid() -> None:
    protocol = {
        "model": {"num_hidden_layers": 24, "intermediate_size": 4864},
        "training_data": {"seed": 20260821},
    }

    spec = verification_spec(protocol)

    assert spec == {
        "seed": 20260821,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "budgets": [
            {"fraction": 0.4, "k": 1946},
            {"fraction": 0.6, "k": 2918},
            {"fraction": 0.8, "k": 3891},
            {"fraction": 1.0, "k": 4864},
        ],
    }


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


def test_readme_documents_the_day1b_verification_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m scripts.verify_qwen2_mask" in readme
    assert "results/day1b/qwen2-mask-verification.json" in readme
