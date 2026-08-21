from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def load_protocol() -> dict[str, object]:
    with (ROOT / "protocol/benchmark-v0.1.yaml").open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded

EXPECTED_RUNTIME = {
    "mlx": "0.31.2",
    "mlx-lm": "0.31.3",
    "huggingface_hub": "1.27.0",
}
EXPECTED_DEV = {
    "PyYAML": "6.0.3",
    "jsonschema": "4.26.0",
    "pytest": "9.1.1",
}
FORBIDDEN = {"torch", "torchvision", "torchtune"}


def read_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"direct dependency is not equality-pinned: {line}"
        name, version = line.split("==", maxsplit=1)
        pins[name] = version
    return pins


def test_runtime_dependencies_are_exact() -> None:
    assert read_pins(ROOT / "requirements.txt") == EXPECTED_RUNTIME


def test_dev_dependencies_are_exact() -> None:
    assert read_pins(ROOT / "requirements-dev.txt") == EXPECTED_DEV


def test_forbidden_runtime_dependencies_are_absent() -> None:
    names = {name.lower() for name in read_pins(ROOT / "requirements.txt")}
    assert names.isdisjoint(FORBIDDEN)


def test_setup_installs_runtime_and_dev_requirements() -> None:
    setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert "python -m pip install -r requirements.txt -r requirements-dev.txt" in setup


def test_generation_resolves_the_immutable_cached_snapshot() -> None:
    generate = (ROOT / "scripts/generate.py").read_text(encoding="utf-8")
    assert 'DEFAULT_MODEL_REVISION = "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"' in generate
    assert "revision=args.revision" in generate
    assert "local_files_only=True" in generate
    assert "load(snapshot_path)" in generate


def test_protocol_freezes_immutable_sources() -> None:
    protocol = load_protocol()
    assert protocol["protocol_version"] == "0.1.0"
    assert protocol["model"]["repository"] == "Qwen/Qwen2-0.5B-Instruct"
    assert protocol["model"]["revision"] == "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    assert protocol["model"]["tokenizer_revision"] == protocol["model"]["revision"]
    assert protocol["training_data"]["revision"] == "feb6109c23dc5bb14eaea059d14b9879284c9234"
    assert protocol["evaluation"]["revision"] == "13ec2c53411ad214f13709a2fcc1c1b730c605ff"
    assert all(
        SHA40.fullmatch(revision)
        for revision in (
            protocol["model"]["revision"],
            protocol["training_data"]["revision"],
            protocol["evaluation"]["revision"],
        )
    )


def test_protocol_freezes_data_counts() -> None:
    data = load_protocol()["training_data"]
    assert data["categories"] == [
        "classification",
        "summarization",
        "information_extraction",
        "brainstorming",
    ]
    assert data["train_per_category"] == 100
    assert data["held_out_per_category"] == 25
    assert len(data["categories"]) * data["train_per_category"] == 400
    assert len(data["categories"]) * data["held_out_per_category"] == 100


def test_protocol_freezes_selection_and_serialization_algorithms() -> None:
    protocol = load_protocol()
    data = protocol["training_data"]
    assert data["canonicalization"] == {
        "encoding": "utf-8",
        "sort_keys": True,
        "json_separators": [",", ":"],
        "hash_prefix": "20260821\n",
        "line_endings": "lf",
    }
    assert data["selection"] == {
        "sort": "sha256_ascending_within_category",
        "train_slice": [0, 100],
        "held_out_slice": [100, 125],
    }
    assert data["serialization"]["context_separator"] == "\n\nContext:\n"
    evaluation = protocol["evaluation"]
    assert evaluation["selection"]["tie_break"] == "prompt_sha256_ascending"
    assert evaluation["selection"]["fill"] == "prompt_sha256_ascending"
    assert evaluation["manifest_fields"] == ["original_key", "prompt_sha256", "instruction_ids"]


def test_protocol_keeps_effective_update_size_constant() -> None:
    profiles = load_protocol()["training"]["sequence_profiles"]
    assert profiles == {
        256: {"microbatch_size": 1, "gradient_accumulation_steps": 8},
        512: {"microbatch_size": 1, "gradient_accumulation_steps": 4},
    }
    for length, profile in profiles.items():
        assert length * profile["microbatch_size"] * profile["gradient_accumulation_steps"] == 2048


def test_protocol_freezes_p6_and_p8_budgets() -> None:
    protocol = load_protocol()
    assert protocol["p6"]["activation_grid"] == [0.4, 0.6, 0.8, 1.0]
    assert protocol["p6"]["realized_dimensions_for_qwen2_0_5b"] == [1946, 2918, 3891, 4864]
    assert protocol["p6"]["day3_fixed_fraction"] == 0.6
    assert protocol["p6"]["day3_train_tokens"] == 250_000
    assert protocol["p8"]["train_tokens"] == 250_000
    assert protocol["p8"]["lora_rank"] == 8
    assert protocol["p8"]["qlora_rank"] == 8
    assert protocol["p8"]["qlora_quantization_bits"] == 4


def test_protocol_declares_every_required_metric() -> None:
    measurements = {
        item["name"]: (item["unit"], item["definition"])
        for item in load_protocol()["measurements"]
    }
    assert measurements == {
        "training_wall_clock_seconds": ("seconds", "training_only_excluding_evaluation"),
        "train_tokens": ("tokens", "non_padding_tokens"),
        "train_tokens_per_second": ("tokens_per_second", "train_tokens_divided_by_wall_clock"),
        "step_time_p50_ms": ("milliseconds", "measured_steps_only"),
        "step_time_p95_ms": ("milliseconds", "measured_steps_only"),
        "mlx_peak_memory_bytes": ("bytes", "mlx_allocator_peak"),
        "mlx_active_memory_bytes": ("bytes", "mlx_allocator_active"),
        "mlx_cache_memory_bytes": ("bytes", "mlx_allocator_cache"),
        "os_peak_rss_bytes": ("bytes", "process_rss_sampled_each_second"),
        "memory_free_percent_min": ("percent", "memory_pressure_q_minimum"),
        "swap_used_start_bytes": ("bytes", "sysctl_vm_swapusage_at_start"),
        "swap_used_end_bytes": ("bytes", "sysctl_vm_swapusage_at_end"),
        "swap_delta_bytes": ("bytes", "end_minus_start"),
        "checkpoint_size_bytes": ("bytes", "recursive_regular_file_sum"),
        "held_out_nll": ("nats_per_token", "assistant_token_negative_log_likelihood"),
        "held_out_tokens": ("tokens", "evaluated_assistant_tokens"),
        "ifeval_strict_prompt_accuracy": ("ratio", "official_strict_prompt_accuracy"),
        "ifeval_strict_instruction_accuracy": ("ratio", "official_strict_instruction_accuracy"),
        "ifeval_loose_prompt_accuracy": ("ratio", "official_loose_prompt_accuracy"),
        "ifeval_loose_instruction_accuracy": ("ratio", "official_loose_instruction_accuracy"),
    }


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert isinstance(loaded, dict)
    return loaded


def schema_validator() -> Draft202012Validator:
    schema = load_json(ROOT / "protocol/run-result.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def dense_example() -> dict[str, object]:
    return load_json(ROOT / "protocol/examples/dense-baseline.json")


def test_dense_example_validates_against_schema() -> None:
    schema_validator().validate(dense_example())


@pytest.mark.parametrize(
    "required_key",
    [
        "schema_version",
        "protocol_version",
        "run_id",
        "status",
        "provenance",
        "hardware",
        "config",
        "metrics",
        "evaluation",
        "artifacts",
        "deviation_ids",
        "failure",
    ],
)
def test_schema_rejects_missing_top_level_contract(required_key: str) -> None:
    result = dense_example()
    del result[required_key]
    errors = list(schema_validator().iter_errors(result))
    assert errors, f"missing {required_key} unexpectedly validated"


def test_schema_rejects_negative_bytes() -> None:
    result = dense_example()
    result["metrics"]["os_peak_rss_bytes"] = -1
    assert list(schema_validator().iter_errors(result))


def test_schema_rejects_ratio_above_one() -> None:
    result = dense_example()
    result["evaluation"]["ifeval_strict_prompt_accuracy"] = 1.01
    assert list(schema_validator().iter_errors(result))


def test_schema_rejects_moving_source_revision() -> None:
    result = dense_example()
    result["provenance"]["model"]["revision"] = "main"
    assert list(schema_validator().iter_errors(result))


def test_schema_rejects_unsupported_method() -> None:
    result = dense_example()
    result["config"]["method"] = "combined_p6_p8"
    assert list(schema_validator().iter_errors(result))


def test_schema_requires_failure_details_for_failed_run() -> None:
    result = dense_example()
    result["status"] = "failed"
    result["failure"] = None
    assert list(schema_validator().iter_errors(result))


def test_schema_enforces_method_specific_groups() -> None:
    result = dense_example()
    result["config"]["method"] = "p8_lora"
    result["config"]["p8"] = {"regime": "lora", "rank": 8, "quantization_bits": None}
    result["config"]["p6"] = {
        "activation_fraction": 0.6,
        "activation_grid": [0.4, 0.6, 0.8, 1.0],
        "realized_dimensions_per_layer": [2918] * 24,
        "mean_activation_fraction": 0.6,
    }
    assert list(schema_validator().iter_errors(result))


def test_schema_requires_p8_group_for_p8_method() -> None:
    result = dense_example()
    result["config"]["method"] = "p8_lora"
    assert list(schema_validator().iter_errors(result))


@pytest.mark.parametrize(
    ("method", "p6", "p8"),
    [
        ("dense_baseline", None, None),
        ("p6_random_mask", {"activation_fraction": 0.6, "activation_grid": [0.4, 0.6, 0.8, 1.0], "realized_dimensions_per_layer": [2918] * 24, "mean_activation_fraction": 0.6}, None),
        ("p6_static_mask", {"activation_fraction": 0.6, "activation_grid": [0.4, 0.6, 0.8, 1.0], "realized_dimensions_per_layer": [2918] * 24, "mean_activation_fraction": 0.6}, None),
        ("p6_learned_fixed_k", {"activation_fraction": 0.6, "activation_grid": [0.4, 0.6, 0.8, 1.0], "realized_dimensions_per_layer": [2918] * 24, "mean_activation_fraction": 0.6}, None),
        ("p6_variable_k", {"activation_fraction": None, "activation_grid": [0.4, 0.6, 0.8, 1.0], "realized_dimensions_per_layer": [2918] * 24, "mean_activation_fraction": 0.6}, None),
        ("p8_full", None, {"regime": "full", "rank": None, "quantization_bits": None}),
        ("p8_lora", None, {"regime": "lora", "rank": 8, "quantization_bits": None}),
        ("p8_qlora", None, {"regime": "qlora", "rank": 8, "quantization_bits": 4}),
    ],
)
def test_schema_accepts_each_method_group(method: str, p6: object, p8: object) -> None:
    result = dense_example()
    result["config"]["method"] = method
    result["config"]["p6"] = p6
    result["config"]["p8"] = p8
    schema_validator().validate(result)


@pytest.mark.parametrize(
    ("group", "field"),
    [
        ("metrics", "training_wall_clock_seconds"),
        ("metrics", "train_tokens"),
        ("metrics", "train_tokens_per_second"),
        ("metrics", "step_time_p50_ms"),
        ("metrics", "step_time_p95_ms"),
        ("metrics", "mlx_peak_memory_bytes"),
        ("metrics", "mlx_active_memory_bytes"),
        ("metrics", "mlx_cache_memory_bytes"),
        ("metrics", "os_peak_rss_bytes"),
        ("metrics", "memory_free_percent_min"),
        ("metrics", "swap_used_start_bytes"),
        ("metrics", "swap_used_end_bytes"),
        ("metrics", "swap_delta_bytes"),
        ("metrics", "checkpoint_size_bytes"),
        ("evaluation", "held_out_nll"),
        ("evaluation", "held_out_tokens"),
        ("evaluation", "ifeval_strict_prompt_accuracy"),
        ("evaluation", "ifeval_strict_instruction_accuracy"),
        ("evaluation", "ifeval_loose_prompt_accuracy"),
        ("evaluation", "ifeval_loose_instruction_accuracy"),
        ("artifacts", "raw_step_times"),
        ("artifacts", "raw_memory_samples"),
        ("artifacts", "ifeval_subset_manifest"),
        ("artifacts", "ifeval_responses"),
        ("artifacts", "checkpoint"),
    ],
)
def test_completed_run_rejects_null_measurements_and_artifacts(group: str, field: str) -> None:
    result = dense_example()
    result[group][field] = None
    assert list(schema_validator().iter_errors(result))
