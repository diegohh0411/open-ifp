from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from scripts.validate_protocol import ProtocolValidationError, load_deviations, load_json_strict, validate_result
from scripts.materialize_protocol_manifests import resolve_output_path, write_manifest


ROOT = Path(__file__).resolve().parents[2]
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def load_protocol() -> dict[str, object]:
    with (ROOT / "protocol/benchmark-v0.1.yaml").open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

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


def test_setup_installs_released_environment_lock() -> None:
    setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
    assert '"${PYTHON_BIN}" -m venv --clear .venv' in setup
    assert "python -m pip install -r requirements-lock.txt" in setup


def test_manifest_output_must_stay_inside_repository() -> None:
    with pytest.raises(ValueError, match="escapes repository"):
        resolve_output_path("../outside.json")


def test_manifest_digest_mismatch_does_not_overwrite_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("original\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generated manifest digest mismatch"):
        write_manifest(path, {"entries": []}, expected_sha256="0" * 64)
    assert path.read_text(encoding="utf-8") == "original\n"


@pytest.mark.parametrize(
    ("actual", "expected_returncode"),
    [("3.12.14", 0), ("3.12.13", 1), ("3.11.9", 1)],
)
def test_released_python_version_checker(
    actual: str,
    expected_returncode: int,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_python_version.py",
            "--expected",
            "3.12.14",
            "--actual",
            actual,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == expected_returncode, completed.stderr


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
        "ensure_ascii": False,
        "unicode_normalization": "none",
        "sort_keys": True,
        "json_separators": [",", ":"],
        "hash_prefix": "20260821\n",
        "line_endings": "lf",
    }
    assert data["selection"] == {
        "sort": "sha256_ascending_within_category",
        "tie_break": "source_index_ascending",
        "train_slice": [0, 100],
        "held_out_slice": [100, 125],
    }
    assert data["serialization"]["context_separator"] == "\n\nContext:\n"
    evaluation = protocol["evaluation"]
    assert evaluation["selection"]["tie_break"] == "prompt_sha256_ascending"
    assert evaluation["selection"]["fill"] == "prompt_sha256_ascending"
    assert evaluation["manifest_fields"] == ["original_key", "prompt_sha256", "instruction_ids"]


def test_canonical_manifests_exist_and_match_protocol() -> None:
    protocol = load_protocol()
    for section in ("training_data", "evaluation"):
        manifest = protocol[section].get("manifest")
        assert manifest is not None
        path = ROOT / manifest["path"]
        assert path.is_file()
        assert sha256_file(path) == manifest["sha256"]


def test_training_manifest_freezes_exact_train_and_held_out_rows() -> None:
    protocol = load_protocol()
    manifest_config = protocol["training_data"].get("manifest")
    assert manifest_config is not None
    path = ROOT / manifest_config["path"]
    manifest = load_json(path)
    assert manifest["source_sha256"] == "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
    entries = manifest["entries"]
    assert len(entries) == 500
    assert {entry["split"] for entry in entries} == {"train", "held_out"}
    assert len({entry["row_sha256"] for entry in entries}) == 500
    for category in protocol["training_data"]["categories"]:
        selected = [entry for entry in entries if entry["category"] == category]
        assert sum(entry["split"] == "train" for entry in selected) == 100
        assert sum(entry["split"] == "held_out" for entry in selected) == 25


def test_ifeval_manifest_freezes_exact_objective_subset() -> None:
    protocol = load_protocol()
    manifest_config = protocol["evaluation"].get("manifest")
    assert manifest_config is not None
    path = ROOT / manifest_config["path"]
    manifest = load_json(path)
    assert manifest["source_sha256"] == "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
    entries = manifest["entries"]
    assert len(entries) == 100
    assert len({entry["original_key"] for entry in entries}) == 100
    assert len({entry["prompt_sha256"] for entry in entries}) == 100


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
        "swap_increase_streak_max": (
            "samples",
            "maximum_consecutive_one_minute_increases",
        ),
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
        "started_at",
        "ended_at",
        "provenance",
        "hardware",
        "config",
        "metrics",
        "evaluation",
        "artifacts",
        "compute_gate",
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


@pytest.mark.parametrize(
    ("sequence_length", "gradient_accumulation_steps"),
    [(256, 4), (512, 8)],
)
def test_schema_rejects_inconsistent_sequence_accumulation_pair(
    sequence_length: int, gradient_accumulation_steps: int
) -> None:
    result = dense_example()
    result["config"]["sequence_length"] = sequence_length
    result["config"]["gradient_accumulation_steps"] = gradient_accumulation_steps
    assert list(schema_validator().iter_errors(result))


def test_schema_requires_failure_details_for_failed_run() -> None:
    result = dense_example()
    result["status"] = "failed"
    result["failure"] = None
    assert list(schema_validator().iter_errors(result))


def test_schema_allows_failed_run_with_unknown_compute_gate() -> None:
    result = dense_example()
    result["status"] = "failed"
    result["failure"] = {
        "stage": "setup",
        "type": "MissingAsset",
        "message": "pinned asset unavailable",
    }
    result["compute_gate"] = {"passed": None, "reasons": []}
    for group in ("metrics", "evaluation", "artifacts"):
        for field in result[group]:
            result[group][field] = None
    schema_validator().validate(result)


def test_validator_accepts_failed_run_with_partial_measurements() -> None:
    result = dense_example()
    result["status"] = "failed"
    result["failure"] = {
        "stage": "setup",
        "type": "MissingAsset",
        "message": "pinned asset unavailable",
    }
    result["compute_gate"] = {"passed": None, "reasons": []}
    for group in ("metrics", "evaluation", "artifacts"):
        for field in result[group]:
            result[group][field] = None
    validate_result(result, load_protocol(), {})


def test_failed_run_with_raw_evidence_requires_derived_aggregates_and_gate() -> None:
    result = dense_example()
    result["status"] = "failed"
    result["failure"] = {
        "stage": "training",
        "type": "InjectedFailure",
        "message": "raw evidence survived",
    }
    for field in result["metrics"]:
        result["metrics"][field] = None
    for field in result["evaluation"]:
        result["evaluation"][field] = None
    result["compute_gate"] = {"passed": None, "reasons": []}
    with pytest.raises(ProtocolValidationError, match="raw_step_times requires"):
        validate_result(result, load_protocol(), {})


def test_failed_run_rejects_partial_throughput_dependency_group() -> None:
    result = dense_example()
    result["status"] = "failed"
    result["failure"] = {
        "stage": "training",
        "type": "InjectedFailure",
        "message": "partial counters",
    }
    result["metrics"]["training_wall_clock_seconds"] = None
    result["metrics"]["train_tokens"] = None
    result["metrics"]["train_tokens_per_second"] = 999.0
    with pytest.raises(ProtocolValidationError, match="throughput dependency group"):
        validate_result(result, load_protocol(), {})


def test_failed_run_rejects_known_gate_without_raw_memory() -> None:
    result = dense_example()
    result["status"] = "failed"
    result["failure"] = {
        "stage": "training",
        "type": "InjectedFailure",
        "message": "memory artifact missing",
    }
    result["artifacts"]["raw_memory_samples"] = None
    with pytest.raises(ProtocolValidationError, match="raw_memory_samples"):
        validate_result(result, load_protocol(), {})


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
        ("metrics", "swap_increase_streak_max"),
        ("metrics", "checkpoint_size_bytes"),
        ("evaluation", "held_out_nll"),
        ("evaluation", "held_out_tokens"),
        ("evaluation", "ifeval_strict_prompt_accuracy"),
        ("evaluation", "ifeval_strict_instruction_accuracy"),
        ("evaluation", "ifeval_loose_prompt_accuracy"),
        ("evaluation", "ifeval_loose_instruction_accuracy"),
        ("artifacts", "raw_step_times"),
        ("artifacts", "raw_memory_samples"),
        ("artifacts", "training_split_manifest"),
        ("artifacts", "ifeval_subset_manifest"),
        ("artifacts", "held_out_evaluation"),
        ("artifacts", "ifeval_responses"),
        ("artifacts", "checkpoint"),
    ],
)
def test_completed_run_rejects_null_measurements_and_artifacts(group: str, field: str) -> None:
    result = dense_example()
    result[group][field] = None
    assert list(schema_validator().iter_errors(result))


def test_strict_loader_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_json_strict(path)


def test_result_matching_protocol_has_no_deviation() -> None:
    validate_result(dense_example(), load_protocol(), {})


def capacity_example() -> dict[str, object]:
    result = dense_example()
    result["config"].update(
        {
            "run_profile": "capacity_probe",
            "sequence_length": 256,
            "gradient_accumulation_steps": 8,
            "token_budget": None,
            "warmup_steps": 20,
            "measured_steps": 200,
        }
    )
    result["metrics"].update(
        {
            "train_tokens": 409_600,
            "train_tokens_per_second": 40_960.0,
            "checkpoint_size_bytes": None,
        }
    )
    for field in result["evaluation"]:
        result["evaluation"][field] = None
    for field in (
        "training_split_manifest",
        "ifeval_subset_manifest",
        "held_out_evaluation",
        "ifeval_responses",
        "checkpoint",
    ):
        result["artifacts"][field] = None
    return result


def test_capacity_probe_has_an_executable_result_contract() -> None:
    result = capacity_example()
    schema_validator().validate(result)
    validate_result(result, load_protocol(), {})


@pytest.mark.parametrize(
    ("field", "value"),
    [("warmup_steps", 19), ("measured_steps", 199), ("token_budget", 250_000)],
)
def test_capacity_probe_rejects_unfrozen_shape(field: str, value: object) -> None:
    result = capacity_example()
    result["config"][field] = value
    with pytest.raises(ProtocolValidationError, match="schema validation"):
        validate_result(result, load_protocol(), {})


def test_result_rejects_missing_declared_artifact() -> None:
    result = dense_example()
    result["artifacts"]["raw_step_times"] = {
        "path": "results/example/does-not-exist.json",
        "sha256": "0" * 64,
    }
    with pytest.raises(ProtocolValidationError, match="missing result artifact"):
        validate_result(result, load_protocol(), {})


@pytest.mark.parametrize(
    ("group", "field", "value", "message"),
    [
        ("metrics", "step_time_p50_ms", 121.0, "step_time_p50_ms"),
        ("metrics", "os_peak_rss_bytes", 8_999_999_999, "os_peak_rss_bytes"),
        ("metrics", "checkpoint_size_bytes", 999, "checkpoint_size_bytes"),
        ("evaluation", "held_out_nll", 2.6, "held_out_nll"),
        (
            "evaluation",
            "ifeval_strict_prompt_accuracy",
            0.26,
            "ifeval_strict_prompt_accuracy",
        ),
    ],
)
def test_result_aggregates_must_match_hashed_raw_artifacts(
    group: str,
    field: str,
    value: object,
    message: str,
) -> None:
    result = dense_example()
    result[group][field] = value
    with pytest.raises(ProtocolValidationError, match=message):
        validate_result(result, load_protocol(), {})


def test_missing_protocol_field_is_invalid_not_keyerror() -> None:
    protocol = load_protocol()
    del protocol["runtime"]["direct_packages"]["mlx"]
    with pytest.raises(ProtocolValidationError, match="released protocol digest"):
        validate_result(dense_example(), protocol, {})


def test_forbidden_package_in_provenance_is_rejected() -> None:
    result = dense_example()
    result["provenance"]["runtime"]["packages"]["torch"] = "2.0.0"
    with pytest.raises(ProtocolValidationError, match="forbidden packages"):
        validate_result(result, load_protocol(), {})


def test_runtime_package_map_must_match_released_lock() -> None:
    result = dense_example()
    result["provenance"]["runtime"]["packages"]["numpy"] = "0.0.0"
    with pytest.raises(ProtocolValidationError, match="/provenance/runtime/packages"):
        validate_result(result, load_protocol(), {})


def test_released_environment_lock_matches_protocol() -> None:
    runtime = load_protocol()["runtime"]
    lock = runtime.get("lock")
    assert lock is not None
    path = ROOT / lock["path"]
    assert path.is_file()
    assert sha256_file(path) == lock["sha256"]


@pytest.mark.parametrize("package_name", ["Torch", "TorchVision", "TorchTune"])
def test_forbidden_package_name_variants_are_rejected(package_name: str) -> None:
    result = dense_example()
    result["provenance"]["runtime"]["packages"][package_name] = "2.0.0"
    with pytest.raises(ProtocolValidationError, match="forbidden packages"):
        validate_result(result, load_protocol(), {})


def test_unrecorded_seed_change_is_rejected() -> None:
    result = dense_example()
    result["config"]["seed"] = 7
    with pytest.raises(ProtocolValidationError, match="/config/seed"):
        validate_result(result, load_protocol(), {})


def test_unrecorded_token_budget_change_is_rejected() -> None:
    result = dense_example()
    result["config"]["token_budget"] = 1
    with pytest.raises(ProtocolValidationError, match="/config/token_budget"):
        validate_result(result, load_protocol(), {})


def test_quality_result_rejects_capacity_only_sequence_length() -> None:
    result = dense_example()
    result["config"]["sequence_length"] = 256
    result["config"]["gradient_accumulation_steps"] = 8
    with pytest.raises(ProtocolValidationError, match="schema validation"):
        validate_result(result, load_protocol(), {})


def test_unrecorded_training_manifest_change_is_rejected() -> None:
    result = dense_example()
    result["provenance"]["dataset"]["split_manifest_sha256"] = "f" * 64
    with pytest.raises(
        ProtocolValidationError,
        match="/provenance/dataset/split_manifest_sha256",
    ):
        validate_result(result, load_protocol(), {})


@pytest.mark.parametrize("field", ["config_sha256", "tokenizer_config_sha256"])
def test_unrecorded_model_file_digest_change_is_rejected(field: str) -> None:
    result = dense_example()
    result["provenance"]["model"][field] = "f" * 64
    with pytest.raises(
        ProtocolValidationError,
        match=f"/provenance/model/{field}",
    ):
        validate_result(result, load_protocol(), {})


def test_unrecorded_model_snapshot_change_is_rejected() -> None:
    result = dense_example()
    result["provenance"]["model"]["cache_snapshot"] = "some/other/snapshot"
    with pytest.raises(
        ProtocolValidationError,
        match="/provenance/model/cache_snapshot",
    ):
        validate_result(result, load_protocol(), {})


def test_unrecorded_model_payload_digest_change_is_rejected() -> None:
    result = dense_example()
    result["provenance"]["model"]["files_sha256"]["model.safetensors"] = "f" * 64
    with pytest.raises(
        ProtocolValidationError,
        match="/provenance/model/files_sha256",
    ):
        validate_result(result, load_protocol(), {})


def test_unrecorded_ifeval_manifest_change_is_rejected() -> None:
    result = dense_example()
    result["artifacts"]["ifeval_subset_manifest"]["sha256"] = "f" * 64
    with pytest.raises(
        ProtocolValidationError,
        match="/artifacts/ifeval_subset_manifest/sha256",
    ):
        validate_result(result, load_protocol(), {})


def test_changed_metric_definition_invalidates_released_protocol() -> None:
    protocol = load_protocol()
    protocol["measurements"][0]["definition"] = "changed_without_a_release"
    with pytest.raises(ProtocolValidationError, match="released protocol digest"):
        validate_result(dense_example(), protocol, {})


def test_fixed_p6_result_rejects_incorrect_realized_dimensions() -> None:
    result = dense_example()
    result["config"]["method"] = "p6_static_mask"
    result["config"]["p6"] = {
        "activation_fraction": 0.6,
        "activation_grid": [0.4, 0.6, 0.8, 1.0],
        "realized_dimensions_per_layer": [1] * 24,
        "mean_activation_fraction": 0.6,
    }
    with pytest.raises(ProtocolValidationError, match="realized_dimensions_per_layer"):
        validate_result(result, load_protocol(), {})


@pytest.mark.parametrize(
    "method",
    ["p6_random_mask", "p6_static_mask", "p6_learned_fixed_k"],
)
def test_day3_fixed_k_rejects_non_preregistered_fraction(method: str) -> None:
    result = dense_example()
    result["config"]["method"] = method
    result["config"]["p6"] = {
        "activation_fraction": 0.4,
        "activation_grid": [0.4, 0.6, 0.8, 1.0],
        "realized_dimensions_per_layer": [1946] * 24,
        "mean_activation_fraction": 0.4,
    }
    with pytest.raises(ProtocolValidationError, match="/config/p6/activation_fraction"):
        validate_result(result, load_protocol(), {})


@pytest.mark.parametrize(
    ("started_at", "ended_at", "wall_clock", "message"),
    [
        ("2026-08-21T18:00:00Z", "2026-08-21T18:00:01Z", 10.0, "run interval"),
        ("2026-08-21T19:00:00Z", "2026-08-21T19:00:10Z", 10.0, "raw memory sample"),
    ],
)
def test_result_rejects_run_interval_inconsistent_with_measurements(
    started_at: str,
    ended_at: str,
    wall_clock: float,
    message: str,
) -> None:
    result = dense_example()
    result["started_at"] = started_at
    result["ended_at"] = ended_at
    result["metrics"]["training_wall_clock_seconds"] = wall_clock
    result["metrics"]["train_tokens_per_second"] = (
        result["metrics"]["train_tokens"] / wall_clock
    )
    with pytest.raises(ProtocolValidationError, match=message):
        validate_result(result, load_protocol(), {})


@pytest.mark.parametrize(
    ("dimensions", "mean_fraction"),
    [([1] * 24, 0.6), ([2918] * 24, 0.7)],
)
def test_variable_p6_result_rejects_inconsistent_activation_summary(
    dimensions: list[int],
    mean_fraction: float,
) -> None:
    result = dense_example()
    result["config"]["method"] = "p6_variable_k"
    result["config"]["p6"] = {
        "activation_fraction": None,
        "activation_grid": [0.4, 0.6, 0.8, 1.0],
        "realized_dimensions_per_layer": dimensions,
        "mean_activation_fraction": mean_fraction,
    }
    with pytest.raises(ProtocolValidationError, match="variable-k activation summary"):
        validate_result(result, load_protocol(), {})


def test_variable_p6_result_rejects_dimension_off_frozen_grid() -> None:
    result = dense_example()
    dimensions = [2918] * 24
    dimensions[0] = 2919
    result["config"]["method"] = "p6_variable_k"
    result["config"]["p6"] = {
        "activation_fraction": None,
        "activation_grid": [0.4, 0.6, 0.8, 1.0],
        "realized_dimensions_per_layer": dimensions,
        "mean_activation_fraction": sum(dimensions) / len(dimensions) / 4864,
    }
    with pytest.raises(ProtocolValidationError, match="frozen activation grid"):
        validate_result(result, load_protocol(), {})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda result: result["metrics"].__setitem__(
                "train_tokens_per_second", 1.0
            ),
            "train_tokens_per_second",
        ),
        (
            lambda result: result["metrics"].__setitem__("swap_delta_bytes", 1),
            "swap_delta_bytes",
        ),
        (
            lambda result: result["metrics"].update(
                {"step_time_p50_ms": 151.0, "step_time_p95_ms": 150.0}
            ),
            "step_time_p50_ms",
        ),
        (
            lambda result: result.__setitem__(
                "ended_at", "2026-08-21T17:59:59Z"
            ),
            "ended_at",
        ),
        (
            lambda result: result["metrics"].__setitem__("train_tokens", 249_999),
            "train_tokens",
        ),
        (
            lambda result: result["metrics"].__setitem__(
                "os_peak_rss_bytes", 32_212_254_721
            ),
            "os_peak_rss_bytes",
        ),
    ],
)
def test_completed_result_rejects_inconsistent_derived_values(
    mutate: object,
    message: str,
) -> None:
    result = dense_example()
    mutate(result)
    with pytest.raises(ProtocolValidationError, match=message):
        validate_result(result, load_protocol(), {})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda result: result.__setitem__(
                "ended_at", "2026-08-21T17:59:59Z"
            ),
            "ended_at",
        ),
        (
            lambda result: result["metrics"].__setitem__(
                "train_tokens_per_second", 1.0
            ),
            "train_tokens_per_second",
        ),
        (
            lambda result: result["metrics"].__setitem__("swap_delta_bytes", 1),
            "swap_delta_bytes",
        ),
        (
            lambda result: result["compute_gate"].update(
                {"passed": False, "reasons": ["peak_rss_exceeded"]}
            ),
            "compute_gate",
        ),
    ],
)
def test_failed_result_rejects_inconsistent_reported_values(
    mutate: object,
    message: str,
) -> None:
    result = dense_example()
    result["status"] = "failed"
    result["failure"] = {
        "stage": "training",
        "type": "InjectedFailure",
        "message": "failed after measurements were captured",
    }
    mutate(result)
    with pytest.raises(ProtocolValidationError, match=message):
        validate_result(result, load_protocol(), {})


def test_deviation_requires_a_new_effective_major_version() -> None:
    result = dense_example()
    result["config"]["seed"] = 7
    result["deviation_ids"] = ["DEV-0001"]
    deviations = {
        "DEV-0001": {
            "deviation_id": "DEV-0001",
            "timestamp": "2026-08-21T17:30:00Z",
            "author": "Diego Hernandez",
            "affected_run_ids": [result["run_id"]],
            "field_path": "/config/seed",
            "old_value": 20260821,
            "new_value": 7,
            "rationale": "exercise deviation validation",
            "comparability_impact": "not comparable to protocol seed",
            "approval_status": "approved",
            "base_protocol_version": "0.1.0",
            "effective_protocol_version": "0.1.0",
        }
    }
    with pytest.raises(ProtocolValidationError, match="effective major version"):
        validate_result(result, load_protocol(), deviations)


def test_approved_exact_deviation_allows_seed_change() -> None:
    result = dense_example()
    result["protocol_version"] = "1.0.0"
    result["config"]["seed"] = 7
    result["deviation_ids"] = ["DEV-0001"]
    deviations = {
        "DEV-0001": {
            "deviation_id": "DEV-0001",
            "timestamp": "2026-08-21T17:30:00Z",
            "author": "Diego Hernandez",
            "affected_run_ids": [result["run_id"]],
            "field_path": "/config/seed",
            "old_value": 20260821,
            "new_value": 7,
            "rationale": "exercise deviation validation",
            "comparability_impact": "not comparable to protocol seed",
            "approval_status": "approved",
            "base_protocol_version": "0.1.0",
            "effective_protocol_version": "1.0.0",
        }
    }
    validate_result(result, load_protocol(), deviations)


def test_approved_exact_deviation_allows_runtime_package_map_change() -> None:
    result = dense_example()
    original_packages = copy.deepcopy(result["provenance"]["runtime"]["packages"])
    changed_packages = copy.deepcopy(original_packages)
    changed_packages["numpy"] = "2.5.3"
    result["provenance"]["runtime"]["packages"] = changed_packages
    result["protocol_version"] = "1.0.0"
    result["deviation_ids"] = ["DEV-0001"]
    deviations = {
        "DEV-0001": {
            "deviation_id": "DEV-0001",
            "timestamp": "2026-08-21T17:30:00Z",
            "author": "Diego Hernandez",
            "affected_run_ids": [result["run_id"]],
            "field_path": "/provenance/runtime/packages",
            "old_value": original_packages,
            "new_value": changed_packages,
            "rationale": "exercise environment deviation validation",
            "comparability_impact": "not comparable to protocol environment",
            "approval_status": "approved",
            "base_protocol_version": "0.1.0",
            "effective_protocol_version": "1.0.0",
        }
    }
    validate_result(result, load_protocol(), deviations)


def test_cited_deviation_must_authorize_an_actual_mismatch() -> None:
    result = dense_example()
    result["protocol_version"] = "1.0.0"
    result["deviation_ids"] = ["DEV-0001"]
    deviations = {
        "DEV-0001": {
            "deviation_id": "DEV-0001",
            "timestamp": "2026-08-21T17:30:00Z",
            "author": "Diego Hernandez",
            "affected_run_ids": [result["run_id"]],
            "field_path": "/config/seed",
            "old_value": 20260821,
            "new_value": 7,
            "rationale": "not actually used by this run",
            "comparability_impact": "not comparable to protocol seed",
            "approval_status": "approved",
            "base_protocol_version": "0.1.0",
            "effective_protocol_version": "1.0.0",
        }
    }
    with pytest.raises(ProtocolValidationError, match="unused deviation_ids"):
        validate_result(result, load_protocol(), deviations)


def test_deviation_must_be_approved_before_run_starts() -> None:
    result = dense_example()
    result["protocol_version"] = "1.0.0"
    result["config"]["seed"] = 7
    result["deviation_ids"] = ["DEV-0001"]
    deviations = {
        "DEV-0001": {
            "deviation_id": "DEV-0001",
            "timestamp": "2026-08-21T18:30:00Z",
            "author": "Diego Hernandez",
            "affected_run_ids": [result["run_id"]],
            "field_path": "/config/seed",
            "old_value": 20260821,
            "new_value": 7,
            "rationale": "approved after the run began",
            "comparability_impact": "not comparable to protocol seed",
            "approval_status": "approved",
            "base_protocol_version": "0.1.0",
            "effective_protocol_version": "1.0.0",
        }
    }
    with pytest.raises(ProtocolValidationError, match="before run start"):
        validate_result(result, load_protocol(), deviations)


def test_deviation_must_match_run_path_and_value() -> None:
    result = dense_example()
    result["protocol_version"] = "1.0.0"
    result["config"]["seed"] = 7
    result["deviation_ids"] = ["DEV-0001"]
    deviations = {
        "DEV-0001": {
            "deviation_id": "DEV-0001",
            "timestamp": "2026-08-21T17:30:00Z",
            "author": "Diego Hernandez",
            "affected_run_ids": ["another-run"],
            "field_path": "/config/seed",
            "old_value": 20260821,
            "new_value": 8,
            "rationale": "wrong target",
            "comparability_impact": "none",
            "approval_status": "approved",
            "base_protocol_version": "0.1.0",
            "effective_protocol_version": "1.0.0",
        }
    }
    with pytest.raises(ProtocolValidationError, match="/config/seed"):
        validate_result(result, load_protocol(), deviations)


def test_deviation_matching_distinguishes_boolean_from_numeric_value() -> None:
    result = dense_example()
    result["protocol_version"] = "1.0.0"
    result["config"]["method"] = "p6_learned_fixed_k"
    result["config"]["p6"] = {
        "activation_fraction": 1.0,
        "activation_grid": [0.4, 0.6, 0.8, 1.0],
        "realized_dimensions_per_layer": [4864] * 24,
        "mean_activation_fraction": 1.0,
    }
    result["deviation_ids"] = ["DEV-0001"]
    deviations = {
        "DEV-0001": {
            "deviation_id": "DEV-0001",
            "timestamp": "2026-08-21T17:30:00Z",
            "author": "Diego Hernandez",
            "affected_run_ids": [result["run_id"]],
            "field_path": "/config/p6/activation_fraction",
            "old_value": 0.6,
            "new_value": True,
            "rationale": "type-confusion probe",
            "comparability_impact": "not comparable",
            "approval_status": "approved",
            "base_protocol_version": "0.1.0",
            "effective_protocol_version": "1.0.0",
        }
    }
    with pytest.raises(ProtocolValidationError, match="activation_fraction"):
        validate_result(result, load_protocol(), deviations)


def test_deviation_reader_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = json.dumps({
        "deviation_id": "DEV-0001",
        "timestamp": "2026-08-21T18:30:00Z",
        "author": "Diego Hernandez",
        "affected_run_ids": ["run-12345678"],
        "field_path": "/config/seed",
        "old_value": 20260821,
        "new_value": 7,
        "rationale": "duplicate test",
        "comparability_impact": "not comparable",
        "approval_status": "approved",
        "base_protocol_version": "0.1.0",
        "effective_protocol_version": "1.0.0",
    })
    path = tmp_path / "deviations.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ProtocolValidationError, match="duplicate deviation_id"):
        load_deviations(path)


def test_deviation_reader_rejects_missing_contract_field(tmp_path: Path) -> None:
    path = tmp_path / "deviations.jsonl"
    path.write_text(
        json.dumps({"deviation_id": "DEV-0001", "approval_status": "approved"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolValidationError, match="missing fields"):
        load_deviations(path)


def test_cli_validates_dense_example() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/validate_protocol.py", "protocol/examples/dense-baseline.json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "valid: poc-20260821-dense-20260821-99554284" in completed.stdout
