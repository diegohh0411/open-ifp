#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocol/benchmark-v0.1.yaml"
DEFAULT_SCHEMA = ROOT / "protocol/run-result.schema.json"
DEFAULT_DEVIATIONS = ROOT / "protocol/deviations.jsonl"
DEFAULT_RELEASES = ROOT / "protocol/releases.json"
DEVIATION_ID = re.compile(r"^DEV-[0-9]{4}$")

DEVIATION_REQUIRED = {
    "deviation_id",
    "timestamp",
    "author",
    "affected_run_ids",
    "field_path",
    "old_value",
    "new_value",
    "rationale",
    "comparability_impact",
    "approval_status",
    "base_protocol_version",
    "effective_protocol_version",
}


class ProtocolValidationError(ValueError):
    pass


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_json_strict(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"expected JSON object: {path}")
    return value


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"expected YAML object: {path}")
    return value


def canonical_digest(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def load_releases(path: Path = DEFAULT_RELEASES) -> dict[str, Any]:
    value = load_json_strict(path)
    releases = value.get("releases")
    if not isinstance(releases, dict):
        raise ProtocolValidationError(f"missing releases object: {path}")
    return releases


def parse_utc_timestamp(value: str, *, context: str) -> datetime:
    if not value.endswith("Z"):
        raise ProtocolValidationError(f"{context} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProtocolValidationError(
            f"{context} must be an RFC3339 UTC timestamp"
        ) from exc
    if parsed.tzinfo != timezone.utc:
        raise ProtocolValidationError(f"{context} must be an RFC3339 UTC timestamp")
    return parsed


def load_deviations(path: Path = DEFAULT_DEVIATIONS) -> dict[str, dict[str, Any]]:
    deviations: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise ProtocolValidationError(f"missing deviation ledger: {path}")
    record_number = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProtocolValidationError(f"invalid deviation at line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ProtocolValidationError(f"deviation line {line_number} is not an object")
        record_number += 1
        missing = sorted(DEVIATION_REQUIRED - set(item))
        if missing:
            raise ProtocolValidationError(
                f"deviation line {line_number} missing fields: {', '.join(missing)}"
            )
        extra = sorted(set(item) - DEVIATION_REQUIRED)
        if extra:
            raise ProtocolValidationError(
                f"deviation line {line_number} has unknown fields: {', '.join(extra)}"
            )
        deviation_id = item.get("deviation_id")
        if not isinstance(deviation_id, str) or not DEVIATION_ID.fullmatch(deviation_id):
            raise ProtocolValidationError(f"deviation line {line_number} has invalid deviation_id")
        if deviation_id in deviations:
            raise ProtocolValidationError(f"duplicate deviation_id: {deviation_id}")
        expected_id = f"DEV-{record_number:04d}"
        if deviation_id != expected_id:
            raise ProtocolValidationError(
                f"deviation line {line_number} expected sequential id {expected_id}"
            )
        if not isinstance(item["affected_run_ids"], list) or not all(
            isinstance(run_id, str) for run_id in item["affected_run_ids"]
        ) or not item["affected_run_ids"]:
            raise ProtocolValidationError(
                f"deviation line {line_number} has invalid affected_run_ids"
            )
        if not isinstance(item["field_path"], str) or not item["field_path"].startswith("/"):
            raise ProtocolValidationError(f"deviation line {line_number} has invalid field_path")
        for field in ("timestamp", "author", "rationale", "comparability_impact"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ProtocolValidationError(
                    f"deviation line {line_number} has invalid {field}"
                )
        parse_utc_timestamp(
            item["timestamp"],
            context=f"deviation line {line_number} timestamp",
        )
        for field in ("base_protocol_version", "effective_protocol_version"):
            if not isinstance(item[field], str) or not re.fullmatch(
                r"[0-9]+\.[0-9]+\.[0-9]+",
                item[field],
            ):
                raise ProtocolValidationError(
                    f"deviation line {line_number} has invalid {field}"
                )
        if item["approval_status"] not in {"pending", "approved", "rejected"}:
            raise ProtocolValidationError(
                f"deviation line {line_number} has invalid approval_status"
            )
        deviations[deviation_id] = item
    return deviations


def walk_finite(value: Any, path: str = "") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolValidationError(f"non-finite number at {path or '/'}")
    if isinstance(value, dict):
        for key, child in value.items():
            walk_finite(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_finite(child, f"{path}/{index}")


def method_token_budget(protocol: dict[str, Any], method: str) -> int:
    if method.startswith("p8_"):
        return protocol["p8"]["train_tokens"]
    return protocol["p6"]["day3_train_tokens"]


def expected_values(
    protocol: dict[str, Any],
    result: dict[str, Any],
    release: dict[str, Any],
) -> dict[str, Any]:
    try:
        length = result["config"]["sequence_length"]
        profile = protocol["training"]["sequence_profiles"][length]
        expected = {
            "/schema_version": protocol["schema_version"],
            "/provenance/protocol/canonical_sha256": release["protocol_canonical_sha256"],
            "/provenance/protocol/schema_sha256": release["schema_canonical_sha256"],
            "/provenance/model/repository": protocol["model"]["repository"],
            "/provenance/model/revision": protocol["model"]["revision"],
            "/provenance/model/cache_snapshot": protocol["model"]["cache_snapshot"],
            "/provenance/model/config_sha256": protocol["model"]["config_sha256"],
            "/provenance/model/tokenizer_config_sha256": protocol["model"]["tokenizer_config_sha256"],
            "/provenance/model/files_sha256": protocol["model"]["files_sha256"],
            "/provenance/tokenizer/repository": protocol["model"]["tokenizer_repository"],
            "/provenance/tokenizer/revision": protocol["model"]["tokenizer_revision"],
            "/provenance/dataset/repository": protocol["training_data"]["repository"],
            "/provenance/dataset/revision": protocol["training_data"]["revision"],
            "/provenance/dataset/split_manifest_sha256": protocol["training_data"]["manifest"]["sha256"],
            "/provenance/evaluator/repository": protocol["evaluation"]["repository"],
            "/provenance/evaluator/revision": protocol["evaluation"]["revision"],
            "/provenance/evaluator/implementation_path": protocol["evaluation"]["implementation_path"],
            "/provenance/runtime/python": protocol["runtime"]["python"],
            "/provenance/runtime/lock_sha256": protocol["runtime"]["lock"]["sha256"],
            "/provenance/runtime/packages": load_lock_packages(
                ROOT / protocol["runtime"]["lock"]["path"]
            ),
            "/hardware/chip": protocol["platform"]["chip"],
            "/hardware/cpu_cores": protocol["platform"]["cpu_cores"],
            "/hardware/gpu_cores": protocol["platform"]["gpu_cores"],
            "/hardware/unified_memory_bytes": protocol["platform"]["unified_memory_bytes"],
            "/config/seed": protocol["training"]["seed"],
            "/config/microbatch_size": profile["microbatch_size"],
            "/config/gradient_accumulation_steps": profile["gradient_accumulation_steps"],
            "/config/effective_tokens_per_update": protocol["training"]["effective_tokens_per_update"],
        }
        run_profile = result["config"]["run_profile"]
        profile_config = protocol["result_contract"]["run_profiles"][run_profile]
        expected["/config/run_profile"] = run_profile
        expected["/config/warmup_steps"] = profile_config["warmup_steps"]
        expected["/config/measured_steps"] = profile_config["measured_steps"]
        if run_profile == "quality_train":
            expected["/config/sequence_length"] = profile_config["sequence_length"]
            expected["/config/token_budget"] = method_token_budget(
                protocol,
                result["config"]["method"],
            )
        elif run_profile == "capacity_probe":
            expected["/config/method"] = profile_config["method"]
            expected["/config/token_budget"] = profile_config["token_budget"]
        if result["config"]["p6"] is not None:
            expected["/config/p6/activation_grid"] = protocol["p6"]["activation_grid"]
        if result["config"]["method"] in {
            "p6_random_mask",
            "p6_static_mask",
            "p6_learned_fixed_k",
        }:
            expected["/config/p6/activation_fraction"] = protocol["p6"]["day3_fixed_fraction"]
        for artifact_name, section_name in (
            ("training_split_manifest", "training_data"),
            ("ifeval_subset_manifest", "evaluation"),
        ):
            if result["artifacts"][artifact_name] is not None:
                manifest = protocol[section_name]["manifest"]
                expected[f"/artifacts/{artifact_name}/path"] = manifest["path"]
                expected[f"/artifacts/{artifact_name}/sha256"] = manifest["sha256"]
        return expected
    except (KeyError, TypeError) as exc:
        raise ProtocolValidationError("missing field required for protocol comparison") from exc


def pointer_get(value: dict[str, Any], pointer: str) -> Any:
    current: Any = value
    for part in pointer.strip("/").split("/"):
        try:
            current = current[part]
        except (KeyError, TypeError) as exc:
            raise ProtocolValidationError(f"missing field at {pointer}") from exc
    return current


def normalize_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_lock_packages(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    normalized_names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ProtocolValidationError(f"invalid exact package pin: {line}")
        name, version = line.split("==", maxsplit=1)
        normalized = normalize_package(name)
        if normalized in normalized_names:
            raise ProtocolValidationError(f"duplicate package in lock: {name}")
        normalized_names.add(normalized)
        packages[name] = version
    return packages


def validate_release(
    protocol: dict[str, Any],
    schema: dict[str, Any],
    releases: dict[str, Any],
) -> dict[str, Any]:
    try:
        version = protocol["protocol_version"]
        release = releases[version]
        expected_protocol_digest = release["protocol_canonical_sha256"]
        expected_schema_digest = release["schema_canonical_sha256"]
    except (KeyError, TypeError) as exc:
        raise ProtocolValidationError("missing released protocol identity") from exc
    actual_protocol_digest = canonical_digest(protocol)
    if actual_protocol_digest != expected_protocol_digest:
        raise ProtocolValidationError(
            "released protocol digest mismatch: "
            f"expected {expected_protocol_digest}, got {actual_protocol_digest}"
        )
    actual_schema_digest = canonical_digest(schema)
    if actual_schema_digest != expected_schema_digest:
        raise ProtocolValidationError(
            "released schema digest mismatch: "
            f"expected {expected_schema_digest}, got {actual_schema_digest}"
        )
    for section_name in ("training_data", "evaluation"):
        try:
            manifest = protocol[section_name]["manifest"]
            path = ROOT / manifest["path"]
            expected = manifest["sha256"]
        except (KeyError, TypeError) as exc:
            raise ProtocolValidationError(
                f"missing {section_name} manifest identity"
            ) from exc
        if not path.is_file():
            raise ProtocolValidationError(f"missing canonical manifest: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ProtocolValidationError(
                f"canonical manifest digest mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )
    try:
        lock = protocol["runtime"]["lock"]
        lock_path = ROOT / lock["path"]
        expected_lock_digest = lock["sha256"]
    except (KeyError, TypeError) as exc:
        raise ProtocolValidationError("missing released environment lock") from exc
    if not lock_path.is_file():
        raise ProtocolValidationError(f"missing released environment lock: {lock_path}")
    actual_lock_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if actual_lock_digest != expected_lock_digest:
        raise ProtocolValidationError(
            "released environment lock digest mismatch: "
            f"expected {expected_lock_digest}, got {actual_lock_digest}"
        )
    return release


def semver_major(value: str) -> int:
    try:
        return int(value.split(".", maxsplit=1)[0])
    except (AttributeError, ValueError) as exc:
        raise ProtocolValidationError(f"invalid protocol version: {value!r}") from exc


def validate_deviation_versions(
    result: dict[str, Any],
    protocol: dict[str, Any],
    cited_ids: list[str],
    deviations: dict[str, dict[str, Any]],
) -> None:
    base_version = protocol["protocol_version"]
    if not cited_ids:
        if result["protocol_version"] != base_version:
            raise ProtocolValidationError(
                f"/protocol_version: expected {base_version!r}, "
                f"got {result['protocol_version']!r}"
            )
        return

    run_started_at = parse_utc_timestamp(result["started_at"], context="started_at")
    effective_versions: set[str] = set()
    for deviation_id in cited_ids:
        item = deviations[deviation_id]
        effective = item.get("effective_protocol_version")
        if item.get("approval_status") != "approved":
            raise ProtocolValidationError(f"deviation {deviation_id} is not approved")
        if item.get("base_protocol_version") != base_version:
            raise ProtocolValidationError(
                f"deviation {deviation_id} has wrong base protocol version"
            )
        deviation_timestamp = parse_utc_timestamp(
            item["timestamp"],
            context=f"deviation {deviation_id} timestamp",
        )
        if deviation_timestamp > run_started_at:
            raise ProtocolValidationError(
                f"deviation {deviation_id} must be approved before run start"
            )
        if not isinstance(effective, str) or semver_major(effective) <= semver_major(
            base_version
        ):
            raise ProtocolValidationError(
                f"deviation {deviation_id} requires a new effective major version"
            )
        effective_versions.add(effective)
    if len(effective_versions) != 1:
        raise ProtocolValidationError("cited deviations disagree on effective protocol version")
    effective_version = effective_versions.pop()
    if result["protocol_version"] != effective_version:
        raise ProtocolValidationError(
            f"/protocol_version: expected effective version {effective_version!r}, "
            f"got {result['protocol_version']!r}"
        )


def round_half_up(value: float, multiplier: int) -> int:
    return int(
        (Decimal(str(value)) * Decimal(multiplier)).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ProtocolValidationError("raw_step_times contains no samples")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def require_numeric(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolValidationError(f"{context} must be numeric")
    if not math.isfinite(float(value)):
        raise ProtocolValidationError(f"{context} must be finite")
    if float(value) < 0:
        raise ProtocolValidationError(f"{context} must be nonnegative")
    return float(value)


def assert_reported_number(
    reported: Any,
    expected: float | int,
    *,
    field: str,
) -> None:
    if reported is None:
        return
    if isinstance(expected, int):
        matches = reported == expected
    else:
        matches = math.isclose(float(reported), expected, rel_tol=1e-9, abs_tol=1e-9)
    if not matches:
        raise ProtocolValidationError(
            f"{field} does not match hashed raw artifact: expected {expected!r}, "
            f"got {reported!r}"
        )


def require_reported_fields(
    values: dict[str, Any],
    fields: tuple[str, ...],
    *,
    source: str,
) -> None:
    missing = [field for field in fields if values[field] is None]
    if missing:
        raise ProtocolValidationError(
            f"{source} requires reported aggregates: {', '.join(missing)}"
        )


def verify_result_artifacts(
    result: dict[str, Any],
    protocol: dict[str, Any],
    root: Path = ROOT,
) -> None:
    artifact_paths: dict[str, Path] = {}
    resolved_root = root.resolve()
    for name, artifact in result["artifacts"].items():
        if artifact is None:
            continue
        path = (resolved_root / artifact["path"]).resolve()
        if not path.is_relative_to(resolved_root):
            raise ProtocolValidationError(f"result artifact escapes root: {artifact['path']}")
        if not path.is_file():
            raise ProtocolValidationError(f"missing result artifact: {path}")
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != artifact["sha256"]:
            raise ProtocolValidationError(
                f"result artifact digest mismatch for {path}: "
                f"expected {artifact['sha256']}, got {actual_digest}"
            )
        artifact_paths[name] = path

    metrics = result["metrics"]
    evaluation = result["evaluation"]
    step_path = artifact_paths.get("raw_step_times")
    if step_path is not None:
        require_reported_fields(
            metrics,
            ("step_time_p50_ms", "step_time_p95_ms"),
            source="raw_step_times",
        )
        raw_steps = load_json_strict(step_path)
        if raw_steps.get("format_version") != protocol["artifact_formats"]["version"]:
            raise ProtocolValidationError("raw_step_times has wrong format_version")
        values = raw_steps.get("step_times_ms")
        if not isinstance(values, list):
            raise ProtocolValidationError("raw_step_times.step_times_ms must be an array")
        step_times = [
            require_numeric(value, context="raw_step_times.step_times_ms")
            for value in values
        ]
        if (
            result["config"]["run_profile"] == "capacity_probe"
            and len(step_times) != result["config"]["measured_steps"]
        ):
            raise ProtocolValidationError(
                "raw_step_times count does not match measured_steps"
            )
        wall_clock = metrics["training_wall_clock_seconds"]
        if wall_clock is not None and sum(step_times) / 1000 > wall_clock:
            raise ProtocolValidationError(
                "raw step-time total exceeds training_wall_clock_seconds"
            )
        assert_reported_number(
            metrics["step_time_p50_ms"],
            nearest_rank(step_times, 0.50),
            field="step_time_p50_ms",
        )
        assert_reported_number(
            metrics["step_time_p95_ms"],
            nearest_rank(step_times, 0.95),
            field="step_time_p95_ms",
        )

    memory_path = artifact_paths.get("raw_memory_samples")
    if memory_path is not None:
        memory_fields = (
            "mlx_peak_memory_bytes",
            "mlx_active_memory_bytes",
            "mlx_cache_memory_bytes",
            "os_peak_rss_bytes",
            "memory_free_percent_min",
            "swap_used_start_bytes",
            "swap_used_end_bytes",
            "swap_delta_bytes",
            "swap_increase_streak_max",
        )
        require_reported_fields(
            metrics,
            memory_fields,
            source="raw_memory_samples",
        )
        if result["compute_gate"]["passed"] is None:
            raise ProtocolValidationError(
                "raw_memory_samples requires a known compute_gate outcome"
            )
        raw_memory = load_json_strict(memory_path)
        if raw_memory.get("format_version") != protocol["artifact_formats"]["version"]:
            raise ProtocolValidationError("raw_memory_samples has wrong format_version")
        run_started_at = parse_utc_timestamp(result["started_at"], context="started_at")
        run_ended_at = parse_utc_timestamp(result["ended_at"], context="ended_at")
        run_interval_seconds = (run_ended_at - run_started_at).total_seconds()

        def sample_stream(
            name: str,
            fields: set[str],
            interval_seconds: int,
        ) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
            samples = raw_memory.get(name)
            if not isinstance(samples, list) or not samples:
                raise ProtocolValidationError(
                    f"raw_memory_samples.{name} must be nonempty"
                )
            timestamps: list[datetime] = []
            for index, sample in enumerate(samples):
                if (
                    not isinstance(sample, dict)
                    or not fields.issubset(sample)
                    or not isinstance(sample.get("timestamp"), str)
                ):
                    raise ProtocolValidationError(
                        f"raw_memory_samples.{name}[{index}] is incomplete"
                    )
                timestamps.append(
                    parse_utc_timestamp(
                        sample["timestamp"],
                        context=f"raw_memory_samples.{name}[{index}].timestamp",
                    )
                )
            jitter_tolerance_seconds = (
                min(interval_seconds, run_interval_seconds) * 0.1
            )
            start_jitter_seconds = abs(
                (timestamps[0] - run_started_at).total_seconds()
            )
            end_jitter_seconds = abs(
                (timestamps[-1] - run_ended_at).total_seconds()
            )
            if (
                start_jitter_seconds > jitter_tolerance_seconds
                or end_jitter_seconds > jitter_tolerance_seconds
            ):
                raise ProtocolValidationError(
                    f"raw memory sample stream {name} does not span started_at/ended_at run interval"
                )
            for previous, current in zip(timestamps, timestamps[1:]):
                elapsed = (current - previous).total_seconds()
                if not 0 < elapsed <= interval_seconds + jitter_tolerance_seconds:
                    raise ProtocolValidationError(
                        f"raw memory sample stream {name} violates sampling cadence"
                    )
            numeric = {
                field: [
                    require_numeric(
                        sample[field],
                        context=f"raw_memory_samples.{name}.{field}",
                    )
                    for sample in samples
                ]
                for field in fields
            }
            return samples, numeric

        _, rss = sample_stream(
            "rss_samples",
            {
                "mlx_peak_memory_bytes",
                "mlx_active_memory_bytes",
                "mlx_cache_memory_bytes",
                "os_rss_bytes",
            },
            protocol["sampling"]["rss_interval_seconds"],
        )
        _, pressure = sample_stream(
            "memory_pressure_samples",
            {"memory_free_percent"},
            protocol["sampling"]["memory_pressure_interval_seconds"],
        )
        _, swap = sample_stream(
            "swap_samples",
            {"swap_used_bytes"},
            protocol["sampling"]["swap_interval_seconds"],
        )
        swaps = swap["swap_used_bytes"]
        streak = 0
        max_streak = 0
        for previous, current in zip(swaps, swaps[1:]):
            streak = streak + 1 if current > previous else 0
            max_streak = max(max_streak, streak)
        derived_memory = {
            "mlx_peak_memory_bytes": max(rss["mlx_peak_memory_bytes"]),
            "mlx_active_memory_bytes": rss["mlx_active_memory_bytes"][-1],
            "mlx_cache_memory_bytes": rss["mlx_cache_memory_bytes"][-1],
            "os_peak_rss_bytes": max(rss["os_rss_bytes"]),
            "memory_free_percent_min": min(pressure["memory_free_percent"]),
            "swap_used_start_bytes": swaps[0],
            "swap_used_end_bytes": swaps[-1],
            "swap_delta_bytes": swaps[-1] - swaps[0],
            "swap_increase_streak_max": max_streak,
        }
        for field, expected in derived_memory.items():
            if field != "memory_free_percent_min":
                expected = int(expected)
            assert_reported_number(metrics[field], expected, field=field)

    checkpoint_path = artifact_paths.get("checkpoint")
    if checkpoint_path is not None:
        require_reported_fields(
            metrics,
            ("checkpoint_size_bytes",),
            source="checkpoint manifest",
        )
        checkpoint_manifest = load_json_strict(checkpoint_path)
        if checkpoint_manifest.get("format_version") != protocol["artifact_formats"]["version"]:
            raise ProtocolValidationError("checkpoint manifest has wrong format_version")
        checkpoint_root_value = checkpoint_manifest.get("root")
        entries = checkpoint_manifest.get("files")
        if not isinstance(checkpoint_root_value, str) or not isinstance(entries, list):
            raise ProtocolValidationError("checkpoint manifest is incomplete")
        checkpoint_root = (resolved_root / checkpoint_root_value).resolve()
        if not checkpoint_root.is_relative_to(resolved_root) or not checkpoint_root.is_dir():
            raise ProtocolValidationError("checkpoint root is missing or escapes artifact root")
        expected_relative_paths: set[str] = set()
        checkpoint_size = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
                raise ProtocolValidationError(
                    f"checkpoint manifest files[{index}] is invalid"
                )
            relative_path = entry["path"]
            if not isinstance(relative_path, str):
                raise ProtocolValidationError(
                    f"checkpoint manifest files[{index}].path is invalid"
                )
            payload_path = (checkpoint_root / relative_path).resolve()
            if (
                not payload_path.is_relative_to(checkpoint_root)
                or not payload_path.is_file()
                or payload_path.is_symlink()
            ):
                raise ProtocolValidationError(
                    f"checkpoint payload is missing or escapes root: {relative_path}"
                )
            normalized_relative_path = payload_path.relative_to(checkpoint_root).as_posix()
            if normalized_relative_path in expected_relative_paths:
                raise ProtocolValidationError("checkpoint manifest contains duplicate paths")
            expected_relative_paths.add(normalized_relative_path)
            actual_size = payload_path.stat().st_size
            actual_digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            if entry["size_bytes"] != actual_size or entry["sha256"] != actual_digest:
                raise ProtocolValidationError(
                    f"checkpoint payload metadata mismatch: {relative_path}"
                )
            checkpoint_size += actual_size
        actual_relative_paths = {
            path.relative_to(checkpoint_root).as_posix()
            for path in checkpoint_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual_relative_paths != expected_relative_paths:
            raise ProtocolValidationError(
                "checkpoint manifest does not enumerate every regular file"
            )
        assert_reported_number(
            metrics["checkpoint_size_bytes"],
            checkpoint_size,
            field="checkpoint_size_bytes",
        )

    held_out_path = artifact_paths.get("held_out_evaluation")
    if held_out_path is not None:
        require_reported_fields(
            evaluation,
            ("held_out_nll", "held_out_tokens"),
            source="held_out_evaluation",
        )
        held_out = load_json_strict(held_out_path)
        if held_out.get("format_version") != protocol["artifact_formats"]["version"]:
            raise ProtocolValidationError("held_out_evaluation has wrong format_version")
        tokens = held_out.get("assistant_tokens")
        nll_sum = held_out.get("negative_log_likelihood_sum")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ProtocolValidationError("held_out_evaluation.assistant_tokens must be positive")
        nll_sum_value = require_numeric(
            nll_sum,
            context="held_out_evaluation.negative_log_likelihood_sum",
        )
        assert_reported_number(evaluation["held_out_tokens"], tokens, field="held_out_tokens")
        assert_reported_number(
            evaluation["held_out_nll"],
            nll_sum_value / tokens,
            field="held_out_nll",
        )

    ifeval_path = artifact_paths.get("ifeval_responses")
    if ifeval_path is not None:
        require_reported_fields(
            evaluation,
            (
                "ifeval_strict_prompt_accuracy",
                "ifeval_strict_instruction_accuracy",
                "ifeval_loose_prompt_accuracy",
                "ifeval_loose_instruction_accuracy",
            ),
            source="ifeval_responses",
        )
        ifeval = load_json_strict(ifeval_path)
        if ifeval.get("format_version") != protocol["artifact_formats"]["version"]:
            raise ProtocolValidationError("ifeval_responses has wrong format_version")
        prompt_total = ifeval.get("prompt_total")
        instruction_total = ifeval.get("instruction_total")
        if not isinstance(prompt_total, int) or prompt_total <= 0:
            raise ProtocolValidationError("ifeval_responses.prompt_total must be positive")
        if not isinstance(instruction_total, int) or instruction_total <= 0:
            raise ProtocolValidationError("ifeval_responses.instruction_total must be positive")
        subset_path = artifact_paths.get("ifeval_subset_manifest")
        if subset_path is None:
            raise ProtocolValidationError(
                "ifeval_subset_manifest is required to verify IFEval aggregates"
            )
        subset = load_json_strict(subset_path)
        entries = subset.get("entries")
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ProtocolValidationError("ifeval_subset_manifest entries are invalid")
        expected_instruction_total = sum(
            len(entry.get("instruction_ids", [])) for entry in entries
        )
        if prompt_total != len(entries) or instruction_total != expected_instruction_total:
            raise ProtocolValidationError(
                "IFEval totals do not match the frozen subset manifest"
            )
        score_fields = {
            "ifeval_strict_prompt_accuracy": ("strict_prompt_passed", prompt_total),
            "ifeval_strict_instruction_accuracy": ("strict_instruction_passed", instruction_total),
            "ifeval_loose_prompt_accuracy": ("loose_prompt_passed", prompt_total),
            "ifeval_loose_instruction_accuracy": ("loose_instruction_passed", instruction_total),
        }
        for field, (count_field, total) in score_fields.items():
            count = ifeval.get(count_field)
            if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= total:
                raise ProtocolValidationError(f"ifeval_responses.{count_field} is invalid")
            assert_reported_number(
                evaluation[field],
                count / total,
                field=field,
            )


def validate_result_semantics(
    result: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    started_at = parse_utc_timestamp(result["started_at"], context="started_at")
    ended_at = parse_utc_timestamp(result["ended_at"], context="ended_at")
    if ended_at < started_at:
        raise ProtocolValidationError("ended_at precedes started_at")
    run_interval_seconds = (ended_at - started_at).total_seconds()
    if run_interval_seconds <= 0:
        raise ProtocolValidationError("run interval must be positive")

    metrics = result["metrics"]
    wall_clock = metrics["training_wall_clock_seconds"]
    train_tokens = metrics["train_tokens"]
    throughput = metrics["train_tokens_per_second"]
    if wall_clock is not None and wall_clock <= 0:
        raise ProtocolValidationError("training_wall_clock_seconds must be positive")
    if wall_clock is not None and wall_clock > run_interval_seconds:
        raise ProtocolValidationError(
            "training_wall_clock_seconds exceeds run interval"
        )
    throughput_group = (wall_clock, train_tokens, throughput)
    if any(value is None for value in throughput_group) and any(
        value is not None for value in throughput_group
    ):
        raise ProtocolValidationError(
            "throughput dependency group must be entirely known or unknown"
        )
    run_profile = result["config"]["run_profile"]
    token_budget = result["config"]["token_budget"]
    if run_profile == "quality_train":
        if result["status"] == "completed" and train_tokens != token_budget:
            raise ProtocolValidationError("train_tokens does not match token_budget")
        if train_tokens is not None and train_tokens > token_budget:
            raise ProtocolValidationError("train_tokens exceeds token_budget")
    elif run_profile == "capacity_probe" and result["status"] == "completed":
        expected_capacity_tokens = (
            result["config"]["measured_steps"]
            * result["config"]["effective_tokens_per_update"]
        )
        if train_tokens != expected_capacity_tokens:
            raise ProtocolValidationError(
                "capacity train_tokens does not match measured_steps"
            )
    if wall_clock is not None and train_tokens is not None and throughput is not None:
        expected_throughput = train_tokens / wall_clock
        if not math.isclose(
            throughput,
            expected_throughput,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ProtocolValidationError(
                "train_tokens_per_second does not equal train_tokens / training_wall_clock_seconds"
            )
    p50 = metrics["step_time_p50_ms"]
    p95 = metrics["step_time_p95_ms"]
    if p50 is not None and p95 is not None and p50 > p95:
        raise ProtocolValidationError("step_time_p50_ms exceeds step_time_p95_ms")
    if (p50 is None) != (p95 is None):
        raise ProtocolValidationError(
            "step-time dependency group must be entirely known or unknown"
        )
    if (
        (p50 is not None or p95 is not None)
        and result["artifacts"]["raw_step_times"] is None
    ):
        raise ProtocolValidationError(
            "step-time aggregates require raw_step_times"
        )
    swap_start = metrics["swap_used_start_bytes"]
    swap_end = metrics["swap_used_end_bytes"]
    swap_delta = metrics["swap_delta_bytes"]
    swap_group = (swap_start, swap_end, swap_delta)
    if any(value is None for value in swap_group) and any(
        value is not None for value in swap_group
    ):
        raise ProtocolValidationError(
            "swap dependency group must be entirely known or unknown"
        )
    if swap_start is not None and swap_end is not None and swap_delta is not None:
        expected_swap_delta = swap_end - swap_start
        if swap_delta != expected_swap_delta:
            raise ProtocolValidationError("swap_delta_bytes does not equal end minus start")

    p6 = result["config"]["p6"]
    method = result["config"]["method"]
    fixed_methods = {"p6_random_mask", "p6_static_mask", "p6_learned_fixed_k"}
    if method in fixed_methods:
        fraction = p6["activation_fraction"]
        expected_dimension = round_half_up(
            fraction,
            protocol["model"]["intermediate_size"],
        )
        expected_dimensions = [expected_dimension] * protocol["model"]["num_hidden_layers"]
        if p6["realized_dimensions_per_layer"] != expected_dimensions:
            raise ProtocolValidationError(
                "realized_dimensions_per_layer does not match activation_fraction"
            )
        if not math.isclose(p6["mean_activation_fraction"], fraction, abs_tol=1e-12):
            raise ProtocolValidationError(
                "mean_activation_fraction does not match fixed activation_fraction"
            )
    elif method == "p6_variable_k":
        dimensions = p6["realized_dimensions_per_layer"]
        allowed_dimensions = set(protocol["p6"]["realized_dimensions_for_qwen2_0_5b"])
        if any(dimension not in allowed_dimensions for dimension in dimensions):
            raise ProtocolValidationError(
                "variable-k activation summary has realized dimensions outside the frozen activation grid"
            )
        expected_mean = (
            sum(dimensions)
            / len(dimensions)
            / protocol["model"]["intermediate_size"]
        )
        rounding_tolerance = 0.5 / protocol["model"]["intermediate_size"]
        if (
            not math.isclose(
                p6["mean_activation_fraction"],
                expected_mean,
                abs_tol=rounding_tolerance,
            )
        ):
            raise ProtocolValidationError(
                "variable-k activation summary is inconsistent with realized dimensions"
            )

    gate_inputs = (
        metrics["os_peak_rss_bytes"],
        swap_start,
        swap_end,
        swap_delta,
        metrics["swap_increase_streak_max"],
    )
    reported_gate = result["compute_gate"]
    has_raw_memory = result["artifacts"]["raw_memory_samples"] is not None
    raw_memory_derived_fields = (
        "mlx_peak_memory_bytes",
        "mlx_active_memory_bytes",
        "mlx_cache_memory_bytes",
        "os_peak_rss_bytes",
        "memory_free_percent_min",
        "swap_used_start_bytes",
        "swap_used_end_bytes",
        "swap_delta_bytes",
        "swap_increase_streak_max",
    )
    if not has_raw_memory and any(
        metrics[field] is not None for field in raw_memory_derived_fields
    ):
        raise ProtocolValidationError(
            "memory and swap aggregates require raw_memory_samples"
        )
    gate_is_known = reported_gate["passed"] is not None or bool(reported_gate["reasons"])
    if gate_is_known and (not has_raw_memory or any(value is None for value in gate_inputs)):
        raise ProtocolValidationError(
            "known compute_gate requires raw memory and every underlying measurement"
        )
    if any(value is None for value in gate_inputs):
        if reported_gate["passed"] is not None or reported_gate["reasons"]:
            raise ProtocolValidationError(
                "compute_gate must be unknown when required measurements are missing"
            )
        return

    gate_reasons: list[str] = []
    if metrics["os_peak_rss_bytes"] > protocol["gates"]["max_peak_rss_bytes"]:
        gate_reasons.append("peak_rss_exceeded")
    if (
        metrics["swap_delta_bytes"] > 0
        or metrics["swap_increase_streak_max"] >= 3
    ):
        gate_reasons.append("sustained_swap")
    if sorted(reported_gate["reasons"]) != sorted(gate_reasons):
        raise ProtocolValidationError("compute_gate reasons do not match measurements")
    if reported_gate["passed"] != (not gate_reasons):
        raise ProtocolValidationError("compute_gate passed does not match measurements")


def approved_deviation(
    *,
    pointer: str,
    expected: Any,
    actual: Any,
    run_id: str,
    cited_ids: list[str],
    deviations: dict[str, dict[str, Any]],
) -> str | None:
    for deviation_id in cited_ids:
        item = deviations.get(deviation_id)
        if item is None:
            continue
        if (
            item.get("approval_status") == "approved"
            and run_id in item.get("affected_run_ids", [])
            and item.get("field_path") == pointer
            and json_equal(item.get("old_value"), expected)
            and json_equal(item.get("new_value"), actual)
        ):
            return deviation_id
    return None


def validate_result(
    result: dict[str, Any],
    protocol: dict[str, Any],
    deviations: dict[str, dict[str, Any]],
    schema_path: Path = DEFAULT_SCHEMA,
    releases_path: Path = DEFAULT_RELEASES,
) -> None:
    walk_finite(result)
    schema = load_json_strict(schema_path)
    releases = load_releases(releases_path)
    release = validate_release(protocol, schema, releases)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(result), key=lambda error: list(error.path))
    if schema_errors:
        rendered = "; ".join(error.message for error in schema_errors)
        raise ProtocolValidationError(f"schema validation failed: {rendered}")

    try:
        packages = result["provenance"]["runtime"]["packages"]
        forbidden = {normalize_package(name) for name in protocol["runtime"]["forbidden_packages"]}
    except (KeyError, TypeError) as exc:
        raise ProtocolValidationError("missing field required for protocol comparison") from exc
    present_forbidden = sorted(
        name for name in packages if normalize_package(name) in forbidden
    )
    if present_forbidden:
        raise ProtocolValidationError(
            "forbidden packages in provenance.runtime.packages: "
            + ", ".join(present_forbidden)
        )
    normalized_package_names = [normalize_package(name) for name in packages]
    if len(normalized_package_names) != len(set(normalized_package_names)):
        raise ProtocolValidationError("runtime package map has duplicate normalized names")

    run_id = result["run_id"]
    cited_ids = result["deviation_ids"]
    missing_ids = sorted(set(cited_ids) - set(deviations))
    if missing_ids:
        raise ProtocolValidationError(f"unknown deviation_ids: {', '.join(missing_ids)}")
    validate_deviation_versions(result, protocol, cited_ids, deviations)

    mismatches: list[str] = []
    used_deviation_ids: set[str] = set()
    for pointer, expected in expected_values(protocol, result, release).items():
        actual = pointer_get(result, pointer)
        if json_equal(actual, expected):
            continue
        matching_deviation_id = approved_deviation(
            pointer=pointer,
            expected=expected,
            actual=actual,
            run_id=run_id,
            cited_ids=cited_ids,
            deviations=deviations,
        )
        if matching_deviation_id is None:
            mismatches.append(f"{pointer}: expected {expected!r}, got {actual!r}")
        else:
            used_deviation_ids.add(matching_deviation_id)
    if mismatches:
        raise ProtocolValidationError("unrecorded protocol deviations: " + "; ".join(mismatches))
    unused_deviation_ids = sorted(set(cited_ids) - used_deviation_ids)
    if unused_deviation_ids:
        raise ProtocolValidationError(
            "unused deviation_ids: " + ", ".join(unused_deviation_ids)
        )
    verify_result_artifacts(result, protocol)
    validate_result_semantics(result, protocol)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one P6/P8 result against protocol 0.1.0")
    parser.add_argument("result", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--deviations", type=Path, default=DEFAULT_DEVIATIONS)
    parser.add_argument("--releases", type=Path, default=DEFAULT_RELEASES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = load_json_strict(args.result)
        protocol = load_protocol(args.protocol)
        deviations = load_deviations(args.deviations)
        validate_result(result, protocol, deviations, args.schema, args.releases)
    except (OSError, ValueError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    print(f"valid: {result['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
