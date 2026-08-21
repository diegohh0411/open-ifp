#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocol/benchmark-v0.1.yaml"
DEFAULT_SCHEMA = ROOT / "protocol/run-result.schema.json"
DEFAULT_DEVIATIONS = ROOT / "protocol/deviations.jsonl"
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


def load_deviations(path: Path = DEFAULT_DEVIATIONS) -> dict[str, dict[str, Any]]:
    deviations: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise ProtocolValidationError(f"missing deviation ledger: {path}")
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProtocolValidationError(f"invalid deviation at line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ProtocolValidationError(f"deviation line {line_number} is not an object")
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
        if item["approval_status"] not in {"pending", "approved", "rejected"}:
            raise ProtocolValidationError(
                f"deviation line {line_number} has invalid approval_status"
            )
        if deviation_id in deviations:
            raise ProtocolValidationError(f"duplicate deviation_id: {deviation_id}")
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


def expected_values(protocol: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    length = result["config"]["sequence_length"]
    profile = protocol["training"]["sequence_profiles"][length]
    return {
        "/schema_version": protocol["schema_version"],
        "/protocol_version": protocol["protocol_version"],
        "/provenance/model/repository": protocol["model"]["repository"],
        "/provenance/model/revision": protocol["model"]["revision"],
        "/provenance/tokenizer/repository": protocol["model"]["tokenizer_repository"],
        "/provenance/tokenizer/revision": protocol["model"]["tokenizer_revision"],
        "/provenance/dataset/repository": protocol["training_data"]["repository"],
        "/provenance/dataset/revision": protocol["training_data"]["revision"],
        "/provenance/evaluator/repository": protocol["evaluation"]["repository"],
        "/provenance/evaluator/revision": protocol["evaluation"]["revision"],
        "/provenance/evaluator/implementation_path": protocol["evaluation"]["implementation_path"],
        "/provenance/runtime/python": protocol["runtime"]["python"],
        "/provenance/runtime/packages/mlx": protocol["runtime"]["direct_packages"]["mlx"],
        "/provenance/runtime/packages/mlx-lm": protocol["runtime"]["direct_packages"]["mlx-lm"],
        "/provenance/runtime/packages/huggingface_hub": protocol["runtime"]["direct_packages"]["huggingface_hub"],
        "/hardware/chip": protocol["platform"]["chip"],
        "/hardware/cpu_cores": protocol["platform"]["cpu_cores"],
        "/hardware/gpu_cores": protocol["platform"]["gpu_cores"],
        "/hardware/unified_memory_bytes": protocol["platform"]["unified_memory_bytes"],
        "/config/seed": protocol["training"]["seed"],
        "/config/microbatch_size": profile["microbatch_size"],
        "/config/gradient_accumulation_steps": profile["gradient_accumulation_steps"],
        "/config/effective_tokens_per_update": protocol["training"]["effective_tokens_per_update"],
    }


def pointer_get(value: dict[str, Any], pointer: str) -> Any:
    current: Any = value
    for part in pointer.strip("/").split("/"):
        current = current[part]
    return current


def approved_deviation(
    *,
    pointer: str,
    expected: Any,
    actual: Any,
    run_id: str,
    cited_ids: list[str],
    deviations: dict[str, dict[str, Any]],
) -> bool:
    for deviation_id in cited_ids:
        item = deviations.get(deviation_id)
        if item is None:
            continue
        if (
            item.get("approval_status") == "approved"
            and run_id in item.get("affected_run_ids", [])
            and item.get("field_path") == pointer
            and item.get("old_value") == expected
            and item.get("new_value") == actual
        ):
            return True
    return False


def validate_result(
    result: dict[str, Any],
    protocol: dict[str, Any],
    deviations: dict[str, dict[str, Any]],
    schema_path: Path = DEFAULT_SCHEMA,
) -> None:
    walk_finite(result)
    schema = load_json_strict(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(result), key=lambda error: list(error.path))
    if schema_errors:
        rendered = "; ".join(error.message for error in schema_errors)
        raise ProtocolValidationError(f"schema validation failed: {rendered}")

    run_id = result["run_id"]
    cited_ids = result["deviation_ids"]
    missing_ids = sorted(set(cited_ids) - set(deviations))
    if missing_ids:
        raise ProtocolValidationError(f"unknown deviation_ids: {', '.join(missing_ids)}")

    mismatches: list[str] = []
    for pointer, expected in expected_values(protocol, result).items():
        actual = pointer_get(result, pointer)
        if actual == expected:
            continue
        if not approved_deviation(
            pointer=pointer,
            expected=expected,
            actual=actual,
            run_id=run_id,
            cited_ids=cited_ids,
            deviations=deviations,
        ):
            mismatches.append(f"{pointer}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise ProtocolValidationError("unrecorded protocol deviations: " + "; ".join(mismatches))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one P6/P8 result against protocol 0.1.0")
    parser.add_argument("result", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--deviations", type=Path, default=DEFAULT_DEVIATIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = load_json_strict(args.result)
        protocol = load_protocol(args.protocol)
        deviations = load_deviations(args.deviations)
        validate_result(result, protocol, deviations, args.schema)
    except (OSError, ValueError) as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    print(f"valid: {result['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
