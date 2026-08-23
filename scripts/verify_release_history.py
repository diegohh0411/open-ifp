#!/usr/bin/env python3
"""Reject removal or rewriting of protocol versions released on the base branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


class ReleaseHistoryError(ValueError):
    pass


def load_releases(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("releases"), dict):
        raise ReleaseHistoryError(f"invalid release registry: {path}")
    return value["releases"]


def verify_release_history(base_path: Path, current_path: Path) -> None:
    base = load_releases(base_path)
    current = load_releases(current_path)
    for version, released_entry in base.items():
        if current.get(version) != released_entry:
            raise ReleaseHistoryError(
                f"released entry {version} was removed or rewritten"
            )


def verify_deviation_history(base_path: Path, current_path: Path) -> None:
    base = base_path.read_bytes()
    current = current_path.read_bytes()
    if base and not base.endswith(b"\n"):
        raise ReleaseHistoryError("base deviation ledger lacks a final newline")
    if not current.startswith(base):
        raise ReleaseHistoryError("deviation ledger is not an immutable byte prefix")
    appended = current[len(base) :]
    if appended and not appended.endswith(b"\n"):
        raise ReleaseHistoryError("appended deviation ledger record is incomplete")


def semver_major(value: str) -> int:
    try:
        return int(value.split(".", maxsplit=1)[0])
    except (AttributeError, ValueError) as exc:
        raise ReleaseHistoryError(f"invalid protocol version: {value!r}") from exc


def pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


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


def semantic_changes(
    base: Any,
    current: Any,
    pointer: str = "",
) -> list[tuple[str, Any, Any]]:
    if isinstance(base, dict) and isinstance(current, dict):
        changes: list[tuple[str, Any, Any]] = []
        for key in sorted(set(base) | set(current)):
            child_pointer = f"{pointer}/{pointer_escape(str(key))}"
            if key not in base:
                changes.append((child_pointer, None, current[key]))
            elif key not in current:
                changes.append((child_pointer, base[key], None))
            else:
                changes.extend(
                    semantic_changes(base[key], current[key], child_pointer)
                )
        return changes
    if not json_equal(base, current):
        return [(pointer or "/", base, current)]
    return []


def verify_protocol_transition(
    base: dict[str, Any],
    current: dict[str, Any],
    appended_deviations: list[dict[str, Any]],
) -> None:
    base_version = base.get("protocol_version")
    current_version = current.get("protocol_version")
    changes = [
        change
        for change in semantic_changes(base, current)
        if change[0] != "/protocol_version"
    ]
    if not changes:
        return
    if semver_major(current_version) <= semver_major(base_version):
        raise ReleaseHistoryError(
            "frozen semantic changes require a new major version"
        )
    for pointer, old_value, new_value in changes:
        matching = any(
            deviation.get("field_path") == pointer
            and json_equal(deviation.get("old_value"), old_value)
            and json_equal(deviation.get("new_value"), new_value)
            and deviation.get("approval_status") == "approved"
            and deviation.get("base_protocol_version") == base_version
            and deviation.get("effective_protocol_version") == current_version
            for deviation in appended_deviations
        )
        if not matching:
            raise ReleaseHistoryError(
                f"frozen change {pointer} lacks an exact approved deviation"
            )


def verify_schema_transition(
    base: dict[str, Any],
    current: dict[str, Any],
    *,
    base_version: str,
    current_version: str,
    appended_deviations: list[dict[str, Any]],
) -> None:
    changes = semantic_changes(base, current, "/result_schema")
    if not changes:
        return
    if semver_major(current_version) <= semver_major(base_version):
        raise ReleaseHistoryError(
            "result-schema semantic changes require a new major version"
        )
    for pointer, old_value, new_value in changes:
        matching = any(
            deviation.get("field_path") == pointer
            and json_equal(deviation.get("old_value"), old_value)
            and json_equal(deviation.get("new_value"), new_value)
            and deviation.get("approval_status") == "approved"
            and deviation.get("base_protocol_version") == base_version
            and deviation.get("effective_protocol_version") == current_version
            for deviation in appended_deviations
        )
        if not matching:
            raise ReleaseHistoryError(
                f"result-schema change {pointer} lacks an exact approved deviation"
            )


def load_appended_deviations(
    base_path: Path,
    current_path: Path,
) -> list[dict[str, Any]]:
    base = base_path.read_bytes()
    current = current_path.read_bytes()
    appended = current[len(base) :].decode("utf-8")
    deviations: list[dict[str, Any]] = []
    for line_number, line in enumerate(appended.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReleaseHistoryError(
                f"appended deviation line {line_number} is not an object"
            )
        deviations.append(value)
    return deviations


def load_protocol(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseHistoryError(f"invalid protocol document: {path}")
    return value


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseHistoryError(f"invalid JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--base-deviations", type=Path)
    parser.add_argument("--current-deviations", type=Path)
    parser.add_argument("--base-protocol", type=Path)
    parser.add_argument("--current-protocol", type=Path)
    parser.add_argument("--base-schema", type=Path)
    parser.add_argument("--current-schema", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        verify_release_history(args.base, args.current)
        if (args.base_deviations is None) != (args.current_deviations is None):
            raise ReleaseHistoryError(
                "both deviation ledger paths must be provided together"
            )
        if args.base_deviations is not None:
            verify_deviation_history(
                args.base_deviations,
                args.current_deviations,
            )
        if (args.base_protocol is None) != (args.current_protocol is None):
            raise ReleaseHistoryError(
                "both protocol document paths must be provided together"
            )
        if args.base_protocol is not None:
            if args.base_deviations is None:
                raise ReleaseHistoryError(
                    "protocol transition verification requires deviation ledgers"
                )
            base_protocol = load_protocol(args.base_protocol)
            current_protocol = load_protocol(args.current_protocol)
            appended_deviations = load_appended_deviations(
                args.base_deviations,
                args.current_deviations,
            )
            verify_protocol_transition(
                base_protocol,
                current_protocol,
                appended_deviations,
            )
            if (args.base_schema is None) != (args.current_schema is None):
                raise ReleaseHistoryError(
                    "both result schema paths must be provided together"
                )
            if args.base_schema is not None:
                verify_schema_transition(
                    load_json_object(args.base_schema),
                    load_json_object(args.current_schema),
                    base_version=base_protocol["protocol_version"],
                    current_version=current_protocol["protocol_version"],
                    appended_deviations=appended_deviations,
                )
    except (OSError, json.JSONDecodeError, ReleaseHistoryError) as exc:
        print(f"invalid: {exc}")
        return 1
    print("release history is append-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
