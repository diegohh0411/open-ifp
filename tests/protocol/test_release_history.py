from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_release_history import (
    ReleaseHistoryError,
    verify_deviation_history,
    verify_protocol_transition,
    verify_schema_transition,
    verify_release_history,
)


def write_releases(path: Path, releases: dict[str, object]) -> None:
    path.write_text(json.dumps({"releases": releases}), encoding="utf-8")


def test_release_history_allows_new_versions(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    current = tmp_path / "current.json"
    write_releases(base, {"0.1.0": {"digest": "a"}})
    write_releases(
        current,
        {"0.1.0": {"digest": "a"}, "1.0.0": {"digest": "b"}},
    )
    verify_release_history(base, current)


@pytest.mark.parametrize(
    "current_releases",
    [
        {},
        {"0.1.0": {"digest": "rewritten"}},
    ],
)
def test_release_history_rejects_removed_or_rewritten_versions(
    tmp_path: Path,
    current_releases: dict[str, object],
) -> None:
    base = tmp_path / "base.json"
    current = tmp_path / "current.json"
    write_releases(base, {"0.1.0": {"digest": "a"}})
    write_releases(current, current_releases)
    with pytest.raises(ReleaseHistoryError, match="released entry 0.1.0"):
        verify_release_history(base, current)


def test_deviation_history_allows_appending_complete_lines(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    current = tmp_path / "current.jsonl"
    base.write_bytes(b'{"deviation_id":"DEV-0001"}\n')
    current.write_bytes(
        b'{"deviation_id":"DEV-0001"}\n{"deviation_id":"DEV-0002"}\n'
    )
    verify_deviation_history(base, current)


def test_deviation_history_rejects_rewritten_prefix(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    current = tmp_path / "current.jsonl"
    base.write_bytes(b'{"deviation_id":"DEV-0001"}\n')
    current.write_bytes(b'{"deviation_id":"DEV-9999"}\n')
    with pytest.raises(ReleaseHistoryError, match="deviation ledger"):
        verify_deviation_history(base, current)


def test_frozen_semantic_change_rejects_patch_release() -> None:
    base = {"protocol_version": "0.1.0", "training": {"seed": 1}}
    current = {"protocol_version": "0.1.1", "training": {"seed": 2}}
    with pytest.raises(ReleaseHistoryError, match="major version"):
        verify_protocol_transition(base, current, [])


def test_frozen_semantic_change_requires_exact_approved_deviation() -> None:
    base = {"protocol_version": "0.1.0", "training": {"seed": 1}}
    current = {"protocol_version": "1.0.0", "training": {"seed": 2}}
    with pytest.raises(ReleaseHistoryError, match="approved deviation"):
        verify_protocol_transition(base, current, [])
    verify_protocol_transition(
        base,
        current,
        [
            {
                "field_path": "/training/seed",
                "old_value": 1,
                "new_value": 2,
                "approval_status": "approved",
                "base_protocol_version": "0.1.0",
                "effective_protocol_version": "1.0.0",
            }
        ],
    )


def test_schema_change_rejects_patch_release_without_deviation() -> None:
    base = {"type": "object"}
    current = {"type": "array"}
    with pytest.raises(ReleaseHistoryError, match="major version"):
        verify_schema_transition(
            base,
            current,
            base_version="0.1.0",
            current_version="0.1.1",
            appended_deviations=[],
        )


@pytest.mark.parametrize(
    ("base_value", "current_value"),
    [(1, True), (0, False)],
)
def test_semantic_diff_is_type_sensitive_for_booleans_and_integers(
    base_value: object,
    current_value: object,
) -> None:
    base = {"protocol_version": "0.1.0", "value": base_value}
    current = {"protocol_version": "0.1.1", "value": current_value}
    with pytest.raises(ReleaseHistoryError, match="major version"):
        verify_protocol_transition(base, current, [])
