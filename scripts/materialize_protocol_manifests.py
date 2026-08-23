#!/usr/bin/env python3
"""Materialize the exact Dolly and IFEval subsets frozen by protocol 0.1.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocol/benchmark-v0.1.yaml"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def canonical_dolly_row(row: dict[str, Any], fields: list[str]) -> dict[str, str]:
    canonical: dict[str, str] = {}
    for field in fields:
        value = row.get(field, "")
        if not isinstance(value, str):
            raise ValueError(f"Dolly field {field!r} must be a string")
        canonical[field] = normalize_newlines(value)
    return canonical


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def materialize_dolly(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    prefix = config["canonicalization"]["hash_prefix"].encode("utf-8")
    fields = config["canonical_json_fields"]
    required = config["require_nonempty"]
    entries: list[dict[str, Any]] = []

    for category in config["categories"]:
        eligible: list[tuple[str, int]] = []
        for source_index, row in enumerate(rows):
            canonical = canonical_dolly_row(row, fields)
            if canonical["category"] != category:
                continue
            if any(not canonical[field].strip() for field in required):
                continue
            row_sha256 = sha256_bytes(prefix + canonical_json(canonical))
            eligible.append((row_sha256, source_index))
        eligible.sort(key=lambda item: (item[0], item[1]))

        train_count = config["train_per_category"]
        held_out_count = config["held_out_per_category"]
        required_count = train_count + held_out_count
        if len(eligible) < required_count:
            raise ValueError(
                f"category {category!r} has {len(eligible)} eligible rows; "
                f"need {required_count}"
            )
        for selection_index, (row_sha256, source_index) in enumerate(
            eligible[:required_count]
        ):
            entries.append(
                {
                    "category": category,
                    "row_sha256": row_sha256,
                    "source_index": source_index,
                    "split": "train" if selection_index < train_count else "held_out",
                }
            )

    return {
        "manifest_version": "0.1.0",
        "kind": "dolly_train_held_out",
        "repository": config["repository"],
        "revision": config["revision"],
        "source_sha256": source_sha256,
        "entries": entries,
    }


def materialize_ifeval(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    prefix = config["prompt_hash_prefix"].encode("utf-8")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt")
        instruction_ids = row.get("instruction_id_list")
        if not isinstance(prompt, str) or not isinstance(instruction_ids, list):
            raise ValueError("IFEval rows require prompt and instruction_id_list")
        if not all(isinstance(item, str) for item in instruction_ids):
            raise ValueError("IFEval instruction IDs must be strings")
        candidates.append(
            {
                "original_key": row.get("key"),
                "prompt_sha256": sha256_bytes(prefix + prompt.encode("utf-8")),
                "instruction_ids": instruction_ids,
            }
        )

    uncovered = {
        instruction_id
        for candidate in candidates
        for instruction_id in candidate["instruction_ids"]
    }
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    subset_size = config["subset_size"]

    while uncovered and len(selected) < subset_size:
        chosen = min(
            remaining,
            key=lambda candidate: (
                -len(uncovered.intersection(candidate["instruction_ids"])),
                candidate["prompt_sha256"],
                str(candidate["original_key"]),
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        uncovered.difference_update(chosen["instruction_ids"])

    remaining.sort(
        key=lambda candidate: (
            candidate["prompt_sha256"],
            str(candidate["original_key"]),
        )
    )
    selected.extend(remaining[: subset_size - len(selected)])
    if len(selected) != subset_size:
        raise ValueError(f"IFEval has only {len(selected)} selectable prompts")

    return {
        "manifest_version": "0.1.0",
        "kind": "ifeval_objective_subset",
        "repository": config["repository"],
        "revision": config["revision"],
        "source_sha256": source_sha256,
        "entries": selected,
    }


def render_manifest(manifest: dict[str, Any]) -> bytes:
    rendered = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    return rendered.encode("utf-8")


def resolve_output_path(relative_path: str) -> Path:
    manifest_root = (ROOT / "protocol/manifests").resolve()
    path = (ROOT / relative_path).resolve()
    if not path.is_relative_to(manifest_root) or path == manifest_root:
        raise ValueError(f"manifest output escapes repository manifest directory: {relative_path}")
    return path


def atomic_write(path: Path, rendered: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    expected_sha256: str,
) -> str:
    rendered = render_manifest(manifest)
    digest = sha256_bytes(rendered)
    if digest != expected_sha256:
        raise ValueError(
            "generated manifest digest mismatch: "
            f"expected {expected_sha256}, got {digest}"
        )
    atomic_write(path, rendered)
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dolly-source", type=Path, required=True)
    parser.add_argument("--ifeval-source", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    for source_path, section in (
        (args.dolly_source, protocol["training_data"]),
        (args.ifeval_source, protocol["evaluation"]),
    ):
        actual = sha256_file(source_path)
        if actual != section["source_sha256"]:
            raise ValueError(
                f"source digest mismatch for {source_path}: "
                f"expected {section['source_sha256']}, got {actual}"
            )

    dolly_manifest = materialize_dolly(
        load_jsonl(args.dolly_source),
        protocol["training_data"],
        protocol["training_data"]["source_sha256"],
    )
    ifeval_manifest = materialize_ifeval(
        load_jsonl(args.ifeval_source),
        protocol["evaluation"],
        protocol["evaluation"]["source_sha256"],
    )
    prepared: list[tuple[Path, dict[str, Any], str]] = []
    for manifest, section in (
        (dolly_manifest, protocol["training_data"]),
        (ifeval_manifest, protocol["evaluation"]),
    ):
        path = resolve_output_path(section["manifest"]["path"])
        expected_digest = section["manifest"]["sha256"]
        actual_digest = sha256_bytes(render_manifest(manifest))
        if actual_digest != expected_digest:
            raise ValueError(
                "generated manifest digest mismatch: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        prepared.append((path, manifest, expected_digest))

    for path, manifest, expected_digest in prepared:
        digest = write_manifest(
            path,
            manifest,
            expected_sha256=expected_digest,
        )
        print(f"{path.relative_to(ROOT)} {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
