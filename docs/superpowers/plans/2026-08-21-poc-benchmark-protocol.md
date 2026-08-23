# P6/P8 Proof-of-Concept Benchmark Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the versioned Day 1A protocol, machine-readable result contract, deviation enforcement, and offline validation needed to compare later P6 and P8 proof-of-concept runs.

**Architecture:** Keep the subsystem file-based and offline. A canonical YAML file owns experiment constants, a JSON Schema owns serialization and method-specific types, and a small Python CLI validates results against both the schema and approved deviation records. Tests exercise each boundary without downloading weights or datasets.

**Tech Stack:** CPython 3.12.14; candidate direct pins Apple MLX 0.31.2, `mlx-lm` 0.31.3, `huggingface_hub` 1.27.0, PyYAML 6.0.3, jsonschema 4.26.0, and pytest 9.1.1, whose compatibility is accepted only after Task 1's clean install and smoke generation; JSON Schema Draft 2020-12.

**Spec:** `docs/superpowers/specs/2026-08-21-poc-benchmark-protocol-design.md`

## Global Constraints

- Target the donated MacBook Pro: Apple M4 Max, 14-core CPU, 32-core GPU, 36 GB unified memory.
- Use CPython 3.12; record the full patch version in protocol and results.
- Execute models only through Apple MLX/Metal and `mlx-lm`.
- Do not add PyTorch, `torchvision`, `torchtune`, or a Transformers model backend.
- Pin every direct dependency with `==`; do not leave version ranges in either requirements file.
- Pin `Qwen/Qwen2-0.5B-Instruct` and its tokenizer to revision `c540970f9e29518b1d8f06ab8b24cba66ad77b6d`.
- Pin Dolly to revision `feb6109c23dc5bb14eaea059d14b9879284c9234` and Google Research IFEval to revision `13ec2c53411ad214f13709a2fcc1c1b730c605ff`.
- Keep Day 1A offline after dependency installation: no tests download models, Dolly, or IFEval.
- Treat P6 as the accepted semester lock and P8 as a candidate; shared infrastructure must not combine their method scopes.
- Preserve the user's untracked `AGENTS.md`; never stage it in a task commit.

## File Map

- `requirements.txt` — exact direct runtime pins.
- `requirements-dev.txt` — exact direct protocol-validation and test pins.
- `setup.sh` — creates the Python 3.12 environment and installs both requirements files.
- `protocol/benchmark-v0.1.yaml` — canonical protocol constants and metric definitions.
- `protocol/run-result.schema.json` — Draft 2020-12 result contract.
- `protocol/examples/dense-baseline.json` — complete valid result example.
- `protocol/deviations.jsonl` — initially empty append-only deviation ledger.
- `scripts/validate_protocol.py` — strict JSON loading, schema validation, protocol consistency, and deviation enforcement.
- `tests/protocol/test_protocol.py` — runtime, constants, schema, failure, and deviation tests.
- `README.md` — protocol validation commands and artifact map.

---

### Task 1: Lock and verify the Python/MLX environment

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/protocol/__init__.py`
- Create: `tests/protocol/test_protocol.py`
- Modify: `requirements.txt`
- Modify: `setup.sh`
- Modify: `scripts/generate.py`

**Interfaces:**
- Consumes: the existing `setup.sh`, `requirements.txt`, and `scripts/smoke.sh` flow.
- Produces: exact runtime/dev dependency files and a reproducible `.venv` used by all later plan tasks.

- [ ] **Step 1: Write the failing dependency-contract tests**

Create `tests/protocol/__init__.py` as an empty package marker. Create `tests/protocol/test_protocol.py` with:

```python
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

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
```

- [ ] **Step 2: Run the dependency-contract tests and confirm the red state**

Run:

```bash
python3.12 -m pytest tests/protocol/test_protocol.py -q
```

Expected: the command cannot yet import pytest, or the tests fail because `requirements-dev.txt` is missing and `requirements.txt` contains ranges. Either result is the required red state; do not change test expectations to accommodate the current files.

- [ ] **Step 3: Replace runtime ranges and add exact dev pins**

Replace `requirements.txt` with:

```text
# Apple MLX stack. Do not add torch / torchvision / torchtune.
mlx==0.31.2
mlx-lm==0.31.3
huggingface_hub==1.27.0
```

Create `requirements-dev.txt` with:

```text
# Offline protocol validation and tests.
PyYAML==6.0.3
jsonschema==4.26.0
pytest==9.1.1
```

In `setup.sh`, replace the single requirements installation line with:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Keep the existing Darwin, arm64, and Python checks unchanged.

In `scripts/generate.py`, import `snapshot_download` from `huggingface_hub`, add the pinned default revision, and resolve a local immutable snapshot before invoking MLX:

```python
DEFAULT_MODEL = "Qwen/Qwen2-0.5B-Instruct"
DEFAULT_MODEL_REVISION = "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"

# In main(), after --model:
parser.add_argument(
    "--revision",
    default=DEFAULT_MODEL_REVISION,
    help=f"immutable Hugging Face revision (default: {DEFAULT_MODEL_REVISION})",
)

snapshot_path = snapshot_download(
    repo_id=args.model,
    revision=args.revision,
    local_files_only=True,
)
print(f"loading {args.model}@{args.revision} from {snapshot_path} via mlx-lm…")
model, tokenizer = load(snapshot_path)
```

Remove the old `load(args.model)` call. A missing immutable cache snapshot must raise; do not retry `main` or another revision.

- [ ] **Step 4: Build the clean environment**

Run:

```bash
test ! -e .venv
./setup.sh
```

Expected: the precondition exits 0 because the repository inspection found no `.venv`; setup then exits 0 with output containing `mlx 0.31.2` and `mlx-lm ok`. If the precondition fails at execution time, stop and ask before moving or replacing that user-owned environment.

- [ ] **Step 5: Run the smoke generation**

Run:

```bash
./scripts/smoke.sh
```

Expected: exit 0, `imports ok, mlx 0.31.2`, generated text, and `smoke ok`. The model must resolve from the existing immutable cache snapshot; a moving-revision download is a failure.

- [ ] **Step 6: Run the dependency tests in the locked environment**

Run:

```bash
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
```

Expected: `5 passed`.

- [ ] **Step 7: Record the resolved environment for review**

Run:

```bash
.venv/bin/python --version
.venv/bin/python -m pip freeze
.venv/bin/python -m pip show mlx mlx-lm huggingface_hub
```

Expected: Python 3.12.14; direct package versions exactly match `requirements.txt`; the freeze contains no `torch`, `torchvision`, or `torchtune`. Save the complete freeze output in the task notes, not as a new repository artifact.

- [ ] **Step 8: Commit the environment contract**

```bash
git add requirements.txt requirements-dev.txt setup.sh scripts/generate.py tests/protocol/__init__.py tests/protocol/test_protocol.py
git commit -m "build: lock protocol validation environment"
```

Before committing, run `git diff --cached --name-only` and verify `AGENTS.md` is absent.

---

### Task 2: Add the canonical protocol and constant-level validation

**Files:**
- Create: `protocol/benchmark-v0.1.yaml`
- Create: `protocol/deviations.jsonl`
- Modify: `tests/protocol/test_protocol.py`

**Interfaces:**
- Consumes: PyYAML from Task 1.
- Produces: `load_protocol() -> dict[str, object]` in the tests and canonical constants later consumed by the JSON Schema validator.

- [ ] **Step 1: Add failing protocol-constant tests**

Append these imports and helpers to `tests/protocol/test_protocol.py`:

```python
import re

import yaml


SHA40 = re.compile(r"^[0-9a-f]{40}$")


def load_protocol() -> dict[str, object]:
    with (ROOT / "protocol/benchmark-v0.1.yaml").open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict)
    return loaded
```

Append these tests:

```python
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
```

- [ ] **Step 2: Run the new tests and confirm they fail because the protocol is absent**

Run:

```bash
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
```

Expected: the five Task 1 tests pass and the six new tests fail with `FileNotFoundError` for `protocol/benchmark-v0.1.yaml`.

- [ ] **Step 3: Create the canonical protocol YAML**

Create `protocol/benchmark-v0.1.yaml` with:

```yaml
protocol_version: "0.1.0"
schema_version: "0.1.0"
status: released
project: estancia
purpose: p6_p8_five_day_proof_of_concept_gate
governance:
  semester_lock: p6
  p8_status: candidate
  combined_method_scope_allowed: false
platform:
  chip: Apple M4 Max
  cpu_cores: 14
  gpu_cores: 32
  unified_memory_bytes: 38654705664
  rss_gate_bytes: 32212254720
runtime:
  python: "3.12.14"
  direct_packages:
    mlx: "0.31.2"
    mlx-lm: "0.31.3"
    huggingface_hub: "1.27.0"
  forbidden_packages:
    - torch
    - torchvision
    - torchtune
model:
  repository: Qwen/Qwen2-0.5B-Instruct
  revision: c540970f9e29518b1d8f06ab8b24cba66ad77b6d
  tokenizer_repository: Qwen/Qwen2-0.5B-Instruct
  tokenizer_revision: c540970f9e29518b1d8f06ab8b24cba66ad77b6d
  allow_moving_revision: false
  num_hidden_layers: 24
  intermediate_size: 4864
training_data:
  repository: databricks/databricks-dolly-15k
  revision: feb6109c23dc5bb14eaea059d14b9879284c9234
  license: CC-BY-SA-3.0
  seed: 20260821
  categories:
    - classification
    - summarization
    - information_extraction
    - brainstorming
  train_per_category: 100
  held_out_per_category: 25
  canonical_json_fields:
    - instruction
    - context
    - response
    - category
  canonicalization:
    encoding: utf-8
    sort_keys: true
    json_separators:
      - ","
      - ":"
    hash_prefix: "20260821\n"
    line_endings: lf
  require_nonempty:
    - instruction
    - response
  selection:
    sort: sha256_ascending_within_category
    train_slice:
      - 0
      - 100
    held_out_slice:
      - 100
      - 125
  serialization:
    chat_template: pinned_tokenizer
    user_fields:
      - instruction
      - context
    context_separator: "\n\nContext:\n"
    preserve_context_except_crlf_to_lf: true
  loss_tokens: assistant_only
evaluation:
  name: ifeval
  repository: google-research/google-research
  revision: 13ec2c53411ad214f13709a2fcc1c1b730c605ff
  implementation_path: instruction_following_eval
  subset_size: 100
  seed: 20260821
  prompt_hash_prefix: "20260821\n"
  selection:
    primary: largest_uncovered_instruction_id_count
    tie_break: prompt_sha256_ascending
    fill: prompt_sha256_ascending
    coverage_completion: every_available_instruction_id
  manifest_fields:
    - original_key
    - prompt_sha256
    - instruction_ids
  judge: official_rules_only
  decoding:
    strategy: greedy
    temperature: 0.0
    top_p: 1.0
    max_new_tokens: 1024
    stop: eos_only
training:
  seed: 20260821
  seed_targets:
    - python
    - numpy_when_present
    - mlx
    - data_order
    - mask_initialization
    - adapter_initialization
  quality_sequence_length: 512
  capacity_sequence_lengths:
    - 256
    - 512
  effective_tokens_per_update: 2048
  capacity_stream: deterministic_full_non_padding_token_blocks
  sequence_profiles:
    256:
      microbatch_size: 1
      gradient_accumulation_steps: 8
    512:
      microbatch_size: 1
      gradient_accumulation_steps: 4
  capacity_probe:
    warmup_steps: 20
    measured_steps: 200
p6:
  activation_grid:
    - 0.4
    - 0.6
    - 0.8
    - 1.0
  rounding: round_half_up
  realized_dimensions_for_qwen2_0_5b:
    - 1946
    - 2918
    - 3891
    - 4864
  day3_fixed_fraction: 0.6
  day3_train_tokens: 250000
  dense_reference_fraction: 1.0
p8:
  methods:
    - p8_full
    - p8_lora
    - p8_qlora
  train_tokens: 250000
  lora_rank: 8
  qlora_rank: 8
  qlora_quantization_bits: 4
measurements:
  - {name: training_wall_clock_seconds, unit: seconds, definition: training_only_excluding_evaluation}
  - {name: train_tokens, unit: tokens, definition: non_padding_tokens}
  - {name: train_tokens_per_second, unit: tokens_per_second, definition: train_tokens_divided_by_wall_clock}
  - {name: step_time_p50_ms, unit: milliseconds, definition: measured_steps_only}
  - {name: step_time_p95_ms, unit: milliseconds, definition: measured_steps_only}
  - {name: mlx_peak_memory_bytes, unit: bytes, definition: mlx_allocator_peak}
  - {name: mlx_active_memory_bytes, unit: bytes, definition: mlx_allocator_active}
  - {name: mlx_cache_memory_bytes, unit: bytes, definition: mlx_allocator_cache}
  - {name: os_peak_rss_bytes, unit: bytes, definition: process_rss_sampled_each_second}
  - {name: memory_free_percent_min, unit: percent, definition: memory_pressure_q_minimum}
  - {name: swap_used_start_bytes, unit: bytes, definition: sysctl_vm_swapusage_at_start}
  - {name: swap_used_end_bytes, unit: bytes, definition: sysctl_vm_swapusage_at_end}
  - {name: swap_delta_bytes, unit: bytes, definition: end_minus_start}
  - {name: checkpoint_size_bytes, unit: bytes, definition: recursive_regular_file_sum}
  - {name: held_out_nll, unit: nats_per_token, definition: assistant_token_negative_log_likelihood}
  - {name: held_out_tokens, unit: tokens, definition: evaluated_assistant_tokens}
  - {name: ifeval_strict_prompt_accuracy, unit: ratio, definition: official_strict_prompt_accuracy}
  - {name: ifeval_strict_instruction_accuracy, unit: ratio, definition: official_strict_instruction_accuracy}
  - {name: ifeval_loose_prompt_accuracy, unit: ratio, definition: official_loose_prompt_accuracy}
  - {name: ifeval_loose_instruction_accuracy, unit: ratio, definition: official_loose_instruction_accuracy}
sampling:
  rss_interval_seconds: 1
  memory_pressure_interval_seconds: 60
  swap_interval_seconds: 60
gates:
  sustained_swap_allowed: false
  sustained_swap_definition: end_above_start_or_three_consecutive_increases
  max_peak_rss_bytes: 32212254720
result_contract:
  schema_path: protocol/run-result.schema.json
  example_path: protocol/examples/dense-baseline.json
  deviations_path: protocol/deviations.jsonl
  raw_artifacts_required: true
```

Create `protocol/deviations.jsonl` as an empty file. Blank lines are ignored by the future reader, but do not add comments because each nonblank line must be JSON.

- [ ] **Step 4: Run the protocol tests**

Run:

```bash
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
```

Expected: `11 passed`.

- [ ] **Step 5: Commit the canonical protocol**

```bash
git add protocol/benchmark-v0.1.yaml protocol/deviations.jsonl tests/protocol/test_protocol.py
git commit -m "docs: freeze P6 P8 PoC benchmark protocol"
```

Before committing, verify the staged file list contains only these three paths.

---

### Task 3: Define and test the run-result JSON contract

**Files:**
- Create: `protocol/run-result.schema.json`
- Create: `protocol/examples/dense-baseline.json`
- Modify: `tests/protocol/test_protocol.py`

**Interfaces:**
- Consumes: canonical versions and method names from `protocol/benchmark-v0.1.yaml`.
- Produces: Draft 2020-12 schema and a completed `dense_baseline` result consumed by Task 4's cross-file validator.

- [ ] **Step 1: Add failing schema and example tests**

Append these imports and helpers to `tests/protocol/test_protocol.py`:

```python
import copy
import json

import pytest
from jsonschema import Draft202012Validator


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
```

Append these tests:

```python
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
```

- [ ] **Step 2: Run the schema tests and confirm the red state**

Run:

```bash
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
```

Expected: Task 1 and Task 2 tests pass; schema tests fail with `FileNotFoundError` for `protocol/run-result.schema.json`.

- [ ] **Step 3: Create the Draft 2020-12 JSON Schema**

Create `protocol/run-result.schema.json`. Use this exact top-level structure and field definitions; keep `additionalProperties: false` at every fixed object boundary:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://open-ifp.local/schemas/run-result-v0.1.json",
  "title": "P6/P8 PoC Run Result",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version", "protocol_version", "run_id", "status",
    "started_at", "ended_at", "provenance", "hardware", "config",
    "metrics", "evaluation", "artifacts", "deviation_ids", "failure"
  ],
  "properties": {
    "schema_version": {"const": "0.1.0"},
    "protocol_version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"},
    "run_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{7,127}$"},
    "status": {"enum": ["completed", "failed"]},
    "started_at": {"type": "string", "format": "date-time"},
    "ended_at": {"type": "string", "format": "date-time"},
    "provenance": {"$ref": "#/$defs/provenance"},
    "hardware": {"$ref": "#/$defs/hardware"},
    "config": {"$ref": "#/$defs/config"},
    "metrics": {"$ref": "#/$defs/metrics"},
    "evaluation": {"$ref": "#/$defs/evaluation"},
    "artifacts": {"$ref": "#/$defs/artifacts"},
    "deviation_ids": {
      "type": "array",
      "uniqueItems": true,
      "items": {"type": "string", "pattern": "^DEV-[0-9]{4}$"}
    },
    "failure": {
      "oneOf": [
        {"type": "null"},
        {"$ref": "#/$defs/failure"}
      ]
    }
  },
  "$defs": {
    "sha40": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "nonnegative_number_or_null": {"type": ["number", "null"], "minimum": 0},
    "nonnegative_integer_or_null": {"type": ["integer", "null"], "minimum": 0},
    "ratio_or_null": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "artifact": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path", "sha256"],
      "properties": {
        "path": {"type": "string", "pattern": "^(?!/)(?!.*\\.\\./).+$"},
        "sha256": {"$ref": "#/$defs/sha256"}
      }
    },
    "source": {
      "type": "object",
      "additionalProperties": false,
      "required": ["repository", "revision"],
      "properties": {
        "repository": {"type": "string", "minLength": 1},
        "revision": {"$ref": "#/$defs/sha40"}
      }
    },
    "dataset_source": {
      "type": "object",
      "additionalProperties": false,
      "required": ["repository", "revision", "split_manifest_sha256"],
      "properties": {
        "repository": {"type": "string", "minLength": 1},
        "revision": {"$ref": "#/$defs/sha40"},
        "split_manifest_sha256": {"$ref": "#/$defs/sha256"}
      }
    },
    "evaluator_source": {
      "type": "object",
      "additionalProperties": false,
      "required": ["repository", "revision", "implementation_path"],
      "properties": {
        "repository": {"type": "string", "minLength": 1},
        "revision": {"$ref": "#/$defs/sha40"},
        "implementation_path": {"type": "string", "minLength": 1}
      }
    },
    "model_source": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "repository", "revision", "cache_snapshot",
        "config_sha256", "tokenizer_config_sha256"
      ],
      "properties": {
        "repository": {"type": "string", "minLength": 1},
        "revision": {"$ref": "#/$defs/sha40"},
        "cache_snapshot": {"type": "string", "pattern": "^(?!/)(?!.*\\.\\./).+$"},
        "config_sha256": {"$ref": "#/$defs/sha256"},
        "tokenizer_config_sha256": {"$ref": "#/$defs/sha256"}
      }
    },
    "provenance": {
      "type": "object",
      "additionalProperties": false,
      "required": ["code_commit", "model", "tokenizer", "dataset", "evaluator", "runtime"],
      "properties": {
        "code_commit": {"$ref": "#/$defs/sha40"},
        "model": {"$ref": "#/$defs/model_source"},
        "tokenizer": {"$ref": "#/$defs/source"},
        "dataset": {"$ref": "#/$defs/dataset_source"},
        "evaluator": {"$ref": "#/$defs/evaluator_source"},
        "runtime": {
          "type": "object",
          "additionalProperties": false,
          "required": ["python", "packages"],
          "properties": {
            "python": {"type": "string", "pattern": "^3\\.12\\.[0-9]+$"},
            "packages": {
              "type": "object",
              "minProperties": 3,
              "propertyNames": {"pattern": "^[A-Za-z0-9_.-]+$"},
              "additionalProperties": {"type": "string", "minLength": 1},
              "required": ["mlx", "mlx-lm", "huggingface_hub"]
            }
          }
        }
      }
    },
    "hardware": {
      "type": "object",
      "additionalProperties": false,
      "required": ["chip", "cpu_cores", "gpu_cores", "unified_memory_bytes", "macos_version", "macos_build", "host_id"],
      "properties": {
        "chip": {"const": "Apple M4 Max"},
        "cpu_cores": {"const": 14},
        "gpu_cores": {"const": 32},
        "unified_memory_bytes": {"type": "integer", "minimum": 1},
        "macos_version": {"type": "string", "minLength": 1},
        "macos_build": {"type": "string", "minLength": 1},
        "host_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{2,63}$"}
      }
    },
    "config": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "method", "seed", "sequence_length", "microbatch_size",
        "gradient_accumulation_steps", "effective_tokens_per_update",
        "token_budget", "optimizer", "p6", "p8"
      ],
      "properties": {
        "method": {"enum": ["dense_baseline", "p6_random_mask", "p6_static_mask", "p6_learned_fixed_k", "p6_variable_k", "p8_full", "p8_lora", "p8_qlora"]},
        "seed": {"type": "integer", "minimum": 0},
        "sequence_length": {"enum": [256, 512]},
        "microbatch_size": {"const": 1},
        "gradient_accumulation_steps": {"enum": [4, 8]},
        "effective_tokens_per_update": {"const": 2048},
        "token_budget": {"type": "integer", "minimum": 1},
        "optimizer": {
          "type": "object",
          "additionalProperties": false,
          "required": ["name", "learning_rate", "weight_decay"],
          "properties": {
            "name": {"type": "string", "minLength": 1},
            "learning_rate": {"type": "number", "exclusiveMinimum": 0},
            "weight_decay": {"type": "number", "minimum": 0}
          }
        },
        "p6": {
          "oneOf": [
            {"type": "null"},
            {
              "type": "object",
              "additionalProperties": false,
              "required": ["activation_fraction", "activation_grid", "realized_dimensions_per_layer", "mean_activation_fraction"],
              "properties": {
                "activation_fraction": {"enum": [null, 0.4, 0.6, 0.8, 1.0]},
                "activation_grid": {
                  "type": "array", "minItems": 4, "maxItems": 4,
                  "prefixItems": [{"const": 0.4}, {"const": 0.6}, {"const": 0.8}, {"const": 1.0}]
                },
                "realized_dimensions_per_layer": {"type": "array", "minItems": 24, "maxItems": 24, "items": {"type": "integer", "minimum": 1, "maximum": 4864}},
                "mean_activation_fraction": {"type": "number", "minimum": 0, "maximum": 1}
              }
            }
          ]
        },
        "p8": {
          "oneOf": [
            {"type": "null"},
            {
              "type": "object",
              "additionalProperties": false,
              "required": ["regime", "rank", "quantization_bits"],
              "properties": {
                "regime": {"enum": ["full", "lora", "qlora"]},
                "rank": {"type": ["integer", "null"], "minimum": 1},
                "quantization_bits": {"type": ["integer", "null"], "minimum": 1}
              }
            }
          ]
        }
      },
      "allOf": [
        {
          "if": {"properties": {"method": {"const": "dense_baseline"}}},
          "then": {"properties": {"p6": {"type": "null"}, "p8": {"type": "null"}}}
        },
        {
          "if": {"properties": {"method": {"pattern": "^p6_"}}},
          "then": {"properties": {"p6": {"type": "object"}, "p8": {"type": "null"}}}
        },
        {
          "if": {"properties": {"method": {"enum": ["p6_random_mask", "p6_static_mask", "p6_learned_fixed_k"]}}},
          "then": {"properties": {"p6": {"properties": {"activation_fraction": {"enum": [0.4, 0.6, 0.8, 1.0]}}}}}
        },
        {
          "if": {"properties": {"method": {"const": "p6_variable_k"}}},
          "then": {"properties": {"p6": {"properties": {"activation_fraction": {"type": "null"}}}}}
        },
        {
          "if": {"properties": {"method": {"const": "p8_full"}}},
          "then": {"properties": {"p6": {"type": "null"}, "p8": {"type": "object", "properties": {"regime": {"const": "full"}, "rank": {"type": "null"}, "quantization_bits": {"type": "null"}}}}}
        },
        {
          "if": {"properties": {"method": {"const": "p8_lora"}}},
          "then": {"properties": {"p6": {"type": "null"}, "p8": {"type": "object", "properties": {"regime": {"const": "lora"}, "rank": {"const": 8}, "quantization_bits": {"type": "null"}}}}}
        },
        {
          "if": {"properties": {"method": {"const": "p8_qlora"}}},
          "then": {"properties": {"p6": {"type": "null"}, "p8": {"type": "object", "properties": {"regime": {"const": "qlora"}, "rank": {"const": 8}, "quantization_bits": {"const": 4}}}}}
        }
      ]
    },
    "metrics": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "training_wall_clock_seconds", "train_tokens", "train_tokens_per_second",
        "step_time_p50_ms", "step_time_p95_ms", "mlx_peak_memory_bytes",
        "mlx_active_memory_bytes", "mlx_cache_memory_bytes", "os_peak_rss_bytes",
        "memory_free_percent_min", "swap_used_start_bytes", "swap_used_end_bytes",
        "swap_delta_bytes", "checkpoint_size_bytes"
      ],
      "properties": {
        "training_wall_clock_seconds": {"$ref": "#/$defs/nonnegative_number_or_null"},
        "train_tokens": {"$ref": "#/$defs/nonnegative_integer_or_null"},
        "train_tokens_per_second": {"$ref": "#/$defs/nonnegative_number_or_null"},
        "step_time_p50_ms": {"$ref": "#/$defs/nonnegative_number_or_null"},
        "step_time_p95_ms": {"$ref": "#/$defs/nonnegative_number_or_null"},
        "mlx_peak_memory_bytes": {"$ref": "#/$defs/nonnegative_integer_or_null"},
        "mlx_active_memory_bytes": {"$ref": "#/$defs/nonnegative_integer_or_null"},
        "mlx_cache_memory_bytes": {"$ref": "#/$defs/nonnegative_integer_or_null"},
        "os_peak_rss_bytes": {"$ref": "#/$defs/nonnegative_integer_or_null"},
        "memory_free_percent_min": {"type": ["number", "null"], "minimum": 0, "maximum": 100},
        "swap_used_start_bytes": {"$ref": "#/$defs/nonnegative_integer_or_null"},
        "swap_used_end_bytes": {"$ref": "#/$defs/nonnegative_integer_or_null"},
        "swap_delta_bytes": {"type": ["integer", "null"]},
        "checkpoint_size_bytes": {"$ref": "#/$defs/nonnegative_integer_or_null"}
      }
    },
    "evaluation": {
      "type": "object",
      "additionalProperties": false,
      "required": ["held_out_nll", "held_out_tokens", "ifeval_strict_prompt_accuracy", "ifeval_strict_instruction_accuracy", "ifeval_loose_prompt_accuracy", "ifeval_loose_instruction_accuracy"],
      "properties": {
        "held_out_nll": {"$ref": "#/$defs/nonnegative_number_or_null"},
        "held_out_tokens": {"$ref": "#/$defs/nonnegative_integer_or_null"},
        "ifeval_strict_prompt_accuracy": {"$ref": "#/$defs/ratio_or_null"},
        "ifeval_strict_instruction_accuracy": {"$ref": "#/$defs/ratio_or_null"},
        "ifeval_loose_prompt_accuracy": {"$ref": "#/$defs/ratio_or_null"},
        "ifeval_loose_instruction_accuracy": {"$ref": "#/$defs/ratio_or_null"}
      }
    },
    "artifacts": {
      "type": "object",
      "additionalProperties": false,
      "required": ["raw_step_times", "raw_memory_samples", "ifeval_subset_manifest", "ifeval_responses", "checkpoint"],
      "properties": {
        "raw_step_times": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/artifact"}]},
        "raw_memory_samples": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/artifact"}]},
        "ifeval_subset_manifest": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/artifact"}]},
        "ifeval_responses": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/artifact"}]},
        "checkpoint": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/artifact"}]}
      }
    },
    "completed_metrics": {
      "allOf": [
        {"$ref": "#/$defs/metrics"},
        {
          "properties": {
            "training_wall_clock_seconds": {"type": "number", "minimum": 0},
            "train_tokens": {"type": "integer", "minimum": 0},
            "train_tokens_per_second": {"type": "number", "minimum": 0},
            "step_time_p50_ms": {"type": "number", "minimum": 0},
            "step_time_p95_ms": {"type": "number", "minimum": 0},
            "mlx_peak_memory_bytes": {"type": "integer", "minimum": 0},
            "mlx_active_memory_bytes": {"type": "integer", "minimum": 0},
            "mlx_cache_memory_bytes": {"type": "integer", "minimum": 0},
            "os_peak_rss_bytes": {"type": "integer", "minimum": 0},
            "memory_free_percent_min": {"type": "number", "minimum": 0, "maximum": 100},
            "swap_used_start_bytes": {"type": "integer", "minimum": 0},
            "swap_used_end_bytes": {"type": "integer", "minimum": 0},
            "swap_delta_bytes": {"type": "integer"},
            "checkpoint_size_bytes": {"type": "integer", "minimum": 0}
          }
        }
      ]
    },
    "completed_evaluation": {
      "allOf": [
        {"$ref": "#/$defs/evaluation"},
        {
          "properties": {
            "held_out_nll": {"type": "number", "minimum": 0},
            "held_out_tokens": {"type": "integer", "minimum": 0},
            "ifeval_strict_prompt_accuracy": {"type": "number", "minimum": 0, "maximum": 1},
            "ifeval_strict_instruction_accuracy": {"type": "number", "minimum": 0, "maximum": 1},
            "ifeval_loose_prompt_accuracy": {"type": "number", "minimum": 0, "maximum": 1},
            "ifeval_loose_instruction_accuracy": {"type": "number", "minimum": 0, "maximum": 1}
          }
        }
      ]
    },
    "completed_artifacts": {
      "allOf": [
        {"$ref": "#/$defs/artifacts"},
        {
          "properties": {
            "raw_step_times": {"$ref": "#/$defs/artifact"},
            "raw_memory_samples": {"$ref": "#/$defs/artifact"},
            "ifeval_subset_manifest": {"$ref": "#/$defs/artifact"},
            "ifeval_responses": {"$ref": "#/$defs/artifact"},
            "checkpoint": {"$ref": "#/$defs/artifact"}
          }
        }
      ]
    },
    "failure": {
      "type": "object",
      "additionalProperties": false,
      "required": ["stage", "type", "message"],
      "properties": {
        "stage": {"enum": ["setup", "data", "training", "evaluation", "serialization", "validation"]},
        "type": {"type": "string", "minLength": 1},
        "message": {"type": "string", "minLength": 1}
      }
    }
  },
  "allOf": [
    {
      "if": {"properties": {"status": {"const": "completed"}}},
      "then": {
        "properties": {
          "failure": {"type": "null"},
          "metrics": {"$ref": "#/$defs/completed_metrics"},
          "evaluation": {"$ref": "#/$defs/completed_evaluation"},
          "artifacts": {"$ref": "#/$defs/completed_artifacts"}
        }
      }
    },
    {
      "if": {"properties": {"status": {"const": "failed"}}},
      "then": {"properties": {"failure": {"$ref": "#/$defs/failure"}}}
    }
  ]
}
```

After creating the file, run `.venv/bin/python -c 'import json; json.load(open("protocol/run-result.schema.json"))'` to catch JSON syntax errors before running pytest.

- [ ] **Step 4: Create the complete dense-baseline example**

Create `protocol/examples/dense-baseline.json` with every required field. Use real immutable source revisions and clearly synthetic metric/artifact digests:

```json
{
  "schema_version": "0.1.0",
  "protocol_version": "0.1.0",
  "run_id": "poc-20260821-dense-20260821-99554284",
  "status": "completed",
  "started_at": "2026-08-21T18:00:00Z",
  "ended_at": "2026-08-21T18:10:00Z",
  "provenance": {
    "code_commit": "99554284db9bb899b5693eef0c6b5be913d1901c",
    "model": {
      "repository": "Qwen/Qwen2-0.5B-Instruct",
      "revision": "c540970f9e29518b1d8f06ab8b24cba66ad77b6d",
      "cache_snapshot": "models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d",
      "config_sha256": "0000000000000000000000000000000000000000000000000000000000000001",
      "tokenizer_config_sha256": "0000000000000000000000000000000000000000000000000000000000000002"
    },
    "tokenizer": {
      "repository": "Qwen/Qwen2-0.5B-Instruct",
      "revision": "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
    },
    "dataset": {
      "repository": "databricks/databricks-dolly-15k",
      "revision": "feb6109c23dc5bb14eaea059d14b9879284c9234",
      "split_manifest_sha256": "0000000000000000000000000000000000000000000000000000000000000003"
    },
    "evaluator": {
      "repository": "google-research/google-research",
      "revision": "13ec2c53411ad214f13709a2fcc1c1b730c605ff",
      "implementation_path": "instruction_following_eval"
    },
    "runtime": {
      "python": "3.12.14",
      "packages": {
        "mlx": "0.31.2",
        "mlx-lm": "0.31.3",
        "huggingface_hub": "1.27.0"
      }
    }
  },
  "hardware": {
    "chip": "Apple M4 Max",
    "cpu_cores": 14,
    "gpu_cores": 32,
    "unified_memory_bytes": 38654705664,
    "macos_version": "15.6",
    "macos_build": "24G84",
    "host_id": "m4max-36gb-01"
  },
  "config": {
    "method": "dense_baseline",
    "seed": 20260821,
    "sequence_length": 512,
    "microbatch_size": 1,
    "gradient_accumulation_steps": 4,
    "effective_tokens_per_update": 2048,
    "token_budget": 250000,
    "optimizer": {"name": "adamw", "learning_rate": 0.00001, "weight_decay": 0.0},
    "p6": null,
    "p8": null
  },
  "metrics": {
    "training_wall_clock_seconds": 600.0,
    "train_tokens": 250000,
    "train_tokens_per_second": 416.6666666667,
    "step_time_p50_ms": 120.0,
    "step_time_p95_ms": 150.0,
    "mlx_peak_memory_bytes": 8000000000,
    "mlx_active_memory_bytes": 7000000000,
    "mlx_cache_memory_bytes": 500000000,
    "os_peak_rss_bytes": 9000000000,
    "memory_free_percent_min": 48.0,
    "swap_used_start_bytes": 0,
    "swap_used_end_bytes": 0,
    "swap_delta_bytes": 0,
    "checkpoint_size_bytes": 1000000000
  },
  "evaluation": {
    "held_out_nll": 2.5,
    "held_out_tokens": 10000,
    "ifeval_strict_prompt_accuracy": 0.25,
    "ifeval_strict_instruction_accuracy": 0.3,
    "ifeval_loose_prompt_accuracy": 0.32,
    "ifeval_loose_instruction_accuracy": 0.38
  },
  "artifacts": {
    "raw_step_times": {"path": "results/example/raw-step-times.jsonl", "sha256": "0000000000000000000000000000000000000000000000000000000000000004"},
    "raw_memory_samples": {"path": "results/example/raw-memory.jsonl", "sha256": "0000000000000000000000000000000000000000000000000000000000000005"},
    "ifeval_subset_manifest": {"path": "results/example/ifeval-subset.jsonl", "sha256": "0000000000000000000000000000000000000000000000000000000000000006"},
    "ifeval_responses": {"path": "results/example/ifeval-responses.jsonl", "sha256": "0000000000000000000000000000000000000000000000000000000000000007"},
    "checkpoint": {"path": "results/example/checkpoint.safetensors", "sha256": "0000000000000000000000000000000000000000000000000000000000000008"}
  },
  "deviation_ids": [],
  "failure": null
}
```

- [ ] **Step 5: Run schema tests and repair only schema defects**

Run:

```bash
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
```

Expected: all tests pass. The completed-run tests must prove that every metric, evaluation value, and raw artifact is non-null; failed runs may retain nulls while carrying a non-null `failure` object.

- [ ] **Step 6: Commit the schema contract**

```bash
git add protocol/run-result.schema.json protocol/examples/dense-baseline.json tests/protocol/test_protocol.py
git commit -m "feat: define benchmark run result contract"
```

Before committing, run `git diff --cached --check` and verify only these three files are staged.

---

### Task 4: Enforce protocol consistency and deviations

**Files:**
- Create: `scripts/validate_protocol.py`
- Modify: `tests/protocol/test_protocol.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `protocol/benchmark-v0.1.yaml`, `protocol/run-result.schema.json`, `protocol/deviations.jsonl`, and a result JSON path.
- Produces: `validate_result(result: dict[str, object], protocol: dict[str, object], deviations: dict[str, dict[str, object]]) -> None` and CLI exit 0 for valid results or exit 1 with deterministic error messages.

- [ ] **Step 1: Add failing strict-loading and deviation tests**

Append these imports to `tests/protocol/test_protocol.py`:

```python
import subprocess
import sys

from scripts.validate_protocol import ProtocolValidationError, load_deviations, load_json_strict, validate_result
```

Append these tests:

```python
def test_strict_loader_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_json_strict(path)


def test_result_matching_protocol_has_no_deviation() -> None:
    validate_result(dense_example(), load_protocol(), {})


def test_unrecorded_seed_change_is_rejected() -> None:
    result = dense_example()
    result["config"]["seed"] = 7
    with pytest.raises(ProtocolValidationError, match="/config/seed"):
        validate_result(result, load_protocol(), {})


def test_approved_exact_deviation_allows_seed_change() -> None:
    result = dense_example()
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
            "rationale": "exercise deviation validation",
            "comparability_impact": "not comparable to protocol seed",
            "approval_status": "approved"
        }
    }
    validate_result(result, load_protocol(), deviations)


def test_deviation_must_match_run_path_and_value() -> None:
    result = dense_example()
    result["config"]["seed"] = 7
    result["deviation_ids"] = ["DEV-0001"]
    deviations = {
        "DEV-0001": {
            "deviation_id": "DEV-0001",
            "timestamp": "2026-08-21T18:30:00Z",
            "author": "Diego Hernandez",
            "affected_run_ids": ["another-run"],
            "field_path": "/config/seed",
            "old_value": 20260821,
            "new_value": 8,
            "rationale": "wrong target",
            "comparability_impact": "none",
            "approval_status": "approved"
        }
    }
    with pytest.raises(ProtocolValidationError, match="/config/seed"):
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
        "approval_status": "approved"
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
```

- [ ] **Step 2: Run the new tests and confirm import failure**

Run:

```bash
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.validate_protocol'`.

- [ ] **Step 3: Implement the strict cross-file validator**

Create `scripts/validate_protocol.py` with:

```python
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
```

- [ ] **Step 4: Run focused validator tests**

Run:

```bash
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
```

Expected: all tests pass, including strict non-finite rejection, unrecorded mismatch rejection, exact approved deviation acceptance, duplicate deviation rejection, and CLI success.

- [ ] **Step 5: Add user-facing protocol commands to the README**

First replace the moving-revision and automatic-download language in the existing README with:

````markdown
**Weights:** Hugging Face [`Qwen/Qwen2-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) at immutable revision `c540970f9e29518b1d8f06ab8b24cba66ad77b6d`.

The smoke command is offline and loads only that cached revision. If the snapshot is missing, it fails instead of falling back to `main` or downloading another revision.
````

Delete the optional community 4-bit model paragraph and command because that model is outside protocol `0.1.0`. Then append this section:

````markdown
## Benchmark protocol

Day 1A's shared P6/P8 proof-of-concept contract lives under `protocol/`:

- `benchmark-v0.1.yaml` freezes sources, splits, budgets, shapes, and metrics.
- `run-result.schema.json` defines completed and failed run records.
- `examples/dense-baseline.json` is a schema-valid illustrative result.
- `deviations.jsonl` is the append-only exception ledger.

The instruction subset comes from [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) under CC BY-SA 3.0; preserve that attribution in derived dataset manifests and documentation.

Validate the protocol and example without network access:

```bash
source .venv/bin/activate
python -m pytest tests/protocol/test_protocol.py -q
python scripts/validate_protocol.py protocol/examples/dense-baseline.json
```

A real result is comparable only when the validator exits successfully. Never change a source, selected example, seed, training shape, or metric definition without adding an approved deviation and updating the protocol version as specified in the design.
````

- [ ] **Step 6: Run the full Day 1A verification suite**

Run:

```bash
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
.venv/bin/python scripts/validate_protocol.py protocol/examples/dense-baseline.json
git diff --check
git status --short
```

Expected:

- All protocol tests pass with zero failures.
- Validator prints `valid: poc-20260821-dense-20260821-99554284` and exits 0.
- `git diff --check` prints nothing.
- `git status --short` lists only the Task 4 paths plus the user's untracked `AGENTS.md` before staging.

- [ ] **Step 7: Review the implementation against Day 1A acceptance criteria**

Confirm each requirement with a file and test:

```text
Versioned model/data/seed/length/budget/metric protocol -> protocol/benchmark-v0.1.yaml
Machine-readable timing/memory/swap/checkpoint result -> protocol/run-result.schema.json
Valid concrete serialization -> protocol/examples/dense-baseline.json
No silent metric or dataset changes -> scripts/validate_protocol.py + protocol/deviations.jsonl
Offline verification -> tests/protocol/test_protocol.py
```

Every mapping must have a passing test. Add a specific failing test before implementation whenever a mapping is not yet covered.

- [ ] **Step 8: Commit the validator and documentation**

```bash
git add scripts/validate_protocol.py tests/protocol/test_protocol.py README.md
git commit -m "feat: validate protocol results and deviations"
```

Run `git diff --cached --name-only` immediately before the commit and verify `AGENTS.md` is absent.

## Final verification

After all four task commits, run fresh commands from the repository root:

```bash
./scripts/smoke.sh
.venv/bin/python -m pytest tests/protocol/test_protocol.py -q
.venv/bin/python scripts/validate_protocol.py protocol/examples/dense-baseline.json
git log -5 --oneline
git status --short
```

Required evidence before claiming completion:

- Smoke generation exits 0 through MLX.
- All protocol tests pass with zero failures.
- The dense example validates and prints its run ID.
- The four implementation commits appear after this plan's documentation commit, with design commit `dd84215` still in their ancestry.
- The only unrelated worktree entry is the preserved untracked `AGENTS.md`.
