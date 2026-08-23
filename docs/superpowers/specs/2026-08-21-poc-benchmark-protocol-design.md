# P6/P8 Proof-of-Concept Benchmark Protocol Design

**Status:** Approved by the user on 2026-08-21

**Strict executable amendment:** Approved by the user on 2026-08-22 after PR review. Protocol `0.1.0` now materializes canonical Dolly/IFEval manifests, pins the complete Python environment and model-file digests, fingerprints the released YAML/schema, and validates cross-field result semantics and compute-gate outcomes.

**Protocol release:** `0.1.0`

**Project:** Estancia de investigación

**Primary task:** [Day 1A — Freeze the P6/P8 benchmark protocol](https://app.notion.com/p/3c3e954db47681e09852c12aa95f67e5)

## Purpose

Day 1A must make every P6 and P8 proof-of-concept run comparable before their implementations diverge. The protocol is intentionally smaller than a publication benchmark: it freezes one model, one small training corpus, one held-out split, one objective evaluation, one seed, two capacity-probe sequence lengths, the P6 activation grid, and a shared result contract.

The protocol is a feasibility gate, not evidence for a final paper claim. It must reveal whether the implementation trains, whether P6 masks learn non-random structure, whether P8 adaptation regimes produce a measurable trade-off, and whether the required runs fit the donated Mac. It does not need enough models, tasks, or seeds for a publishable conclusion.

Supporting P8 in this shared protocol does not change project governance: P6 remains the accepted semester lock, P8 remains a candidate, and the five-day gate may select P6, P8, or neither without combining their method scopes.

## Architectural decision

The protocol will be a small, file-based subsystem with four responsibilities:

1. An immutable protocol document defines every comparability-sensitive input and metric.
2. A JSON Schema defines the result interface shared by later P6 and P8 runners.
3. A valid example result proves that the schema is usable before training begins.
4. An append-only deviation log records every intentional departure from protocol `0.1.0`.

Later data, training, evaluation, and measurement code consumes these files. Those runners may change internally without changing the protocol interface. A run is comparable only when it validates against the schema and either matches the protocol exactly or cites a recorded deviation.

## Scope

### Included in Day 1A

- Freeze protocol version `0.1.0`.
- Freeze model, data, split, evaluation, seed, sequence lengths, effective update size, P6 activation fractions, decoding, and metric definitions.
- Create the protocol YAML, result JSON Schema, valid example result, append-only deviation log, and validation tests.
- Resolve the current MLX environment once in a clean Python 3.12 virtual environment, replace direct dependency ranges with exact equality pins, and record the complete environment in the protocol.
- Validate that an example dense-baseline result satisfies the shared contract.

### Excluded from Day 1A

- Implementing the Qwen FFN mask, training loop, LoRA, QLoRA, or full fine-tuning.
- Implementing the IFEval generation runner or macOS memory sampler.
- Running the five-day experiments.
- Adding Qwen 1.5B, extra tasks, additional seeds, energy measurement, or paper-scale statistics.
- Adding PyTorch, Transformers-based model execution, or NVIDIA-specific code.

Those items belong to Day 1B and later gate tasks. Day 1A defines their required inputs and outputs.

The strict amendment makes one exception to the original scope boundary: Day 1A materializes selection manifests—not dataset contents—so later runners can prove that they used the exact same rows and prompts.

## Canonical artifacts

Implementation will create these files:

- `protocol/benchmark-v0.1.yaml` — human-readable canonical protocol.
- `protocol/run-result.schema.json` — machine-readable result contract.
- `protocol/examples/dense-baseline.json` — complete valid example.
- `protocol/deviations.jsonl` — append-only deviation records; initially empty.
- `tests/protocol/test_protocol.py` — protocol/schema consistency and negative validation tests.
- `requirements-dev.txt` — exact pins for protocol validation and testing tools only.

The YAML and JSON Schema are authoritative together. The YAML owns experiment constants and metric definitions. The schema owns serialization types, required fields, conditional fields, units, and failure representation. The example is illustrative and cannot override either authoritative file.

## Frozen protocol

### Platform and runtime

- Hardware target: MacBook Pro with Apple M4 Max, 14-core CPU, 32-core GPU, and 36 GB unified memory.
- Operating system: the exact macOS product version and build are recorded in every run.
- Python: CPython 3.12, with the full patch version recorded in every run.
- Execution stack: Apple MLX and `mlx-lm`; model tensors and generation run through MLX/Metal.
- PyTorch, `torchvision`, `torchtune`, and a Transformers model backend are prohibited.
- Direct Python dependencies use exact equality pins. Version ranges are not valid for a released protocol.
- The Day 1A implementation resolves the existing dependency floors in a fresh virtual environment, verifies one dense generation, records `python --version` and `pip freeze`, and then writes the resolved direct versions into both `requirements.txt` and the protocol. Protocol `0.1.0` remains unreleased until those exact values exist.
- Every run records all installed packages as a name-to-version map. A missing Python, MLX, `mlx-lm`, or `huggingface_hub` version makes the result invalid.

This resolution rule avoids inventing an untested MLX/`mlx-lm` combination while still making the first accepted environment immutable.

### Model and tokenizer

- Repository: `Qwen/Qwen2-0.5B-Instruct`.
- Hugging Face revision: `c540970f9e29518b1d8f06ab8b24cba66ad77b6d`.
- Tokenizer: loaded from the same repository and revision.
- Model and tokenizer files must be acquired by immutable revision, never by the moving name `main`.
- The run records the repository, revision, exact cache-relative snapshot locator, and SHA-256 digest of every model/tokenizer payload used by MLX, including `model.safetensors`, tokenizer vocabulary/merges, and configuration files. The locator must not contain a username or absolute home-directory path.
- Qwen 1.5B is not part of protocol `0.1.0`; it is a later capacity probe and requires a recorded deviation or a new protocol version.

### Instruction-training data

- Dataset: `databricks/databricks-dolly-15k`.
- Revision: `feb6109c23dc5bb14eaea059d14b9879284c9234`.
- License: CC BY-SA 3.0; attribution must remain in repository documentation.
- Categories: `classification`, `summarization`, `information_extraction`, and `brainstorming`.
- Per category: 100 training examples and 25 held-out examples.
- Totals: 400 training examples and 100 held-out examples.
- The held-out rows are never used for optimization, early stopping, mask selection, or hyperparameter choice.
- IFEval prompts are never used for optimization or mask selection.

Selection is deterministic. For every eligible Dolly row, create a canonical UTF-8 JSON representation containing `instruction`, `context`, `response`, and `category`, with sorted keys and no insignificant whitespace. Compute SHA-256 over `20260821\n` followed by that canonical JSON. Within each category, sort ascending by the digest; assign the first 100 rows to training and the next 25 to held-out. Reject rows whose instruction or response is empty. Preserve the optional context exactly except for normalizing CRLF to LF.

Training serialization uses the tokenizer's pinned chat template. The user message is the instruction followed, when context is non-empty, by `\n\nContext:\n` and the context. The assistant message is the reference response. Loss is computed on assistant tokens only.

### Objective evaluation

- Benchmark: Google Research IFEval.
- Repository: `google-research/google-research`.
- Revision: `13ec2c53411ad214f13709a2fcc1c1b730c605ff`.
- Implementation path: `instruction_following_eval/` from that revision.
- Evaluator: the official strict and loose instruction-following checks; no LLM judge and no locally invented scoring rules.
- Evaluation subset: exactly 100 prompts selected deterministically from the pinned `input_data.jsonl`.

Subset selection first sorts candidate prompts by SHA-256 over `20260821\n` followed by the exact prompt text. It then greedily selects the prompt that covers the largest number of instruction IDs not yet represented, breaking ties by the digest. After every available instruction ID has at least one selected prompt, it fills the remaining slots in digest order. The subset manifest records each selected prompt's original key, prompt digest, and instruction IDs. Any change to the selected keys is a dataset change and requires a deviation record.

Generation is deterministic: apply the pinned Qwen chat template, use greedy decoding, set temperature to `0`, set top-p to `1`, stop only on the model's EOS token, and allow at most 1,024 new tokens. Record both prompt-level and instruction-level strict and loose accuracy. Store all prompts, raw responses, and per-instruction outcomes for auditability.

### Seed and training shapes

- Proof-of-concept seed: `20260821`.
- Seed Python, NumPy when present, MLX, data ordering, mask initialization, and adapter initialization from the same recorded seed.
- Microbatch size: `1` sequence.
- Capacity-probe sequence lengths: `256` and `512` tokens.
- Gradient accumulation: `8` at length 256 and `4` at length 512.
- Effective tokens per optimizer update: `2,048` in both cases.
- Pack the deterministic training stream into full token blocks for capacity measurements; do not count padding as training tokens.
- Quality and learning-signal runs use length 512. Length 256 exists only for the Day 2 capacity comparison.
- P8's matched-budget micro-study uses 250,000 non-padding training tokens for full fine-tuning, LoRA rank 8, and 4-bit QLoRA rank 8.
- P6's Day 3 fixed-k learning-signal study uses the 60% activation budget and exactly 250,000 non-padding training tokens.

### P6 activation grid

- Activated FFN intermediate dimensions: 40%, 60%, 80%, and 100% of each layer's intermediate dimension.
- Convert each fraction to an integer `k` with `round_half_up(fraction * intermediate_dimension)`.
- Apply the same `k` rule independently to every layer.
- The 100% path is the dense equivalence reference.
- The 60% point is the preregistered fixed-k point for the Day 3 random-versus-static-versus-learned comparison; later variable-k work evaluates the full grid.
- Random, static, learned fixed-k, and variable-k methods report the realized activated dimension count per layer and the mean activated fraction over tokens.
- P8 runs set all P6 mask fields to JSON `null`; they must not silently populate them with 100%.

## Measurements and units

Every completed training run records:

- Training wall-clock seconds, excluding evaluation.
- Non-padding training tokens.
- Training tokens per second as `train_tokens / training_wall_clock_seconds`.
- Per-step duration p50 and p95 in milliseconds over measured steps.
- MLX peak, active, and cache memory in bytes.
- Peak operating-system RSS in bytes, sampled once per second for the benchmark process.
- Minimum system memory-free percentage reported by `memory_pressure -Q` during the run, sampled once per minute and at start/end.
- Swap used at start, once per minute, and at end plus signed start-to-end delta in bytes, parsed from `sysctl vm.swapusage`.
- Checkpoint size in bytes, defined as the recursive sum of regular checkpoint files.
- Held-out assistant-token negative log-likelihood and evaluated token count.
- IFEval strict prompt accuracy, strict instruction accuracy, loose prompt accuracy, and loose instruction accuracy.

Day 2 capacity runs use 20 warm-up steps followed by 200 measured steps for Qwen 0.5B. Warm-up steps do not contribute to step-time percentiles or throughput. Raw step durations and raw memory samples are saved as artifacts; aggregate values alone are insufficient. The raw measured-step total cannot exceed training wall-clock time. Raw memory uses distinct timestamped streams for one-second RSS/MLX samples, one-minute memory-pressure samples, and one-minute swap samples, with start/end samples in every stream.

All sizes use bytes, all durations state their unit in the field name, and all ratios are JSON numbers between `0` and `1`. JSON `NaN`, positive infinity, and negative infinity are forbidden.

## Result contract

Every result has these top-level objects:

- `schema_version` and `protocol_version`.
- `run_id`, `status`, `started_at`, and `ended_at`.
- `provenance`: code commit, model, tokenizer, data, evaluator, and runtime versions.
- `hardware`: chip, core counts, unified memory, macOS version/build, and host identifier.
- `config`: either the `quality_train` or `capacity_probe` run profile, method, seed, sequence length, microbatch, accumulation, effective tokens/update, token budget, warm-up/measured steps, optimizer settings, P6 mask settings, and P8 adapter settings. Capacity probes are step-budgeted dense baselines with 20 warm-up and 200 measured steps; quality runs are token-budgeted at length 512.
- `metrics`: timing, throughput, MLX memory, OS RSS, memory pressure, swap, and checkpoint size.
- `evaluation`: held-out loss/token count and IFEval scores.
- `artifacts`: repository-contained paths and SHA-256 digests for raw step and memory samples, held-out evaluation totals, IFEval counts, checkpoint manifests, and subset manifests. A checkpoint manifest must enumerate the exact recursive regular-file set with per-file sizes and hashes. The validator recomputes every reported aggregate that these artifacts support.
- `deviation_ids`: zero or more identifiers present in `protocol/deviations.jsonl`.
- `failure`: `null` for completed runs or a structured failure object for failed runs.

`status` is either `completed` or `failed`. Completed runs require every protocol metric. Failed runs retain all available partial measurements and require `failure.stage`, `failure.type`, and `failure.message`. A crash, out-of-memory event, sustained swap, evaluator error, or invalid output is saved as a failed result rather than omitted.

Method-specific schema conditions prevent impossible combinations. P6 mask methods require an activation fraction or variable-budget summary and prohibit P8 adapter fields. P8 LoRA and QLoRA require rank 8 and prohibit P6 mask fields. Dense baselines require both groups to be `null`.

## Versioning and deviations

`protocol/deviations.jsonl` is append-only. Each record contains:

- `deviation_id` in sequential `DEV-0001` form.
- UTC timestamp.
- Author.
- Affected run IDs.
- Exact protocol field path.
- Old and new JSON values.
- Rationale.
- Expected comparability impact.
- Approval status.

A run that differs from the YAML without citing a matching deviation fails validation. Silent fallbacks are prohibited. If a model, dataset, selected example, seed, metric definition, or training-shape value changes, record a deviation and increment the protocol major version. Adding an optional method-specific field increments the minor version. Clarifying prose without changing behavior increments the patch version.

`protocol/releases.json` is consistency metadata within one checkout. The protocol CI job supplies the external trust boundary by comparing it with the pull request's base branch and rejecting removal or modification of any released version; only new version entries may be appended. The same job requires the base branch's deviation ledger to remain an exact byte prefix of the proposed ledger. A changed frozen YAML value requires a new major protocol version and an exact approved deviation for every changed field; a patch-version registry entry cannot launder a semantic change.

The five-day gate compares protocol-matching runs by default. Deviated runs are displayed separately unless the deviation explicitly states and justifies comparability.

## Failure handling

- Missing immutable assets fail before the run; code must not fall back from a revision to a moving branch.
- Dataset rows that fail the frozen eligibility rules are excluded before hashing; later manual filtering is a dataset change.
- A schema validation error prevents a run from being reported as completed.
- A nonzero evaluator exit, missing IFEval response, or unsupported instruction ID marks evaluation failed.
- Sustained swap means the end sample is above the start sample or swap-used values increase in three consecutive one-minute samples. The run remains recorded but fails the compute gate.
- Peak RSS above 30 GiB (`32,212,254,720` bytes) marks the run as compute-gate failure even if training finishes.
- Raw artifacts are retained for both successful and failed runs.

## Validation strategy

Day 1A tests are protocol tests, not model-quality tests:

1. Parse the YAML and require every frozen field above.
2. Validate the dense example against the JSON Schema.
3. Remove each required top-level object in parameterized tests and verify validation fails.
4. Insert `NaN`, a negative byte count, an invalid ratio, a moving source revision, and an unsupported method; verify each fails.
5. Verify both sequence-length/accumulation pairs equal 2,048 tokens per update.
6. Verify the Dolly category counts sum to 400 train and 100 held-out.
7. Verify the activation grid is exactly `[0.4, 0.6, 0.8, 1.0]`.
8. Verify every completed method satisfies its method-specific P6/P8 conditions.
9. Change a protocol constant without a matching deviation and verify consistency validation fails.
10. Add a matching deviation record and verify the same result validates as deviated.

No test downloads model weights or datasets. Network access is unnecessary for protocol validation.

## Acceptance mapping

- **Versioned protocol records model revision, dataset revision, seeds, lengths, budgets, and metric definitions:** `protocol/benchmark-v0.1.yaml` plus its consistency tests.
- **Machine-readable run template includes the requested timing, memory, swap, and checkpoint fields:** `protocol/run-result.schema.json` and `protocol/examples/dense-baseline.json`.
- **No metric or dataset changes after Day 1 without a deviation:** append-only `protocol/deviations.jsonl`, result `deviation_ids`, and cross-file consistency validation.

## Sources

- [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- [Databricks Dolly 15K dataset card](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/feb6109c23dc5bb14eaea059d14b9879284c9234/README.md)
- [Google Research IFEval](https://github.com/google-research/google-research/tree/13ec2c53411ad214f13709a2fcc1c1b730c605ff/instruction_following_eval)
- [EleutherAI IFEval integration](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ifeval), consulted as an alternative but not selected because the official model-agnostic evaluator is smaller and preserves the MLX-only boundary.
- [P6 workstream](https://app.notion.com/p/3c1e954db476814481fbfb9159859cc6)
- [P8 candidate workstream](https://app.notion.com/p/3c3e954db4768182a038f207082caccc)
- [Five-day P6 XOR P8 feasibility gate](https://app.notion.com/p/3c3e954db4768134b14ac8253e4aa27e)
