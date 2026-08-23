# open-ifp

Open-weight instruction-following pruning playground. Locked Estancia paper: **P6** (variable-budget IFP). This repo is not PyTorch.

**Runtime:** Apple [MLX](https://github.com/ml-explore/mlx) via [`mlx-lm`](https://github.com/ml-explore/mlx-lm). Python is only the driver; tensors and generation run on Metal, not `torch`.

**Weights:** Hugging Face [`Qwen/Qwen2-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) at immutable revision `c540970f9e29518b1d8f06ab8b24cba66ad77b6d`.

The smoke command is offline and loads only that cached revision. If the snapshot is missing, it fails instead of falling back to `main` or downloading another revision.

## Setup (macOS, Apple Silicon)

```bash
cd open-ifp
./setup.sh
source .venv/bin/activate
./scripts/smoke.sh
```

## Commands

```bash
source .venv/bin/activate
python scripts/generate.py
python scripts/generate.py --prompt "Explain instruction-following pruning in one sentence."
python scripts/generate.py --model Qwen/Qwen2-0.5B-Instruct --max-tokens 80
```

## Layout

```
open-ifp/
  setup.sh
  requirements.txt          # direct MLX runtime pins (no torch)
  requirements-dev.txt      # direct protocol-test pins
  requirements-lock.txt     # complete released Python environment
  protocol/                 # released recipe, manifests, schema, and ledger
  scripts/generate.py
  scripts/smoke.sh
```

Do not add PyTorch. Training an IFP mask comes later; this repository currently provides the released proof-of-concept contract and validation infrastructure.

Upstream: [Qwen2](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct), [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Benchmark protocol

Day 1A's shared P6/P8 proof-of-concept contract lives under `protocol/`:

- `benchmark-v0.1.yaml` freezes sources, splits, budgets, shapes, and metrics.
- `releases.json` pins canonical digests for the released YAML and result schema.
- `run-result.schema.json` defines completed and failed run records.
- `examples/dense-baseline.json` is a validator-valid illustrative result backed by hashed fixtures in `results/example/`.
- `manifests/` pins the exact 400/100 Dolly train/held-out rows and 100 IFEval prompts.
- `deviations.jsonl` is the append-only exception ledger.

The instruction subset comes from [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) under CC BY-SA 3.0; preserve that attribution in derived dataset manifests and documentation.

Validate the protocol and example without network access:

```bash
source .venv/bin/activate
python -m pytest tests/protocol -q
python scripts/validate_protocol.py protocol/examples/dense-baseline.json
```

The validator verifies the released protocol/schema fingerprints, complete environment lock, exact data-manifest hashes, every model/tokenizer payload digest, quality and capacity run profiles, P6 realized dimensions, raw-artifact hashes and aggregates, separate RSS/pressure/swap cadences, checkpoint manifests, IFEval denominators, and the compute-gate outcome. A real result is comparable only when this command exits successfully. CI also compares `releases.json` and the deviation-ledger byte prefix with the base branch so released identities cannot be rewritten in place.

Rebuild the committed manifests from locally downloaded copies of the two pinned source files:

```bash
python scripts/materialize_protocol_manifests.py \
  --dolly-source /path/to/databricks-dolly-15k.jsonl \
  --ifeval-source /path/to/input_data.jsonl
```

The script rejects source files whose SHA-256 digests do not match protocol `0.1.0`. Regenerated manifest digests must remain identical to the values in `benchmark-v0.1.yaml`.

Never change a source, selected example, seed, training shape, budget, metric definition, schema, or released environment in place. Add a sequential deviation record with UTC timestamp, base/effective protocol versions, exact old/new values, rationale, impact, and approval; comparability-changing deviations require a new major effective version.
