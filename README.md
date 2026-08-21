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
  requirements.txt          # mlx + mlx-lm only (no torch)
  scripts/generate.py
  scripts/smoke.sh
```

Do not add PyTorch. Training an IFP mask comes later; this commit is load + generate only.

Upstream: [Qwen2](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct), [mlx-lm](https://github.com/ml-explore/mlx-lm).

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
