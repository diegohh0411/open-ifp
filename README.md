# open-ifp

Open-weight instruction-following pruning playground. Locked Estancia paper: **P6** (variable-budget IFP). This repo is not PyTorch.

**Runtime:** Apple [MLX](https://github.com/ml-explore/mlx) via [`mlx-lm`](https://github.com/ml-explore/mlx-lm). Python is only the driver; tensors and generation run on Metal, not `torch`.

**Weights:** Hugging Face. Default playground model: [`Qwen/Qwen2-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct).

## Setup (macOS, Apple Silicon)

```bash
cd open-ifp
./setup.sh
source .venv/bin/activate
./scripts/smoke.sh
```

First smoke downloads the HF checkpoint into `~/.cache/huggingface` and lets `mlx-lm` convert it. 0.5B is small (about 1 GB on disk).

Optional 4-bit community convert (faster, still Hugging Face):

```bash
python scripts/generate.py --model mlx-community/Qwen2-0.5B-Instruct-4bit
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
