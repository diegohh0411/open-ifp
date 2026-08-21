#!/usr/bin/env bash
# Apple Silicon only. Creates ./.venv with mlx + mlx-lm (no PyTorch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "open-ifp is MLX-only (Apple Silicon). Found $(uname -s)." >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Need Apple Silicon (arm64). Found $(uname -m)." >&2
  exit 1
fi

PYTHON_BIN="$(command -v python3.12 || command -v python3)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Need Python 3.12 (Homebrew python@3.12)." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python -c "import importlib.metadata as m; import mlx, mlx_lm; print('mlx', m.version('mlx')); print('mlx-lm', m.version('mlx-lm'))"
echo
echo "Activate with:  source .venv/bin/activate"
echo "Smoke with:     ./scripts/smoke.sh"
