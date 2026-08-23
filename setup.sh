#!/usr/bin/env bash
# Apple Silicon only. Creates the complete released protocol environment.
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

PYTHON_BIN="$(command -v python3.12 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Need Python 3.12.14 (Homebrew python@3.12)." >&2
  exit 1
fi
"${PYTHON_BIN}" scripts/check_python_version.py --expected 3.12.14

"${PYTHON_BIN}" -m venv --clear .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install pip==26.2.1
python -m pip install -r requirements-lock.txt
python -c "import importlib.metadata as m; import mlx, mlx_lm; print('mlx', m.version('mlx')); print('mlx-lm', m.version('mlx-lm'))"
echo
echo "Activate with:  source .venv/bin/activate"
echo "Smoke with:     ./scripts/smoke.sh"
