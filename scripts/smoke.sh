#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -d .venv ]]; then
  echo "Run ./setup.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import importlib.metadata as m; import mlx; import mlx_lm; print('imports ok, mlx', m.version('mlx'))"
python scripts/generate.py --max-tokens 32
echo "smoke ok"
