#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$ROOT/P3_DeltaChunk"
command -v uv >/dev/null || { echo "Install uv and rerun setup.sh." >&2; exit 2; }
[ -x "$ROOT/.venv/bin/python" ] || uv venv --python 3.12 "$ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" -r "$PROJECT/requirements.txt"
uv pip install --python "$ROOT/.venv/bin/python" psutil==7.2.2 pytest==9.2.0
# Legacy drivers resolve the same interpreter from either workspace level.
[ -e "$PROJECT/.venv" ] || ln -s ../.venv "$PROJECT/.venv"
mkdir -p "$ROOT/logs" "$ROOT/runs" "$PROJECT/logs" "$PROJECT/work"
echo "Environment ready. Run bash reproduce.sh analysis."
