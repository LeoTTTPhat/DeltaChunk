#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$ROOT/tools/memory_guard.py" --limit-gb "${LIMIT_GB:-16}" --reserve-gb "${RESERVE_GB:-4}" --wait-lock 0 --max-restarts 0 --state "$ROOT/runs/reproduction.json" -- bash "$ROOT/reproduce.sh" "$@"
