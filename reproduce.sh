#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$ROOT/P3_DeltaChunk"
PY="${ARTIFACT_PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || { echo "Run bash setup.sh first, or set ARTIFACT_PYTHON." >&2; exit 2; }
export ARTIFACT_PYTHON="$PY" PYTHONPATH="$PROJECT/src:$PROJECT:${PYTHONPATH:-}" MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "$PROJECT"
case "${1:-analysis}" in
 analysis)
  "$PY" experiments/e5b_stats.py
  "$PY" experiments/round2_analysis.py matched
  "$PY" experiments/round2_analysis.py tails
  "$PY" experiments/round2_analysis.py rep
  "$PY" experiments/summary.py
 ;;
 test)
  "$PY" -m pytest -p no:cacheprovider tests/test_core.py tests/test_predict.py -q
 ;;
 test-all)
  "$PY" -m pytest tests/ -q
 ;;
 run)
 shift
 [ "$#" -gt 0 ] || { echo "Usage: bash reproduce.sh run <script.py> [arguments]" >&2; exit 2; }
 exec "$PY" "$@"
 ;;
 *) echo "Usage: bash reproduce.sh [analysis|test|test-all|run <script.py> ...]" >&2; exit 2 ;;
esac
