#!/bin/bash
# Start the ICDE27 project memory watchdog if it is not already running (idempotent).
# Policy: project total <= 200 GB (soft 180 GB blocks new memguard jobs), machine safety cap 470 GB (only if project >= 20 GB;
# two consecutive samples, at most one such stop per 60 s). At 490 GiB stop a project job regardless of its share.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDF="$ROOT/logs/memwatch.pid"
STATUS="$ROOT/logs/memwatch_status.json"
mkdir -p "$ROOT/logs"
# A heartbeat is visible even when the host sandbox prevents ps from inspecting an existing daemon.
if python3 - "$STATUS" "$PIDF" <<'PY'
import json, os, sys, time
try:
    with open(sys.argv[1]) as handle:
        status = json.load(handle)
    with open(sys.argv[2]) as handle:
        pid = int(handle.read())
    active = pid > 1 and time.time() - os.path.getmtime(sys.argv[1]) < 90 and status["cap_gb"] == 200
except (OSError, ValueError, KeyError):
    active = False
sys.exit(0 if active else 1)
PY
then
  echo "memwatch already running (pid $(cat "$PIDF"))"; exit 0
fi
cd "$ROOT"
if ! python3 "$ROOT/tools/memwatch.py" --once >/dev/null; then
  echo "Cannot start project memory accounting from this shell." >&2; exit 75
fi
nohup python3 "$ROOT/tools/memwatch.py" --cap-gb 200 --soft-gb 180 --sys-cap-gb 470 --sys-warn-gb 430 --sys-min-project-gb 20 --sys-hard-gb 490 --sys-grace 60 --interval 5 --cooldown 120 >> "$ROOT/logs/memwatch.out" 2>&1 &
sleep 3
echo "memwatch started (pid $(cat "$PIDF"))"; cat "$ROOT/logs/memwatch_status.json" | head -5
