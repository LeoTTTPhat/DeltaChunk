#!/usr/bin/env bash
# Round 2 long runs for DeltaChunk (C4 band-only ablation, C6 size-matched retrieval with a common background).
#
# Launch from P3_DeltaChunk under the memory guard (the guard restarts a stopped job with MEMGUARD_SHRINK=1):
#   python3 ../tools/memguard.py --cap-gb 40 --system-cap-gb 490 --cooldown 60 --max-restarts 6 \
#       --log logs/memguard_round2.log -- bash experiments/round2_long.sh
# Resumable and idempotent: every step writes a marker in results/round2/.done/ and is skipped when the marker exists;
# the replays and E4 also checkpoint inside a step (per document / per arm), so a restart resumes mid-step.
# Threads: P3_THREADS (default 8, capped at 8; 4 when MEMGUARD_SHRINK=1). SMOKE=1 runs a tiny K8S subset with _smoke
# tags (writes results/round2/LONG_DONE_smoke instead of LONG_DONE).
# Log: logs/round2_long.log (appended). End marker: results/round2/LONG_DONE.
set -euo pipefail
cd "$(dirname "$0")/.."
P3=$(pwd)
mkdir -p logs results/round2/.done
exec > >(tee -a logs/round2_long.log) 2>&1

THREADS=${P3_THREADS:-8}
[ "$THREADS" -gt 8 ] && THREADS=8
if [ "${MEMGUARD_SHRINK:-0}" = "1" ]; then THREADS=$(( THREADS > 4 ? 4 : THREADS )); fi
export P3_THREADS=$THREADS P3_MAX_PROCS=$THREADS OMP_NUM_THREADS=$THREADS RAYON_NUM_THREADS=$THREADS
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src
PY=.venv/bin/python

SMOKE=${SMOKE:-0}
if [ "$SMOKE" = "1" ]; then
  TAG=_smoke; BAND_ARGS="--streams k8s --docs 3 --tag _smoke"; E4_ARGS="--streams k8s --docs 3 --bg 500"; QTAG=_r2smoke
  DONE_FILE=results/round2/LONG_DONE_smoke
else
  TAG=""; BAND_ARGS=""; E4_ARGS="--bg 30000"; QTAG=_r2matched
  DONE_FILE=results/round2/LONG_DONE
fi
MARK=results/round2/.done
log() { echo "[$(date '+%F %T')] round2_long${TAG}: $*"; }
step() {   # step <name> <command...>: run once, mark done
  local name=$1; shift
  if [ -f "$MARK/$name$TAG" ]; then log "skip $name (done)"; return 0; fi
  log "start $name (threads $THREADS, shrink ${MEMGUARD_SHRINK:-0}): $*"
  local t0=$SECONDS
  "$@"
  date '+%F %T' > "$MARK/$name$TAG"
  log "done $name in $((SECONDS - t0)) s"
}
log "begin; memwatch: $(cat ../logs/memwatch_status.json 2>/dev/null | tr -d '\n' | cut -c1-160)"

# ---------------------------------------------------------------- C4: band-only ablation (count replay, CPU only)
step band_calibrate $PY experiments/r2_band.py calibrate $BAND_ARGS
step band_replay    $PY experiments/r2_band.py replay $BAND_ARGS --procs "$THREADS"
step band_analyze   $PY experiments/r2_band.py analyze $BAND_ARGS

# ---------------------------------------------------------------- C6: size-matched retrieval, common REC background, CPU
# Parameters matching REC's realized final-snapshot size (and STRUCT-NM's, for REC@nm) per stream.
if [ ! -s results/round2/matched_params.json ]; then
  step match_params $PY experiments/r2_match_params.py
fi
# Explicit --device cpu: the device enters E4's scoring-checkpoint hash, so this run gets its own checkpoint directory and
# never resumes the MPS (native) or the older device-less CPU checkpoints; every arm, including REC, is scored within
# this run. Final layouts are cached by chunker name and parameters only, so the nominal T=128 arms (KARA included)
# reuse their replayed layouts; KARA@match needs one new full-history replay.
ARMS=REC,FIX,STRUCT-M,SentCDC,STL,KARA,STRUCT-NM,REC@nm,KARA@match,STRUCT-M@match,SentCDC@match,STL@match,FIX@match
step e4_matched $PY experiments/e8q.py --device cpu --T 128 --bg-mode rec $E4_ARGS --chunkers "$ARMS" --tag "$QTAG"
step e4_matched_stats $PY experiments/stats_quality.py e4 --tag "$QTAG"
step e4_matched_sizes $PY - "$QTAG" <<'EOF'
import json, sys
tag = sys.argv[1]
m = json.load(open(f"results/e4_quality{tag}.meta.json"))["achieved_tok"]
t = json.load(open("results/round2/matched_params.json"))["targets"]
out = {}
for arm, by in m.items():
    for s, v in by.items():
        ref = t[s]["STRUCT-NM"] if arm in ("REC@nm", "STRUCT-NM") else t[s]["REC@T128"]
        out.setdefault(arm, {})[s] = dict(realized=v, target=ref, rel_dev=round(v / ref - 1, 4), within_5pct=abs(v / ref - 1) <= 0.05)
json.dump(out, open(f"results/round2/size_check{tag}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
EOF
if [ "$SMOKE" = "1" ]; then
  step span_smoke $PY experiments/r2_span.py --tags "$QTAG" --suffix _smoke
else
  step span_all $PY experiments/r2_span.py
fi

date '+%F %T' > "$DONE_FILE"
log "all steps done -> $DONE_FILE"
