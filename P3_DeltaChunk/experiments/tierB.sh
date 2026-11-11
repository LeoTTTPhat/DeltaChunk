#!/bin/bash
# Tier B driver (runner session 4). Starts only after every Tier A result file exists; launches no new step at or after
# the hard stop (default Nov 10 14:00 local, env TIERB_STOP="YYYY-mm-dd HH:MM"). Cheapest first; every step runs under
# memguard and is resumable (E1/E4/E6 checkpoints), so re-running this script skips finished work.
#   1. REC-O counts (E1, 3 streams, tag _tierB) + e1_merge --tag _tierB
#   2. E8 quality points (E4 at native 30k background): SentCDC lambda=1 at T=128 (experiments/e8q.py; REC/SentCDC
#      resume from the primary E4 checkpoints), then REC/STRUCT-M/SentCDC at T=200 and T=320; stats_quality per tag
#   3. E6 REBUILD tau 0.05 / 0.1, seed 0 (tag _tauB)
# usage: nohup bash experiments/tierB.sh > logs/ephase/tierB.log 2>&1 &   (WAIT_EPLAN=0 skips waiting for eplan.sh)
P3="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P3" || exit 1
PY="$P3/.venv/bin/python"; export PYTHONPATH="$P3/src:$P3/experiments"
G="python3 $P3/../tools/memguard.py --cap-gb ${CAP:-30} --max-restarts 8 --log $P3/logs/memguard.log --"
STOP="${TIERB_STOP:-2026-11-10 14:00}"; STOP_S=$(date -j -f "%Y-%m-%d %H:%M" "$STOP" +%s)
PROCS=$(cat logs/ephase/MAX_PROCS 2>/dev/null || echo 4)
TIER_A_FILES="e5b_policies.json e9_answers.csv e9_answers_stale.csv e7_unit_costs.json e7_timing.csv e7c_llm.json
  e7_analysis.json e4_quality.csv e5a_late.json e5c_ctx_cost.json e6_index.csv e7_rho.json e1_edits.parquet"
log() { echo "=== $(date '+%F %T') $*"; }
ok_time() { [ "$(date +%s)" -lt "$STOP_S" ] || { log "hard stop $STOP reached: not launching $1"; return 1; }; }

if [ "${WAIT_EPLAN:-1}" = 1 ]; then
  while pgrep -f "bash experiments/eplan.sh" > /dev/null; do sleep 60; done
  log "eplan.sh finished"
fi
missing=""; for f in $TIER_A_FILES; do [ -e "results/$f" ] || missing="$missing $f"; done
if [ -n "$missing" ]; then log "Tier A incomplete (missing:$missing): Tier B not started"; exit 2; fi
log "all Tier A files present"
$PY experiments/fresh_stats.py > logs/ephase/fresh_stats.log 2>&1; $PY experiments/tables.py > logs/ephase/tables_tierA.log 2>&1; $PY experiments/summary.py > logs/ephase/summary_tierA.log 2>&1
log "tables/summary regenerated (rc tables/summary in logs/ephase/{tables,summary}_tierA.log)"

ok_time reco || exit 0
log "REC-O counts"
for s in wiki k8s cpy; do $G $PY experiments/e1.py --stream $s --chunkers REC-O --tag _tierB --procs "$PROCS" || log "!!! e1 REC-O $s rc=$?"; done
$PY experiments/e1_merge.py --tag _tierB --runbooks "" || log "!!! e1_merge _tierB rc=$?"

ok_time e8q_lam || exit 0
log "E8 quality: SentCDC lambda=1 at T=128"
$G $PY experiments/e8q.py --chunkers REC,SentCDC,SentCDC@lam=1 --bg 30000 --tag _e8q_lam1 || log "!!! e8q lam1 rc=$?"
$PY experiments/stats_quality.py e4 --tag _e8q_lam1 || log "!!! stats e8q_lam1 rc=$?"
for T in 200 320; do
  ok_time "e8q_T$T" || exit 0
  log "E8 quality: T=$T"
  $G $PY experiments/e4.py --T $T --chunkers REC,STRUCT-M,SentCDC --bg 30000 --tag _e8q_T$T || log "!!! e8q T$T rc=$?"
  $PY experiments/stats_quality.py e4 --tag _e8q_T$T || log "!!! stats e8q_T$T rc=$?"
done

ok_time e6_tauB || exit 0
log "E6 REBUILD tau 0.05/0.1 seed 0"
E6_TAUS=0.05,0.1 $G $PY experiments/e6.py --policies REBUILD --seeds 0 --bg 50000 --tag _tauB || log "!!! e6 tauB rc=$?"
log "tierB done"
