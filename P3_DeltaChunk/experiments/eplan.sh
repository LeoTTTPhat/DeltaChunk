#!/usr/bin/env bash
# E-phase queue (Tier A), launched after the B5 freeze. Two chains run in parallel, then the timing/analysis chain:
#   CPU chain : E1 (3 streams, 8 procs) -> merge -> T1, E1-rep, E1-fid, E2 (+r=3), e2_predict, E3, E8
#   LLM/GPU   : qa_gen -> qa_prox -> E4 (native bg, no-bg control, stale-20) -> E3q, E5a/b/c -> E9 (+stale)
#   index     : (after qa_gen) E6 (bg 50k, 3 seeds, tau 0.05/0.1/0.3) -> E7 rho
#   last      : E7 micro / timing / llm (alone), e7_analysis, stats, plots
# LLM sizes follow the experiments.md re-plan (contended GPU): QA 500 WIKI + 200 K8S + 200 CPY retrieval questions,
# 300 edit-proximal questions, E9 300 questions x REC/SentCDC/KARA + closed-book. Every step is memguarded and resumable;
# rerun this script to resume (finished steps with results are cheap: E1/E3/E8 skip finished documents, LLM cache hits).
set -uo pipefail
P3="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P3"
L="$P3/logs/ephase"; mkdir -p "$L"
export CAP=${CAP:-30} TAG="" SMOKE=0 PROCS=${PROCS:-8}
export QA_N_WIKI=${QA_N_WIKI:-500} QA_N_GIT=${QA_N_GIT:-200} QA_N_PROX=${QA_N_PROX:-300}
export E9_N=${E9_N:-300} E9_CONFIGS=${E9_CONFIGS:-REC,SentCDC,KARA}
# run a frozen copy (bash reads scripts incrementally: editing run_all.sh mid-run would corrupt a running chain)
R="$L/run_all.frozen.sh"; cp experiments/run_all.sh "$R"; sed -i '' "s|^P3=.*|P3=\"$P3\"; cd \"\$P3\"|" "$R"
# Runner session 1 (Nov 9 17:30): steps run one at a time with retries. The first launch lost its whole GPU chain when
# memwatch stopped the llama-server for > 6 min (machine at 470 GB) and qa_gen's client retries ran out. Each step is
# resumable (LLM cache, per-document checkpoints), so a failed step is retried up to RETRIES times after RETRY_SLEEP s;
# a step that still fails is logged as FAILED and the chain moves on (later steps check their own inputs).
RETRIES=${RETRIES:-4}; RETRY_SLEEP=${RETRY_SLEEP:-300}
chain() {  # chain <log> <step,step,...>
  local log="$1" st
  for st in ${2//,/ }; do
    local k=1
    until STEPS=$st bash "$R" >> "$log" 2>&1; do
      echo "!!! step $st attempt $k failed $(date '+%F %T')" >> "$log"
      [ $k -ge $RETRIES ] && { echo "!!! step $st FAILED after $k attempts" >> "$log"; break; }
      k=$((k+1)); sleep $RETRY_SLEEP
    done
  done
}
echo "eplan start $(date '+%F %T')"
# (an E1 run started earlier, e.g. logs/ephase/e1_early.log, is waited for: the chain then only resumes/skips)
( while pgrep -f "experiments/e1.py --stream" >/dev/null; do sleep 60; done
  chain "$L/cpu_chain.log" e1,e1_merge,t1,e1rep,e1fid,e2,e2_predict,e3,e8 ) &
CPU=$!
chain "$L/gpu_chain.log" qa_gen,qa_prox,e4,e4_nobg,e4_stale,e3q,e5a,e5b,e9,e9_stale,e5c &
GPU=$!
# index chain: E6 (long: 3 chunkers x 5 policies x 3 seeds) + rho(N), in parallel once the WIKI questions exist
( until [ -f results/qa/questions_cpy.jsonl ] || ! kill -0 $GPU 2>/dev/null; do sleep 60; done
  [ -f results/qa/questions_cpy.jsonl ] && chain "$L/index_chain.log" e6,e7_rho ) &
IDX=$!
echo "cpu_chain pid $CPU, gpu_chain pid $GPU, index_chain pid $IDX"
wait $CPU; echo "cpu_chain exit $? $(date '+%F %T')"
wait $GPU; echo "gpu_chain exit $? $(date '+%F %T')"
wait $IDX; echo "index_chain exit $? $(date '+%F %T')"
chain "$L/last_chain.log" e7_micro,e7_timing,e7_llm,e7_analysis,stats,plots
echo "last_chain done $(date '+%F %T')"
echo "eplan done $(date '+%F %T')"
