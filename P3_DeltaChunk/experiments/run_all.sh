#!/usr/bin/env bash
# Full EditStream-RAG pipeline (Tier A). Every heavy step runs under the ICDE27 memory guard; every driver is resumable.
# TAG="" for the full run, TAG=_smoke with SMOKE=1 for the 20-document smoke. Order follows experiments.md (schedule)
# and PLAN.md (E-phase). B5 (review finding B4-m run_all): every Tier A step with the exact arguments of the E-phase,
# including the r=3 E2 run, the proximal question set, the stale variants of E4/E9, E3q, E5a/b/c, E7 and the analysis.
set -euo pipefail
P3="$(cd "$(dirname "$0")/.." && pwd)"; cd "$P3"
PY="$P3/.venv/bin/python"; G="python3 $P3/../tools/memguard.py --cap-gb ${CAP:-40} --log $P3/logs/memguard.log --"
TAG="${TAG:-}"; DOCS=""; SM="${SMOKE:-0}"; [ "$SM" = 1 ] && DOCS="--docs 20"
PROCS="${PROCS:-8}"; [ "$SM" = 1 ] && PROCS=3
export PYTHONPATH="$P3/src"
sm() { [ "$SM" = 1 ] && echo "$1" || echo "${2:-}"; }   # smoke-only / full-only arguments

# STEPS=a,b,c runs only the named steps (e.g. STEPS=e1_merge,t1 for a partial rerun); default: all.
want() { [ -z "${STEPS:-}" ] || [[ ",$STEPS," == *",$1,"* ]]; }
step() { echo "=== $(date '+%F %T') $*"; }

want data && { step data; bash experiments/get_data.sh all; }
want wiki && { step wiki; [ -f data/streams/wiki.parquet ] || { $G $PY -m deltachunk.data.wiki_extract scan; $G $PY -m deltachunk.data.wiki_extract convert; $PY -m deltachunk.data.wiki_extract merge; }; }
want git && { step git; for s in k8s cpy; do [ -f data/streams/$s.parquet ] || $G $PY -m deltachunk.data.git_extract $s --procs 5; done; }
want calibrate && { step calibrate; [ -f results/calibration.json ] || $G $PY experiments/calibrate.py; }
want checks && { step checks; $PY experiments/b2_checks.py; }
# ---------------------------------------------------------------- CPU: counts
want e1 && { step e1; for s in wiki k8s cpy; do $G $PY experiments/e1.py --stream $s $DOCS --tag "$TAG" --procs $PROCS; done; }
want e1_merge && { step e1_merge; $G $PY experiments/e1_merge.py --tag "$TAG"; }
want t1 && { step t1; $PY experiments/e0_t1.py --tag "$TAG"; }
want e1rep && { step e1rep; $PY experiments/e1rep.py --tag "$TAG" $(sm "--pairs 10"); }
want e1fid && { step e1fid; $PY experiments/e1fid.py --tag "$TAG" $(sm "--docs 10"); }
want e2 && { step e2; for s in wiki k8s cpy; do $G $PY experiments/e2.py --stream $s $DOCS --tag "$TAG" --procs $PROCS; $G $PY experiments/e2.py --stream $s $DOCS --tag "$TAG" --procs $PROCS --r 3; done; }
want e2_predict && { step e2_predict; $PY experiments/e2_predict.py --tag "$TAG" $(sm "--sample 300 --boot 200"); }
want e3 && { step e3; for s in wiki k8s cpy; do $G $PY experiments/e3.py --stream $s $DOCS --tag "$TAG" --procs $PROCS; done; }
want e8 && { step e8; for s in wiki k8s; do $G $PY experiments/e8.py --stream $s $DOCS --tag "$TAG"; done; }
# ---------------------------------------------------------------- LLM / GPU (llama-server on :8301: logs/llama_cmd.txt)
want qa_gen && { step qa_gen; for s in wiki k8s cpy; do n=${QA_N_WIKI:-1000}; [ $s != wiki ] && n=${QA_N_GIT:-500}; [ "$SM" = 1 ] && n=5; $G $PY experiments/qa_gen.py --stream $s --n $n --tag "$TAG" $DOCS; done; }
want qa_prox && { step qa_prox; $G $PY experiments/qa_gen.py --stream wiki --n $(sm 20 ${QA_N_PROX:-300}) --per_doc 2 --proximal-only --tag "_prox$TAG"; }
want e4 && { step e4; $G $PY experiments/e4.py --tag "$TAG" --qtag "$TAG" $(sm "--bg 2000 --docs 20" "--bg 30000"); }
want e4_nobg && { step e4_nobg; $G $PY experiments/e4.py --tag "_nobg$TAG" --qtag "$TAG" --bg-mode none $(sm "--docs 20"); }
want e4_stale && { step e4_stale; $G $PY experiments/e4.py --tag "${TAG}_stale" --qtag "$TAG" --stale 20 --chunkers REC $(sm "--bg 2000 --docs 20" "--bg 30000"); }
want e3q && { step e3q; $G $PY experiments/e3q.py --tag "$TAG" $(sm "--docs 20 --probes 5"); }
want e5a && { step e5a; $G $PY experiments/e5a.py --tag "$TAG" $(sm "--edits 5"); }
want e5b && { step e5b; $G $PY experiments/e5b.py --tag "$TAG" --qtag "_prox$TAG" --docs-from-q $(sm "--last 5"); }
want e5c && { step e5c; $G $PY experiments/e5c.py --tag "$TAG" $(sm "--chunks 4 --reps 2" "--chunks 200 --reps 3"); }
want e6 && { step e6; $G $PY experiments/e6.py --tag "$TAG" --qtag "$TAG" $(sm "--docs 20 --edits 1000 --bg 2000 --probe 500 --seeds 0" "--bg ${E6_BG:-50000}"); }
want e7_rho && { step e7_rho; $G $PY experiments/e7.py rho --tag "$TAG" $(sm "--sizes 20000"); }
want e9 && { step e9; $G $PY experiments/e9.py --tag "$TAG" --ctag "$TAG" --qtag "$TAG" --configs "${E9_CONFIGS:-REC,STRUCT-M,SentCDC,KARA}" $(sm "--n 5" "--n ${E9_N:-600}"); }
want e9_stale && { step e9_stale; $G $PY experiments/e9.py --tag "${TAG}_stale" --ctag "${TAG}_stale" --qtag "$TAG" --configs REC-stale20 $(sm "--n 5" "--n ${E9_N:-600}"); }
# ---------------------------------------------------------------- timing alone, last
want e7_micro && { step e7_micro; $G $PY experiments/e7.py micro --tag "$TAG" $(sm "--micro-reps 3"); }
want e7_timing && { step e7_timing; $G $PY experiments/e7.py timing --tag "$TAG" $(sm "--docs 3 --reps 1" "--docs 20 --reps 3"); }
want e7_llm && { step e7_llm; $G $PY experiments/e7.py llm --tag "$TAG" $(sm "--n 5" "--n 200"); }
# ---------------------------------------------------------------- analysis
want e7_analysis && { step e7_analysis; $PY experiments/e7_analysis.py --tag "$TAG" --utag "$TAG"; }
want stats && { step stats; $PY experiments/stats_quality.py e4 --tag "$TAG"; $PY experiments/stats_quality.py e9 --tag "$TAG"; }
want plots && { step plots; $PY experiments/plots.py --tag "$TAG"; }
want tables && { step tables; $PY experiments/tables.py --tag "$TAG"; }
want summary && { step summary; $PY experiments/summary.py --tag "$TAG"; }
step done
