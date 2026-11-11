"""Round 2 (C5): resynchronization length of greedy packing versus MAXP on every edit.

For REC, STRUCT-M and SentCDC at the nominal T=128 calibration of each stream, each consecutive retained pair is
re-chunked and the new layout is walked left to right. A new chunk is *reused* if its key is still available in the old
layout's key multiset (consumed in order). For the last edit region (unit-identity diff, as in E2), the *downstream*
chunks are the new chunks that start at or after the end of that region's new text (character offsets in the canonical
text). We record
  down_ins   : downstream chunks that are not reused (inserted chunks entirely after the edited region),
  resync_len : leading run of non-reused downstream chunks before the first reused one (the resynchronization length;
               if no downstream chunk is reused, the whole downstream run),
  ins        : all inserted chunks (S1), as a check against E1.
Per-document checkpoints in results/raw/r2_resync/<stream>/ (resumable); output results/round2/resync_edits.parquet.
usage: PYTHONPATH=src .venv/bin/python experiments/r2_resync.py [--streams wiki,k8s,cpy] [--docs N] [--procs 6] [--tag ""]
"""
import argparse, glob, hashlib, os, time, traceback
from collections import Counter
from multiprocessing import Pool

import pandas as pd
from common import RAW, RESULTS, cap_procs, calibration, checkpoint_dir, code_hash, doc_groups, load_stream, run_meta, worker_threads

from deltachunk.chunkers import Version, make
from deltachunk.diff import regions

CHUNKERS = ("REC", "STRUCT-M", "SentCDC")
CFG = {}


def walk(old_cs, new_cs, region_end):
    pool = Counter(c.key for c in old_cs)
    down, run, ended, ins = 0, 0, False, 0
    for c in new_cs:
        reused = pool[c.key] > 0
        if reused:
            pool[c.key] -= 1
        else:
            ins += 1
        if c.start >= region_end:
            if not reused:
                down += 1
                if not ended:
                    run += 1
            else:
                ended = True
    return ins, down, run, ended


def process(args):
    doc, vers = args
    f = os.path.join(CFG["odir"], hashlib.sha1(str(doc).encode()).hexdigest()[:16] + ".parquet")
    if os.path.exists(f):
        return "skip"
    try:
        V = [Version(v["text"]) for v in vers]
        regs = [None] + [regions(V[i - 1].units, V[i].units) for i in range(1, len(V))]
        rows = []
        for name in CHUNKERS:
            ch = make(name, CFG["cal"].get(name))
            C = [ch.chunk(v) for v in V]
            for i in range(1, len(V)):
                rg = regs[i]
                if rg:
                    last = rg[-1]
                    vn = V[i]
                    if last.b2 > last.a2:
                        end = vn.uoff[last.b2 - 1][1]
                    else:   # pure deletion: the region ends where the next unit starts
                        end = vn.uoff[last.a2][0] if last.a2 < len(vn.uoff) else len(vn.text)
                else:
                    end = len(V[i].text) + 1   # no changed unit: nothing downstream
                ins, down, run, ended = walk(C[i - 1], C[i], end)
                rows.append(dict(stream=CFG["stream"], doc_id=str(doc), chunker=name, rev_new=int(vers[i]["rev_idx"]),
                                 n_regions=len(rg), l_new=sum(g.l_new for g in rg), ins=ins, down_ins=down,
                                 resync_len=run, resynced=ended, n_new=len(C[i])))
        tmp = f + ".tmp"
        pd.DataFrame(rows).to_parquet(tmp, index=False); os.replace(tmp, f)
        return "ok"
    except Exception:
        return "ERROR " + str(doc) + " " + traceback.format_exc()[-600:]


def init(c):
    worker_threads(); CFG.update(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--streams", default="wiki,k8s,cpy"); ap.add_argument("--docs", type=int)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("P3_THREADS", "6"))); ap.add_argument("--tag", default="")
    a = ap.parse_args(); t0 = time.time()
    if os.environ.get("MEMGUARD_SHRINK"):
        a.procs = max(1, a.procs // 2)
    a.procs = min(8, cap_procs(a.procs))
    frames, errors = [], []
    for s in a.streams.split(","):
        cal = calibration(s)["128"]
        odir = checkpoint_dir(os.path.join(RAW, f"r2_resync{a.tag}", s), code_hash("r2_resync", s, a.docs, sorted(cal.items())))
        groups = doc_groups(load_stream(s, a.docs))
        with Pool(a.procs, initializer=init, initargs=(dict(stream=s, cal=cal, odir=odir),)) as pool:
            msgs = list(pool.imap_unordered(process, groups, chunksize=1))
        errors += [m for m in msgs if m.startswith("ERROR")]
        frames += [pd.read_parquet(x) for x in glob.glob(os.path.join(odir, "*.parquet"))]
        print(s, len(groups), "documents", sum(m == "ok" for m in msgs), "new", len(errors), "errors", round(time.time() - t0), "s", flush=True)
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.join(RESULTS, "round2"), exist_ok=True)
    df.to_parquet(os.path.join(RESULTS, "round2", f"resync_edits{a.tag}.parquet"), index=False)
    meta = run_meta(seconds=round(time.time() - t0, 1), procs=a.procs, errors=errors[:5], n_errors=len(errors), rows=len(df))
    pd.Series(meta, dtype=object).to_json(os.path.join(RESULTS, "round2", f"resync_edits{a.tag}.meta.json"), indent=1)
    print(df.groupby(["stream", "chunker"])[["ins", "down_ins", "resync_len"]].describe(percentiles=[0.5, 0.95, 0.99]).T.round(2))
    if errors:
        raise SystemExit(f"{len(errors)} documents failed: {errors[0]}")


if __name__ == "__main__":
    main()
