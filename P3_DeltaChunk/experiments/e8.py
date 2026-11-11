"""E8: sensitivity counts (S1, per revision): SentCDC r in 1..6 and lambda in {1, 0.5}; target size T for REC/STRUCT-M/
SentCDC-matched configs; GCDC min-guard grid; Steadlith min/max grid; unit granularity (sentence-unit MAXP vs
word-level Steadlith vs byte-level FastCDC at matched mean size). Resumable per document.
usage: python experiments/e8.py --stream wiki [--docs 1000] [--tag _smoke]
"""
import argparse, hashlib, json, os, time, traceback
from collections import Counter
from multiprocessing import Pool

import pandas as pd
from common import cap_procs, RAW, RESULTS, checkpoint_dir, code_hash, doc_groups, load_stream, run_meta, worker_threads

from deltachunk.chunkers import ByteCDC, GuardedCDC, SentCDC, Version
from deltachunk.chunkers.base import Recursive, StructMerge
from deltachunk.chunkers.stateful import SteadlithAdapter
from deltachunk.text import ntok_many


def configs():
    c = {}
    for r in range(1, 7):
        for lam in (1.0, 0.5):
            c[f"SentCDC r={r} lam={lam}"] = lambda r=r, lam=lam: SentCDC(r, lam)
    for T in (128, 200, 320):
        c[f"REC T={T}"] = lambda T=T: Recursive(T)
        c[f"STRUCT-M T={T}"] = lambda T=T: StructMerge(T)
    for mg in (25, 50, 100, 150):
        for p in (0.15, 0.3):
            c[f"GCDC Tmin={mg} p={p}"] = lambda mg=mg, p=p: GuardedCDC(mg, p)
    for lo, hi in ((60, 120), (100, 200), (120, 320), (176, 352), (180, 480)):
        c[f"STL min={lo} max={hi}"] = lambda lo=lo, hi=hi: SteadlithAdapter(lo, hi, False)
    for avg in (450, 900, 1800):
        c[f"BCDC avg={avg}"] = lambda avg=avg: ByteCDC(avg)
    return c


CFG = {}


def process(args):
    doc, vers = args
    odir = CFG["odir"]; f = os.path.join(odir, hashlib.sha1(str(doc).encode()).hexdigest()[:16] + ".parquet")
    if os.path.exists(f):
        return "skip"
    try:
        V = [Version(v["text"]) for v in vers]; rows = []
        for label, mk in configs().items():
            ch = mk(); prev = None
            for i, v in enumerate(V):
                cs = ch.chunk(v); keys = [c.key for c in cs]; toks = ntok_many([c.text for c in cs])
                if prev is not None:
                    ins = Counter(keys) - Counter(prev[0]); tk = dict(zip(keys, toks))
                    rows.append(dict(stream=CFG["stream"], doc_id=str(doc), config=label, rev_new=int(vers[i]["rev_idx"]),
                                     ins=sum(ins.values()), dele=sum((Counter(prev[0]) - Counter(keys)).values()),
                                     toks=sum(tk[k] * c for k, c in ins.items()), tok_new=sum(toks), n_new=len(cs),
                                     doc_tok_new=sum(u.tok for u in v.units)))
                prev = (keys, toks)
        os.makedirs(odir, exist_ok=True); pd.DataFrame(rows).to_parquet(f, index=False)
        return "ok"
    except Exception:
        return "ERROR " + traceback.format_exc()[-800:]


def init(c):
    worker_threads(); CFG.update(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stream", required=True); ap.add_argument("--docs", type=int)
    ap.add_argument("--tag", default=""); ap.add_argument("--procs", type=int, default=8); a = ap.parse_args(); t0 = time.time()
    a.procs = cap_procs(a.procs)
    CFG.update(stream=a.stream, tag=a.tag)
    groups = doc_groups(load_stream(a.stream, a.docs))
    CFG["odir"] = checkpoint_dir(os.path.join(RAW, f"e8{a.tag}", a.stream), code_hash("e8", a.stream))
    with Pool(a.procs, initializer=init, initargs=(dict(CFG),)) as pool:
        msgs = list(pool.imap_unordered(process, groups, chunksize=1))
    errs = [m for m in msgs if m.startswith("ERROR")]
    import glob
    df = pd.concat([pd.read_parquet(x) for x in glob.glob(os.path.join(CFG["odir"], "*.parquet"))], ignore_index=True)
    df["toks_frac"] = df.toks / df.doc_tok_new.clip(lower=1)          # document-token denominator (as E1)
    df["toks_frac_workload"] = df.toks / df.tok_new.clip(lower=1)
    s = df.groupby("config").agg(edits=("ins", "size"), ins=("ins", "mean"), toks_frac=("toks_frac", "mean"), toks_frac_workload=("toks_frac_workload", "mean"),
                                 mean_chunk_tok=("tok_new", "sum")).reset_index()
    s["mean_chunk_tok"] = s.mean_chunk_tok / df.groupby("config").n_new.sum().values
    s.to_csv(os.path.join(RESULTS, f"e8_sens_{a.stream}{a.tag}.csv"), index=False)
    json.dump(run_meta(seconds=round(time.time() - t0, 1), errors=errs[:3], docs=len(groups)),
              open(os.path.join(RESULTS, f"e8_sens_{a.stream}{a.tag}.meta.json"), "w"), indent=1)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
