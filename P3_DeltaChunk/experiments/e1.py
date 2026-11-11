"""E1: per-edit chunk deltas for every Tier A chunker on a stream, across sync intervals (+ T sweep at per-revision).

Resumable: one checkpoint per document under results/raw/e1<tag>/<stream>/ (counts + S1 key deltas). The S2/S3
global-scope counts and the runbook exports are computed afterwards by e1_merge.py in global timestamp order.

usage: python experiments/e1.py --stream wiki [--docs 20] [--tag _smoke] [--procs 8]
"""
import argparse, hashlib, json, os, sys, time, traceback
from collections import Counter
from multiprocessing import Pool

from common import cap_procs, P3, RAW, RESULTS, calibration, checkpoint_dir, code_hash, doc_groups, load_stream, release_ts, run_meta, worker_threads

from deltachunk.chunkers import TIER_A, Version, make
from deltachunk.delta import coalesce
from deltachunk.diff import edit_class, regions
from deltachunk.text import ntok_many

# chunkers that consume the shared unit segmentation (Version units): their timed cost must include seg_ms
UNIT_BASED = {"SENTWIN", "STRUCT-M", "STRUCT-NM", "SentCDC", "GCDC", "KARA", "DIFF[REC]", "DIFF[STRUCT-M]"}
SIZE_PARAM = ["FIX", "FIXO", "REC", "STRUCT-M", "SentCDC", "GCDC", "BCDC", "STL", "STL+snap", "KARA", "DIFF[REC]", "DIFF[STRUCT-M]", "DOC"]
CFG = {}


def k64(hexkey: str) -> int:
    return int(hexkey[:16], 16) - (1 << 63)   # signed int64 for parquet


def chunk_seq(name, params, versions, idxs):
    """Chunk versions[idxs] in order; returns {i: (keys, toks, pieces, ms)} (stateful chunkers get the sequence)."""
    ch = make(name, params); out = {}; state = None
    for i in idxs:
        t = time.perf_counter()
        if ch.stateful:
            cs, state = ch.update(versions[i], state)
        else:
            cs = ch.chunk(versions[i])
        ms = (time.perf_counter() - t) * 1000
        out[i] = ([c.key for c in cs], ntok_many([c.text for c in cs]), [c.pieces for c in cs], ms)
    return out


def process(args):
    doc, vers = args
    stream, tag = CFG["stream"], CFG["tag"]
    safe = hashlib.sha1(str(doc).encode()).hexdigest()[:16]
    odir = CFG["odir"]
    fc, fd = os.path.join(odir, f"{safe}.counts.parquet"), os.path.join(odir, f"{safe}.delta.parquet")
    if os.path.exists(fc):
        return doc, "skip"
    import pandas as pd
    try:
        from deltachunk.chunkers.stateful import SteadlithAdapter
        f0 = SteadlithAdapter.fallbacks
        t0 = time.time()
        V = [Version(v["text"]) for v in vers]
        seg_ms = (time.time() - t0) * 1000 / max(1, len(V))
        meta = [dict(i=i, ts=int(v["ts"]), rev_idx=int(v["rev_idx"])) for i, v in enumerate(vers)]
        plans = {iv: [m["i"] for m in coalesce(meta, iv, CFG["release"])] for iv in CFG["intervals"]}
        # chunker-independent edit info per interval pair
        einfo = {}
        for iv, idxs in plans.items():
            for p, c in zip(idxs, idxs[1:]):
                if (p, c) in einfo:
                    continue
                regs = regions(V[p].units, V[c].units)
                n_old = len(V[p].units)
                einfo[(p, c)] = dict(edit_class=edit_class(vers[p]["text"], vers[c]["text"], V[p].units, V[c].units, regs),
                                     n_regions=len(regs), l_old=sum(r.l_old for r in regs), l_new=sum(r.l_new for r in regs),
                                     rel_pos=(regs[0].a / n_old if regs and n_old else -1.0),
                                     regions=json.dumps([[r.a, r.b, r.a2, r.b2] for r in regs]),
                                     units_new=len(V[c].units), doc_tok_new=sum(u.tok for u in V[c].units))
        rows, deltas = [], []
        for T in CFG["Ts"]:
            cal = CFG["cal"].get(str(T), {})
            names = CFG["chunkers"] if T == CFG["Ts"][0] else [n for n in CFG["chunkers"] if n in SIZE_PARAM]
            ivs = CFG["intervals"] if T == CFG["Ts"][0] else ["revision"]
            for name in names:
                params = cal.get(name, {"T": T} if T != 200 else {})
                stateless_cache = None
                for iv in ivs:
                    idxs = plans[iv]
                    if not idxs:
                        continue
                    if make(name, params).stateful or stateless_cache is None:
                        res = chunk_seq(name, params, V, idxs if make(name, params).stateful else sorted(set(i for ix in plans.values() for i in ix)))
                        if not make(name, params).stateful:
                            stateless_cache = res
                    else:
                        res = stateless_cache
                    if iv == "revision" and T == CFG["Ts"][0]:
                        k0, t0_, _, _ = res[idxs[0]]
                        deltas.append(dict(chunker=name, doc_id=str(doc), rev_idx=meta[idxs[0]]["rev_idx"], ts=meta[idxs[0]]["ts"],
                                           ins_keys=[k64(k) for k in k0], ins_toks=list(t0_), del_keys=[]))
                    for p, c in zip(idxs, idxs[1:]):
                        ko, to, _, _ = res[p]; kn, tn, pn, ms = res[c]
                        tok = dict(zip(kn, tn)); tok.update({k: t for k, t in zip(ko, to) if k not in tok})
                        co, cn = Counter(ko), Counter(kn)
                        ins, dele = cn - co, co - cn
                        insk = list(ins.elements())
                        piece_of = dict(zip(kn, pn))
                        rows.append(dict(stream=stream, doc_id=str(doc), chunker=name, T=T, interval=iv,
                                         rev_old=meta[p]["rev_idx"], rev_new=meta[c]["rev_idx"], ts=meta[c]["ts"],
                                         n_old=len(ko), n_new=len(kn), ins=len(insk), dele=sum(dele.values()),
                                         toks=sum(tok[k] for k in insk), tok_new=sum(tn),
                                         guard_ins=sum(1 for k in insk if piece_of.get(k, 1) > 1), chunk_ms=ms, seg_ms=seg_ms,
                                         chunk_total_ms=ms + (seg_ms if name in UNIT_BASED else 0.0),
                                         **einfo[(p, c)]))
                        if iv == "revision" and T == CFG["Ts"][0]:
                            deltas.append(dict(chunker=name, doc_id=str(doc), rev_idx=meta[c]["rev_idx"], ts=meta[c]["ts"],
                                               ins_keys=[k64(k) for k in insk], ins_toks=[tok[k] for k in insk],
                                               del_keys=[k64(k) for k in dele.elements()]))
        for r in rows:
            r["stl_fallbacks_doc"] = SteadlithAdapter.fallbacks - f0   # Steadlith calls that needed the long-word split
        os.makedirs(odir, exist_ok=True)
        pd.DataFrame(deltas).to_parquet(fd + ".tmp", index=False); os.replace(fd + ".tmp", fd)
        pd.DataFrame(rows).to_parquet(fc + ".tmp", index=False); os.replace(fc + ".tmp", fc)
        return doc, f"ok {len(vers)} versions {time.time() - t0:.1f}s"
    except Exception:
        return doc, "ERROR " + traceback.format_exc()[-1500:]


def init(cfg):
    worker_threads(); CFG.update(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True); ap.add_argument("--docs", type=int, default=None)
    ap.add_argument("--tag", default=""); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--chunkers", default=",".join(TIER_A)); ap.add_argument("--Ts", default="200,128,320")
    ap.add_argument("--intervals", default=None)
    a = ap.parse_args()
    if os.environ.get("MEMGUARD_SHRINK"):
        a.procs = max(2, a.procs // 2)
    a.procs = cap_procs(a.procs)
    intervals = a.intervals.split(",") if a.intervals else ["revision", "daily", "weekly", "monthly"] + (["release"] if a.stream == "cpy" else [])
    cfg = dict(stream=a.stream, tag=a.tag, chunkers=a.chunkers.split(","), Ts=[int(x) for x in a.Ts.split(",")],
               intervals=intervals, release=release_ts(a.stream), cal=calibration(a.stream))
    if not cfg["cal"]:
        print("WARNING: results/calibration.json missing; using default parameters", flush=True)
    # checkpoints are keyed by code + configuration (stale rows from an older code/calibration are moved aside)
    cfg["odir"] = checkpoint_dir(os.path.join(RAW, f"e1{a.tag}", a.stream), code_hash("e1", {k: v for k, v in cfg.items()}))
    df = load_stream(a.stream, a.docs)
    groups = doc_groups(df)
    t0 = time.time(); done = err = 0
    with Pool(a.procs, initializer=init, initargs=(cfg,)) as pool:
        for doc, msg in pool.imap_unordered(process, groups, chunksize=1):
            done += 1; err += msg.startswith("ERROR")
            if msg.startswith("ERROR") or done % 10 == 0 or done == len(groups):
                print(f"[{time.time() - t0:7.1f}s] {done}/{len(groups)} {doc}: {msg[:2000]}", flush=True)
    meta = run_meta(stream=a.stream, docs=len(groups), procs=a.procs, seconds=round(time.time() - t0, 1), errors=err, **{k: v for k, v in cfg.items() if k not in ("cal", "release", "stream")})
    json.dump(meta, open(os.path.join(RAW, f"e1{a.tag}", a.stream, "_meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("docs", "seconds", "errors")}))


if __name__ == "__main__":
    main()
