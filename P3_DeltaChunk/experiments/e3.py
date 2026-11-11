"""E3 (drift, counts part): history-dependent chunkers (KARA, DIFF[REC], DIFF[STRUCT-M]) replayed over each document's
history vs a fresh chunking of the same version by the same base chunker (KARA: fresh KARA layout; DIFF[b]: b).
At checkpoints t (edit index) record per document: maintained and fresh chunk sizes, the key divergence
1 - |K(C_t) & K(C*)| / |K(C*)|, and the share of chunks outside [T_lo, T_hi]. KS distances are computed on pooled sizes
at the end. Periodic re-normalization cost: INS of switching from C_t to C*(d_t).
B5 fix (review finding B4-E3): the document population shrinks and gets longer with t, so curves are also reported for
FIXED cohorts (documents with >= 100 and >= 200 edits: every point of a cohort curve has the same documents) next to
the all-documents curve; every row carries `cohort` and the number of documents.
usage: python experiments/e3.py --stream wiki [--docs 1000] [--tag _smoke]
"""
import argparse, hashlib, json, os, time, traceback
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from common import cap_procs, RAW, RESULTS, calibration, checkpoint_dir, code_hash, doc_groups, load_stream, run_meta, worker_threads
from scipy.stats import ks_2samp

from deltachunk.chunkers import Version, make
from deltachunk.text import ntok_many

CHECK = [0, 25, 50, 100, 150, 200, 250, 299]
PAIRS = {"KARA": "KARA", "DIFF[REC]": "REC", "DIFF[STRUCT-M]": "STRUCT-M"}
CFG = {}


def process(args):
    doc, vers = args
    odir = CFG["odir"]; f = os.path.join(odir, hashlib.sha1(str(doc).encode()).hexdigest()[:16] + ".parquet")
    if os.path.exists(f):
        return "skip"
    try:
        cal = CFG["cal"]; rows = []
        V = [Version(v["text"]) for v in vers]
        checks = sorted(set([c for c in CHECK if c < len(V)] + [len(V) - 1]))
        for hd, base in PAIRS.items():
            ch, fresh = make(hd, cal.get(hd)), make(base, cal.get(base)); st = None
            for t, v in enumerate(V):
                cs, st = ch.update(v, st)
                if t in checks:
                    fs = fresh.update(v, None)[0] if fresh.stateful else fresh.chunk(v)
                    km, kf = Counter(c.key for c in cs), Counter(c.key for c in fs)
                    tm, tf = ntok_many([c.text for c in cs]), ntok_many([c.text for c in fs])
                    rows.append(dict(stream=CFG["stream"], doc_id=str(doc), chunker=hd, t=t, n_edits_doc=len(V) - 1, n_m=len(cs), n_f=len(fs),
                                     sizes_m=tm, sizes_f=tf, divergence=1 - sum((km & kf).values()) / max(1, sum(kf.values())),
                                     renorm_ins=sum((kf - km).values()), renorm_toks=sum(x for c, x in zip(fs, tf) if km[c.key] == 0)))
        os.makedirs(odir, exist_ok=True); pd.DataFrame(rows).to_parquet(f, index=False)
        return "ok"
    except Exception:
        return "ERROR " + traceback.format_exc()[-800:]


def init(c):
    worker_threads(); CFG.update(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stream", required=True); ap.add_argument("--docs", type=int)
    ap.add_argument("--tag", default=""); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--lo", type=int, default=50); ap.add_argument("--hi", type=int, default=480); a = ap.parse_args(); t0 = time.time()
    a.procs = cap_procs(a.procs)
    CFG.update(stream=a.stream, tag=a.tag, cal=calibration(a.stream).get("200", {}))
    groups = doc_groups(load_stream(a.stream, a.docs))
    CFG["odir"] = checkpoint_dir(os.path.join(RAW, f"e3{a.tag}", a.stream), code_hash("e3", CFG["cal"], CHECK, PAIRS))
    with Pool(a.procs, initializer=init, initargs=(dict(CFG),)) as pool:
        msgs = list(pool.imap_unordered(process, groups, chunksize=1))
    import glob
    df = pd.concat([pd.read_parquet(x) for x in glob.glob(os.path.join(CFG["odir"], "*.parquet"))], ignore_index=True)
    out = []
    df["t"] = df.t.where(df.t.isin(CHECK), -1)   # -1 = each document's final version (if not a checkpoint)
    groups_c = [("all", df)] + [(f">={k}", df[df.n_edits_doc >= k]) for k in (100, 200)]
    for cohort, dfc in groups_c:
      for (ch, t), g in dfc.groupby(["chunker", "t"]):
        if cohort != "all" and (t == -1 or t > int(cohort[2:])):
            continue
        sm = np.concatenate(g.sizes_m.values); sf = np.concatenate(g.sizes_f.values)
        out.append(dict(stream=a.stream, cohort=cohort, chunker=ch, t=t, docs=len(g), ks=float(ks_2samp(sm, sf).statistic),
                        mean_m=float(sm.mean()), mean_f=float(sf.mean()),
                        out_of_range_m=float(((sm < a.lo) | (sm > a.hi)).mean()), out_of_range_f=float(((sf < a.lo) | (sf > a.hi)).mean()),
                        divergence=float(g.divergence.mean()), renorm_toks_frac=float(g.renorm_toks.sum() / max(1, sf.sum()))))
    s = pd.DataFrame(out); s.to_csv(os.path.join(RESULTS, f"e3_drift_{a.stream}{a.tag}.csv"), index=False)
    json.dump(run_meta(seconds=round(time.time() - t0, 1), errors=[m for m in msgs if m.startswith("ERROR")][:3], docs=len(groups)),
              open(os.path.join(RESULTS, f"e3_drift_{a.stream}{a.tag}.meta.json"), "w"), indent=1)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
