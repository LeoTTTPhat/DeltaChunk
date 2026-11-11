"""History dependence of the history-dependent chunkers (pre-draft review, optional items 5 and 6), at the size-matched
T=128 calibration, on the first N documents (salted-hash order) of a stream:
  (a) cross-replica overlap: the final layout after a per-revision replay vs after a monthly replay (same final text,
      different histories; delta.coalesce 'monthly'), and each vs a fresh chunking of the final text. Divergence uses
      E3's definition, 1 - |K(A) & K(B)| / |K(B)| on chunk-key multisets.
  (b) one-edit divergence: start from a FRESH layout of version i, apply one update to version i+1, compare with the
      fresh layout of version i+1 (up to --per-doc evenly spaced edits per document). Fresh = KARA with no state for
      KARA, the base chunker for DIFF[b].
usage: PYTHONPATH=src python experiments/kara_stability.py --stream wiki [--docs 100] [--per-doc 10] [--procs 4]
-> results/kara_stability_<stream>.csv (one row per doc x chunker x measure) and .json (means)
"""
import argparse, json, os, time, traceback
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
from common import RESULTS, calibration, cap_procs, doc_groups, load_stream, run_meta, worker_threads

from deltachunk.chunkers import Version, make
from deltachunk.delta import coalesce

PAIRS = {"KARA": "KARA", "DIFF[REC]": "REC", "DIFF[STRUCT-M]": "STRUCT-M"}
CFG = {}


def div(a, b):
    ka, kb = Counter(c.key for c in a), Counter(c.key for c in b)
    return 1 - sum((ka & kb).values()) / max(1, sum(kb.values()))


def fresh_of(base, cal, V):
    f = make(base, cal.get(base))
    return f.update(V, None)[0] if f.stateful else f.chunk(V)


def process(args):
    doc, vers = args
    try:
        cal = CFG["cal"]; rows = []
        meta = [dict(i=i, ts=int(v["ts"]), rev_idx=int(v["rev_idx"])) for i, v in enumerate(vers)]
        mon = [m["i"] for m in coalesce(meta, "monthly")]
        V = [Version(v["text"]) for v in vers]
        n = len(V) - 1
        for hd, base in PAIRS.items():
            ch = make(hd, cal.get(hd))
            st = None
            for v in V:
                cr, st = ch.update(v, st)
            st = None
            for i in mon:
                cm, st = ch.update(V[i], st)
            fr = fresh_of(base, cal, V[-1])
            rows.append(dict(doc_id=str(doc), chunker=hd, measure="final_revision_vs_monthly", value=div(cr, cm), edits=n,
                             monthly_syncs=len(mon) - 1))
            rows.append(dict(doc_id=str(doc), chunker=hd, measure="final_revision_vs_fresh", value=div(cr, fr), edits=n))
            rows.append(dict(doc_id=str(doc), chunker=hd, measure="final_monthly_vs_fresh", value=div(cm, fr), edits=n))
            if n >= 1:
                idx = sorted(set(np.linspace(0, n - 1, min(CFG["per_doc"], n)).round().astype(int)))
                vals = []
                for i in idx:
                    _, s0 = ch.update(V[i], None)   # fresh layout of version i (DIFF[b]: the base layout)
                    c1, _ = ch.update(V[i + 1], s0)
                    vals.append(div(c1, fresh_of(base, cal, V[i + 1])))
                rows.append(dict(doc_id=str(doc), chunker=hd, measure="one_edit_vs_fresh", value=float(np.mean(vals)),
                                 edits=len(vals)))
        return rows
    except Exception:
        return "ERROR " + str(doc) + " " + traceback.format_exc()[-800:]


def init(c):
    worker_threads(); CFG.update(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stream", required=True); ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--per-doc", type=int, default=10); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--T", default="128"); a = ap.parse_args(); t0 = time.time()
    cal = calibration(a.stream).get(a.T, {})
    groups = doc_groups(load_stream(a.stream))[:a.docs]
    procs = cap_procs(a.procs)
    rows, errs = [], []
    with Pool(procs, initializer=init, initargs=(dict(cal=cal, per_doc=a.per_doc),)) as pool:
        for r in pool.imap_unordered(process, groups):
            (errs.append(r) if isinstance(r, str) else rows.extend(r))
    D = pd.DataFrame(rows); D.insert(0, "stream", a.stream)
    D.to_csv(os.path.join(RESULTS, f"kara_stability_{a.stream}.csv"), index=False)
    S = D.groupby(["chunker", "measure"]).value.agg(["mean", "median", "size"]).reset_index()
    json.dump(dict(summary=S.to_dict("records"), docs=len(groups), T=a.T, per_doc=a.per_doc, procs=procs, errors=errs,
                   seconds=round(time.time() - t0, 1), meta=run_meta()),
              open(os.path.join(RESULTS, f"kara_stability_{a.stream}.json"), "w"), indent=1)
    print(S.round(4).to_string()); print("errors", errs[:3])


if __name__ == "__main__":
    main()
