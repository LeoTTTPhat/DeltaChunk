"""Round 2 (C6): per-stream parameters that match realized final-snapshot chunk sizes for the size-matched quality runs.

Targets are the realized mean chunk sizes (bge tokens over all chunks of the final snapshots) of
  REC at nominal T=128 (the common-background run's reference arm), and
  STRUCT-NM (paragraph chunks; no size parameter),
as saved in results/e4_quality_sharedbg.meta.json / e4_quality.meta.json.
Stateless arms are matched by bisection on their size parameter, measuring the realized mean on every final snapshot
(the indexed corpus itself, not a calibration slice). KARA's final layout depends on the whole history, so its
chunk_size is interpolated (log-log) from the three replayed calibration points of E1 (final-layout means at nominal
T=128/200/320, which equal E4's realized sizes) and checked after the run.
Writes results/round2/matched_params.json: {stream: {variant name: params}} plus the realized sizes.
usage: PYTHONPATH=src .venv/bin/python experiments/r2_match_params.py
"""
import json, os, time

import numpy as np
import pandas as pd
from common import RESULTS, calibration, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.text import ntok_many

OUT = os.path.join(RESULTS, "round2", "matched_params.json")


def realized(name, params, versions):
    ch = make(name, params); toks = []
    for v in versions:
        cs = ch.update(v, None)[0] if ch.stateful else ch.chunk(v)
        toks += ntok_many([c.text for c in cs])
    return float(np.mean(toks))


def bisect_int(name, key, lo, hi, target, versions, fixed=None, tie=None):
    fixed = dict(fixed or {}); best = None; seen = {}
    while lo <= hi:
        mid = (lo + hi) // 2
        p = dict(fixed, **{key: mid})
        if tie:
            p.update(tie(mid))
        m = seen.setdefault(mid, realized(name, p, versions))
        if best is None or abs(m - target) < abs(best[1] - target):
            best = (p, m)
        if m < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def kara_param(stream, target):
    e1 = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=["stream", "doc_id", "chunker", "T", "rev_new", "n_new", "chunk_tok_mean"],
                         filters=[("interval", "==", "revision"), ("chunker", "==", "KARA"), ("stream", "==", stream)])
    last = e1.sort_values("rev_new").groupby(["doc_id", "T"]).tail(1)
    last = last.assign(tt=last.chunk_tok_mean * last.n_new)
    g = last.groupby("T")[["tt", "n_new"]].sum()
    sizes = (g.tt / g.n_new).to_dict()
    cal = calibration(stream)
    pts = sorted((sizes[T], cal[str(T)]["KARA"]["chunk_size"]) for T in sizes)
    xs, ys = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    cs = int(round(float(np.exp(np.interp(np.log(target), xs, ys)))))
    return {"chunk_size": cs}, {str(int(T)): round(s, 1) for T, s in sizes.items()}, pts


def main():
    t0 = time.time()
    shared = json.load(open(os.path.join(RESULTS, "e4_quality_sharedbg.meta.json")))["achieved_tok"]
    native = json.load(open(os.path.join(RESULTS, "e4_quality.meta.json")))["achieved_tok"]
    res = {"targets": {}, "params": {}, "realized": {}, "kara_points": {}}
    for s in ("wiki", "k8s", "cpy"):
        versions = [Version(vers[-1]["text"]) for _, vers in doc_groups(load_stream(s))]
        rec_t, nm_t = shared["REC"][s], native["STRUCT-NM"][s]
        res["targets"][s] = {"REC@T128": rec_t, "STRUCT-NM": nm_t}
        cal = calibration(s)["128"]
        P, R = {}, {}
        P["STRUCT-M@match"], R["STRUCT-M@match"] = bisect_int("STRUCT-M", "T", 60, 900, rec_t, versions)
        best = min(((r, realized("SentCDC", {"r": r, "lam": 0.5}, versions)) for r in range(1, 25)), key=lambda x: abs(x[1] - rec_t))
        P["SentCDC@match"], R["SentCDC@match"] = {"r": best[0], "lam": 0.5}, best[1]
        P["FIX@match"], R["FIX@match"] = bisect_int("FIX", "T", 60, 600, rec_t, versions)
        P["STL@match"], R["STL@match"] = bisect_int("STL", "min_tokens", 20, 240, rec_t, versions,
                                                    tie=lambda m: {"max_tokens": min(480, 2 * m)})
        P["REC@nm"], R["REC@nm"] = bisect_int("REC", "T", 20, 600, nm_t, versions)
        kp, ksizes, pts = kara_param(s, rec_t)
        P["KARA@match"] = kp; R["KARA@match"] = None
        res["kara_points"][s] = dict(final_layout_means=ksizes, points=pts)
        res["params"][s] = P; res["realized"][s] = R
        res["calibration_T128"] = res.get("calibration_T128", {}); res["calibration_T128"][s] = cal
        print(s, json.dumps(P), {k: (round(v, 1) if v else v) for k, v in R.items()}, round(time.time() - t0), "s", flush=True)
    res["meta"] = run_meta(seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
