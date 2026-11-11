"""B3.3: calibrate each size-parameterized chunker to a mean of T tokens (bge-small) PER STREAM on that stream's 10%
calibration slice (final versions of 10% of its documents, salted-hash order), for T in {200, 128, 320}.
Writes results/calibration.json: params[T][stream][chunker] (per_stream=true) and achieved mean/p50/p95 per stream.
(B5 fix of review finding B4-C2: the build calibrated on the pooled slice, leaving chunkers up to 1.6x apart in size
within a stream.) SentCDC cannot reach every target (heading anchors cap its mean); the r closest to T is chosen and
`reachable` records whether the achieved mean is within 10% of T.
usage: python experiments/calibrate.py [--frac 0.1]
"""
import argparse, json, os, time
import numpy as np
from common import P3, RESULTS, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.text import ntok_many

SPACES = {  # chunker -> (param name, lo, hi, integer?, extra fixed params)
    "REC": ("T", 60, 900, True, {}),
    "STRUCT-M": ("T", 60, 900, True, {}),
    "GCDC": ("T_min", 5, 450, True, {"p": 0.25}),
    "BCDC": ("avg", 100, 6000, True, {}),
    "STL": ("min_tokens", 20, 240, True, {}),       # max_tokens = 2 * min_tokens (Bian et al.'s 176/352 ratio)
    "KARA": ("chunk_size", 60, 480, True, {}),
}


def mean_tok(name, params, versions):
    ch = make(name, params)
    toks = []
    for v in versions:
        cs = ch.update(v, None)[0] if ch.stateful else ch.chunk(v)
        toks += ntok_many([c.text for c in cs])
    return float(np.mean(toks)), toks


def with_param(name, x):
    p, lo, hi, integer, fixed = SPACES[name]
    d = dict(fixed); d[p] = int(x) if integer else x
    if name == "STL":
        d["max_tokens"] = min(480, 2 * d["min_tokens"])
    return d


def bisect(name, target, versions):
    p, lo, hi, integer, fixed = SPACES[name]
    best = None
    for _ in range(14):
        mid = (lo + hi) // 2 if integer else (lo + hi) / 2
        m, _ = mean_tok(name, with_param(name, mid), versions)
        if best is None or abs(m - target) < abs(best[1] - target):
            best = (mid, m)
        if m < target: lo = mid + 1 if integer else mid
        else: hi = mid - 1 if integer else mid
        if lo > hi:
            break
    return with_param(name, best[0]), best[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--targets", default="200,128,320"); a = ap.parse_args()
    t0 = time.time()
    per_stream = {}
    for s in ("wiki", "k8s", "cpy"):
        try:
            df = load_stream(s)
        except Exception:
            continue
        groups = doc_groups(df)
        k = max(1, int(round(a.frac * len(groups))))
        per_stream[s] = [Version(g[1][-1]["text"]) for g in groups[:k]]
    out = {"per_stream": True, "params": {}, "achieved": {}, "sentcdc_reachable": {},
           "slice_docs": {s: len(v) for s, v in per_stream.items()}}
    sent_curve = {s: [(r, mean_tok("SentCDC", {"r": r, "lam": 0.5}, vs)[0]) for r in range(1, 21)] for s, vs in per_stream.items()}
    out["sentcdc_r_curve"] = {s: [[r, round(m, 1)] for r, m in c] for s, c in sent_curve.items()}
    for T in [int(x) for x in a.targets.split(",")]:
        PS = {}
        for s, vs in per_stream.items():
            P = {"FIX": {"T": T}, "FIXO": {"T": T, "overlap": max(1, round(T / 5))}}
            for name in SPACES:
                P[name], m = bisect(name, T, vs)
                print(T, s, name, P[name], round(m, 1), flush=True)
            # SentCDC: integer r closest to target (lambda = 0.5)
            best = min(sent_curve[s], key=lambda x: abs(x[1] - T))
            P["SentCDC"] = {"r": best[0], "lam": 0.5}
            out["sentcdc_reachable"].setdefault(str(T), {})[s] = bool(abs(best[1] - T) <= 0.1 * T)
            for alias, base in (("DIFF[REC]", "REC"), ("DOC", "REC"), ("REC-O", "REC"), ("DIFF[STRUCT-M]", "STRUCT-M"), ("STL+snap", "STL")):
                P[alias] = dict(P[base])
            P["REC-O"]["overlap"] = 40
            PS[s] = P
        out["params"][str(T)] = PS
        ach = {}
        for name in ["FIX", "FIXO", "REC", "STRUCT-M", "SentCDC", "GCDC", "BCDC", "STL", "STL+snap", "KARA", "STRUCT-NM", "SENTWIN"]:
            ach[name] = {}
            for s, vs in per_stream.items():
                _, toks = mean_tok(name, PS[s].get(name, {}), vs)
                ach[name][s] = dict(mean=round(float(np.mean(toks)), 1), p50=float(np.median(toks)), p95=float(np.percentile(toks, 95)),
                                    max=int(max(toks)), over512=int(sum(t > 512 for t in toks)))
        out["achieved"][str(T)] = ach
    out["meta"] = run_meta(seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(RESULTS, "calibration.json"), "w"), indent=1)
    print(json.dumps(out["params"]["200"], indent=0))
    print(json.dumps({n: {s: v["mean"] for s, v in d.items()} for n, d in out["achieved"]["200"].items()}))


if __name__ == "__main__":
    main()
