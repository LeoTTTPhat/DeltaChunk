"""E2: Theorem 1 (corrected) checked on every real SentCDC edit, plus the Monte Carlo expectation on a sample of edits.

For each consecutive retained pair: regions (unit-identity diff), zones Z+ merged when they overlap (composed bound),
anchors A = heading-anchor cuts inside Z' of each zone, guard firing (an inserted chunk produced by a max split; such
edits are reported separately because the theorem assumes no guard splits).
usage: python experiments/e2.py --stream wiki [--docs 20] [--tag _smoke] [--mc 300]
"""
import argparse, json, os, random, time
from collections import Counter
from multiprocessing import Pool

import pandas as pd
from common import cap_procs, RESULTS, calibration, doc_groups, load_stream, run_meta

from deltachunk.chunkers import SentCDC, Version
from deltachunk.diff import regions
from deltachunk.predict import sentcdc_ins_mc

CFG = {}


def bound(l, r):
    return (l + 2 * r - 1) // (r + 1) + 2


def zones(regs, r, side):
    """Merge zones Z+ = [a-r, b'+r+1] (new side) or [a-r, b+r+1] (old side) that overlap; returns [(z0, z1, s, e)]."""
    out = []
    for g in regs:
        s, e = (g.a2, g.b2) if side == "new" else (g.a, g.b)
        z = [s - r, e - 1 + r + 1, s, e]
        if out and out[-1][1] >= z[0]:
            out[-1][1] = z[1]; out[-1][3] = e
        else:
            out.append(z)
    return out


def anchors_in(v, z0, z1, r):
    """Heading-anchor cut positions i in Z' = [z0, z1-1] (cut after i because unit i+1 is a heading)."""
    n = len(v.units)
    return sum(1 for i in range(max(0, z0), min(n - 1, z1)) if v.units[i + 1].is_heading)


def process(args):
    doc, vers = args
    r, lam = CFG["r"], CFG["lam"]
    ch = SentCDC(r=r, lam=lam)
    V = [Version(v["text"]) for v in vers]
    C = [ch.chunk(v) for v in V]
    rows = []
    for i in range(1, len(V)):
        vo, vn = V[i - 1], V[i]
        ko, kn = Counter(c.key for c in C[i - 1]), Counter(c.key for c in C[i])
        insk = kn - ko; delk = ko - kn
        ins, dele = sum(insk.values()), sum(delk.values())
        regs = regions(vo.units, vn.units)
        guard = any(c.pieces > 1 and c.key in insk for c in C[i]) or any(c.pieces > 1 and c.key in delk for c in C[i - 1])
        zn, zo = zones(regs, r, "new"), zones(regs, r, "old")
        b_ins = sum(bound(z[3] - z[2], r) + anchors_in(vn, z[0], z[1], r) for z in zn)
        b_del = sum(bound(z[3] - z[2], r) + anchors_in(vo, z[0], z[1], r) for z in zo)
        rows.append(dict(stream=CFG["stream"], doc_id=str(doc), rev_new=int(vers[i]["rev_idx"]), ins=ins, dele=dele,
                         bound_ins=b_ins, bound_del=b_del, guard=guard, n_regions=len(regs), n_zones=len(zn),
                         l_new=sum(g.l_new for g in regs), l_old=sum(g.l_old for g in regs), n_units=len(vo.units),
                         a=(regs[0].a if regs else -1), p_par=sum(u.par_end for u in vo.units) / max(1, len(vo.units)),
                         viol_ins=(not guard) and ins > b_ins, viol_del=(not guard) and dele > b_del))
    return rows


def init(c):
    from common import worker_threads
    worker_threads(); CFG.update(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stream", required=True); ap.add_argument("--docs", type=int)
    ap.add_argument("--tag", default=""); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--mc", type=int, default=300)
    ap.add_argument("--mc_samples", type=int, default=400)
    ap.add_argument("--r", type=int, default=None, help="override the calibrated window (Thm 1 is also checked at r=3, where the T_max guard rarely fires)")
    a = ap.parse_args(); t0 = time.time()
    a.procs = cap_procs(a.procs)
    p = calibration(a.stream).get("200", {}).get("SentCDC", {"r": 3, "lam": 0.5})
    cfg = dict(stream=a.stream, r=a.r or p["r"], lam=p["lam"])
    if a.r:
        a.tag = f"_r{a.r}{a.tag}"
    groups = doc_groups(load_stream(a.stream, a.docs))
    from common import worker_threads
    with Pool(a.procs, initializer=init, initargs=(cfg,)) as pool:
        rows = [r for rs in pool.imap_unordered(process, groups, chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    df["r"], df["lam"] = cfg["r"], cfg["lam"]     # every row carries its SentCDC configuration (F2 must not pool r values)
    # Monte Carlo expectation on a sample of single-region, guard-free edits
    rng = random.Random(0)
    single = df[(df.n_regions == 1) & (~df.guard)]
    idx = rng.sample(list(single.index), min(a.mc, len(single)))
    df["mc_mean"] = float("nan"); df["mc_se"] = float("nan")
    for k in idx:
        row = df.loc[k]
        m, se = sentcdc_ins_mc(int(row.n_units), int(row.a), int(row.l_old), int(row.l_new), cfg["r"], cfg["lam"], float(row.p_par),
                               samples=a.mc_samples, seed=int(k))
        df.loc[k, "mc_mean"] = m; df.loc[k, "mc_se"] = se
    df.to_parquet(os.path.join(RESULTS, f"e2_thm1_{a.stream}{a.tag}.parquet"), index=False)
    s = dict(stream=a.stream, r=cfg["r"], lam=cfg["lam"], edits=len(df), guard_edits=int(df.guard.sum()),
             covered_edits=int((~df.guard).sum()), covered_share=float((~df.guard).mean()),   # Thm 1 denominator
             viol_ins=int(df.viol_ins.sum()), viol_del=int(df.viol_del.sum()),
             mean_ins=float(df.ins.mean()), mean_bound=float(df.bound_ins.mean()),
             mc_edits=len(idx), mc_mean_pred=float(df.mc_mean.mean()), mc_mean_meas=float(df.loc[idx, "ins"].mean()) if idx else None,
             meta=run_meta(seconds=round(time.time() - t0, 1), procs=a.procs))
    json.dump(s, open(os.path.join(RESULTS, f"e2_thm1_{a.stream}{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "meta"}))


if __name__ == "__main__":
    main()
