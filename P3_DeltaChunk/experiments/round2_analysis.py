"""Round 2 analyses from saved measurements (no new chunking, no model inference).

Subcommands (outputs in results/round2/):
  matched   C3: per-stream matched-size table (achieved sizes at T=128/200/320, support, interpolated cost with a paired
            document bootstrap), spreads with and without Steadlith (absolute and relative), KARA lead over the best
            stateless configuration, FIX ratios, Steadlith excess at nominal T=128 and at the common size,
            FIX/REC ratio by document length.
  tails     C5: per-edit INS and re-embedded-token tails (p95, p99, max) for REC, STRUCT-M and SentCDC at nominal T=128
            and interpolated at the common size; share of edits whose INS exceeds the SentCDC bound of the same edit.
  resync    C5: resynchronization lengths from experiments/r2_resync.py output.
  rep       C7: pair bootstrap for the Bian-style replication gaps.
  timing    C7: dispersion of the timed totals and ratios across repetitions; achieved sizes of the timed configs.
  predict   C7: chunker- and document-level predictor errors with document-bootstrap intervals.
  minors    C8: drift cohorts, sentence-probe design, TOST for STALE/WINDOW-D, Theorem 1 bound terms, unit-cost fit
            diagnostics, Steadlith workaround shares.
usage: PYTHONPATH=src .venv/bin/python experiments/round2_analysis.py matched|tails|resync|rep|timing|predict|minors|tex|all
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

from common import P3, RESULTS

OUT = os.path.join(RESULTS, "round2")
SEED = 20261110
B = 2000
STREAMS = ("wiki", "k8s", "cpy")
SIZED = ("FIX", "FIXO", "REC", "STRUCT-M", "SentCDC", "GCDC", "BCDC", "STL", "STL+snap", "KARA", "DIFF[REC]",
         "DIFF[STRUCT-M]")
ALIGNED = ("BCDC", "DIFF[REC]", "DIFF[STRUCT-M]", "GCDC", "KARA", "REC", "STRUCT-M", "SentCDC")   # Steadlith excluded
STATELESS = ("BCDC", "GCDC", "REC", "STRUCT-M", "SentCDC")                                       # content-aligned, invariant
STEADLITH = ("STL", "STL+snap")
PROVENANCE = {
    "FIX": "authors (token windows)", "FIXO": "authors (token windows)", "REC": "third-party (LangChain 1.1.3)",
    "STRUCT-M": "authors", "STRUCT-NM": "authors", "SENTWIN": "authors", "SentCDC": "authors (MAXP rule [BBG10])",
    "GCDC": "authors (FastCDC-style guards)", "BCDC": "third-party (fastcdc 1.7.0) + whitespace snap",
    "STL": "third-party (Steadlith 74c5b55b), adapted", "STL+snap": "third-party (Steadlith 74c5b55b), adapted",
    "KARA": "third-party (kara-toolkit 0.3.2), adapted", "DIFF[REC]": "authors", "DIFF[STRUCT-M]": "authors",
    "DOC": "authors (document replace)"}


def out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def q(x, lo=0.025, hi=0.975):
    a, b = np.quantile(x, [lo, hi])
    return float(a), float(b)


# --------------------------------------------------------------------------------------------------- matched size (C3)
def doc_table():
    cols = ["stream", "doc_id", "chunker", "T", "toks_frac", "chunk_tok_mean", "tok_new", "ins"]
    raw = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=cols, filters=[("interval", "==", "revision")])
    doc = raw.groupby(["stream", "doc_id", "chunker", "T"], observed=True).agg(
        edits=("toks_frac", "size"), cost=("toks_frac", "sum"), size=("chunk_tok_mean", "sum"),
        doc_tok=("tok_new", "mean")).reset_index()
    return doc


def interp_weights(size, achieved):
    """Log-log interpolation between the two bracketing achieved sizes; None if `size` is outside the sweep."""
    order = np.argsort(achieved)
    a = np.asarray(achieved)[order]
    if len(np.unique(a)) < 2 or size < a[0] or size > a[-1]:
        return None
    right = int(np.clip(np.searchsorted(a, size), 1, len(a) - 1))
    left = right - 1
    while a[right] == a[left] and left > 0:
        left -= 1
    w = (np.log(size) - np.log(a[left])) / (np.log(a[right]) - np.log(a[left]))
    return order[left], order[right], float(w)


def matched():
    doc = doc_table()
    rng = np.random.default_rng(SEED)
    table, spreads, leads, excess, lengths = [], [], [], [], []
    for stream in STREAMS:
        f = doc[doc.stream.eq(stream)]
        ids = pd.Index(sorted(f.doc_id.unique()))
        draws = rng.integers(len(ids), size=(B, len(ids)))
        series = {}
        for ch in SIZED:
            pts = []
            for T, g in f[f.chunker.eq(ch)].groupby("T"):
                g = g.set_index("doc_id").reindex(ids)
                e, c, s = g.edits.to_numpy(), g.cost.to_numpy(), g["size"].to_numpy()
                achieved = s.sum() / e.sum()
                boot = c[draws].sum(1) / e[draws].sum(1)
                pts.append(dict(T=int(T), achieved=float(achieved), obs=float(c.sum() / e.sum()), boot=boot))
            series[ch] = sorted(pts, key=lambda p: p["T"])
        # common size: smallest integer size >= every content-aligned configuration's achieved size at nominal T=128;
        # largest common size: largest integer size <= the smallest sweep maximum among the configurations it supports
        c_lo = int(np.ceil(max(next(p["achieved"] for p in series[ch] if p["T"] == 128) for ch in ALIGNED)))
        sup = [ch for ch in ALIGNED if max(p["achieved"] for p in series[ch]) >= c_lo]
        c_hi = int(np.floor(min(max(p["achieved"] for p in series[ch]) for ch in sup)))

        def est(ch, size):
            pts = series[ch]
            w = interp_weights(size, [p["achieved"] for p in pts])
            if w is None:
                return None
            i, j, wt = w
            yi = np.r_[pts[i]["obs"], pts[i]["boot"]]
            yj = np.r_[pts[j]["obs"], pts[j]["boot"]]
            return np.exp((1 - wt) * np.log(np.clip(yi, 1e-9, None)) + wt * np.log(np.clip(yj, 1e-9, None)))
        for size, label in ((c_lo, "common"), (c_hi, "largest")):
            E = {ch: est(ch, size) for ch in SIZED}
            for ch in SIZED:
                row = dict(stream=stream, size=size, size_label=label, chunker=ch, provenance=PROVENANCE[ch],
                           group=("Steadlith" if ch in STEADLITH else "offset" if ch in ("FIX", "FIXO") else
                                  "history-dependent" if ch in ("KARA", "DIFF[REC]", "DIFF[STRUCT-M]") else "stateless"))
                for p in series[ch]:
                    row[f"achieved_T{p['T']}"] = round(p["achieved"], 1)
                    row[f"cost_T{p['T']}"] = p["obs"]
                y = E[ch]
                row["supported"] = y is not None
                if y is not None:
                    row["cost"] = float(y[0]); row["ci_lo"], row["ci_hi"] = q(y[1:])
                table.append(row)
            for fam_name, fam in (("content-aligned excl. Steadlith", ALIGNED), ("content-aligned incl. STL",
                                  ALIGNED + ("STL",)), ("content-aligned incl. STL and STL+snap", ALIGNED + STEADLITH)):
                mem = [ch for ch in fam if E[ch] is not None]
                Y = np.vstack([E[ch] for ch in mem])
                lo_, hi_ = Y.min(0), Y.max(0)
                spreads.append(dict(stream=stream, size=size, size_label=label, family=fam_name, members=" ".join(mem),
                                    excluded=" ".join(ch for ch in fam if E[ch] is None), n=len(mem),
                                    low=float(lo_[0]), high=float(hi_[0]),
                                    low_cfg=mem[int(Y[:, 0].argmin())], high_cfg=mem[int(Y[:, 0].argmax())],
                                    spread_pp=100 * float(hi_[0] - lo_[0]),
                                    spread_ci=[round(100 * v, 2) for v in q((hi_ - lo_)[1:])],
                                    rel_spread=float(hi_[0] / lo_[0] - 1),
                                    rel_ci=[round(v, 3) for v in q((hi_ / lo_ - 1)[1:])],
                                    fix_over_low=float(E["FIX"][0] / lo_[0]) if E["FIX"] is not None else None,
                                    fix_over_high=float(E["FIX"][0] / hi_[0]) if E["FIX"] is not None else None))
            st = [ch for ch in STATELESS if E[ch] is not None]
            S = np.vstack([E[ch] for ch in st])
            best = st[int(S[:, 0].argmin())]
            if E["KARA"] is not None:
                d_fixed = E["KARA"] - E[best]
                d_min = E["KARA"] - S.min(0)
                leads.append(dict(stream=stream, size=size, size_label=label, best_stateless=best,
                                  kara=float(E["KARA"][0]), best=float(E[best][0]),
                                  lead_pp=-100 * float(d_fixed[0]), lead_ci=[round(-100 * v, 2) for v in q(d_fixed[1:])[::-1]],
                                  lead_min_ci=[round(-100 * v, 2) for v in q(d_min[1:])[::-1]],
                                  p_kara_not_cheaper=float(np.mean(d_fixed[1:] >= 0))))
            for ch in STEADLITH:
                if E[ch] is not None and E["REC"] is not None:
                    r = E[ch] / E["REC"] - 1
                    excess.append(dict(stream=stream, setting=f"{label} size {size}", chunker=ch, rel=float(r[0]),
                                       rel_ci=[round(v, 3) for v in q(r[1:])],
                                       abs_pp=100 * float(E[ch][0] - E["REC"][0])))
        # nominal T=128 Steadlith excess (paired over documents)
        for ch in STEADLITH:
            a = next(p for p in series[ch] if p["T"] == 128); b = next(p for p in series["REC"] if p["T"] == 128)
            r = np.r_[a["obs"], a["boot"]] / np.r_[b["obs"], b["boot"]] - 1
            excess.append(dict(stream=stream, setting="nominal T=128", chunker=ch, rel=float(r[0]),
                               rel_ci=[round(v, 3) for v in q(r[1:])], abs_pp=100 * (a["obs"] - b["obs"])))
        # FIX/REC ratio by document length (nominal T=128, per-document mean cost)
        g = f[f["T"].eq(128) & f.chunker.isin(["FIX", "REC"])].pivot_table(index="doc_id", columns="chunker",
                                                                           values=["cost", "edits", "doc_tok"])
        for lo, hi in ((0, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, 10 ** 9)):
            m = (g["doc_tok"]["REC"] >= lo) & (g["doc_tok"]["REC"] < hi)
            if m.sum() >= 5:
                fx = g["cost"]["FIX"][m].sum() / g["edits"]["FIX"][m].sum()
                rc = g["cost"]["REC"][m].sum() / g["edits"]["REC"][m].sum()
                lengths.append(dict(stream=stream, doc_tokens=f"{lo}-{hi}", docs=int(m.sum()), fix=fx, rec=rc, ratio=fx / rc))
    pd.DataFrame(table).to_csv(out("matched_size_table.csv"), index=False)
    pd.DataFrame(spreads).to_csv(out("matched_size_spreads.csv"), index=False)
    pd.DataFrame(leads).to_csv(out("kara_lead.csv"), index=False)
    pd.DataFrame(excess).to_csv(out("steadlith_excess.csv"), index=False)
    pd.DataFrame(lengths).to_csv(out("fix_ratio_by_length.csv"), index=False)
    print(pd.DataFrame(spreads).to_string())
    print(pd.DataFrame(leads).to_string())
    print(pd.DataFrame(excess).to_string())
    print(pd.DataFrame(lengths).to_string())


# ------------------------------------------------------------------------------------------------------------ tails (C5)
def wquantile(sorted_vals, weights_sorted, qs):
    cw = np.cumsum(weights_sorted)
    tot = cw[-1]
    return np.array([sorted_vals[min(len(sorted_vals) - 1, np.searchsorted(cw, qq * tot))] for qq in qs])


def tails(nboot=400):
    cols = ["stream", "doc_id", "chunker", "T", "rev_new", "ins", "toks", "chunk_tok_mean"]
    chs = ["REC", "STRUCT-M", "SentCDC"]
    raw = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=cols,
                          filters=[("interval", "==", "revision"), ("chunker", "in", chs)])
    rng = np.random.default_rng(SEED + 1)
    rows, share_rows = [], []
    qs = (0.5, 0.95, 0.99)
    for stream in STREAMS:
        f = raw[raw.stream.eq(stream)]
        ids = pd.Index(sorted(f.doc_id.unique()))
        draws = rng.integers(len(ids), size=(nboot, len(ids)))
        counts = np.stack([np.bincount(d, minlength=len(ids)) for d in draws])      # nboot x docs
        for (ch, T), g in f.groupby(["chunker", "T"]):
            pos = ids.get_indexer(g.doc_id)
            row = dict(stream=stream, chunker=ch, T=int(T), edits=len(g),
                       achieved=float(g.chunk_tok_mean.mean()))
            for col in ("ins", "toks"):
                v = g[col].to_numpy(float); o = np.argsort(v, kind="stable"); vs = v[o]; ps = pos[o]
                obs = np.quantile(v, qs, method="inverted_cdf")
                boot = np.array([wquantile(vs, counts[b, ps], qs) for b in range(nboot)])
                for k, qq in enumerate(qs):
                    lab = f"{col}_p{int(qq * 100)}"
                    row[lab] = float(obs[k]); row[lab + "_ci"] = list(q(boot[:, k]))
                row[f"{col}_max"] = float(v.max()); row[f"{col}_mean"] = float(v.mean())
            rows.append(row)
        # share of edits whose INS exceeds the SentCDC bound of the same edit (bound at r=3 and at the T=200 window)
        for suffix, label in (("_r3", "r=3"), ("", "calibrated r (T=200)")):
            e2 = pd.read_parquet(os.path.join(RESULTS, f"e2_thm1_{stream}{suffix}.parquet"),
                                 columns=["doc_id", "rev_new", "bound_ins", "guard", "r"])
            for ch in chs:
                g = f[f.chunker.eq(ch) & f["T"].eq(128)].merge(e2, on=["doc_id", "rev_new"], how="inner")
                over = (g.ins > g.bound_ins).to_numpy(float)
                pos = ids.get_indexer(g.doc_id)
                tot = np.bincount(pos, minlength=len(ids)); ov = np.bincount(pos, weights=over, minlength=len(ids))
                boot = (counts @ ov) / (counts @ tot)
                gf = g[~g.guard]
                share_rows.append(dict(stream=stream, bound=label, r=int(e2.r.iloc[0]), chunker=ch, T=128,
                                       edits=len(g), share_over=float(over.mean()), ci=list(q(boot)),
                                       share_over_covered=float((gf.ins > gf.bound_ins).mean()),
                                       max_excess=int((g.ins - g.bound_ins).max())))
    t = pd.DataFrame(rows)
    t.to_csv(out("tails.csv"), index=False)
    s = pd.DataFrame(share_rows)
    s.to_csv(out("tails_over_bound.csv"), index=False)
    print(t[["stream", "chunker", "T", "achieved", "ins_p95", "ins_p99", "ins_max", "toks_p95", "toks_p99", "toks_max"]].to_string())
    print(s.to_string())


def resync():
    f = os.path.join(OUT, "resync_edits.parquet")
    if not os.path.exists(f):
        raise SystemExit("run experiments/r2_resync.py first")
    d = pd.read_parquet(f)
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for (stream, ch), g in d.groupby(["stream", "chunker"]):
        ids = pd.Index(sorted(g.doc_id.unique()))
        draws = rng.integers(len(ids), size=(400, len(ids)))
        counts = np.stack([np.bincount(x, minlength=len(ids)) for x in draws])
        for scope, h in (("single-region", g[g.n_regions.eq(1)]), ("all edits", g)):
            pos = ids.get_indexer(h.doc_id)
            row = dict(stream=stream, chunker=ch, scope=scope, edits=len(h))
            for col in ("down_ins", "resync_len"):
                v = h[col].to_numpy(float); o = np.argsort(v, kind="stable")
                obs = np.quantile(v, (0.5, 0.95, 0.99), method="inverted_cdf")
                boot = np.array([wquantile(v[o], counts[b, pos[o]], (0.95, 0.99)) for b in range(400)])
                row.update({f"{col}_p50": float(obs[0]), f"{col}_p95": float(obs[1]), f"{col}_p99": float(obs[2]),
                            f"{col}_p95_ci": list(q(boot[:, 0])), f"{col}_p99_ci": list(q(boot[:, 1])),
                            f"{col}_max": float(v.max()), f"{col}_mean": float(v.mean())})
                for k in (2, 5, 10):
                    row[f"{col}_share_gt{k}"] = float((v > k).mean())
                gt = (v > 2).astype(float)
                tot = np.bincount(pos, minlength=len(ids)); hit = np.bincount(pos, weights=gt, minlength=len(ids))
                row[f"{col}_share_gt2_ci"] = list(q((counts @ hit) / (counts @ tot)))
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out("resync_summary.csv"), index=False)
    print(t.to_string())


# --------------------------------------------------------------------------------------------------------- C7 analyses
def rep():
    d = pd.read_csv(os.path.join(RESULTS, "e1_rep.csv"))
    rng = np.random.default_rng(SEED + 3)
    idx = rng.integers(len(d), size=(B, len(d)))
    pairs = [("Bian-STL176/352", "Bian-Recursive400"), ("Bian-Fixed304", "Bian-Recursive400"),
             ("Bian-Fixed304", "Bian-STL176/352"), ("Bian-STL176/352", "LangChain-REC400")]
    rows = []
    for a, b in pairs:
        diff = (d[a] - d[b]).to_numpy()
        boot = diff[idx].mean(1)
        rows.append(dict(a=a, b=b, mean_a=100 * d[a].mean(), mean_b=100 * d[b].mean(), diff_pp=100 * diff.mean(),
                         ci95=[round(100 * v, 2) for v in q(boot)], share_a_gt_b=float((diff > 0).mean()),
                         share_equal=float((diff == 0).mean())))
    info = dict(pairs=len(d), documents=int(d.doc_id.nunique()), tok_new_median=float(d.tok_new.median()),
                rule="one pair per article: articles in salted-hash order (doc_groups); per article, candidate revision "
                     "indices sorted by sha256('bian-rep:<doc>:<i>'); the first of the first 20 candidates whose old and new "
                     "versions both have 1,024-20,000 tokens and differ is taken; stop at 120 pairs",
                comparisons=rows)
    json.dump(info, open(out("rep_gap.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


def timing():
    t = pd.read_csv(os.path.join(RESULTS, "e7_timing.csv"))
    rows = []
    for (ch, reg), g in t.groupby(["chunker", "regime"]):
        v = g.per_1k_edits_s.to_numpy()
        rows.append(dict(chunker=ch, regime=reg, reps=len(v), mean=v.mean(), min=v.min(), max=v.max(),
                         sd=v.std(ddof=1), per_rep=[round(x, 2) for x in g.sort_values("rep").per_1k_edits_s]))
    s = pd.DataFrame(rows)
    piv = t.pivot_table(index=["rep", "regime"], columns="chunker", values="per_1k_edits_s").reset_index()
    ratios = []
    for reg, g in piv.groupby("regime"):
        red = 1 - g["REC"] / g["FIX"]
        ratios.append(dict(regime=reg, reduction_mean_of_totals=float(1 - g["REC"].mean() / g["FIX"].mean()),
                           per_rep=[round(x, 3) for x in red], min=float(red.min()), max=float(red.max())))
    # achieved sizes of the timed configurations: E1 T=200 per-edit chunk sizes on the same 20 WIKI histories
    sys.path.insert(0, os.path.dirname(__file__))
    from common import load_stream, doc_groups
    docs = [str(d) for d, _ in doc_groups(load_stream("wiki", 20, columns=["doc_id", "rev_idx", "ts", "text"]))]
    e1 = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=["stream", "doc_id", "chunker", "T", "chunk_tok_mean"],
                         filters=[("interval", "==", "revision"), ("stream", "==", "wiki"), ("T", "==", 200)])
    e1 = e1[e1.doc_id.isin(docs)]
    sizes = e1.groupby("chunker").chunk_tok_mean.mean().round(1).to_dict()
    res = dict(per_config=s.to_dict("records"), reductions=ratios, timed_docs=len(docs),
               achieved_T200_timed_docs={k: sizes.get(k) for k in ("FIX", "REC", "STRUCT-M", "SentCDC", "KARA", "DOC")},
               contention="shared machine; 3 repetitions in rotated order; load snapshots in e7_timing.csv 'load'")
    json.dump(res, open(out("timing_dispersion.json"), "w"), indent=1, default=float)
    print(s.to_string()); print(ratios); print(res["achieved_T200_timed_docs"])


def kendall(a, b):
    n = len(a); s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


def predict(nboot=1000):
    d = pd.read_parquet(os.path.join(RESULTS, "e2_predict_edits.parquet"))
    nontriv = ["SentCDC", "REC", "STRUCT-M", "SENTWIN"]
    rng = np.random.default_rng(SEED + 4)
    agg = d.groupby(["stream", "chunker", "doc_id"]).agg(pred=("predicted_raw", "sum"), meas=("measured", "sum"),
                                                          n=("measured", "size")).reset_index()
    def tables(stream):
        g = agg[agg.stream.eq(stream)]
        ids = pd.Index(sorted(g.doc_id.unique()))
        P = g.pivot(index="doc_id", columns="chunker", values="pred").reindex(ids)
        M = g.pivot(index="doc_id", columns="chunker", values="meas").reindex(ids)
        N = g.pivot(index="doc_id", columns="chunker", values="n").reindex(ids)
        return ids, P, M, N
    W = tables("wiki")
    res = {"chunker_level": [], "stream_level": [], "doc_level": []}
    for stream in ("k8s", "cpy"):
        T = tables(stream)
        wd = rng.integers(len(W[0]), size=(nboot, len(W[0])))
        td = rng.integers(len(T[0]), size=(nboot, len(T[0])))
        boot_ape = {c: [] for c in nontriv}; taus, md3, md4 = [], [], []
        def compute(wi, ti):
            fac = {c: W[2][c].to_numpy()[wi].sum() / W[1][c].to_numpy()[wi].sum() for c in nontriv}
            pm = {c: T[1][c].to_numpy()[ti].sum() / T[3][c].to_numpy()[ti].sum() * fac[c] for c in nontriv}
            mm = {c: T[2][c].to_numpy()[ti].sum() / T[3][c].to_numpy()[ti].sum() for c in nontriv}
            ape = {c: abs(pm[c] - mm[c]) / mm[c] for c in nontriv}
            tau = kendall([pm[c] for c in nontriv], [mm[c] for c in nontriv])
            return ape, tau
        ape0, tau0 = compute(np.arange(len(W[0])), np.arange(len(T[0])))
        for b in range(nboot):
            ape, tau = compute(wd[b], td[b])
            for c in nontriv:
                boot_ape[c].append(ape[c])
            taus.append(tau); md3.append(np.median([ape[c] for c in ("SentCDC", "REC", "STRUCT-M")]))
            md4.append(np.median([ape[c] for c in nontriv]))
        for c in nontriv:
            res["chunker_level"].append(dict(stream=stream, chunker=c, ape=ape0[c], ci=list(q(boot_ape[c]))))
        vals, cnt = np.unique(np.round(taus, 3), return_counts=True)
        res["stream_level"].append(dict(stream=stream, docs=len(T[0]), mdape3=float(np.median([ape0[c] for c in ("SentCDC", "REC", "STRUCT-M")])),
                                        mdape3_ci=list(q(md3)), mdape4=float(np.median([ape0[c] for c in nontriv])),
                                        mdape4_ci=list(q(md4)), tau=tau0, tau_ci=list(q(taus)),
                                        tau_distribution={str(v): int(k) for v, k in zip(vals, cnt)}))
        # document level: per-document mean prediction (WIKI factor) vs per-document mean measurement
        fac = {c: W[2][c].sum() / W[1][c].sum() for c in nontriv}
        for c in nontriv:
            pm = T[1][c] / T[3][c] * fac[c]; mm = T[2][c] / T[3][c]
            ok = mm > 0
            ape = (abs(pm - mm) / mm)[ok].to_numpy()
            bt = [np.median(ape[rng.integers(len(ape), size=len(ape))]) for _ in range(nboot)]
            res["doc_level"].append(dict(stream=stream, chunker=c, docs=int(ok.sum()), mdape_docs=float(np.median(ape)),
                                         ci=list(q(bt)), share_within_25pct=float((ape <= 0.25).mean())))
    json.dump(res, open(out("predictor_intervals.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------------------------------------------------ C8 minors
def minors():
    res = {}
    # drift cohorts (Fig. 4 vs text): values at t = 25 and 100 for both cohorts, with document counts
    rows = []
    for s in STREAMS:
        d = pd.read_csv(os.path.join(RESULTS, f"e3_drift_{s}.csv"))
        for _, r in d[d.t.isin([25, 100])].iterrows():
            rows.append(dict(stream=s, cohort=r.cohort, chunker=r.chunker, t=int(r.t), docs=int(r.docs),
                             ks=round(r.ks, 3), divergence=round(r.divergence, 3), renorm=round(r.renorm_toks_frac, 3)))
    pd.DataFrame(rows).to_csv(out("drift_cohorts.csv"), index=False)
    res["drift_cohorts"] = "drift_cohorts.csv"
    # sentence-probe design
    p = pd.read_csv(os.path.join(RESULTS, "e3_quality.csv"))
    design = dict(documents=int(p.doc_id.nunique()), probes=int(len(p[["probe", "doc_id", "t", "chunker"]].drop_duplicates())),
                  checkpoints=sorted(int(x) for x in p.t.unique()),
                  docs_at_t={int(t): int(g.doc_id.nunique()) for t, g in p.groupby("t")},
                  probes_per_doc_t=float(p.groupby(["t", "chunker", "doc_id"]).size().median()))
    rng = np.random.default_rng(SEED + 5)
    per_t = []
    for (ch, t), g in p.groupby(["chunker", "t"]):
        g = g.assign(diff=g.maintained - g.fresh)
        dd = g.groupby("doc_id")["diff"].agg(["sum", "size"])
        ii = rng.integers(len(dd), size=(B, len(dd)))
        boot = dd["sum"].to_numpy()[ii].sum(1) / dd["size"].to_numpy()[ii].sum(1)
        per_t.append(dict(chunker=ch, t=int(t), docs=len(dd), probes=int(dd["size"].sum()), diff_pp=100 * g["diff"].mean(),
                          ci=[round(100 * v, 2) for v in q(boot)]))
    design["per_checkpoint"] = per_t
    res["probe_design"] = design
    # TOST for STALE and WINDOW-D against EXACT (clustered by document; 90% and 95% intervals; margin 2 pp)
    pol = pd.read_csv(os.path.join(RESULTS, "e5b_policies.csv"))
    qd = {json.loads(l)["qid"]: json.loads(l)["doc_id"] for l in open(os.path.join(RESULTS, "qa", "questions_wiki_prox.jsonl"))}
    w = pol.pivot(index="qid", columns="policy", values="recall1024")
    w["doc"] = [str(qd.get(x)) for x in w.index]
    tost = []
    for pl in [c for c in w.columns if c not in ("EXACT", "doc")]:
        diff = (w[pl] - w["EXACT"]).astype(float)
        dd = diff.groupby(w["doc"]).agg(["sum", "size"])
        ii = rng.integers(len(dd), size=(B, len(dd)))
        boot = dd["sum"].to_numpy()[ii].sum(1) / dd["size"].to_numpy()[ii].sum(1)
        c90, c95 = q(boot, 0.05, 0.95), q(boot)
        tost.append(dict(policy=pl, questions=int(diff.size), documents=len(dd), diff_pp=100 * diff.mean(),
                         ci90=[round(100 * v, 2) for v in c90], ci95=[round(100 * v, 2) for v in c95],
                         tost_equivalent_2pp=bool(c90[0] > -0.02 and c90[1] < 0.02),
                         discordant=int((diff != 0).sum())))
    res["tost"] = tost
    # Theorem 1: per-edit bound terms (single-zone edits) and edits without any changed region
    th = []
    for s in STREAMS:
        for suffix in ("", "_r3"):
            e = pd.read_parquet(os.path.join(RESULTS, f"e2_thm1_{s}{suffix}.parquet"))
            r = int(e.r.iloc[0]); cov = e[~e.guard]
            one = cov[cov.n_zones.eq(1)]
            hash_term = (one.l_new + 2 * r - 1) // (r + 1) + 2
            anchor = one.bound_ins - hash_term
            nz = cov[cov.n_zones.eq(0)]
            th.append(dict(stream=s, r=r, covered=len(cov), mean_ins=float(cov.ins.mean()), mean_bound=float(cov.bound_ins.mean()),
                           ratio_of_means=float(cov.ins.mean() / cov.bound_ins.mean()),
                           mean_of_ratios=float((cov.ins / cov.bound_ins.where(cov.bound_ins > 0)).mean()),
                           single_zone=len(one), mean_hash_term=float(hash_term.mean()), mean_anchor_term=float(anchor.mean()),
                           share_with_anchor=float((anchor > 0).mean()), no_zone_edits=len(nz),
                           no_zone_ins=int(nz.ins.sum()), l0_with_zone=int(((cov.l_new == 0) & (cov.n_zones > 0)).sum())))
    res["thm1_terms"] = th
    # unit-cost fit diagnostics by batch size (Theil-Sen line from e7_unit_costs.json)
    uc = json.load(open(os.path.join(RESULTS, "e7_unit_costs.json")))
    calls = pd.read_csv(os.path.join(RESULTS, "e7_unit_calls.csv"))
    a0, a = uc.get("alpha0"), uc.get("alpha")
    diag = []
    for size, g in calls.groupby("size"):
        pred = a0 + a * g.tokens
        diag.append(dict(batch=int(size), calls=len(g), median_tokens=float(g.tokens.median()),
                         median_s=float(g.seconds.median()), median_pred_s=float(pred.median()),
                         median_rel_err=float(((pred - g.seconds) / g.seconds).median())))
    res["unit_cost_fit"] = dict(alpha0=a0, alpha=a, by_batch=diag, keys=sorted(uc.keys()))
    # Steadlith workaround shares (E1 per-edit column stl_fallbacks_doc; T=128)
    e1 = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=["stream", "doc_id", "chunker", "T", "stl_fallbacks_doc"],
                         filters=[("interval", "==", "revision"), ("chunker", "in", ["STL", "STL+snap"])])
    wk = []
    for (s, ch, T), g in e1.groupby(["stream", "chunker", "T"]):
        per_doc = g.groupby("doc_id").stl_fallbacks_doc.max()
        wk.append(dict(stream=s, chunker=ch, T=int(T), docs=len(per_doc), docs_with_fallback=int((per_doc > 0).sum()),
                       edits=len(g), edits_in_docs_with_fallback=int(g.doc_id.isin(per_doc[per_doc > 0].index).sum())))
    res["steadlith_workarounds"] = wk
    json.dump(res, open(out("minors.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "probe_design"}, indent=1, default=float)[:6000])
    print(json.dumps({k: v for k, v in design.items() if k != "per_checkpoint"}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = dict(matched=matched, tails=tails, resync=resync, rep=rep, timing=timing, predict=predict, minors=minors)
    if cmd == "tex":
        fns = {}
    for name in (fns if cmd == "all" else ([cmd] if cmd in fns else [])):
        if cmd == "all" and name == "resync" and not os.path.exists(os.path.join(OUT, "resync_edits.parquet")):
            continue
        fns[name]()


# ------------------------------------------------------------------------------------------------ LaTeX table (C3)
def tex():
    """paper/tab_matched.tex from results/round2/matched_size_*.csv (and nominal rows for unmatched configurations)."""
    t = pd.read_csv(os.path.join(OUT, "matched_size_table.csv"))
    sp = pd.read_csv(os.path.join(OUT, "matched_size_spreads.csv"))
    kl = pd.read_csv(os.path.join(OUT, "kara_lead.csv"))
    t = t[t.size_label.eq("common")]
    # Achieved sizes come from the unrounded edit-weighted means: formatting the CSV's one-decimal values again rounded
    # twice (122.506 -> 122.5 -> 122), so six cells disagreed by one token with Fig. 1, which uses the unrounded means.
    ach_e1 = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=["stream", "chunker", "T", "chunk_tok_mean"],
                             filters=[("interval", "==", "revision"), ("T", "in", [128, 200, 320])]
                             ).groupby(["chunker", "stream", "T"]).chunk_tok_mean.mean()
    order = ["FIX", "FIXO", "REC", "STRUCT-M", "SentCDC", "GCDC", "BCDC", "KARA", "DIFF[REC]", "DIFF[STRUCT-M]", "STL", "STL+snap"]
    prov = {"REC": "3rd", "BCDC": "3rd+", "STL": "3rd*", "STL+snap": "3rd*", "KARA": "3rd*"}
    lines = []
    for ch in order:
        cells = [f"\\cn{{{ch}}}"]
        for s in STREAMS:
            r = t[(t.stream == s) & (t.chunker == ch)].iloc[0]
            ach = "/".join(f"{ach_e1.get((ch, s, T), r[f'achieved_T{T}']):.0f}" for T in (128, 200, 320))
            if r.supported:
                cost = f"{100 * r.cost:.1f} [{100 * r.ci_lo:.1f}, {100 * r.ci_hi:.1f}]"
            else:
                cost = "\\multicolumn{1}{c}{---$^\\ddagger$}"
            cells += [ach, cost]
        lines.append(" & ".join(cells) + " \\\\")
    # unmatched configurations (no size parameter): nominal values
    e1 = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=["stream", "chunker", "T", "toks_frac", "chunk_tok_mean"],
                         filters=[("interval", "==", "revision"), ("chunker", "in", ["STRUCT-NM", "SENTWIN"]), ("T", "==", 200)])
    for ch in ("STRUCT-NM", "SENTWIN"):
        cells = [f"\\cn{{{ch}}}$^\\dagger$"]
        for s in STREAMS:
            g = e1[(e1.stream == s) & (e1.chunker == ch)]
            cells += [f"{g.chunk_tok_mean.mean():.0f}", f"{100 * g.toks_frac.mean():.1f} (nominal)"]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")

    def fam(name, label):
        cells = [label]
        for s in STREAMS:
            r = sp[(sp.stream == s) & (sp.size_label == "common") & (sp.family == name)].iloc[0]
            ci = json.loads(r.spread_ci) if isinstance(r.spread_ci, str) else r.spread_ci
            cells += [f"{int(r.n)} cfg", f"{r.spread_pp:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]; {100 * r.rel_spread:.0f}\\%"]
        return " & ".join(cells) + " \\\\"
    lines.append(fam("content-aligned excl. Steadlith", "Spread excl.\\ Steadlith"))
    lines.append(fam("content-aligned incl. STL and STL+snap", "Spread incl.\\ Steadlith"))
    cells = ["\\cn{KARA} lead"]
    for s in STREAMS:
        r = kl[(kl.stream == s) & (kl.size_label == "common")].iloc[0]
        ci = json.loads(r.lead_ci) if isinstance(r.lead_ci, str) else r.lead_ci
        cells += [f"vs.\\ \\cn{{{r.best_stateless}}}", f"{r.lead_pp:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"]
    lines.append(" & ".join(cells) + " \\\\")
    import re
    body = re.sub(r"(?<![\w$])-(\d)", r"$-$\1", "\n".join(lines))
    os.makedirs(os.path.join(P3, "paper"), exist_ok=True)   # the release has no manuscript folder
    with open(os.path.join(P3, "paper", "tab_matched.tex"), "w") as f:
        f.write("% generated by experiments/round2_analysis.py tex from results/round2/matched_size_{table,spreads}.csv,\n"
                "% results/round2/kara_lead.csv and results/e1_edits.parquet (unmatched rows, T=200); do not edit by hand\n")
        f.write(body + "\n")
    print(body)


def tex_tails():
    """paper/tab_tails.tex from results/round2/tails.csv, tails_over_bound.csv and resync_summary.csv."""
    t = pd.read_csv(os.path.join(OUT, "tails.csv"))
    o = pd.read_csv(os.path.join(OUT, "tails_over_bound.csv"))
    r = pd.read_csv(os.path.join(OUT, "resync_summary.csv"))
    L = lambda v: json.loads(v) if isinstance(v, str) else v
    fmt = lambda x: f"{100 * x:.2f}" if 100 * x < 0.1 else f"{100 * x:.1f}"
    lines = []
    for s in STREAMS:
        for k, ch in enumerate(("REC", "STRUCT-M", "SentCDC")):
            a = t[(t.stream == s) & (t.chunker == ch) & (t["T"] == 128)].iloc[0]
            b = o[(o.stream == s) & (o.chunker == ch) & (o.bound == "r=3")].iloc[0]
            c = r[(r.stream == s) & (r.chunker == ch) & (r.scope == "single-region")].iloc[0]
            p99 = L(a.ins_p99_ci); bci = L(b.ci); cci = L(c.resync_len_share_gt2_ci)
            lines.append(" & ".join([s.upper() if k == 0 else "", f"\\cn{{{ch}}}", f"{a.ins_p95:.0f}",
                                     f"{a.ins_p99:.0f} [{p99[0]:.0f}, {p99[1]:.0f}]", f"{a.ins_max:.0f}",
                                     f"{a.toks_p95 / 1000:.1f}", f"{a.toks_p99 / 1000:.1f}", f"{a.toks_max / 1000:.0f}",
                                     f"{fmt(b.share_over)} [{fmt(bci[0])}, {fmt(bci[1])}]", f"{c.resync_len_p99:.0f}",
                                     f"{fmt(c.resync_len_share_gt2)} [{fmt(cci[0])}, {fmt(cci[1])}]",
                                     f"{c.resync_len_max:.0f}"]) + " \\\\")
    os.makedirs(os.path.join(P3, "paper"), exist_ok=True)   # the release has no manuscript folder
    with open(os.path.join(P3, "paper", "tab_tails.tex"), "w") as f:
        f.write("% generated by experiments/round2_analysis.py tex from results/round2/tails.csv, tails_over_bound.csv,\n"
                "% resync_summary.csv; do not edit by hand\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tex":
    tex()
    tex_tails()


# ------------------------------------------------------------------------------- 120-pair samples (R2-M1, evaluative value)
def pairs120(draws=1000):
    """How often does a 120-pair sample (one random edit per random document, as in Bian et al.'s design) reproduce the
    full-stream results at nominal T=128? Reports the share of draws with the full-stream order of STL above REC, with
    a paired 95% interval for STL - REC that excludes 0, and Kendall's tau between sample and full-stream rankings."""
    chs = ["FIX", "REC", "STRUCT-M", "SentCDC", "GCDC", "BCDC", "STL", "KARA"]
    e = pd.read_parquet(os.path.join(RESULTS, "e1_edits.parquet"), columns=["stream", "doc_id", "chunker", "T", "rev_new", "toks_frac"],
                        filters=[("interval", "==", "revision"), ("T", "==", 128), ("chunker", "in", chs)])
    rng = np.random.default_rng(SEED + 6)
    out = []
    for s in STREAMS:
        w = e[e.stream.eq(s)].pivot_table(index=["doc_id", "rev_new"], columns="chunker", values="toks_frac").dropna()
        full = w.mean()
        docs = w.index.get_level_values(0).to_numpy()
        groups = pd.Series(np.arange(len(w))).groupby(docs).apply(list).to_dict()
        keys = list(groups)
        order_ok, sig, taus, gaps = 0, 0, [], []
        W = w[chs].to_numpy()
        for _ in range(draws):
            pick = rng.choice(len(keys), size=min(120, len(keys)), replace=False)
            rows = [rng.choice(groups[keys[i]]) for i in pick]
            m = W[rows].mean(0)
            d = W[rows, chs.index("STL")] - W[rows, chs.index("REC")]
            se = d.std(ddof=1) / np.sqrt(len(d))
            order_ok += m[chs.index("STL")] > m[chs.index("REC")]
            sig += (d.mean() - 1.96 * se) > 0
            gaps.append(100 * d.mean())
            taus.append(kendall(list(m), list(full[chs].to_numpy())))
        out.append(dict(stream=s, draws=draws, full_gap_pp=100 * float(full["STL"] - full["REC"]),
                        share_stl_above_rec=order_ok / draws, share_gap_ci_excludes_0=sig / draws,
                        gap_p5_p95=[float(np.percentile(gaps, 5)), float(np.percentile(gaps, 95))],
                        tau_median=float(np.median(taus)), tau_p5=float(np.percentile(taus, 5)),
                        share_tau_1=float(np.mean(np.isclose(taus, 1.0)))))
    json.dump(out, open(out_path := os.path.join(OUT, "pairs120.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "pairs120":
    pairs120()


# ----------------------------------------------------------- size-matched retrieval: direct paired comparisons (C6, 2c)
def sizepairs(tag="_r2matched", pairs=(("STRUCT-NM", "REC@nm"), ("KARA@match", "KARA"))):
    """Direct paired differences between two arms of the common-background run `tag` under the span, answer and anywhere
    criteria recomputed by experiments/r2_span.py (results/round2/span_metrics.csv): STRUCT-NM against REC at
    STRUCT-NM's realized size (does paragraph structure add anything beyond size?) and KARA at REC's size against nominal
    KARA. Document-clustered bootstrap (2,000 resamples, as in r2_span.py) and exact McNemar tests (no multiplicity
    correction: two pre-specified contrasts). Output: results/round2/size_pairs.csv."""
    from scipy.stats import binomtest
    m = pd.read_csv(os.path.join(OUT, "span_metrics.csv"))
    m = m[m.tag.eq(tag)]
    rng = np.random.default_rng(SEED + 7)
    rows = []
    for a, b in pairs:
        X, Y = m[m.chunker.eq(a)].set_index("qid"), m[m.chunker.eq(b)].set_index("qid")
        common = X.index.intersection(Y.index)
        for scope in ("pooled",) + STREAMS:
            idx = common if scope == "pooled" else common[Y.loc[common, "stream"].eq(scope).to_numpy()]
            for crit in ("span", "answer", "anywhere"):
                x, y = X.loc[idx, crit].to_numpy(), Y.loc[idx, crit].to_numpy()
                g = pd.DataFrame(dict(d=x - y, cl=Y.loc[idx, "doc"].to_numpy())).groupby("cl")["d"].agg(["sum", "size"])
                s, n = g["sum"].to_numpy(), g["size"].to_numpy()
                draws = rng.integers(0, len(s), size=(B, len(s)))
                boots = s[draws].sum(1) / n[draws].sum(1)
                bb, cc = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                rows.append(dict(tag=tag, config=a, reference=b, scope=scope, criterion=crit, n=len(idx),
                                 mean_config=float(x.mean()), mean_reference=float(y.mean()), diff=float(s.sum() / n.sum()),
                                 ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)),
                                 mcnemar_p=binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0, discordant=bb + cc))
    t = pd.DataFrame(rows)
    t.to_csv(out("size_pairs.csv"), index=False)
    print(t.round(4).to_string())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "sizepairs":
    sizepairs()


# --------------------------------------------------------------------------------- band-only ablation table (C4, 2c)
BAND_ROWS = [  # (arm in results/round2/band_ablation.csv, label with cut density, maximum chunk size)
    ("STL-narrow", r"\cn{STL} (default)", r"2$\times$"),
    ("STL+snap-narrow", r"\cn{STL+snap}", r"2$\times$"),
    ("STL-narrow-m6", r"\cn{STL}, dense", r"2$\times$"),
    ("STL-wide", r"\cn{STL}", r"16$\times$"),
    ("STL-wide-m6", r"\cn{STL}, dense", r"16$\times$"),
    ("STL+snap-wide-m6", r"\cn{STL+snap}, dense", r"16$\times$"),
    ("GCDC-narrow", r"\cn{GCDC}, $p{=}0.25$", r"2$\times$"),
    ("GCDC-narrow-p10", r"\cn{GCDC}, $p{=}0.10$", r"2$\times$"),
    ("GCDC-wide", r"\cn{GCDC}, $p{=}0.25$ (E1)", "480"),
    ("GCDC-wide-p10", r"\cn{GCDC}, $p{=}0.10$", "480"),
    ("BCDC-wide", r"\cn{BCDC} (E1)", r"16$\times$"),
]


def tex_band():
    """paper/tab_band.tex rows: cost relative to REC at the common size (%, paired document-bootstrap 95% CI) for every
    arm of results/round2/band_ablation.csv (experiments/r2_band.py); unsupported cells are marked."""
    t = pd.read_csv(os.path.join(OUT, "band_ablation.csv"))
    sign = lambda v: f"{v:+.0f}".replace("-", "$-$")
    num = lambda v: f"{v:.0f}".replace("-", "$-$")
    lines = []
    for arm, label, band in BAND_ROWS:
        cells = []
        for s in STREAMS:
            r = t[(t.stream == s) & (t.arm == arm)]
            if r.empty or not bool(r.supported.iloc[0]):
                cells.append(r"\multicolumn{1}{c}{n/s}")
                continue
            lo, hi = json.loads(r.rel_ci.iloc[0])
            cells.append(f"{sign(100 * r.rel_vs_rec.iloc[0])} [{num(100 * lo)}, {num(100 * hi)}]")
        lines.append(" & ".join([label, band] + cells) + r" \\")
    os.makedirs(os.path.join(P3, "paper"), exist_ok=True)   # the release has no manuscript folder
    with open(os.path.join(P3, "paper", "tab_band.tex"), "w") as f:
        f.write("% generated by experiments/round2_analysis.py tex from results/round2/band_ablation.csv "
                "(experiments/r2_band.py); do not edit by hand\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tex":
    tex_band()
