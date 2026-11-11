"""Paired quality statistics for E4 (retrieval) and E9 (answers): per stream x config mean with document-clustered
bootstrap 95% CI; paired difference vs a reference config with a clustered bootstrap CI, exact McNemar p (Holm-corrected
across configs), and TOST equivalence with the pre-declared margin of 2 percentage points. B5 fix (review finding
B4-TOST: a percentile bootstrap is degenerate when no pair differs): TOST uses the analytic Newcombe (1998, method 10)
score interval for a difference of PAIRED proportions at 90%; equivalence is declared only if that interval lies inside
[-0.02, 0.02], there is at least one discordant pair, and n reaches the size a power calculation requires for the 2 pp
margin (alpha 0.05 per side, power 0.8, observed discordance rate floored at 5%). The clustered bootstrap CI is still
reported; `tost_method`, n and n_required are written per stratum.
usage: python experiments/stats_quality.py e4 [--tag ""] [--metric recall@1024] [--ref REC]
       python experiments/stats_quality.py e9 [--tag ""] [--metric contains] [--ref REC]
"""
import argparse, json, os

import numpy as np
import pandas as pd
from common import RESULTS, run_meta
from scipy.stats import binomtest

MARGIN = 0.02


def clustered(values: pd.Series, clusters: pd.Series, B=2000, seed=0, q=(2.5, 97.5)):
    df = pd.DataFrame({"v": values.values, "c": clusters.values}).groupby("c").v.agg(["sum", "size"])
    s, n = df["sum"].values, df["size"].values; rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(B, len(s))); boots = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), [float(np.percentile(boots, x)) for x in q], boots


def wilson(x, n, z):
    if n == 0:
        return 0.0, 1.0
    p = x / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe_paired(a, b, c, d, z=1.6449):
    """Newcombe's method-10 interval for p1 - p2 with paired binary data; a = both 1, b = x only, c = ref only, d = neither."""
    n = a + b + c + d
    if n == 0:
        return float("nan"), float("nan")
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z); l2, u2 = wilson(a + c, n, z)
    prod = (a + b) * (c + d) * (a + c) * (b + d)
    if prod == 0:
        phi = 0.0
    elif a * d > b * c:
        phi = max(a * d - b * c - n / 2, 0) / np.sqrt(prod)
    else:
        phi = (a * d - b * c) / np.sqrt(prod)
    diff = p1 - p2
    dl = np.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = np.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return float(diff - dl), float(diff + du)


def n_required(p_disc, margin=MARGIN, za=1.6449, zb=0.8416):
    """Pairs needed for TOST of paired proportions with true difference 0 (Connor-style, power 0.8)."""
    return int(np.ceil((za + zb) ** 2 * max(p_disc, 0.05) / margin ** 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("exp", choices=["e4", "e9"]); ap.add_argument("--tag", default="")
    ap.add_argument("--metric", default=None); ap.add_argument("--ref", default="REC"); a = ap.parse_args()
    if a.exp == "e4":
        D = pd.read_csv(os.path.join(RESULTS, f"e4_quality{a.tag}.csv")); cfg = "chunker"; metric = a.metric or "recall@1024"
    else:
        D = pd.read_csv(os.path.join(RESULTS, f"e9_answers{a.tag}.csv")); cfg = "config"; metric = a.metric or "contains"
    # cluster = document of the question (qid -> doc via questions files)
    qdoc = {}
    for f in os.listdir(os.path.join(RESULTS, "qa")):
        if f.startswith("questions_") and f.endswith(".jsonl"):
            for l in open(os.path.join(RESULTS, "qa", f)):
                q = json.loads(l); qdoc[q["qid"]] = f"{q['stream']}:{q['doc_id']}"
    D["cluster"] = D.qid.map(qdoc).fillna(D.qid)
    rows = []
    for s, g in list(D.groupby("stream")) + [("all", D)]:
        ref = g[g[cfg] == a.ref].set_index("qid")[metric]
        for c, h in g.groupby(cfg):
            m, ci, _ = clustered(h[metric], h.cluster)
            row = dict(stream=s, config=c, n=len(h), mean=m, ci_lo=ci[0], ci_hi=ci[1])
            if c != a.ref and len(ref):
                x = h.set_index("qid")[metric]; common = x.index.intersection(ref.index)
                d = (x[common] - ref[common]); cl = h.set_index("qid").cluster[common]
                dm, dci, boots = clustered(d, cl, q=(2.5, 97.5))
                lo90, hi90 = np.percentile(boots, 5), np.percentile(boots, 95)
                b, cc = int(((x[common] == 1) & (ref[common] == 0)).sum()), int(((x[common] == 0) & (ref[common] == 1)).sum())
                aa = int(((x[common] == 1) & (ref[common] == 1)).sum()); dd = int(((x[common] == 0) & (ref[common] == 0)).sum())
                p = binomtest(b, b + cc, 0.5).pvalue if b + cc else 1.0
                lo, hi = newcombe_paired(aa, b, cc, dd); n = len(common); nreq = n_required((b + cc) / max(1, n))
                eq = bool(lo > -MARGIN and hi < MARGIN and (b + cc) > 0 and n >= nreq)
                row.update(diff=dm, diff_ci_lo=dci[0], diff_ci_hi=dci[1], mcnemar_p=p, discordant=b + cc, n_pairs=n,
                           tost90_lo=lo, tost90_hi=hi, n_required=nreq, tost_equivalent=eq,
                           tost_method="Newcombe paired score 90% + discordance>0 + n>=n_required",
                           boot90_lo=float(lo90), boot90_hi=float(hi90),
                           # minimum detectable effect (two-sided alpha 0.05, power 0.8): 2.80 x clustered-bootstrap SE of
                           # the paired difference; unclustered McNemar approximation alongside (discordance floored at 1/n)
                           mde_boot=float(2.8016 * boots.std(ddof=1)),
                           mde_mcnemar=float(2.8016 * np.sqrt(max(b + cc, 1) / max(1, n)) / np.sqrt(max(1, n))))
            rows.append(row)
    R = pd.DataFrame(rows)
    for s, g in R.groupby("stream"):   # Holm across configs within a stream
        idx = g.dropna(subset=["mcnemar_p"]).sort_values("mcnemar_p").index
        k = len(idx); prev = 0
        for i, j in enumerate(idx):
            prev = max(prev, min(1.0, (k - i) * R.loc[j, "mcnemar_p"])); R.loc[j, "p_holm"] = prev
    R.to_csv(os.path.join(RESULTS, f"{a.exp}_stats{a.tag}.csv"), index=False)
    json.dump(run_meta(metric=metric, ref=a.ref, margin=MARGIN), open(os.path.join(RESULTS, f"{a.exp}_stats{a.tag}.meta.json"), "w"), indent=1)
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
